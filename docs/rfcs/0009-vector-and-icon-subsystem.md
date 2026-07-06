# RFC-0009: Vector and Icon Subsystem (MSDF, JIT-Dev / AOT-Release Pipeline)

- **Status:** Draft
- **Author(s):** Briany4717
- **Created:** 2026-06-24
- **Last updated:** 2026-06-26
- **Depends on:**
  - RFC-0001 (Core Architecture — zero-GC model, `wgpu` pipelines, four
    subsystems, `frame.rs` boundary, thread layout)
  - RFC-0002 (`byld` Language — Lume surface, automatic reactivity)
  - RFC-0003 (Interactive Events and View Mutation — reflected props, input)
  - RFC-0005 (Intrinsic View Catalog — closed catalog, agnosticism contract)
  - RFC-0006 (`byard` CLI and Dev Runner — file watcher, tick ordering, `.byard/`)
  - RFC-0007 (User-View Instantiation — design-system wrappers are user `View`s)
  - RFC-0008 (Package Ecosystem — asset distribution engine policy)
- **Amends:**
  - RFC-0001 §3.1 (the four-pipeline table becomes five — see
    `0001-erratum-vector-pipeline.md`)
  - RFC-0005 §4 (adds the twelfth intrinsic, `VectorIcon`; closes the
    "Image source type" open item)

> **Editorial note.** An earlier draft of this RFC used the coinage *"MSSDF"* and
> placed field generation on the Tokio I/O pool, GPU uploads on background
> workers, and a `discard` in the fragment shader. Those four points conflicted
> with RFC-0001 (§5.1 pool semantics, §9 `frame.rs` boundary, §3.1 TBDR
> targeting). This revision adopts the corrected design: the standard term
> **MSDF**, generation on a **CPU pool**, atlas uploads that **cross `frame.rs`**,
> and a **`discard`-free** mobile fragment path. The corrections and their
> rationale are tracked as IMPL-58..61 in `support/DESICIONS.md` and specified for
> implementation in `support/IMPLEMENTATION_5.md` §2.

---

## Summary

This RFC specifies Byard's vector and icon subsystem: a fifth, specialised render
pipeline (`VectorMSDF`) that samples a **multi-channel signed distance field
(MSDF)** atlas to draw monochrome icons that stay perfectly crisp at any scale,
plus a two-phase asset pipeline that gives developers raw-SVG ergonomics with
native execution. In **development** (`byard dev`), raw `.svg` files are
intercepted on save, validated, and compiled to MSDF fields on a CPU worker pool,
then blitted into a transient GPU atlas — all without ever blocking the render or
logic threads. In **production** (`byard build`), an AOT pass tree-shakes the
module graph, packs only the icons actually instantiated into one immutable
high-density atlas, and inlines their coordinates as a static `[Rect; N]` table.
The engine core remains strictly design-agnostic: it consumes opaque asset handles
and knows nothing of "search" or "home". Design systems (`byard-material`,
`byard-cupertino`) ship as downstream packages (RFC-0008) that map typed symbols
to the `VectorIcon` primitive.

## Motivation

Existing vector and icon pipelines carry structural overheads Byard rejects:

**Redundant CPU rasterisation.** Drawing raw SVG at runtime forces the CPU to
interpret Bézier paths and fill rules every frame, destroying the frame budget.
Baking PNGs at a target size instead produces blurry, aliased edges during fluid
scale or rotation (the M3-Expressive motion language depends on exactly these
transitions).

**VRAM density explosion.** Storing distinct bitmaps per display density or per
dynamic scale value bloats video memory and saturates the PCIe bus with constant
texture uploads.

**Core bloat and coupling.** Embedding a design-specific icon library (e.g.
Material) into the compiler or runtime adds weight and breaks the agnosticism a
systems framework should preserve.

MSDF solves the first two. By storing contour distance across the R, G, and B
channels, it preserves sharp orthogonal corners under hardware linear filtering —
the corners that a classic single-channel SDF rounds off when magnified — using
only a trivial per-fragment median. The transform from Bézier paths to a field
happens **once** in Dev (and **zero** times in Release); the GPU then handles
arbitrary scale/rotation at the cost of a flat textured quad. The third problem —
coupling — is solved by keeping the core's vocabulary at the level of asset
handles and pushing all design-system semantics into downstream packages.

## Guide-level explanation

### Agnostic usage in `.byd`

The core exposes one low-level intrinsic, `VectorIcon`. It consumes an asset
handle (or relative path resolved to one); it has no semantic notion of what the
icon depicts:

```byld
View SettingsButton() {
    var active = false

    VectorIcon(asset("icons/gear.svg")) #[size: 24, color: 0xFFFFFF]
}
```

`size` drives Taffy geometry; `color` is the single tint the MSDF shader applies.
Animating `size` from `16` to `128` re-samples the **same** field at a larger
screen rect — infinitely crisp, no new texture, no CPU work.

### The ecosystem model (`byard-material` / `byard-cupertino`)

To build stylised apps without authoring raw vectors, developers use ecosystem
packages (RFC-0008) that declare their assets in their own manifest and expose
clean, typed wrappers — themselves ordinary user `View`s (RFC-0007):

```byld
use material as m

View Dashboard() {
    Column #[gap: 12] {
        m.Icon(.search) #[variant: .rounded, size: 24, color: env.theme.primary]
        m.Icon(.home)   #[variant: .sharp, size: 24]
    }
}
```

The `material` package maps the `.search` token + `variant` to an internal asset
handle, which lowers to `VectorIcon`. The developer gets impeccable DX and LSP
auto-completion; the engine core stays entirely decoupled — it never contains the
strings `"search"` or `"material"`.

### Dev runner vs. release mechanics

Running `byard dev` and saving an SVG:

1. The watcher intercepts the raw SVG; a CPU worker validates and generates the
   MSDF field in milliseconds (off the render and logic threads).
2. The finished field crosses back to the logic thread, which reserves an atlas
   slot and records a partial-upload command on the next frame; the render thread
   performs the single `write_texture` and the glyph appears.
3. Until the field is resident, the call site renders a zero-opacity placeholder
   of the correct size, so the frame ships on time (no stall).

Running `byard build`, the compiler analyses exactly which icons are instantiated.
Even if `byard-material` ships 6,000 icons, an app that instantiates 5 bakes a
consolidated atlas with exactly those 5. VRAM overhead is the theoretical minimum;
boot copies one texture and indexes a static table — no parsing, no runtime
coordinate math.

## Reference-level explanation

### 1. The `VectorMSDF` pipeline

A fifth specialised pipeline joins the Encoder subsystem
(`byard-core/src/encoder/vector_msdf.rs`), compiled at engine init alongside the
existing four (RFC-0001 §3.1), inside the same §8 `push_error_scope` window. On
failure it returns `ByardError::PipelineCompilation { pipeline: "VectorMSDF",
reason }` — no panic, no software fallback (RFC-0001 §8).

A render instance is an immutable data unit defined in `frame.rs` (the only
cross-subsystem boundary, RFC-0001 §9):

```rust
pub struct VectorInstance {
    pub atlas_uv_rect: Rect,      // UV within the MSDF atlas
    pub screen_rect:   Rect,      // Taffy-resolved geometry (logical px)
    pub color:         [f32; 4],  // normalized RGBA tint
    pub px_range:      f32,       // distance range baked at generation (§5)
    pub atlas_layer:   u32,       // array-texture layer
}
```

`RenderFrame` gains `vector_instances: Vec<VectorInstance>` and an
`atlas_uploads: Vec<AtlasUpload>` queue (§3). The **same** `VectorInstance` shape
is produced by both the Dev JIT and the AOT path, so the render code is identical
in both modes (the dev/prod parity invariant).

#### The precision fragment shader

The fragment stage reconstructs the contour from the median of the three channels
and anti-aliases analytically from a baked distance range. It outputs
**premultiplied alpha** and contains **no `discard`** on the default path, because
RFC-0001 §3.1 targets TBDR mobile GPUs where `discard` defeats early-Z:

```wgsl
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let msd = textureSample(t_msdf, s_msdf, in.uv).rgb;

    // Median of the three channels (sharp-corner reconstruction)
    let sd = max(min(msd.r, msd.g), min(max(msd.r, msd.g), msd.b)) - 0.5;

    // Screen-space coverage from the baked px_range
    let unit_range = vec2<f32>(in.px_range) / vec2<f32>(textureDimensions(t_msdf, 0));
    let screen_tex_size = vec2<f32>(1.0) / fwidth(in.uv);
    let screen_px_range = max(0.5 * dot(unit_range, screen_tex_size), 1.0);

    let opacity = clamp(sd * screen_px_range + 0.5, 0.0, 1.0);
    let a = in.color.a * opacity;
    return vec4<f32>(in.color.rgb * a, a);   // premultiplied; no discard
}
```

A desktop-only opaque-pass feature flag may re-enable `discard` for zero-coverage
fragments, but it is never in the default mobile path. The per-fragment cost
matches a flat textured quad, preserving Byard's frame-time guarantees.

### 2. JIT vector compiler (`byard dev`)

The generator is **compiler-side** (`byard-compiler/src/vector/`), never in
`byard-core` — the core only consumes a finished field via `frame.rs` (RFC-0001
§9 dependency direction). Evaluating a `VectorIcon` during dev runs:

```
[Logic Thread] ── Icon evaluated ── resident in atlas cache?
                                          │
        ┌─────────────────────────────────┴───────────────────┐
     [YES]                                                    [NO]
        │                                                       │
  Read UV mapping                                  Emit zero-opacity placeholder
  Emit VectorInstance                              (frame still ships — no stall)
                                                          │
                                                  Dispatch one-shot task (dedup):
                                                    Tokio I/O reads SVG bytes
                                                          │
                                                    CPU pool (rayon /
                                                    spawn_blocking):
                                                      usvg parse → validate
                                                      → MSDF generate
                                                          │
                                                    [crossbeam channel]
                                                          │
                                                  Logic thread: reserve UV slot,
                                                    record AtlasUpload on frame
                                                          │
                                                  Render thread: one
                                                    Queue::write_texture, then draw
```

Two properties are load-bearing and were the editorial corrections above:

- **Generation runs on a CPU pool, not the Tokio I/O pool** (RFC-0001 §5.1 reserves
  Tokio for async I/O; field math is CPU-bound). Only the SVG file read uses Tokio.
- **Background workers never touch the `wgpu::Queue`** (RFC-0001 §9). The worker
  emits an owned `MsdfGlyph`; the logic thread records an `AtlasUpload`; the render
  thread alone uploads. This preserves the single-atomic-swap, no-shared-GPU-state
  guarantee.

#### Structural complexity guardrail (hard error)

To keep the JIT bounded, generation validates the parsed tree first:

```rust
fn validate_vector_complexity(svg: &usvg::Tree) -> Result<(), CompileError> {
    let mut total_nodes = 0;
    for node in svg.root.descendants() {
        if let usvg::NodeKind::Path(ref path) = *node.borrow() {
            total_nodes += path.data.len();
            // The MSDF pipeline supports monochromatic paths only.
            if path.paint.has_gradients_or_filters() {
                return Err(CompileError::SvgUnsupportedFeatures { span: path.span });
            }
        }
    }
    if total_nodes > MAX_NODES {              // default 500 (IMPL-62)
        return Err(CompileError::SvgTooComplexForMssdf { found_nodes: total_nodes });
    }
    Ok(())
}
```

An SVG with gradients/filters, or one exceeding `MAX_NODES` path segments, is a
hard `CompileError`, signalling the developer to route the asset through the
fallback `TextureSampler` image pipeline (RFC-0005 `Image`) instead. (The error
*name* `SvgTooComplexForMssdf` is retained for source compatibility with the
original draft; its message says "MSDF".)

### 3. Dev atlas: uploads, growth, eviction

The transient dev atlas is an array-texture managed by a shelf/skyline allocator
on the logic thread. Uploads cross `frame.rs` as data:

```rust
pub struct AtlasUpload { pub layer: u32, pub rect: Rect, pub bytes: Box<[u8]> }
```

When the atlas fills, the allocator adds an array layer up to a cap; past the cap
it **LRU-evicts** the least-recently-sampled glyph and reuses its cell. An evicted
handle falls back to the placeholder + regenerate-on-next-use path, so eviction is
never visible beyond a one-frame placeholder. (This closes a gap in the original
draft, which specified the blit but not the full-atlas behaviour — IMPL-64.)

On hot-reload, a saved `.svg` invalidates its cache entry; if the new field fits
the same cell the UV slot is reused and only the texels change — existing
`VectorInstance`s are untouched, so a size animation in flight stays crisp and the
consuming `View` does not remount.

### 4. Radical AOT packing (`byard build`)

1. The compiler performs a complete static traversal of the resolved module graph
   (RFC-0007 instantiation graph + RFC-0008 imports), collecting every
   `VectorIcon(asset(LITERAL))` reference.
2. **Dynamic-reference guard.** If an `asset(...)` argument is not a
   compile-time-constant handle (derived from a `var` or a runtime `match`), the
   set cannot be statically closed → `CompileError::VectorAssetNotStatic { span }`
   with a fix-it: make it literal, or declare an explicit inclusion list in
   `byard.toml` (`[assets.vectors] include = [...]`). A partial atlas is never
   shipped silently. (Gap in the original draft — IMPL-65.)
3. Dormant assets in the local package cache (`.byard/packages/`) without a live
   reference are stripped entirely.
4. The closed set is generated (reusing the §2 generator), deduplicated, and packed
   via **MaxRects bounding-box packing** into one immutable array-texture.
5. The coordinate lookup is emitted as a fixed-size `static VECTOR_ATLAS: [Rect; N]`
   (with `px_range` and `layer`), inlined into the Phase-4 transpile target
   (RFC-0001 §7.3). Boot uploads exactly one texture and indexes the table — no
   dynamic parsing, no runtime coordinate math, zero overhead.

### 5. Generation parameters

The default generation grid is **32×32** with a baked `px_range` of **4** (the
`px_range` flows into `VectorInstance` for the §1 AA formula); high-PPI targets may
opt into 64×64. Edge-coloring uses an angular threshold of **48°** to separate
channels at sharp joints without wobble under extreme magnification. These defaults
are tunable (IMPL-62) and validated by determinism and sharp-corner tests
(`IMPLEMENTATION_5.md` M45).

A generated field is content-addressed and may be persisted to
`.byard/cache/vectors/` keyed by `hash(svg ‖ grid ‖ px_range ‖ generator-version)`,
so subsequent cold `byard dev` starts skip regeneration entirely (IMPL-68). The key
includes the generator version, so a toolchain bump invalidates the cache safely.

## Drawbacks

- **Monochromatic constraint.** The MSDF shader applies a single tint per instance.
  Multicolour vector art, gradients, and drop-shadows are excluded from this
  high-performance path; they route as flat objects through `TextureSampler` (or,
  later, the `ComputePath` pipeline). This is an explicit, validated boundary, not
  a silent degradation.
- **First-use JIT latency (Dev only).** The first mount of an uncached SVG runs the
  field transform on a worker before the glyph is valid on the GPU, introducing a
  multi-millisecond delay on the async path. The render loop stays hot (placeholder
  until the upload lands), and the persistent cache (§5) removes the cost on
  subsequent starts.
- **Generator mathematical complexity.** Transforming arbitrary cubic Bézier paths
  into continuous multi-channel distance fields — with correct channel coloring at
  sharp joints — is a steep, well-isolated maintenance burden on the compiler crate.
  It is mitigated by leaning on an existing generator (the `msdf` crate, IMPL-63)
  and a strict determinism test harness.

## Rationale and alternatives

**Why MSDF over a CPU-rasterised fallback (`resvg` + `tiny-skia`)?** Software
rasterisation destroys the frame budget during scale animation: the logic thread
would rebuild the bitmap at every intermediate dimension to avoid aliasing,
choking the PCIe bus. MSDF transforms once in Dev (zero times in Release) and lets
the GPU handle arbitrary transforms at flat-quad cost.

**Why not full compute-shader path rendering (e.g. Vello)?** Analytical path
evaluation via compute shaders is the state of the art for complex, dynamic vector
illustration, but it requires multipass pipelines, raises the `wgpu` feature floor,
and adds GPU cost that is unnecessary for simple monochrome UI icons. MSDF yields
the same low-VRAM, infinite-sharpness properties for icon metrics while keeping
lower-tier hardware targets viable. Compute-path rendering remains a future
companion pipeline (`ComputePath`), not a replacement.

**Why a CPU pool instead of the Tokio pool for generation?** RFC-0001 §5.1 reserves
Tokio for async I/O whose results return to the logic thread; field generation is
CPU-bound and would starve the runtime's workers. The split (Tokio for the file
read, a CPU pool for the math) keeps both pools doing what they are for.

**Why route uploads through `frame.rs` instead of a worker-side `write_texture`?**
RFC-0001 §9 makes `frame.rs` the only cross-subsystem boundary and §5.1 gives the
render thread sole ownership of the GPU. Modelling the blit as `AtlasUpload` data
on the frame preserves the single-atomic-swap guarantee and keeps the worker pure.

**Why a fifth pipeline rather than extending `TextureSampler`?** RFC-0001 §3.1's
thesis is small, specialised pipelines over an über-shader (TBDR bandwidth). MSDF
sampling needs its own fragment math (median + analytic AA); folding it into
`TextureSampler` would branch the hot path. A dedicated pipeline keeps each draw
arm branch-free.

## Prior art

- **`msdfgen` (Viktor Chlumský, 2015).** The originating algorithm and reference
  implementation for multi-channel signed distance fields; the median-of-three
  reconstruction and the `px_range` AA model come directly from it.
- **`glyphon` / `cosmic-text`.** Byard already uses a glyph atlas for `TextGlyph`
  (RFC-0001 §3.1); the MSDF atlas is the same atlas discipline applied to vectors.
- **Flutter `VectorGraphics` / `flutter_svg`.** Demonstrates the CPU-rasterisation
  cost this RFC avoids; baked-raster alternatives demonstrate the aliasing cost.
- **Vello / `piet-gpu`.** The state-of-the-art compute-shader path renderer;
  informs the deferred `ComputePath` pipeline and the decision not to require it
  for icons.
- **Material Symbols.** The 6,000-glyph variable icon font that motivates the
  tree-shaking AOT pass and the typed downstream-package model.

## Unresolved questions

- **Before merge:**
  - [ ] **Generation grid size.** Ratify the 32×32 default vs. 64×64 for high-PPI
    (8K) precision against dev-mode memory (IMPL-62).
  - [ ] **Edge-coloring threshold.** Confirm 48° for the channel-separation angle
    so sharp joints never wobble under extreme scaling (IMPL-62).
  - [ ] **Generator dependency.** `msdf` crate vs. a vendored generator, pending a
    licence/quality review (IMPL-63).
- **During implementation:**
  - [ ] **Dev-atlas cap and eviction tuning.** Layer count and LRU threshold
    before eviction visibly churns on a large icon set (IMPL-64).
  - [ ] **Inclusion-list ergonomics.** Final `byard.toml` `[assets.vectors]` shape
    for the dynamic-reference escape hatch (IMPL-65).
  - [ ] **Persistent cache pruning.** On-disk size bound and prune policy for
    `.byard/cache/vectors/` (IMPL-68).

## Future possibilities

- **`ComputePath` pipeline (Phase N).** An analytical compute-shader companion for
  multicolour layouts and arbitrary SVG paths without precomputation bounds.
- **CLI library pre-compilation.** Let design-system authors pre-bake fields before
  publishing (RFC-0008), removing SVG compilation from the end-user's machine
  entirely.
- **Reactive path morphing.** Allow `byld` views to pass reactive bindings into the
  path coordinate buffers, updating the field across thread channels for native
  vector morphing (depends on RFC-0002 D8 reactive styles + `ComputePath`).
