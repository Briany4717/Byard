# Erratum to RFC-0001: the render pipeline set becomes five (adds `VectorMSDF`)

- **Status:** Active erratum (amends, does not replace, RFC-0001)
- **Author(s):** Briany4717
- **Created:** 2026-06-26
- **Applies to:** RFC-0001 §3.1 ("Multi-pipeline architecture"), and any RFC-0001
  prose asserting that the renderer has exactly four pipelines.
- **Authority:** RFC-0009 (Vector and Icon Subsystem).

---

## Why this erratum exists

RFC-0001 §3.1 enumerated **four** specialised render pipelines (`SolidBox`,
`DecoratedBox`, `TextGlyph`, `TextureSampler`) and its surrounding prose treats
that set as fixed. RFC-0009 introduces a **fifth** pipeline, `VectorMSDF`, to draw
resolution-independent monochrome icons from a multi-channel signed distance field
atlas. RFC-0001 is **Active**, so rather than edit its body (which would erase the
record of the original four-pipeline design), this erratum records the addition and
the single sentence it supersedes. Where RFC-0001's "four pipelines" prose and this
erratum disagree, **this erratum and RFC-0009 win.**

This erratum changes **the pipeline count only**. Every other claim in RFC-0001 §3
stands unchanged: the rationale for small, specialised pipelines over an
über-shader (TBDR bandwidth); the Z-bin batching model (§3.2); the dirty-rectangle
scissor-clipping model (§3.3); and the init-time GPU error boundary (§8). The new
pipeline is added *within* those rules — it is exactly the kind of small,
specialised pipeline §3.1 already argues for.

---

## Correction

### C1 — The pipeline table gains a fifth row

RFC-0001 §3.1's table is amended to:

| Pipeline | Draws |
|----------|-------|
| `SolidBox` | Axis-aligned rectangles with solid fill |
| `DecoratedBox` | Rectangles with `border-radius`, gradients, box-shadows |
| `TextGlyph` | Text via a `glyphon` glyph atlas |
| `TextureSampler` | UV-mapped quads (images, raster icons) |
| **`VectorMSDF`** | **Monochrome vector icons via a multi-channel signed distance field atlas (RFC-0009)** |

Any RFC-0001 sentence reading "exactly four pipelines" / "the renderer pipelines
are fixed at four" is superseded by "**five** pipelines"; the set remains closed —
adding a sixth still requires an RFC and an erratum.

### C2 — The §9 crate layout comment is widened

RFC-0001 §9 sketches `encoder/` as owning "(SolidBox, DecoratedBox, TextGlyph,
TextureSampler)". Read it as additionally owning `vector_msdf` (the new
`encoder/vector_msdf.rs` + `vector_msdf.wgsl`). The dependency graph is unchanged:
`encoder` still depends only on `frame.rs`; the MSDF *generator* lives on the
compiler side (`byard-compiler/src/vector/`) and never enters `byard-core`
(RFC-0001 §9 dependency direction; RFC-0009 §2).

### C3 — The §8 error boundary covers the fifth pipeline

`VectorMSDF` is compiled at init inside the same `Device::push_error_scope` /
`pop_error_scope` window as the other four. A failure returns
`ByardError::PipelineCompilation { pipeline: "VectorMSDF", reason }`. No panic, no
software fallback (RFC-0001 §8 unchanged).

---

## What does *not* change

- **The boundary.** `frame.rs` remains the only cross-subsystem boundary. The new
  `VectorInstance` and `AtlasUpload` types are defined there alongside
  `BoxInstance` / `TextureSampler` / `TextLine`; no subsystem reaches across
  (RFC-0001 §9; RFC-0009 §1, §3).
- **The thread layout.** The render thread still owns the GPU and never blocks;
  the logic thread still owns mutation; the Tokio pool is still async-I/O-only.
  MSDF field generation runs on a separate CPU pool, and atlas uploads cross
  `frame.rs` as data — both consistent with RFC-0001 §5.1 (RFC-0009 §2).
- **The hard-GPU requirement.** No software vector rasteriser is added to the
  runtime path; the engine still requires a `wgpu`-compatible GPU (RFC-0001 §8).

---

## Summary table

| RFC-0001 as written | Corrected | Authority |
|---|---|---|
| Four render pipelines | **Five** render pipelines (`+ VectorMSDF`) | RFC-0009 C1 |
| `encoder/` owns four shader modules | owns **five** (`+ vector_msdf`) | RFC-0009 C2 |
| Init error scope wraps four pipelines | wraps **five** | RFC-0009 C3 |
| `frame.rs` is the only cross-subsystem boundary | unchanged | — |
| Render thread owns the GPU; Tokio is I/O-only | unchanged | — |
