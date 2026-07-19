// Off-screen blur passes for the RFC-0023 backdrop pipeline: one 21-tap 1-D
// Gaussian along `dir`, run twice (horizontal then vertical — the classic
// separable pair) into a downsampled scratch target.
//
// `sigma` follows the CSS `backdrop-filter: blur(N)` convention the RFC cites
// as its inspiration: the `blur` prop *is* the Gaussian σ. Taps span ±2.5σ
// (99% of the kernel's mass) at a spacing of σ/4; the encoder adaptively
// deepens the downsample until σ, expressed in destination texels, keeps
// that spacing ≤ 2 — sparse taps at high resolution read as N overlaid
// copies of the image ("double vision"), not a blur, so resolution is
// traded instead: the composite's bilinear upscale hides it (the RFC's own
// resolution rationale). Weights are computed in-shader and normalised, so
// one pipeline covers every radius with a constant tap count. The first
// pass also performs the downsample for free (its destination is the scaled
// target; the filtering sampler minifies the source).

struct BlurParams {
    // 1 / destination (scaled) texture size.
    texel: vec2<f32>,
    // Blur direction: (1,0) or (0,1).
    dir: vec2<f32>,
    // Gaussian σ in destination-scaled texels (encoder-capped ≤ 8).
    sigma: f32,
    _pad0: f32,
    _pad1: vec2<f32>,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_smp: sampler;
@group(0) @binding(2) var<uniform> params: BlurParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle from the vertex index alone — no vertex buffer.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    // Flip Y: framebuffer row 0 is the top, NDC +1 is the top.
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sigma = max(params.sigma, 0.5);
    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var i = -10; i <= 10; i = i + 1) {
        let x = f32(i) / 10.0 * 2.5 * sigma;
        let w = exp(-(x * x) / (2.0 * sigma * sigma));
        let off = params.dir * params.texel * x;
        acc = acc + textureSampleLevel(src_tex, src_smp, in.uv + off, 0.0) * w;
        wsum = wsum + w;
    }
    return acc / wsum;
}
