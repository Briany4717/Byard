//! GPU readback proofs for the `CanvasShape` pipeline (RFC-0020 Tier 1).
//!
//! Renders synthetic frames of programmatic shapes and reads pixels back, so
//! the analytic-SDF fragment shader's stroke/fill/sweep behaviour is pinned
//! down on a real device — the CPU-mirror tests in `byard-core` cover the
//! same geometry deterministically without a GPU.
//!
//! Skips cleanly when no GPU adapter is available.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use byard_core::encoder::EncoderSubsystem;
use byard_core::frame::{
    CANVAS_CAP_ROUND, CANVAS_SHAPE_ARC, CANVAS_SHAPE_CIRCLE, CANVAS_SHAPE_RECT, CanvasShape,
    RenderFrame, Viewport,
};
use std::sync::Arc;

fn try_device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("canvas-shape readback device"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        ..Default::default()
    }))
    .ok()?;
    Some((Arc::new(device), Arc::new(queue)))
}

/// A read-back framebuffer: physical-pixel BGRA bytes plus the row stride.
struct Readback {
    data: Vec<u8>,
    bpr: u32,
    scale: f32,
}

impl Readback {
    /// Samples the BGRA pixel at a *logical* coordinate as `(b, g, r, a)`.
    fn at(&self, lx: f32, ly: f32) -> (u8, u8, u8, u8) {
        let px = (lx * self.scale) as u32;
        let py = (ly * self.scale) as u32;
        let idx = (py * self.bpr + px * 4) as usize;
        (
            self.data[idx],
            self.data[idx + 1],
            self.data[idx + 2],
            self.data[idx + 3],
        )
    }
}

/// Encodes `frame` into an off-screen target and reads the whole thing back.
fn render(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    frame: &RenderFrame,
    logical_w: f32,
    logical_h: f32,
) -> Readback {
    let scale = 2.0_f32;
    let phys_w = (logical_w * scale) as u32;
    let phys_h = (logical_h * scale) as u32;
    let fmt = wgpu::TextureFormat::Bgra8UnormSrgb;
    let mut enc = pollster::block_on(EncoderSubsystem::init(
        Arc::clone(device),
        Arc::clone(queue),
        fmt,
        scale,
        phys_w,
        phys_h,
    ))
    .unwrap();
    enc.update_viewport(
        Viewport {
            width: logical_w,
            height: logical_h,
        },
        phys_w,
        phys_h,
        scale,
    );

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("canvas-shape target"),
        size: wgpu::Extent3d {
            width: phys_w,
            height: phys_h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let cmd = enc.encode_frame_from_relay(&target, frame).unwrap();
    queue.submit(std::iter::once(cmd));

    let bpr = 256 * (phys_w * 4).div_ceil(256);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("canvas-shape readback"),
        size: u64::from(bpr) * u64::from(phys_h),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut ce = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    ce.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(phys_h),
            },
        },
        wgpu::Extent3d {
            width: phys_w,
            height: phys_h,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(ce.finish()));
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().to_vec();
    Readback { data, bpr, scale }
}

/// A stroked circle paints its ring and leaves its interior untouched.
#[test]
fn circle_stroke_paints_the_ring_and_not_the_interior() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping canvas-shape readback");
        return;
    };

    let (w, h) = (200.0_f32, 200.0_f32);
    let mut frame = RenderFrame::new();
    frame.push_canvas_shape(CanvasShape {
        kind: CANVAS_SHAPE_CIRCLE,
        params: [100.0, 100.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        stroke_color: [1.0, 0.0, 0.0, 1.0],
        stroke_width: 8.0,
        ..CanvasShape::default()
    });
    let rb = render(&device, &queue, &frame, w, h);

    // On the ring (east point): strongly red.
    let (b, g, r, a) = rb.at(150.0, 100.0);
    assert!(a > 200, "ring pixel must be opaque, got alpha {a}");
    assert!(
        r > 200 && g < 60 && b < 60,
        "ring pixel must be red, got BGR=({b},{g},{r})"
    );
    // The centre is untouched (no fill).
    let (_, _, _, ca) = rb.at(100.0, 100.0);
    assert!(ca < 10, "unfilled interior must stay clear, got alpha {ca}");
    // Well outside the ring is untouched too.
    let (_, _, _, oa) = rb.at(10.0, 10.0);
    assert!(oa < 10, "outside pixel must stay clear, got alpha {oa}");
}

/// A 90° arc covers its swept quadrant and leaves the opposite one empty; a
/// filled rect covers its interior.
#[test]
fn arc_sweep_and_rect_fill_cover_exactly_their_regions() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping canvas-shape readback");
        return;
    };

    let (w, h) = (200.0_f32, 200.0_f32);
    let mut frame = RenderFrame::new();
    // Quarter arc: 0°..90° sweeps the +X → +Y quadrant (screen-space
    // clockwise from east through south).
    frame.push_canvas_shape(CanvasShape {
        kind: CANVAS_SHAPE_ARC,
        params: [
            100.0,
            100.0,
            60.0,
            0.0,
            std::f32::consts::FRAC_PI_2,
            0.0,
            0.0,
            0.0,
        ],
        stroke_color: [0.0, 1.0, 0.0, 1.0],
        stroke_width: 10.0,
        cap: CANVAS_CAP_ROUND,
        ..CanvasShape::default()
    });
    // Filled blue rect in the top-left corner.
    frame.push_canvas_shape(CanvasShape {
        kind: CANVAS_SHAPE_RECT,
        params: [10.0, 10.0, 40.0, 30.0, 4.0, 0.0, 0.0, 0.0],
        fill_color: [0.0, 0.0, 1.0, 1.0],
        stroke_width: 0.0,
        ..CanvasShape::default()
    });
    let rb = render(&device, &queue, &frame, w, h);

    // 45° into the sweep: on the ring, green.
    let mid = 45f32.to_radians();
    let (b, g, r, a) = rb.at(100.0 + 60.0 * mid.cos(), 100.0 + 60.0 * mid.sin());
    assert!(
        a > 200 && g > 200 && r < 60 && b < 60,
        "in-sweep ring pixel must be green, got BGRA=({b},{g},{r},{a})"
    );
    // The un-swept west point of the same ring stays clear.
    let (_, _, _, wa) = rb.at(100.0 - 60.0, 100.0);
    assert!(
        wa < 10,
        "out-of-sweep ring pixel must stay clear, got alpha {wa}"
    );
    // Inside the filled rect: blue.
    let (fb, fg, fr, fa) = rb.at(30.0, 25.0);
    assert!(
        fa > 200 && fb > 200 && fr < 60 && fg < 60,
        "rect interior must be blue, got BGRA=({fb},{fg},{fr},{fa})"
    );
}
