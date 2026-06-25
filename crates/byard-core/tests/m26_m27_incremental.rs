//! M26/M27 — incremental render correctness on a real GPU (RFC-0001 §3.3).
//!
//! These mirror `m21_pipelines.rs`'s GPU-dependent style: they request a real
//! adapter and **skip gracefully** when none is available (headless CI). They
//! exercise the actual `persistent_color` retain-across-frames behaviour that
//! the pure `compute_scissor` unit tests in `encoder` can only model.
#![allow(clippy::cast_precision_loss)]

use byard_core::encoder::EncoderSubsystem;
use byard_core::frame::{BoxInstance, DecoratedBox, RenderFrame, TextLine, Viewport};
use std::sync::Arc;

/// Returns `(device, queue)` for a real adapter, or `None` if no GPU is present.
fn try_device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("ByardCore - M26/M27 Test Device"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        ..Default::default()
    }))
    .ok()?;
    Some((Arc::new(device), Arc::new(queue)))
}

/// Encodes `frame` onto a fresh `size×size` target via the encoder (whose
/// internal `persistent_color` carries state across calls), then reads back the
/// RGBA8 pixel at `(px, py)`.
fn render_and_read(
    enc: &mut EncoderSubsystem,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    frame: &RenderFrame,
    size: u32,
    px: u32,
    py: u32,
) -> [u8; 4] {
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("readback target"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let cmd = enc.encode_frame_from_relay(&target, frame).unwrap();
    queue.submit(std::iter::once(cmd));

    let bpr = 256u32 * size.div_ceil(64); // round 4*size up to 256.
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback buffer"),
        size: u64::from(bpr) * u64::from(size),
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
                rows_per_image: Some(size),
            },
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(ce.finish()));

    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range();
    let row = (py * bpr) as usize;
    let idx = row + (px * 4) as usize;
    [data[idx], data[idx + 1], data[idx + 2], data[idx + 3]]
}

fn solid_box(rect: [f32; 4], color: [f32; 4]) -> BoxInstance {
    BoxInstance {
        rect,
        color,
        radii: [0.0; 4],
    }
}

fn init_encoder(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, size: u32) -> EncoderSubsystem {
    let mut enc = pollster::block_on(EncoderSubsystem::init(
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        1.0,
        size,
        size,
    ))
    .expect("encoder init");
    enc.update_viewport(Viewport::new(size as f32, size as f32), size, size, 1.0);
    enc
}

/// M26 regression at the pixel level: a **textless** scene whose only change
/// between two frames is a `BoxInstance`'s colour must show the new colour.
///
/// Before M26, the second (incremental) frame produced `scissor == None`
/// (no dirty text) so `should_draw == false`, and the swapchain composite
/// blitted the *stale* `persistent_color` — the new colour never appeared.
#[test]
fn textless_box_colour_mutation_reaches_the_screen_on_an_incremental_frame() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping textless box mutation test");
        return;
    };
    let size = 64;
    let mut enc = init_encoder(Arc::clone(&device), Arc::clone(&queue), size);

    // Frame 1: a red box, no text. (First call is always a full redraw.)
    let mut f1 = RenderFrame::new();
    f1.push_instance(solid_box([0.0, 0.0, 64.0, 64.0], [1.0, 0.0, 0.0, 1.0]));
    let c1 = render_and_read(&mut enc, &device, &queue, &f1, size, 32, 32);
    assert!(
        c1[0] > 200 && c1[1] < 60,
        "frame 1 should be red, got {c1:?}"
    );

    // Frame 2: same box rect, now green. Same version, same primitive counts,
    // still no text → this is the exact case the bug silently dropped.
    let mut f2 = RenderFrame::new();
    f2.push_instance(solid_box([0.0, 0.0, 64.0, 64.0], [0.0, 1.0, 0.0, 1.0]));
    let c2 = render_and_read(&mut enc, &device, &queue, &f2, size, 32, 32);
    assert!(
        c2[1] > 200 && c2[0] < 60,
        "frame 2's green box must reach the screen (M26), got {c2:?}"
    );
}

/// M27: a static (clean) `DecoratedBox` plus a mutating `TextLine` must keep
/// the decoration on screen across incremental frames — the clear-quad +
/// scissor union, now driven by the text alone, must never wipe the
/// decoration's retained pixels in `persistent_color`.
#[test]
fn static_decorated_box_survives_unrelated_incremental_text_frames() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping decorated-box persistence test");
        return;
    };
    let size = 64;
    let mut enc = init_encoder(Arc::clone(&device), Arc::clone(&queue), size);

    // A decorated box (opaque blue fill) parked in the top-left corner, far
    // from where the text will live. It is dirty only on frame 1.
    let decorated_color = [0.0, 0.0, 1.0, 1.0];
    let make_frame = |text: &str, decorated_dirty: bool| {
        let mut f = RenderFrame::new();
        f.push_decorated(DecoratedBox {
            base: solid_box([0.0, 0.0, 16.0, 16.0], decorated_color),
            dirty: decorated_dirty,
            ..Default::default()
        });
        f.push_text(TextLine {
            x: 40.0,
            y: 40.0,
            text: text.to_string(),
            font_size: 12.0,
            color: [1.0, 1.0, 1.0, 1.0],
            dirty: true,
        });
        f
    };

    // Frame 1 (full redraw): decoration is painted.
    let f1 = make_frame("a", true);
    let d1 = render_and_read(&mut enc, &device, &queue, &f1, size, 4, 4);
    assert!(
        d1[2] > 200 && d1[0] < 60,
        "decoration must paint on frame 1, got {d1:?}"
    );

    // Frames 2 & 3 (incremental): only the text changes; the decoration is
    // clean. Its corner pixel must still read blue every frame.
    for text in ["b", "c"] {
        let f = make_frame(text, false);
        let d = render_and_read(&mut enc, &device, &queue, &f, size, 4, 4);
        assert!(
            d[2] > 200 && d[0] < 60,
            "clean decoration must persist on incremental frame (text={text:?}), got {d:?}"
        );
    }
}
