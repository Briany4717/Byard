//! End-to-end GPU readback of the real demo: parse → lower → render →
//! `EncoderSubsystem` → read pixels back. Reproduces what the live window shows
//! so a "widgets don't draw" regression is caught headlessly. Skips when no GPU
//! adapter is available.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use byard_compiler::interp::eval::Interpreter;
use byard_compiler::parser::parse;
use byard_core::encoder::EncoderSubsystem;
use byard_core::frame::{RenderFrame, Viewport};
use std::sync::Arc;

const SRC: &str = include_str!("../../byard-compiler/examples/hello_world.byd");

fn try_device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("readback device"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        ..Default::default()
    }))
    .ok()?;
    Some((Arc::new(device), Arc::new(queue)))
}

#[test]
#[allow(clippy::too_many_lines)]
fn demo_boxes_are_actually_painted_on_screen() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping readback");
        return;
    };

    // ── Logic side: render the real demo into a RenderFrame ───────────────
    let logical_w = 600.0_f32;
    let logical_h = 1000.0_f32;
    let parsed = parse(SRC);
    assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
    let mut interp = Interpreter::new();
    let tree = interp.lower_view(&parsed.views[0], &[]);
    assert!(interp.errors().is_empty(), "{:?}", interp.errors());
    interp.tick();
    let mut frame = RenderFrame::new();
    interp.render(&tree, &mut frame, logical_w, logical_h);

    // Pick the largest opaque box whose blue channel dominates strongly — the
    // 96×56 `0x6495ED` showcase box (a plain SolidBox, no text, no overlap).
    let bluest = frame
        .instances()
        .iter()
        .copied()
        .filter(|b| b.color[3] > 0.9 && b.color[2] > 0.8 && b.color[2] - b.color[0] > 0.4)
        .max_by(|a, b| {
            (a.rect[2] * a.rect[3])
                .partial_cmp(&(b.rect[2] * b.rect[3]))
                .unwrap()
        })
        .expect("the demo emits a large blue solid box");
    assert!(
        bluest.color[2] > bluest.color[0],
        "expected a blue-dominant box, got {:?}",
        bluest.color
    );
    let cx = bluest.rect[0] + bluest.rect[2] / 2.0;
    let cy = bluest.rect[1] + bluest.rect[3] / 2.0;

    // ── GPU side: encode at HiDPI (scale 2) into a BGRA sRGB target ───────
    let scale = 2.0_f32;
    let phys_w = (logical_w * scale) as u32;
    let phys_h = (logical_h * scale) as u32;
    let fmt = wgpu::TextureFormat::Bgra8UnormSrgb;
    let mut enc = pollster::block_on(EncoderSubsystem::init(
        Arc::clone(&device),
        Arc::clone(&queue),
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
        label: Some("target"),
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
    let cmd = enc.encode_frame_from_relay(&target, &frame).unwrap();
    queue.submit(std::iter::once(cmd));

    // ── Read the pixel at the blue box's center (physical) ────────────────
    let px = (cx * scale) as u32;
    let py = (cy * scale) as u32;
    let bpr = 256 * (phys_w * 4).div_ceil(256);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
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
    let data = slice.get_mapped_range();
    let idx = (py * bpr + px * 4) as usize;
    // BGRA byte order.
    let (b, _g, r, a) = (data[idx], data[idx + 1], data[idx + 2], data[idx + 3]);

    assert!(a > 10, "the box pixel must be opaque, got alpha {a}");
    assert!(
        b > r,
        "the blue box must paint blue-dominant pixels, got B={b} R={r}"
    );
}

/// Regression: a widget that sits *inside* an opaque, bordered card
/// (here the `Toggle`, whose ON track is painted in the theme accent) must
/// remain visible on screen. Before the fix the card became an opaque
/// `DecoratedBox` drawn in the decorated pass — *after* the widget's `SolidBox`
/// — so it painted over the toggle and only text showed through. We read the
/// toggle's track pixel and assert it is the bright accent, not the dark card.
#[test]
#[allow(clippy::too_many_lines)]
fn widget_inside_bordered_card_is_not_occluded() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping readback");
        return;
    };

    let logical_w = 600.0_f32;
    let logical_h = 1000.0_f32;
    let parsed = parse(SRC);
    assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
    let mut interp = Interpreter::new();
    let tree = interp.lower_view(&parsed.views[0], &[]);
    assert!(interp.errors().is_empty(), "{:?}", interp.errors());
    interp.tick();
    let mut frame = RenderFrame::new();
    interp.render(&tree, &mut frame, logical_w, logical_h);

    // The toggle ON track: the accent-coloured solid box furthest to the right
    // (the slider's accent pieces sit further left). A point just left of centre
    // lands on the track, clear of the white thumb that rides the right half.
    let accent =
        |c: &[f32; 4]| c[0] < 0.1 && (0.3..0.5).contains(&c[1]) && (0.5..0.7).contains(&c[2]);
    let track = frame
        .instances()
        .iter()
        .copied()
        .filter(|b| accent(&b.color) && b.rect[3] > 20.0)
        .max_by(|a, b| a.rect[0].partial_cmp(&b.rect[0]).unwrap())
        .expect("the demo emits the toggle's accent track");
    let cx = track.rect[0] + track.rect[2] * 0.25;
    let cy = track.rect[1] + track.rect[3] / 2.0;

    let scale = 2.0_f32;
    let phys_w = (logical_w * scale) as u32;
    let phys_h = (logical_h * scale) as u32;
    let fmt = wgpu::TextureFormat::Bgra8UnormSrgb;
    let mut enc = pollster::block_on(EncoderSubsystem::init(
        Arc::clone(&device),
        Arc::clone(&queue),
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
        label: Some("target"),
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
    let cmd = enc.encode_frame_from_relay(&target, &frame).unwrap();
    queue.submit(std::iter::once(cmd));

    let px = (cx * scale) as u32;
    let py = (cy * scale) as u32;
    let bpr = 256 * (phys_w * 4).div_ceil(256);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
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
    let data = slice.get_mapped_range();
    let idx = (py * bpr + px * 4) as usize;
    // BGRA. The dark card fill is ~(B,G,R) = (54,38,38); the accent track is a
    // bright teal-blue with G and B far above that. A clearly bright pixel here
    // proves the toggle painted over the card rather than being hidden by it.
    let (b, g, r) = (data[idx], data[idx + 1], data[idx + 2]);
    assert!(
        b > 110 && g > 90 && b > r + 30,
        "toggle track must show the accent over the card, got BGR=({b},{g},{r})"
    );
}
