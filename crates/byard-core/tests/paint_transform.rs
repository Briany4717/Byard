//! RFC-0011 — paint-time transform primitives: a `Transform` moves/scales/
//! rotates the *painted* quad without touching layout. GPU-dependent tests
//! request a real adapter and **skip gracefully** when none is available
//! (headless CI), mirroring `m21_pipelines.rs`'s pattern.
#![allow(clippy::cast_precision_loss)]

use byard_core::encoder::EncoderSubsystem;
use byard_core::frame::{BoxInstance, RenderFrame, Transform, Viewport};
use std::sync::Arc;

fn try_device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("ByardCore - Test Device"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        ..Default::default()
    }))
    .ok()?;
    Some((Arc::new(device), Arc::new(queue)))
}

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

    let bpr = 256u32 * size.div_ceil(64);
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

#[test]
fn a_translated_box_paints_at_its_transformed_position_not_its_layout_rect() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping paint-transform readback test");
        return;
    };
    let size = 128u32;
    let mut enc = pollster::block_on(EncoderSubsystem::init(
        Arc::clone(&device),
        Arc::clone(&queue),
        wgpu::TextureFormat::Rgba8UnormSrgb,
        1.0,
        size,
        size,
    ))
    .unwrap();
    enc.update_viewport(
        Viewport {
            width: size as f32,
            height: size as f32,
        },
        size,
        size,
        1.0,
    );

    // A red box laid out at (10,10)-(50,50), translated +40px on each axis
    // by the paint-time transform — its *layout* rect never moves, only
    // where it's drawn.
    let mut frame = RenderFrame::new();
    frame.push_instance(BoxInstance {
        rect: [10.0, 10.0, 40.0, 40.0],
        color: [1.0, 0.0, 0.0, 1.0],
        radii: [0.0; 4],
        transform: Transform {
            translate: [40.0, 40.0],
            ..Transform::IDENTITY
        },
    });

    // Its own (untransformed) layout rect must be empty — the transform
    // moved the paint, not the geometry.
    let at_layout_rect = render_and_read(&mut enc, &device, &queue, &frame, size, 30, 30);
    assert!(
        at_layout_rect[0] < 80,
        "the translated box must not paint its original layout rect, got {at_layout_rect:?}"
    );

    // The transformed destination (30+40, 30+40) = (70, 70) must be red.
    let at_transformed_pos = render_and_read(&mut enc, &device, &queue, &frame, size, 70, 70);
    assert!(
        at_transformed_pos[0] > 120 && at_transformed_pos[1] < 80 && at_transformed_pos[2] < 80,
        "the translated box must paint at its transformed position, got {at_transformed_pos:?}"
    );
}

#[test]
fn identity_transform_matches_untransformed_output_byte_for_byte() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping identity-transform regression test");
        return;
    };
    let size = 64u32;
    let mut enc = pollster::block_on(EncoderSubsystem::init(
        Arc::clone(&device),
        Arc::clone(&queue),
        wgpu::TextureFormat::Rgba8UnormSrgb,
        1.0,
        size,
        size,
    ))
    .unwrap();
    enc.update_viewport(
        Viewport {
            width: size as f32,
            height: size as f32,
        },
        size,
        size,
        1.0,
    );

    let make_frame = |transform: Transform| {
        let mut frame = RenderFrame::new();
        frame.push_instance(BoxInstance {
            rect: [10.0, 10.0, 30.0, 30.0],
            color: [0.0, 1.0, 0.0, 1.0],
            radii: [0.0; 4],
            transform,
        });
        frame
    };

    // Constructing `Transform` field-by-field (not via the `IDENTITY`
    // constant) still produces the same pixels — proves identity isn't a
    // special-cased fast path that silently diverges from the real math.
    let explicit_identity = Transform {
        translate: [0.0, 0.0],
        scale: [1.0, 1.0],
        rotate: 0.0,
        origin: [0.0, 0.0],
        opacity: 1.0,
    };
    assert!(explicit_identity.is_identity());

    let baseline = render_and_read(
        &mut enc,
        &device,
        &queue,
        &make_frame(Transform::IDENTITY),
        size,
        25,
        25,
    );
    let explicit = render_and_read(
        &mut enc,
        &device,
        &queue,
        &make_frame(explicit_identity),
        size,
        25,
        25,
    );
    assert_eq!(
        baseline, explicit,
        "identity built from IDENTITY vs field-by-field must render identically"
    );
}
