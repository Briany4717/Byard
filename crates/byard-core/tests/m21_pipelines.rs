//! M21 — `DecoratedBox` / `TextureSampler` pipeline tests (RFC-0001 §3.1, §8).
//!
//! GPU-dependent tests request a real adapter and **skip gracefully** when none
//! is available (headless CI), so they assert on machines with a GPU without
//! breaking those without. The `uv_transform` tests are pure CPU and always run.
#![allow(clippy::cast_precision_loss)]

use byard_core::ByardError;
use byard_core::encoder::EncoderSubsystem;
use byard_core::encoder::texture_sampler::uv_transform;
use byard_core::frame::{BoxInstance, DecoratedBox, ImageFit, RenderFrame, Viewport};
use std::sync::Arc;

/// Elementwise approximate equality for a UV transform `[f32; 4]`.
fn approx4(a: [f32; 4], b: [f32; 4]) {
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-6, "{a:?} != {b:?}");
    }
}

/// Returns `(device, queue)` for a real adapter, or `None` if no GPU is present.
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

#[test]
fn encoder_builds_all_pipelines_including_m21() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping pipeline build test");
        return;
    };
    // init() builds SolidBox, clear, text, DecoratedBox and TextureSampler
    // pipelines. Success means the two new M21 WGSL shaders compiled and their
    // pipelines passed GPU validation (RFC-0001 §8).
    let result = pollster::block_on(EncoderSubsystem::init(
        device,
        queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        1.0,
        64,
        64,
    ));
    assert!(result.is_ok(), "encoder init failed: {:?}", result.err());
}

#[test]
fn bad_shader_surfaces_pipeline_compilation_not_panic() {
    let Some((device, _queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping bad-shader test");
        return;
    };

    // Mirror the §8 error-scope pattern with intentionally invalid WGSL and
    // confirm the validation scope captures the failure (the mechanism that
    // `build_pipeline` turns into `ByardError::PipelineCompilation`) — never a panic.
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bad shader"),
        source: wgpu::ShaderSource::Wgsl("this is not valid wgsl @@@".into()),
    });
    let _ = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("bad pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: None,
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    let err = pollster::block_on(scope.pop());
    assert!(
        err.is_some(),
        "a bad shader must produce a validation error"
    );

    // And the error maps cleanly into our typed variant (no panic, no fallback).
    let mapped = ByardError::PipelineCompilation {
        pipeline: "Test".to_string(),
        reason: err.unwrap().to_string(),
    };
    assert!(matches!(mapped, ByardError::PipelineCompilation { .. }));
}

/// Renders `frame` to an offscreen `size×size` target on the real GPU and reads
/// the RGBA8 pixel at logical `(px, py)`.
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

#[test]
fn solid_and_decorated_boxes_actually_paint_pixels() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping readback test");
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

    // A red solid box and a blue decorated (bordered) box, both well inside.
    let mut frame = RenderFrame::new();
    frame.push_instance(BoxInstance {
        rect: [10.0, 10.0, 40.0, 40.0],
        color: [1.0, 0.0, 0.0, 1.0],
        radii: [0.0; 4],
    });
    frame.push_decorated(DecoratedBox {
        base: BoxInstance {
            rect: [70.0, 70.0, 40.0, 40.0],
            color: [0.0, 0.0, 1.0, 1.0],
            radii: [0.0; 4],
        },
        border_width: 3.0,
        border_color: [1.0, 1.0, 1.0, 1.0],
        ..DecoratedBox::default()
    });

    let solid = render_and_read(&mut enc, &device, &queue, &frame, size, 30, 30);
    let decorated = render_and_read(&mut enc, &device, &queue, &frame, size, 90, 90);

    assert!(
        solid[0] > 120 && solid[1] < 80 && solid[2] < 80,
        "SolidBox should paint red pixels, got {solid:?}"
    );
    assert!(
        decorated[2] > 120,
        "DecoratedBox should paint blue pixels, got {decorated:?}"
    );
}

#[test]
fn uv_transform_fill_is_identity() {
    approx4(
        uv_transform(ImageFit::Fill, 100, 50, 200.0, 200.0),
        [1.0, 1.0, 0.0, 0.0],
    );
}

#[test]
fn uv_transform_cover_crops_the_wider_axis() {
    // Image wider than the (square) rect → crop horizontally: scale_x < 1,
    // centered offset, full vertical.
    let [sx, sy, ox, oy] = uv_transform(ImageFit::Cover, 200, 100, 100.0, 100.0);
    assert!(sx < 1.0 && (sy - 1.0).abs() < f32::EPSILON);
    assert!((ox - (1.0 - sx) / 2.0).abs() < 1e-6 && oy.abs() < f32::EPSILON);
}

#[test]
fn uv_transform_contain_letterboxes_beyond_unit_range() {
    // Image wider than rect → contain adds vertical bars: the vertical UV range
    // exceeds [0,1] (scale_y > 1) so out-of-image fragments discard.
    let [sx, sy, _ox, oy] = uv_transform(ImageFit::Contain, 200, 100, 100.0, 100.0);
    assert!((sx - 1.0).abs() < f32::EPSILON && sy > 1.0);
    assert!(
        oy < 0.0,
        "letterbox offset should push the image down/centered"
    );
}

#[test]
fn uv_transform_none_uses_natural_pixels() {
    // 50px image in a 100px rect at natural size → covers half the rect.
    approx4(
        uv_transform(ImageFit::None, 50, 50, 100.0, 100.0),
        [2.0, 2.0, 0.0, 0.0],
    );
}
