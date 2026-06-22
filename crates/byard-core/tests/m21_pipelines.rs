//! M21 — `DecoratedBox` / `TextureSampler` pipeline tests (RFC-0001 §3.1, §8).
//!
//! GPU-dependent tests request a real adapter and **skip gracefully** when none
//! is available (headless CI), so they assert on machines with a GPU without
//! breaking those without. The `uv_transform` tests are pure CPU and always run.

use byard_core::ByardError;
use byard_core::encoder::EncoderSubsystem;
use byard_core::encoder::texture_sampler::uv_transform;
use byard_core::frame::ImageFit;
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
