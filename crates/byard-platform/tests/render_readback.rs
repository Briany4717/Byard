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
    let info = adapter.get_info();
    eprintln!(
        "  GPU adapter: {} ({:?}, {:?} backend, driver: {})",
        info.name, info.device_type, info.backend, info.driver
    );
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
    let logical_w = 1280.0_f32;
    let logical_h = 720.0_f32;
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
        .filter(|b| {
            // Opaque, strongly blue-dominant …
            let blue = b.color[3] > 0.9 && b.color[2] > 0.8 && b.color[2] - b.color[0] > 0.4;
            // … and fully on-screen, so the sampled centre lands inside the
            // readback buffer no matter how tall the demo grows: it extends past
            // the 720px frame (there is no ScrollView yet), and an off-screen
            // box's centre would index out of bounds.
            let cx = b.rect[0] + b.rect[2] / 2.0;
            let cy = b.rect[1] + b.rect[3] / 2.0;
            let on_screen = cx >= 0.0 && cx < logical_w && cy >= 0.0 && cy < logical_h;
            blue && on_screen
        })
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

/// End-to-end proof of the RFC-0009 dev JIT pipeline: a `VectorIcon` starts as
/// a zero-opacity placeholder, the background generation lands within a few
/// ticks, and the resulting `AtlasUpload` actually reaches the GPU texture and
/// paints an opaque pixel — the whole chain from `.byd` source to a real
/// screen pixel, not just unit-level cache bookkeeping.
#[test]
#[allow(clippy::too_many_lines)]
fn vector_icon_paints_after_generation_lands() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping readback");
        return;
    };

    // A guaranteed fully-opaque icon (no holes) so any interior pixel is
    // unambiguously "inside" the shape, regardless of MSDF channel mixing.
    let svg_path = std::env::temp_dir().join("byard_vector_readback_solid_square.svg");
    std::fs::write(
        &svg_path,
        br##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
              <path d="M2 2 L22 2 L22 22 L2 22 Z" fill="#000000"/>
            </svg>"##,
    )
    .unwrap();
    let svg_path = svg_path.to_str().unwrap();

    let logical_w = 400.0_f32;
    let logical_h = 400.0_f32;
    let src = format!(r#"View App() {{ VectorIcon("{svg_path}") #[size: 200, color: 0xFFFFFF] }}"#);
    let parsed = parse(&src);
    assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
    let mut interp = Interpreter::new();
    let tree = interp.lower_view(&parsed.views[0], &[]);
    assert!(interp.errors().is_empty(), "{:?}", interp.errors());

    // Poll ticks until the background generation lands (first tick dispatches
    // it and only ever emits the INV-9 placeholder).
    let mut frame = RenderFrame::new();
    let mut resident_rect = None;
    for _ in 0..200 {
        interp.tick();
        frame = RenderFrame::new();
        interp.render(&tree, &mut frame, logical_w, logical_h);
        let inst = frame.vector_instances()[0];
        if inst.color[3] > 0.0 {
            resident_rect = Some(inst.screen_rect);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let rect = resident_rect.expect("the icon must become resident within the poll window");
    let cx = rect[0] + rect[2] / 2.0;
    let cy = rect[1] + rect[3] / 2.0;

    let scale = 1.0_f32;
    let phys_w = logical_w as u32;
    let phys_h = logical_h as u32;
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

    let px = cx as u32;
    let py = cy as u32;
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
    let (b, g, r, a) = (data[idx], data[idx + 1], data[idx + 2], data[idx + 3]);

    let _ = std::fs::remove_file(svg_path);

    assert!(
        a > 200,
        "the generated icon's interior must be opaque, got alpha {a}"
    );
    assert!(
        b > 200 && g > 200 && r > 200,
        "color: 0xFFFFFF must tint the field white, got BGR=({b},{g},{r})"
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

    let logical_w = 1280.0_f32;
    let logical_h = 720.0_f32;
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

/// RFC-0005 `ScrollView` content clip: a box (and text) emitted inside
/// `begin_clip` is scissored to the clip rect — visible inside it, gone outside.
/// Set `BYARD_DUMP_PNG=1` to also write the frame to a PNG for eyeballing.
#[test]
#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
fn content_clip_scissors_overflow_to_the_viewport() {
    use byard_core::BoxInstance;
    use byard_core::frame::{Rect, TextLine, Transform};

    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping readback");
        return;
    };

    let (lw, lh) = (400.0_f32, 300.0_f32);
    let scale = 1.0_f32;
    let (pw, ph) = (lw as u32, lh as u32);

    let solid = |x: f32, y: f32, w: f32, h: f32, c: [f32; 4]| BoxInstance {
        rect: [x, y, w, h],
        color: c,
        radii: [0.0; 4],
        transform: Transform::IDENTITY,
    };

    // Scene: a dark full-frame background, then a 200×140 "scroll window" clip
    // at (100,80) holding a bright box + text that overflow it on every side.
    let mut frame = RenderFrame::new();
    frame.set_version(1);
    frame.push_instance(solid(0.0, 0.0, lw, lh, [0.07, 0.08, 0.11, 1.0]));
    let window = Rect::new(100.0, 80.0, 200.0, 140.0);
    frame.begin_clip(window);
    frame.push_instance(solid(60.0, 40.0, 280.0, 240.0, [0.39, 0.58, 0.93, 1.0]));
    for (i, s) in ["clipped to the", "scroll viewport", "— overflow hidden"]
        .iter()
        .enumerate()
    {
        frame.push_text(TextLine {
            x: 70.0,
            y: 110.0 + i as f32 * 28.0,
            text: (*s).to_string(),
            font_size: 18.0,
            color: [1.0, 1.0, 1.0, 1.0],
            dirty: true,
        });
    }
    frame.end_clip();

    let fmt = wgpu::TextureFormat::Bgra8UnormSrgb;
    let mut enc = pollster::block_on(EncoderSubsystem::init(
        Arc::clone(&device),
        Arc::clone(&queue),
        fmt,
        scale,
        pw,
        ph,
    ))
    .unwrap();
    enc.update_viewport(
        Viewport {
            width: lw,
            height: lh,
        },
        pw,
        ph,
        scale,
    );
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("clip target"),
        size: wgpu::Extent3d {
            width: pw,
            height: ph,
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

    // Read the whole frame back.
    let bpr = 256 * (pw * 4).div_ceil(256);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: u64::from(bpr) * u64::from(ph),
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
                rows_per_image: Some(ph),
            },
        },
        wgpu::Extent3d {
            width: pw,
            height: ph,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(ce.finish()));
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range();

    // BGRA sample at a physical pixel.
    let px = |x: u32, y: u32| -> (u8, u8, u8, u8) {
        let i = (y * bpr + x * 4) as usize;
        (data[i], data[i + 1], data[i + 2], data[i + 3])
    };

    // Inside the window, on the blue content → blue-dominant.
    let (b_in, _g, r_in, _a) = px(160, 130);
    assert!(
        b_in > 150 && b_in > r_in + 40,
        "content inside the clip must paint blue, got B={b_in} R={r_in}"
    );

    // Inside the blue content's rect but OUTSIDE the window (top-left of it):
    // must be clipped away → the dark background, not the blue content. (The bg
    // is `[0.07,0.08,0.11]` linear, which sRGB-encodes to ~BGR(93,80,75) in the
    // sRGB target — dark and near-neutral, unlike the strongly-blue content.)
    let (b_out, _g_out, r_out, _a) = px(70, 50);
    assert!(
        b_out < 120 && b_out < r_out + 40,
        "content outside the clip must be scissored away (dark bg, not blue), got B={b_out} R={r_out}"
    );

    // Optional visual dump (BGRA → RGBA), for eyeballing the clip.
    if std::env::var("BYARD_DUMP_PNG").is_ok() {
        let mut rgba = vec![0u8; (pw * ph * 4) as usize];
        for y in 0..ph {
            for x in 0..pw {
                let (b, g, r, a) = px(x, y);
                let o = ((y * pw + x) * 4) as usize;
                rgba[o] = r;
                rgba[o + 1] = g;
                rgba[o + 2] = b;
                rgba[o + 3] = a;
            }
        }
        let out = std::env::var("BYARD_PNG_PATH").unwrap_or_else(|_| {
            std::env::temp_dir()
                .join("byard_clip.png")
                .display()
                .to_string()
        });
        image::save_buffer(&out, &rgba, pw, ph, image::ColorType::Rgba8).unwrap();
        eprintln!("wrote clip readback PNG to {out}");
    }
}
