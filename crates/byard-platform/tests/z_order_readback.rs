//! GPU readback proofs for cross-pass paint order (RFC-0011 draw-order depth).
//!
//! These render *synthetic* frames with a known emission order and read pixels
//! back, so they pin down the exact behaviours the flat 4-pass encoder used to
//! get wrong:
//!   1. a later-emitted child paints *over* an earlier container's border
//!      (previously the decorated border pass always sat on top of solids);
//!   2. text is depth-sorted like everything else, so a later-emitted opaque box
//!      occludes earlier text (previously text always drew on top).
//!
//! Skips cleanly when no GPU adapter is available.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]

use byard_core::encoder::EncoderSubsystem;
use byard_core::frame::{BoxInstance, DecoratedBox, RenderFrame, TextLine, Transform, Viewport};
use std::sync::Arc;

fn try_device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("z-order readback device"),
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
        label: Some("z-order target"),
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
        label: Some("z-order readback"),
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

fn solid(rect: [f32; 4], color: [f32; 4]) -> BoxInstance {
    BoxInstance {
        rect,
        color,
        radii: [0.0; 4],
        transform: Transform::IDENTITY,
    }
}

/// A later-emitted child box that overlaps its container's border must paint
/// *over* the border — the exact "cards draw under the container border" bug.
/// The container border is bright red; the child is bright green and is emitted
/// after it, straddling the top border ring. The overlap pixel must read green.
#[test]
fn later_child_paints_over_container_border() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping z-order readback");
        return;
    };

    let (w, h) = (300.0_f32, 300.0_f32);
    let mut frame = RenderFrame::new();

    // 1. Container fill (solid, dark), emitted first.
    frame.push_instance(solid([50.0, 50.0, 200.0, 150.0], [0.1, 0.1, 0.12, 1.0]));
    // 2. Container border (decorated overlay, transparent interior, red ring),
    //    emitted second — the pass that used to sit unconditionally on top.
    frame.push_decorated(DecoratedBox {
        base: solid([50.0, 50.0, 200.0, 150.0], [0.0, 0.0, 0.0, 0.0]),
        border_width: 10.0,
        border_color: [1.0, 0.0, 0.0, 1.0],
        opacity: 1.0,
        dirty: true,
        ..Default::default()
    });
    // 3. Child box (solid, green), emitted LAST, straddling the top border ring
    //    at y≈50 (covers y 40..90, x 100..200).
    frame.push_instance(solid([100.0, 40.0, 100.0, 50.0], [0.0, 0.9, 0.2, 1.0]));

    let rb = render(&device, &queue, &frame, w, h);

    // A point on the top border line (y≈55) inside the child's x-span: the child
    // is emitted after the border, so its green must win over the red border.
    let (b, g, r, a) = rb.at(150.0, 55.0);
    assert!(a > 10, "overlap pixel must be opaque, got alpha {a}");
    assert!(
        g > 110 && g > r + 40 && g > b + 40,
        "later child must paint over the container border (expected green-dominant), got BGR=({b},{g},{r})"
    );
}

/// Text is depth-sorted like every other primitive: a later-emitted opaque box
/// occludes earlier text instead of the text drawing on top. We first render
/// text alone to locate a glyph pixel (a strongly red one), then render the same
/// text with a green box emitted *after* it covering that spot — the pixel must
/// flip from red (glyph) to green (box).
#[test]
fn later_box_occludes_earlier_text() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping z-order readback");
        return;
    };

    let (w, h) = (400.0_f32, 200.0_f32);
    let text = || TextLine {
        x: 40.0,
        y: 60.0,
        text: "OCCLUSION".to_string(),
        font_size: 64.0,
        color: [1.0, 0.0, 0.0, 1.0],
        dirty: true,
    };

    // ── Render A: text only. Find the reddest glyph pixel in the text band. ──
    let mut frame_a = RenderFrame::new();
    frame_a.push_text(text());
    let rb_a = render(&device, &queue, &frame_a, w, h);

    let mut best = None;
    let mut best_red = 0i32;
    let mut ly = 40.0_f32;
    while ly < 130.0 {
        let mut lx = 40.0_f32;
        while lx < 360.0 {
            let (b, g, r, _a) = rb_a.at(lx, ly);
            // Redness of a glyph pixel: red present, blue/green suppressed.
            let redness = i32::from(r) - i32::from(b).max(i32::from(g));
            if redness > best_red {
                best_red = redness;
                best = Some((lx, ly));
            }
            lx += 2.0;
        }
        ly += 2.0;
    }
    let (gx, gy) = best.expect("text-only render must contain a glyph pixel");
    assert!(
        best_red > 40,
        "expected a clearly red glyph pixel in the text-only render, got redness {best_red}"
    );

    // ── Render B: same text, then a green opaque box emitted AFTER it, ───────
    // covering the whole text band. The glyph pixel must now read green.
    let mut frame_b = RenderFrame::new();
    frame_b.push_text(text());
    frame_b.push_instance(solid([30.0, 20.0, 340.0, 100.0], [0.0, 0.85, 0.2, 1.0]));
    let rb_b = render(&device, &queue, &frame_b, w, h);

    let (b, g, r, a) = rb_b.at(gx, gy);
    assert!(a > 10, "covered pixel must be opaque, got alpha {a}");
    assert!(
        g > 110 && g > r + 40,
        "a box emitted after the text must occlude it (text no longer always-on-top); \
         expected green at ({gx},{gy}), got BGR=({b},{g},{r})"
    );
}

/// RFC-0017 regression: a **translucent** box emitted after text must NOT cull
/// it. A translucent box (an overlay scrim, a see-through fill) blends over what
/// is already drawn; it must not write draw-order depth, or every earlier text
/// glyph beneath it — drawn in the later text pass at a farther depth — would
/// fail the depth test and vanish. This is the "all app text disappears under a
/// modal scrim" bug. Contrast with `later_box_occludes_earlier_text`, where an
/// *opaque* box legitimately occludes the text.
#[test]
fn translucent_box_over_text_does_not_cull_it() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping z-order readback");
        return;
    };

    let (w, h) = (400.0_f32, 200.0_f32);
    let text = || TextLine {
        x: 40.0,
        y: 60.0,
        text: "SCRIM".to_string(),
        font_size: 64.0,
        color: [1.0, 0.0, 0.0, 1.0],
        dirty: true,
    };

    // ── Render A: text only. Find the reddest glyph pixel. ──
    let mut frame_a = RenderFrame::new();
    frame_a.push_text(text());
    let rb_a = render(&device, &queue, &frame_a, w, h);

    let mut best = None;
    let mut best_red = 0i32;
    let mut ly = 40.0_f32;
    while ly < 130.0 {
        let mut lx = 40.0_f32;
        while lx < 360.0 {
            let (b, g, r, _a) = rb_a.at(lx, ly);
            let redness = i32::from(r) - i32::from(b).max(i32::from(g));
            if redness > best_red {
                best_red = redness;
                best = Some((lx, ly));
            }
            lx += 2.0;
        }
        ly += 2.0;
    }
    let (gx, gy) = best.expect("text-only render must contain a glyph pixel");
    assert!(
        best_red > 40,
        "expected a clearly red glyph pixel, got {best_red}"
    );

    // ── Render B: same text, then a TRANSLUCENT green box (opacity 0.5) emitted
    // AFTER it covering the whole band. Because the box does not write depth, the
    // text still passes the depth test and paints on top — the glyph pixel stays
    // red-dominant (not culled to green).
    let mut frame_b = RenderFrame::new();
    frame_b.push_text(text());
    frame_b.push_decorated(DecoratedBox {
        base: solid([30.0, 20.0, 340.0, 100.0], [0.0, 0.85, 0.2, 1.0]),
        opacity: 0.5,
        dirty: true,
        ..Default::default()
    });
    let rb_b = render(&device, &queue, &frame_b, w, h);

    let (b, g, r, a) = rb_b.at(gx, gy);
    assert!(a > 10, "pixel must be opaque, got alpha {a}");
    assert!(
        r > g && r > b,
        "text under a translucent box must survive (red-dominant), not be culled; \
         got BGR=({b},{g},{r}) at ({gx},{gy})"
    );
}
