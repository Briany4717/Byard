//! GPU readback proofs for the `Ripple` pipeline (RFC-0023).
//!
//! Renders synthetic frames of ripple ink over solid backgrounds and reads
//! pixels back, pinning down on a real device the visual contracts the
//! CPU-side tests in `byard-compiler` cannot see: the ink composites over
//! the background (light ink brightens a dark surface, dark ink darkens a
//! light one), it clips to the element's rounded rect, and its draw-order
//! depth keeps it above the background but beneath children.
//!
//! Skips cleanly when no GPU adapter is available.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use byard_core::BoxInstance;
use byard_core::encoder::EncoderSubsystem;
use byard_core::frame::{RenderFrame, RippleInstance, Transform, Viewport};
use std::sync::Arc;

fn try_device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("ripple readback device"),
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
        label: Some("ripple target"),
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
        label: Some("ripple readback"),
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

/// A [`RippleInstance`] with an identity transform on `rect`.
fn ripple(rect: [f32; 4], params: [f32; 4], color: [f32; 4], radii: [f32; 4]) -> RippleInstance {
    let t = Transform::IDENTITY;
    RippleInstance {
        rect,
        params,
        color,
        radii,
        t_translate: t.translate,
        t_scale: t.scale,
        t_rotate: t.rotate,
        t_origin: t.origin,
        depth: 0.0, // stamped by `push_ripple`
    }
}

/// The ink composites over the background inside its circle — brightening a
/// dark surface with light ink and *darkening* a light surface with dark ink
/// (the regression a purely additive blend causes: addition can only ever add
/// light, so dark ink on a light card was invisible) — leaves the rest of the
/// element untouched, and never bleeds past a rounded corner (RFC-0023
/// resolved question: always clip).
#[test]
#[allow(clippy::many_single_char_names)]
fn ripple_ink_composites_over_light_and_dark_and_clips_to_the_rounded_corner() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping ripple readback");
        return;
    };

    let (w, h) = (200.0_f32, 200.0_f32);
    let rect = [20.0, 20.0, 160.0, 160.0];
    let radii = [40.0; 4];

    // Element background: a dark opaque rounded box.
    let mut frame = RenderFrame::new();
    frame.push_instance(BoxInstance {
        rect,
        color: [0.15, 0.05, 0.3, 1.0],
        radii,
        transform: Transform::IDENTITY,
    });
    // Mid-flight ripple from the element centre: 60px radius, full fade
    // alpha, 50% white ink.
    frame.push_ripple(ripple(
        rect,
        [100.0, 100.0, 60.0, 1.0],
        [1.0, 1.0, 1.0, 0.5],
        radii,
    ));
    let rb = render(&device, &queue, &frame, w, h);

    // Inside the circle: strictly brighter than the bare background outside it
    // (white ink over the dark fill).
    let (ib, ig, ir, ia) = rb.at(100.0, 100.0);
    let (ob, og, or_, oa) = rb.at(170.0, 100.0); // in-rect, outside the circle
    assert!(
        ia > 200 && oa > 200,
        "both samples sit on the opaque element"
    );
    assert!(
        i32::from(ir) > i32::from(or_) + 40
            && i32::from(ig) > i32::from(og) + 40
            && i32::from(ib) > i32::from(ob) + 40,
        "ink must brighten the background: inked BGR=({ib},{ig},{ir}) vs bare BGR=({ob},{og},{or_})"
    );

    // Dark ink on a *light* surface must darken it — the exact case a purely
    // additive blend gets wrong (dark ink was invisible on light cards).
    let mut light = RenderFrame::new();
    light.push_instance(BoxInstance {
        rect,
        color: [0.98, 0.96, 1.0, 1.0],
        radii,
        transform: Transform::IDENTITY,
    });
    light.push_ripple(ripple(
        rect,
        [100.0, 100.0, 60.0, 1.0],
        [0.1, 0.05, 0.25, 0.4], // translucent dark-purple ink
        radii,
    ));
    let rb = render(&device, &queue, &light, w, h);
    let (db, dg, dr, da) = rb.at(100.0, 100.0);
    let (lb, lg, lr, la) = rb.at(170.0, 100.0); // bare light surface
    assert!(
        da > 200 && la > 200,
        "both samples sit on the opaque element"
    );
    assert!(
        i32::from(dr) < i32::from(lr) - 30
            && i32::from(dg) < i32::from(lg) - 30
            && i32::from(db) < i32::from(lb) - 30,
        "dark ink must darken a light surface: inked BGR=({db},{dg},{dr}) vs bare BGR=({lb},{lg},{lr})"
    );

    // A second, fully-covering ripple pinned at a huge radius must still
    // respect the rounded corner: the pixel just inside the rect's square
    // corner (but outside the 40px corner round) stays clear.
    let mut clipped = RenderFrame::new();
    clipped.push_ripple(ripple(
        rect,
        [100.0, 100.0, 400.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        radii,
    ));
    let rb = render(&device, &queue, &clipped, w, h);
    let (_, _, _, corner_a) = rb.at(26.0, 26.0);
    assert!(
        corner_a < 10,
        "ink must clip to the border radius, got corner alpha {corner_a}"
    );
    // While the straight edge midpoint (inside the shape) is inked.
    let (_, _, edge_r, edge_a) = rb.at(26.0, 100.0);
    assert!(
        edge_a > 100 && edge_r > 100,
        "the in-shape edge must be inked, got r={edge_r} a={edge_a}"
    );
}

/// The ripple's stamped draw-order depth composites it above the element's
/// background but beneath the element's children (RFC-0023 §1: the label
/// stays crisp on top of the ink).
#[test]
fn ripple_depth_keeps_children_crisp_above_the_ink() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter — skipping ripple readback");
        return;
    };

    let (w, h) = (200.0_f32, 200.0_f32);
    let mut frame = RenderFrame::new();
    // Background (emitted first = farthest).
    frame.push_instance(BoxInstance {
        rect: [0.0, 0.0, 200.0, 200.0],
        color: [0.1, 0.1, 0.1, 1.0],
        radii: [0.0; 4],
        transform: Transform::IDENTITY,
    });
    // Ripple over the whole element (emitted second = between).
    frame.push_ripple(ripple(
        [0.0, 0.0, 200.0, 200.0],
        [100.0, 100.0, 300.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0; 4],
    ));
    // A pure-blue child quad in the middle (emitted last = nearest).
    frame.push_instance(BoxInstance {
        rect: [80.0, 80.0, 40.0, 40.0],
        color: [0.0, 0.0, 1.0, 1.0],
        radii: [0.0; 4],
        transform: Transform::IDENTITY,
    });
    let rb = render(&device, &queue, &frame, w, h);

    // Outside the child: white ink over the dark background.
    let (_, _, ink_r, ink_a) = rb.at(40.0, 40.0);
    assert!(
        ink_a > 200 && ink_r > 150,
        "ink must cover the background, got r={ink_r} a={ink_a}"
    );
    // On the child: pure blue — the ink must NOT wash over it (depth test
    // rejects the ripple fragment where the nearer child already drew).
    let (cb, cg, cr, ca) = rb.at(100.0, 100.0);
    assert!(
        ca > 200 && cb > 200 && cr < 60 && cg < 60,
        "the child must stay crisp above the ink, got BGRA=({cb},{cg},{cr},{ca})"
    );
}
