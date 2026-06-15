//! # Encoder
//!
//! Multi-pipeline `wgpu` command dispatch.
//!
//! This subsystem owns the specialised render pipelines compiled at startup:
//!
//! - **`SolidBox`** — Axis-aligned rectangles with solid fill **and per-corner
//!   `border-radius` via an analytical SDF**. This absorbs the basic rounded-rect
//!   case from the RFC §3.1 `DecoratedBox` column by design: a single instanced
//!   pipeline handles both plain and rounded rectangles with zero extra GPU state
//!   switches, because the radius parameters are part of the per-instance vertex
//!   data rather than a pipeline variant.
//! - **`DecoratedBox`** — Rectangles with gradients, box-shadows, and parametric
//!   decorations (future sub-issue).
//! - **`TextGlyph`** — Text rendering via a `glyphon` glyph atlas (future).
//! - **`TextureSampler`** — UV-mapped quads for decoded images and icons (future).
//!
//! Primitives are batched into Z-bins (stacking contexts) and ordered first by
//! pipeline, then by local Z-index, to minimise GPU context switches.
//!
//! Pipeline creation is wrapped in `Device::push_error_scope` /
//! `Device::pop_error_scope` with `ErrorFilter::Validation`. Failures are
//! surfaced as [`ByardError::PipelineCompilation`](crate::ByardError::PipelineCompilation)
//! — the engine never panics on a GPU error.

use std::sync::Arc;

use bytemuck;
use wgpu::util::DeviceExt;

use crate::ByardError;
use crate::frame::Viewport;

/// GPU-ready instance data for a single solid rectangle.
///
/// Each field maps directly to a WGSL `@location` in `solid_box.wgsl`.
/// The layout is `#[repr(C)]` and implements `bytemuck::Pod` so the slice
/// can be cast to `&[u8]` and uploaded to the instance buffer with zero copy.
///
/// Field order matches the `@location` indices in the vertex shader:
/// - `@location(1)` — `rect`
/// - `@location(2)` — `color`
/// - `@location(3)` — `radii`
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BoxInstance {
    /// Rectangle in logical pixels: `[x, y, width, height]`.
    pub rect: [f32; 4],
    /// Linear-space fill colour: `[r, g, b, a]`.
    pub color: [f32; 4],
    /// Per-corner border radii: `[top_left, top_right, bottom_right, bottom_left]`.
    pub radii: [f32; 4],
}

impl BoxInstance {
    /// Returns the `wgpu` vertex buffer layout for the instance buffer.
    ///
    /// Step mode is [`wgpu::VertexStepMode::Instance`] — the GPU advances one
    /// entry per drawn rectangle, not per vertex of the shared unit quad.
    #[must_use]
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // rect: [x, y, w, h]
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // color: [r, g, b, a]
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // radii: [tl, tr, br, bl]
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

/// Vertices of the shared unit quad (two triangles via `TriangleStrip`).
///
/// All rectangle instances share this single buffer. The vertex shader scales
/// and translates the quad to match each instance's `rect` field.
const QUAD_VERTICES: &[f32] = &[
    0.0, 0.0, // Top-Left
    1.0, 0.0, // Top-Right
    0.0, 1.0, // Bottom-Left
    1.0, 1.0, // Bottom-Right
];

/// Owns all wgpu resources for the `SolidBox` render pipeline.
///
/// Initialised once via [`EncoderSubsystem::init`]. Holds `Arc` handles to
/// the device and queue so the render thread can submit commands without
/// locking the logic thread.
pub struct EncoderSubsystem {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    render_pipeline: wgpu::RenderPipeline,
    quad_buffer: wgpu::Buffer,
    viewport_buffer: wgpu::Buffer,
    viewport_bind_group: wgpu::BindGroup,
}

impl EncoderSubsystem {
    /// Initialises the GPU context and compiles the `SolidBox` pipeline.
    ///
    /// Shader compilation and pipeline creation are wrapped in a
    /// `push_error_scope` / `pop_error_scope` pair (RFC §8). Any GPU-side
    /// validation failure is returned as
    /// [`ByardError::PipelineCompilation`](crate::ByardError::PipelineCompilation)
    /// — this method never panics on a GPU error.
    ///
    /// # Errors
    ///
    /// - [`ByardError::UnsupportedBackend`] — no compatible adapter found or
    ///   device creation failed.
    /// - [`ByardError::PipelineCompilation`] — the WGSL shader or the pipeline
    ///   descriptor failed GPU-side validation.
    pub async fn init(
        instance: &wgpu::Instance,
        surface: &wgpu::Surface<'static>,
        surface_format: wgpu::TextureFormat,
    ) -> Result<Self, ByardError> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or(ByardError::UnsupportedBackend)?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("ByardCore - Encoder Device"),
                    required_features: wgpu::Features::empty(),
                    // Use the adapter's own limits — no artificial WebGL2 cap.
                    required_limits: adapter.limits(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|_| ByardError::UnsupportedBackend)?;

        // Static geometry shared by every SolidBox instance.
        let quad_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ByardCore - Static Quad Buffer"),
            contents: bytemuck::cast_slice(QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let viewport_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ByardCore - Viewport Uniform Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ByardCore - Viewport Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(16),
                },
                count: None,
            }],
        });

        let viewport_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ByardCore - Viewport Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: viewport_buffer.as_entire_binding(),
            }],
        });

        let quad_layout = wgpu::VertexBufferLayout {
            array_stride: 8, // 2 × f32
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
        };

        // `bind_group_layout` is passed into the helper so that `pipeline_layout`
        // can be created inside the error scope alongside the shader and pipeline,
        // matching the full sequence required by RFC §8.
        let render_pipeline =
            build_solid_box_pipeline(&device, &bind_group_layout, quad_layout, surface_format)
                .await?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            render_pipeline,
            quad_buffer,
            viewport_buffer,
            viewport_bind_group,
        })
    }

    /// Uploads updated viewport dimensions to the GPU uniform buffer.
    ///
    /// Must be called whenever the surface is resized before the next frame
    /// that uses this encoder.
    pub fn update_viewport(&self, viewport: Viewport) {
        // Write 16 bytes to match the padded uniform buffer size.
        // The shader only reads the first 8 bytes (vec2<f32>); the trailing
        // two floats are zero padding required by the 16-byte alignment rule.
        let size_data = [viewport.width, viewport.height, 0.0_f32, 0.0];
        self.queue
            .write_buffer(&self.viewport_buffer, 0, bytemuck::cast_slice(&size_data));
    }

    /// Encodes a single UI frame into a `CommandBuffer` ready for queue submission.
    ///
    /// Creates a transient instance buffer from `instances`, records a render pass
    /// that clears the target and draws every rectangle, then returns the finished
    /// command buffer ready for submission. The caller must submit it on the
    /// same [`wgpu::Queue`] that was used to initialise this encoder.
    ///
    /// If `instances` is empty the pass still runs (clearing the target to
    /// transparent) but no draw call is issued.
    ///
    /// # Instance buffer lifetime
    ///
    /// The buffer is allocated per call and dropped after `encoder.finish()`.
    /// For Phase 1 frame counts this is acceptable; a persistent ring-buffer
    /// upload strategy can replace this in a future sub-issue.
    #[must_use]
    pub fn encode_frame(
        &self,
        target_view: &wgpu::TextureView,
        instances: &[BoxInstance],
    ) -> wgpu::CommandBuffer {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ByardCore - Frame Command Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ByardCore - UI Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if !instances.is_empty() {
                let instance_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("ByardCore - SolidBox Instance Buffer"),
                            contents: bytemuck::cast_slice(instances),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                render_pass.set_pipeline(&self.render_pipeline);
                render_pass.set_bind_group(0, &self.viewport_bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.quad_buffer.slice(..));
                render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
                // Safety: no UI frame will ever hold 2^32 instances. The cast is
                // bounded by system memory (each BoxInstance is 48 bytes, so
                // 2^32 would require 192 GiB of RAM before reaching this code).
                #[allow(clippy::cast_possible_truncation)]
                render_pass.draw(0..4, 0..instances.len() as u32);
            }
        }

        encoder.finish()
    }
}

/// Compiles the WGSL shader and assembles the `SolidBox` render pipeline.
///
/// Separated from [`EncoderSubsystem::init`] to keep that function under the
/// 100-line lint threshold.
///
/// Per RFC §8, the full creation sequence — `create_pipeline_layout`,
/// `create_shader_module`, and `create_render_pipeline` — is wrapped inside a
/// single `push_error_scope` / `pop_error_scope` pair so that any GPU-side
/// validation failure is captured and returned as
/// [`ByardError::PipelineCompilation`].
async fn build_solid_box_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    quad_layout: wgpu::VertexBufferLayout<'static>,
    surface_format: wgpu::TextureFormat,
) -> Result<wgpu::RenderPipeline, ByardError> {
    // --- GPU VALIDATION ERROR SCOPE (RFC §8) ---
    // Covers create_pipeline_layout + create_shader_module + create_render_pipeline,
    // the three operations listed in RFC §8 as requiring capture.
    device.push_error_scope(wgpu::ErrorFilter::Validation);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ByardCore - SolidBox Pipeline Layout"),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ByardCore - SolidBox WGSL Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "solid_box.wgsl"
        ))),
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("ByardCore - SolidBox Render Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader_module,
            entry_point: Some("vs_main"),
            buffers: &[quad_layout, BoxInstance::layout()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader_module,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    if let Some(error) = device.pop_error_scope().await {
        return Err(ByardError::PipelineCompilation {
            pipeline: "SolidBox".to_string(),
            reason: error.to_string(),
        });
    }

    Ok(pipeline)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── BoxInstance layout ────────────────────────────────────────────────────
    //
    // The GPU relies on the byte layout of BoxInstance being exactly what
    // `BoxInstance::layout()` declares. Any mismatch silently corrupts every
    // rendered rectangle. These tests catch such regressions at compile time.

    #[test]
    fn box_instance_size_and_alignment() {
        // 3 fields × [f32; 4] × 4 bytes = 48 bytes total.
        // The GPU stride declaration in `layout()` hardcodes this value.
        assert_eq!(
            std::mem::size_of::<BoxInstance>(),
            48,
            "BoxInstance must be exactly 48 bytes"
        );
        // f32 requires 4-byte alignment; wgpu vertex attributes assume this.
        assert_eq!(std::mem::align_of::<BoxInstance>(), 4);
    }

    #[test]
    fn box_instance_field_offsets_match_shader_locations() {
        // `BoxInstance::layout()` declares offsets 0, 16, 32 for rect/color/radii.
        // If any field is reordered or padded, the shader sees garbage.
        assert_eq!(std::mem::offset_of!(BoxInstance, rect), 0);
        assert_eq!(std::mem::offset_of!(BoxInstance, color), 16);
        assert_eq!(std::mem::offset_of!(BoxInstance, radii), 32);
    }

    #[test]
    fn box_instance_layout_stride_step_mode_and_attributes() {
        let layout = BoxInstance::layout();

        assert_eq!(
            layout.array_stride, 48,
            "stride must equal size_of::<BoxInstance>()"
        );
        assert_eq!(
            layout.step_mode,
            wgpu::VertexStepMode::Instance,
            "must advance per instance, not per vertex"
        );

        // Verify each attribute's (shader_location, offset) pair.
        let attrs = layout.attributes;
        assert_eq!(attrs.len(), 3);

        assert_eq!(attrs[0].shader_location, 1); // rect
        assert_eq!(attrs[0].offset, 0);
        assert_eq!(attrs[0].format, wgpu::VertexFormat::Float32x4);

        assert_eq!(attrs[1].shader_location, 2); // color
        assert_eq!(attrs[1].offset, 16);
        assert_eq!(attrs[1].format, wgpu::VertexFormat::Float32x4);

        assert_eq!(attrs[2].shader_location, 3); // radii
        assert_eq!(attrs[2].offset, 32);
        assert_eq!(attrs[2].format, wgpu::VertexFormat::Float32x4);
    }

    #[test]
    fn box_instance_bytemuck_cast_produces_correct_byte_count() {
        // bytemuck::cast_slice is used in encode_frame to upload instances.
        // A wrong Pod impl (e.g. accidental padding) would give the wrong length.
        let instances = [
            BoxInstance {
                rect: [0.0, 0.0, 100.0, 50.0],
                color: [1.0, 0.0, 0.5, 1.0],
                radii: [8.0, 8.0, 8.0, 8.0],
            },
            BoxInstance {
                rect: [10.0, 20.0, 200.0, 80.0],
                color: [0.0, 1.0, 0.0, 0.8],
                radii: [0.0; 4],
            },
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&instances);
        assert_eq!(bytes.len(), 2 * 48, "2 instances × 48 bytes each");
    }

    #[test]
    // Exact bit-level zero is what Zeroable guarantees, so strict equality is
    // correct here: we are not comparing computed floats but literal bit patterns.
    #[allow(clippy::float_cmp)]
    fn box_instance_zeroed_is_valid_and_all_zero() {
        // bytemuck::Zeroable guarantees that all-zero bytes form a valid
        // BoxInstance. Used implicitly when zero-filling instance buffers.
        let z: BoxInstance = bytemuck::Zeroable::zeroed();
        assert_eq!(z.rect, [0.0; 4]);
        assert_eq!(z.color, [0.0; 4]);
        assert_eq!(z.radii, [0.0; 4]);
    }

    // ── QUAD_VERTICES ─────────────────────────────────────────────────────────

    #[test]
    // QUAD_VERTICES is a compile-time constant array of exact integer-valued
    // floats (0.0 and 1.0). Strict equality is intentional: we are verifying
    // that no rounding crept in, not comparing computed results.
    #[allow(clippy::float_cmp)]
    fn quad_vertices_form_unit_square() {
        // 4 vertices × 2 coords (x, y) = 8 floats.
        assert_eq!(QUAD_VERTICES.len(), 8);

        // Every coordinate must be exactly 0.0 or 1.0.
        for &v in QUAD_VERTICES {
            assert!(
                v == 0.0 || v == 1.0,
                "unexpected quad vertex coordinate: {v}"
            );
        }

        // All four corners of the unit square [0,1]² must be present.
        let pairs: Vec<(f32, f32)> = QUAD_VERTICES.chunks(2).map(|c| (c[0], c[1])).collect();

        assert!(pairs.contains(&(0.0, 0.0)), "missing top-left  (0,0)");
        assert!(pairs.contains(&(1.0, 0.0)), "missing top-right  (1,0)");
        assert!(pairs.contains(&(0.0, 1.0)), "missing bottom-left  (0,1)");
        assert!(pairs.contains(&(1.0, 1.0)), "missing bottom-right (1,1)");
    }

    // ── SDF ───────────────────────────────────────────────────────────────────

    /// CPU reimplementation of the `sd_rounded_box` WGSL function.
    ///
    /// Mirrors the shader logic exactly so algebraic properties can be
    /// asserted in unit tests without spinning up a GPU backend.
    fn cpu_sd_rounded_box(p: [f32; 2], b: [f32; 2], r: [f32; 4]) -> f32 {
        // Screen Y increases downward, so top half has p[1] < 0.
        // Default case (p.x <= 0 && p.y <= 0) → top-left radius.
        let mut r_corner = r[0]; // Top-Left
        if p[0] > 0.0 && p[1] < 0.0 {
            r_corner = r[1]; // Top-Right
        }
        if p[0] > 0.0 && p[1] > 0.0 {
            r_corner = r[2]; // Bottom-Right
        }
        if p[0] < 0.0 && p[1] > 0.0 {
            r_corner = r[3]; // Bottom-Left
        }

        let q_x = p[0].abs() - b[0] + r_corner;
        let q_y = p[1].abs() - b[1] + r_corner;

        let length_max_q = (q_x.max(0.0) * q_x.max(0.0) + q_y.max(0.0) * q_y.max(0.0)).sqrt();

        q_x.max(q_y).min(0.0) + length_max_q - r_corner
    }

    #[test]
    fn sdf_zero_radii_degenerates_to_axis_aligned_rect() {
        // With r=[0,0,0,0] the SDF must equal the plain AABB SDF.
        // This is the most common production case (solid rectangle, no rounding).
        let half = [50.0_f32, 50.0];
        let r = [0.0_f32; 4];

        // Centre: strictly inside.
        assert!(cpu_sd_rounded_box([0.0, 0.0], half, r) < 0.0);

        // Right edge midpoint: on the boundary → SDF = 0.
        // q_x = 50 − 50 + 0 = 0, q_y = 0 − 50 + 0 = −50
        // → min(max(0, −50), 0) + length((0, 0)) − 0 = 0
        let d = cpu_sd_rounded_box([50.0, 0.0], half, r);
        assert!(d.abs() < 0.001, "right edge: {d}");

        // Outside right edge (x=55, y=0): SDF = 5.
        // q_x = 55 − 50 = 5, q_y = 0 − 50 = −50
        // → min(max(5, −50), 0) + length((5, 0)) − 0 = 0 + 5 = 5
        let d = cpu_sd_rounded_box([55.0, 0.0], half, r);
        assert!((d - 5.0).abs() < 0.001, "right exterior: {d}");

        // Outside corner (x=55, y=55): SDF = √(5²+5²) ≈ 7.071.
        // q_x = 5, q_y = 5 → 0 + √50 ≈ 7.071
        let d = cpu_sd_rounded_box([55.0, 55.0], half, r);
        assert!((d - (50.0_f32).sqrt()).abs() < 0.001, "sharp corner: {d}");
    }

    #[test]
    fn sdf_all_four_quadrants_select_correct_radius() {
        // Fully asymmetric radii: TL=10, TR=20, BR=30, BL=40.
        // For each quadrant, place a point at a distance that depends only
        // on the corner radius of that quadrant and verify the expected SDF.
        //
        // Strategy: at (±45, ∓45) (all inside the box), the SDF depends on
        // r_corner because q_* includes `+ r_corner`. By varying only the
        // active radius we can isolate each quadrant.
        //
        // Each expected value computed analytically:
        // q = 45 − 50 + r = r − 5  (same for both axes at this symmetric point)
        // When r ≥ 5: both q components ≥ 0 → result = √2·(r−5) − r
        // When r < 5: both q components < 0 → result = (r−5) − r = −5
        fn expected(r: f32) -> f32 {
            let q = 45.0 - 50.0 + r; // = r - 5
            if q <= 0.0 {
                q - r
            } else {
                (2.0_f32).sqrt() * q - r
            }
        }

        let half = [50.0_f32, 50.0];
        let r = [10.0_f32, 20.0, 30.0, 40.0]; // TL, TR, BR, BL

        // Top-Left (p.x < 0, p.y < 0) → r[0] = 10
        let d = cpu_sd_rounded_box([-45.0, -45.0], half, r);
        assert!((d - expected(10.0)).abs() < 0.001, "TL: {d}");

        // Top-Right (p.x > 0, p.y < 0) → r[1] = 20
        let d = cpu_sd_rounded_box([45.0, -45.0], half, r);
        assert!((d - expected(20.0)).abs() < 0.001, "TR: {d}");

        // Bottom-Right (p.x > 0, p.y > 0) → r[2] = 30
        let d = cpu_sd_rounded_box([45.0, 45.0], half, r);
        assert!((d - expected(30.0)).abs() < 0.001, "BR: {d}");

        // Bottom-Left (p.x < 0, p.y > 0) → r[3] = 40
        let d = cpu_sd_rounded_box([-45.0, 45.0], half, r);
        assert!((d - expected(40.0)).abs() < 0.001, "BL: {d}");
    }

    #[test]
    // The recovered values are the same bytes written as the original —
    // no arithmetic involved, so strict bit-equality is the correct assertion.
    #[allow(clippy::float_cmp)]
    fn box_instance_bytemuck_round_trip_preserves_values() {
        // Verifies that casting BoxInstance → &[u8] → BoxInstance returns
        // identical field values. Catches any Pod impl that shuffles bytes.
        let original = BoxInstance {
            rect: [1.0, 2.0, 300.0, 400.0],
            color: [0.25, 0.5, 0.75, 1.0],
            radii: [8.0, 16.0, 24.0, 32.0],
        };

        let bytes: &[u8] = bytemuck::bytes_of(&original);
        let recovered: &BoxInstance = bytemuck::from_bytes(bytes);

        assert_eq!(recovered.rect, original.rect);
        assert_eq!(recovered.color, original.color);
        assert_eq!(recovered.radii, original.radii);
    }

    #[test]
    fn sdf_mathematical_quadrants_and_boundaries() {
        // 100×100 box centred at origin → half_size = (50, 50).
        let half_size = [50.0_f32, 50.0];
        // Asymmetric radii to verify per-corner selection.
        let radii = [10.0_f32, 15.0, 20.0, 25.0];

        // Centre is deep inside — SDF must be strongly negative.
        let dist_center = cpu_sd_rounded_box([0.0, 0.0], half_size, radii);
        assert!(dist_center < -20.0, "centre: {dist_center}");

        // Top edge midpoint (x=0, y=−50): SDF ≈ 0.
        // q_x = 0 − 50 + 10 = −40, q_y = 50 − 50 + 10 = 10
        // → min(max(−40, 10), 0) + length((0, 10)) − 10 = 0 + 10 − 10 = 0
        let dist_edge = cpu_sd_rounded_box([0.0, -50.0], half_size, radii);
        assert!(dist_edge.abs() < 0.001, "top edge: {dist_edge}");

        // Far outside corner — SDF must be substantially positive.
        let dist_outer = cpu_sd_rounded_box([100.0, 100.0], half_size, radii);
        assert!(dist_outer > 40.0, "outer: {dist_outer}");

        // Asymmetry: TL radius=10 gives a different SDF than BR radius=20
        // at the same distance from their respective corners.
        let dist_tl = cpu_sd_rounded_box([-45.0, -45.0], half_size, radii);
        let dist_br = cpu_sd_rounded_box([45.0, 45.0], half_size, radii);
        assert!(
            (dist_tl - dist_br).abs() > 1.0,
            "TL={dist_tl}, BR={dist_br} should differ"
        );

        // Axis boundary (x=0, y=−45): default case (TL radius=10) must apply.
        // q_x = 0 − 50 + 10 = −40, q_y = 45 − 50 + 10 = 5
        // → min(max(−40, 5), 0) + length((0, 5)) − 10 = 0 + 5 − 10 = −5
        let dist_boundary = cpu_sd_rounded_box([0.0, -45.0], half_size, radii);
        assert!(
            (dist_boundary - (-5.0)).abs() < 0.001,
            "axis boundary: {dist_boundary}"
        );
    }

    // ── SDF: corner arc ───────────────────────────────────────────────────────

    #[test]
    fn sdf_point_exactly_on_corner_arc_has_zero_distance() {
        // For the BR corner with r=20 and half=[50,50]:
        //   arc centre = (50−20, 50−20) = (30, 30)
        //   point at 45° on the arc: p = (30 + 20/√2, 30 + 20/√2) ≈ (44.14, 44.14)
        //
        // Manual calculation (BR quadrant → r_corner=20):
        //   q_x = 44.14 − 50 + 20 = 14.14
        //   q_y = same
        //   result = 0 + √(14.14² + 14.14²) − 20 = √400 − 20 = 0
        let half = [50.0_f32, 50.0];
        let r = [0.0_f32, 0.0, 20.0, 0.0]; // only BR has radius
        let offset = 20.0_f32 / (2.0_f32).sqrt();
        let p = [30.0 + offset, 30.0 + offset];
        let d = cpu_sd_rounded_box(p, half, r);
        assert!(d.abs() < 0.001, "on arc: {d}");
    }

    // ── SDF: degenerate radii ─────────────────────────────────────────────────

    #[test]
    fn sdf_full_radius_equals_half_size_acts_like_circle() {
        // When r == half for all corners the rounded rect degenerates to a circle.
        // For half=[25,25] and r=[25,25,25,25] the boundary lies at distance 25
        // from the origin in every direction.
        //
        // Right midpoint p=(25,0) — falls into the TL default case since p.y==0:
        //   q_x = 25−25+25 = 25, q_y = 0−25+25 = 0
        //   result = 0 + length((25,0)) − 25 = 25 − 25 = 0 (on boundary) ✓
        //
        // TR diagonal p=(25/√2, −25/√2) — TR case (p.x>0, p.y<0):
        //   q_x = q_y = 25/√2 − 25 + 25 = 25/√2 ≈ 17.68
        //   result = 0 + √(17.68²+17.68²) − 25 = 25 − 25 = 0 (on boundary) ✓
        let half = [25.0_f32, 25.0];
        let r = [25.0_f32; 4];

        let d = cpu_sd_rounded_box([25.0, 0.0], half, r);
        assert!(d.abs() < 0.001, "right midpoint: {d}");

        let diag = 25.0_f32 / (2.0_f32).sqrt();
        let d = cpu_sd_rounded_box([diag, -diag], half, r);
        assert!(d.abs() < 0.001, "TR diagonal: {d}");

        let d = cpu_sd_rounded_box([0.0, 0.0], half, r);
        assert!((d - (-25.0)).abs() < 0.001, "centre: {d}");
    }

    #[test]
    fn sdf_radius_exceeding_half_size_is_finite() {
        // The SDF function does not clamp radii to half-size. A radius larger
        // than half-size produces a mathematically valid (though visually odd)
        // value. This test documents that the function is total — it never
        // panics or returns NaN/±inf for any finite inputs.
        let half = [50.0_f32, 50.0];
        let r_big = [60.0_f32; 4];
        let r_huge = [1000.0_f32; 4];

        assert!(cpu_sd_rounded_box([0.0, 0.0], half, r_big).is_finite());
        assert!(cpu_sd_rounded_box([0.0, 0.0], half, r_huge).is_finite());
        assert!(cpu_sd_rounded_box([100.0, 100.0], half, r_big).is_finite());
    }

    // ── bytemuck: empty slice and non-finite values ───────────────────────────

    #[test]
    fn box_instance_cast_slice_empty_gives_zero_bytes() {
        // encode_frame guards with `if !instances.is_empty()` before creating
        // a buffer. Verify that casting an empty slice is safe and produces
        // zero bytes — not UB, not a panic.
        let empty: &[BoxInstance] = &[];
        let bytes: &[u8] = bytemuck::cast_slice(empty);
        assert_eq!(bytes.len(), 0);
    }

    #[test]
    fn box_instance_pod_accepts_non_finite_floats() {
        // bytemuck::Pod requires every bit pattern to be a valid value.
        // NaN and ±inf are valid f32 bit patterns, so Pod must accept them.
        // encode_frame calls bytemuck::cast_slice on instances it receives;
        // if the caller passes NaN coordinates, the cast must not panic.
        let inst = BoxInstance {
            rect: [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0],
            color: [f32::NAN; 4],
            radii: [f32::INFINITY; 4],
        };
        let bytes = bytemuck::bytes_of(&inst);
        assert_eq!(bytes.len(), 48, "NaN/inf must not change struct size");
    }

    // ── QUAD_VERTICES: TriangleStrip geometry ────────────────────────────────

    #[test]
    fn quad_vertices_triangle_strip_tiles_unit_square_without_gaps() {
        // TriangleStrip with 4 vertices produces exactly 2 triangles:
        //   T1: indices 0,1,2 → (TL, TR, BL)
        //   T2: indices 1,2,3 → (TR, BL, BR)
        //
        // Verify their combined area equals 1.0 (the unit square), which
        // proves they tile the surface without gaps or overlaps.
        fn tri_area(ax: f32, ay: f32, bx: f32, by: f32, cx: f32, cy: f32) -> f32 {
            ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)).abs() * 0.5
        }

        let p: Vec<(f32, f32)> = QUAD_VERTICES.chunks(2).map(|c| (c[0], c[1])).collect();

        let a1 = tri_area(p[0].0, p[0].1, p[1].0, p[1].1, p[2].0, p[2].1);
        let a2 = tri_area(p[1].0, p[1].1, p[2].0, p[2].1, p[3].0, p[3].1);

        assert!((a1 - 0.5).abs() < 0.001, "T1 area = {a1} (expected 0.5)");
        assert!((a2 - 0.5).abs() < 0.001, "T2 area = {a2} (expected 0.5)");
        assert!(
            (a1 + a2 - 1.0).abs() < 0.001,
            "total area = {} (expected 1.0)",
            a1 + a2
        );
    }
}
