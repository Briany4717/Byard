//! Async GPU pass timing via `wgpu` timestamp queries (RFC-0013 §"GPU
//! timing").
//!
//! The CPU never blocks on the GPU to read a timing back: a frame's
//! timestamps are resolved into one of a small ring of readback buffers and
//! `map_async`'d; [`GpuTimer::drain_ready`] only ever polls
//! [`wgpu::PollType::Poll`] (non-blocking) and reports whichever slots have
//! already completed — by construction that is always at least
//! [`READBACK_LAG`] frames after the frame that produced them (RFC-0013:
//! "read it two frames later").
//!
//! Degrades to `None` from [`GpuTimer::new`] when the device lacks
//! [`wgpu::Features::TIMESTAMP_QUERY`] — GPU timing is reported as
//! unavailable rather than fabricated (RFC-0013 **P5**).

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::telemetry::{Sample, ScopeId, ScopeKind, scope_id_tagged};

#[cfg(test)]
use crate::telemetry::scope_kind;

/// How many frames the async readback lags behind the frame it measures.
const READBACK_LAG: usize = 2;
/// One slot per in-flight frame: the one currently being written, plus one
/// per frame of lag before its readback can be reused.
const SLOTS: usize = READBACK_LAG + 1;

const SLOT_IDLE: u8 = 0;
const SLOT_PENDING: u8 = 1;
const SLOT_READY: u8 = 2;
const SLOT_FAILED: u8 = 3;

struct Slot {
    buffer: wgpu::Buffer,
    state: Arc<AtomicU8>,
}

/// One named GPU pass, tracked as a pair of timestamp-query slots.
struct GpuScope {
    name: &'static str,
    scope_id: ScopeId,
    begin_index: u32,
    end_index: u32,
}

/// Owns the timestamp query set and the readback ring for a fixed list of
/// named GPU passes, established once at construction.
pub struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    slots: [Slot; SLOTS],
    scopes: Vec<GpuScope>,
    period_ns: f64,
    frame_index: usize,
}

impl GpuTimer {
    /// Builds a timer for `scope_names`, one GPU pass each. Returns `None`
    /// if `device` lacks [`wgpu::Features::TIMESTAMP_QUERY`] (RFC-0013
    /// **P5**) — the caller keeps rendering with GPU timing simply absent.
    ///
    /// # Panics
    ///
    /// Panics if `scope_names` has more than `u32::MAX / 2` entries — this
    /// codebase never registers more than a handful of named GPU passes.
    #[must_use]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scope_names: &[&'static str],
    ) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let count = u32::try_from(scope_names.len() * 2)
            .expect("far fewer than u32::MAX GPU scopes are ever registered");
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("ByardCore - GPU Timestamp Queries"),
            ty: wgpu::QueryType::Timestamp,
            count,
        });
        let resolve_size = u64::from(count) * 8;
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ByardCore - GPU Timestamp Resolve"),
            size: resolve_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let slots = std::array::from_fn(|_| Slot {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ByardCore - GPU Timestamp Readback"),
                size: resolve_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            state: Arc::new(AtomicU8::new(SLOT_IDLE)),
        });
        let scopes = scope_names
            .iter()
            .enumerate()
            .map(|(i, &name)| GpuScope {
                name,
                scope_id: scope_id_tagged(name, ScopeKind::Gpu),
                begin_index: u32::try_from(i * 2).expect("index fits u32"),
                end_index: u32::try_from(i * 2 + 1).expect("index fits u32"),
            })
            .collect();
        Some(Self {
            query_set,
            resolve_buffer,
            slots,
            scopes,
            period_ns: f64::from(queue.get_timestamp_period()),
            frame_index: 0,
        })
    }

    /// Returns the `RenderPassDescriptor::timestamp_writes` value for the
    /// named pass, or `None` if `name` was never registered with
    /// [`GpuTimer::new`].
    #[must_use]
    pub fn timestamp_writes(&self, name: &str) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let scope = self.scopes.iter().find(|s| s.name == name)?;
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(scope.begin_index),
            end_of_pass_write_index: Some(scope.end_index),
        })
    }

    /// Resolves this frame's timestamps into the current slot's readback
    /// buffer. Call once per frame, after every timestamp-writing pass has
    /// been recorded on `encoder`, and before the command buffer is
    /// finished. Pair with [`GpuTimer::request_map`] once that command
    /// buffer has actually been submitted — `wgpu` rejects a submission that
    /// writes into a buffer already under an active `map_async` request, so
    /// the two must not be merged into one call.
    pub fn resolve_and_copy(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let slot_idx = self.frame_index % SLOTS;
        // A slot still holding an unread mapping from an earlier lap of the
        // ring is reclaimed here rather than overwritten silently — its
        // (stale) result is simply dropped, since a caller that never called
        // `drain_ready` in `READBACK_LAG` frames wasn't going to read it.
        if self.slots[slot_idx].state.load(Ordering::Acquire) != SLOT_IDLE {
            self.slots[slot_idx].buffer.unmap();
            self.slots[slot_idx]
                .state
                .store(SLOT_IDLE, Ordering::Release);
        }

        let count = u32::try_from(self.scopes.len() * 2).unwrap_or(0);
        encoder.resolve_query_set(&self.query_set, 0..count, &self.resolve_buffer, 0);
        let size = u64::from(count) * 8;
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.slots[slot_idx].buffer,
            0,
            size,
        );
        self.frame_index += 1;
    }

    /// Requests the async map for the slot [`GpuTimer::resolve_and_copy`]
    /// just filled. Call once per frame, immediately after the command
    /// buffer containing that copy has been submitted to the queue.
    pub fn request_map(&mut self) {
        let slot_idx = (self.frame_index - 1) % SLOTS;
        let state = Arc::clone(&self.slots[slot_idx].state);
        state.store(SLOT_PENDING, Ordering::Release);
        self.slots[slot_idx]
            .buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                state.store(
                    if result.is_ok() {
                        SLOT_READY
                    } else {
                        SLOT_FAILED
                    },
                    Ordering::Release,
                );
            });
    }

    /// Non-blocking: drives any completed `map_async` callbacks
    /// (`wgpu::PollType::Poll`, never `Wait`) and appends a `Gpu`-tagged
    /// [`Sample`] to `out` for every ready slot's pass. A slot that isn't
    /// ready yet is left pending and checked again on a later call.
    pub fn drain_ready(&mut self, device: &wgpu::Device, out: &mut Vec<Sample>) {
        let _ = device.poll(wgpu::PollType::Poll);
        for slot in &mut self.slots {
            let state = slot.state.load(Ordering::Acquire);
            if state != SLOT_READY && state != SLOT_FAILED {
                continue;
            }
            if state == SLOT_READY {
                {
                    let view = slot.buffer.slice(..).get_mapped_range();
                    let raw: &[u64] = bytemuck::cast_slice(&view);
                    for scope in &self.scopes {
                        let begin = raw[scope.begin_index as usize];
                        let end = raw[scope.end_index as usize];
                        #[allow(
                            clippy::cast_precision_loss,
                            clippy::cast_sign_loss,
                            clippy::cast_possible_truncation
                        )]
                        // GPU pass durations never approach f64's precision or u64's range
                        let duration_ns =
                            (end.saturating_sub(begin) as f64 * self.period_ns) as u64;
                        out.push(Sample::gpu_duration(scope.scope_id, duration_ns));
                    }
                }
                slot.buffer.unmap();
            }
            slot.state.store(SLOT_IDLE, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("ByardCore - GpuTimer Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }))
        .ok()
    }

    #[test]
    fn new_returns_none_without_the_timestamp_query_feature() {
        // RFC-0013 P5: a device with no TIMESTAMP_QUERY support (this test's
        // device requests `Features::empty()`) degrades to unavailable
        // rather than fabricating GPU timings.
        let Some((device, queue)) = try_device() else {
            eprintln!("no GPU adapter — skipping GpuTimer capability test");
            return;
        };
        assert!(
            !device.features().contains(wgpu::Features::TIMESTAMP_QUERY),
            "this test assumes a device created without TIMESTAMP_QUERY"
        );
        assert!(GpuTimer::new(&device, &queue, &["gpu.test_pass"]).is_none());
    }

    /// Returns `(device, queue)` created with `TIMESTAMP_QUERY`, or `None` if
    /// no adapter is present or none supports it (headless CI safe, mirrors
    /// `m21_pipelines.rs`'s `try_device`).
    fn try_timestamp_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        if !adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("ByardCore - GpuTimer Test Device (timestamps)"),
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }))
        .ok()
    }

    #[test]
    fn new_succeeds_and_tags_scopes_gpu_when_the_feature_is_available() {
        let Some((device, queue)) = try_timestamp_device() else {
            eprintln!("no TIMESTAMP_QUERY-capable adapter — skipping GpuTimer availability test");
            return;
        };

        let timer = GpuTimer::new(&device, &queue, &["gpu.test_pass"])
            .expect("TIMESTAMP_QUERY is enabled on this device");
        assert!(timer.timestamp_writes("gpu.test_pass").is_some());
        assert!(timer.timestamp_writes("gpu.nonexistent_pass").is_none());
        assert_eq!(
            crate::telemetry::scope_kind(timer.scopes[0].scope_id),
            Some(ScopeKind::Gpu)
        );
    }

    #[test]
    fn a_timed_pass_eventually_yields_a_gpu_tagged_sample() {
        // End-to-end: record a real (trivial) render pass with timestamp
        // writes, resolve + request its readback, then poll non-blockingly
        // (RFC-0013: never `PollType::Wait`) until the async map completes.
        let Some((device, queue)) = try_timestamp_device() else {
            eprintln!("no TIMESTAMP_QUERY-capable adapter — skipping GpuTimer round-trip test");
            return;
        };
        let mut timer = GpuTimer::new(&device, &queue, &["gpu.test_pass"])
            .expect("TIMESTAMP_QUERY is enabled on this device");

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ByardCore - GpuTimer Test Target"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ByardCore - GpuTimer Test Encoder"),
        });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ByardCore - GpuTimer Test Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: timer.timestamp_writes("gpu.test_pass"),
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        timer.resolve_and_copy(&mut encoder);
        queue.submit(std::iter::once(encoder.finish()));
        timer.request_map();

        let mut samples = Vec::new();
        for _ in 0..200 {
            timer.drain_ready(&device, &mut samples);
            if !samples.is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(samples.len(), 1, "the single timed pass yields one sample");
        assert_eq!(scope_kind(samples[0].scope), Some(ScopeKind::Gpu));
    }
}
