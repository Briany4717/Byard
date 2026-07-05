//! Dev JIT pipeline: cache, dedup, and background dispatch for MSDF vector
//! icons (RFC-0009 §2 as corrected by §2-B/§2-C).
//!
//! Generation runs on its own one-shot worker thread — never the logic or
//! render thread (INV-9). Results cross a `crossbeam` channel to whoever
//! calls [`VectorJit::drain_ready`]; that must be the **logic thread**, the
//! only place a UV slot is allocated and an `AtlasUpload` recorded (INV-2).
//! This module never touches a `wgpu::Queue` (INV-8) — the render thread
//! alone applies the resulting uploads.
//!
//! The atlas allocator here is a minimal fixed-grid bump allocator (no reuse,
//! no eviction) — a stand-in until the shelf/skyline + LRU allocator lands.

use std::collections::HashMap;

use byard_core::encoder::vector_msdf::ATLAS_SIZE;
use byard_core::frame::{AtlasUpload, Rect};
use crossbeam_channel::{Receiver, Sender};

use super::generate::{GRID_SIZE, PX_RANGE, generate};
use crate::diagnostics::Span;

/// A resident glyph's location in the atlas — everything a `VectorInstance`
/// needs besides its screen rect and tint (RFC-0009 §1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidentGlyph {
    /// Normalized UV rect within the atlas.
    pub uv_rect: Rect,
    /// Array-texture layer.
    pub layer: u32,
    /// Baked distance range (RFC-0009 §2-E).
    pub px_range: f32,
}

enum CacheEntry {
    Pending,
    Resident {
        glyph: ResidentGlyph,
        /// Owned field bytes, kept around so the upload can be **re-emitted**
        /// on every tick until [`ack_receiver`](VectorJit) confirms the render
        /// thread actually applied it.
        bytes: std::sync::Arc<[u8]>,
        width: u32,
        height: u32,
        /// This upload's identity, echoed back through the ack channel.
        upload_id: u64,
        /// Set once the render thread confirms it applied `upload_id`.
        acked: bool,
    },
    /// Generation failed; stays a permanent placeholder until a hot-reload
    /// invalidates the entry.
    Failed,
}

struct JitMessage {
    handle: String,
    result: Result<super::generate::MsdfGlyph, String>,
}

const CELLS_PER_ROW: u32 = ATLAS_SIZE / GRID_SIZE;
const CELLS_PER_LAYER: u32 = CELLS_PER_ROW * CELLS_PER_ROW;
/// Fixed layer cap for this minimal allocator; the shelf/skyline + LRU
/// allocator replaces this with growth + eviction.
const MAX_LAYERS: u32 = 4;

/// Cache + dispatcher for dev-mode MSDF generation, owned by the interpreter.
pub struct VectorJit {
    entries: HashMap<String, CacheEntry>,
    sender: Sender<JitMessage>,
    receiver: Receiver<JitMessage>,
    next_cell: u32,
    next_upload_id: u64,
    /// Receives the ids of uploads the render thread has actually applied
    /// (wired in by the host via [`VectorJit::set_ack_receiver`]; `None` in
    /// contexts with no render thread, e.g. most unit tests — an upload then
    /// simply keeps resending forever, which is harmless there).
    ack_receiver: Option<Receiver<u64>>,
}

impl Default for VectorJit {
    fn default() -> Self {
        let (sender, receiver) = crossbeam_channel::unbounded();
        Self {
            entries: HashMap::new(),
            sender,
            receiver,
            next_cell: 0,
            next_upload_id: 0,
            ack_receiver: None,
        }
    }
}

impl VectorJit {
    /// Creates an empty cache with no in-flight or resident glyphs.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Wires in the channel the render thread reports applied-upload ids
    /// through (see [`byard_core::encoder::EncoderSubsystem::set_vector_ack_sender`]).
    /// Without this, a resident glyph's upload is re-attached to every tick's
    /// frame forever rather than only until acknowledged — correct but
    /// wasteful, so callers with a real render thread should always wire it.
    pub fn set_ack_receiver(&mut self, rx: Receiver<u64>) {
        self.ack_receiver = Some(rx);
    }

    /// Looks up `handle` (an SVG file path). Returns its resident atlas
    /// location if already generated; otherwise dispatches a one-shot
    /// generation task (deduped — a second miss on the same handle while one
    /// is already pending does not spawn another) and returns `None`, so the
    /// caller emits a placeholder this tick (INV-9).
    pub fn lookup_or_dispatch(&mut self, handle: &str) -> Option<ResidentGlyph> {
        if handle.is_empty() {
            return None;
        }
        match self.entries.get(handle) {
            Some(CacheEntry::Resident { glyph, .. }) => return Some(*glyph),
            Some(CacheEntry::Pending | CacheEntry::Failed) => return None,
            None => {}
        }
        self.entries.insert(handle.to_string(), CacheEntry::Pending);
        self.dispatch(handle.to_string());
        None
    }

    fn dispatch(&self, handle: String) {
        eprintln!("vector: dispatching generation for {handle:?}");
        let tx = self.sender.clone();
        std::thread::spawn(move || {
            let result = std::fs::read(&handle)
                .map_err(|e| format!("failed to read {handle}: {e}"))
                .and_then(|bytes| {
                    generate(&bytes, GRID_SIZE, PX_RANGE, Span::new(0, 0)).map_err(|e| e.headline())
                });
            let _ = tx.send(JitMessage { handle, result });
        });
    }

    /// Drains every generation that completed since the last call, allocates
    /// each a fresh atlas cell, marks it resident, and returns the
    /// [`AtlasUpload`]s to attach to this tick's frame — plus a re-send of
    /// every still-unacknowledged resident upload, so a `RenderFrame` the
    /// render thread happens to skip never permanently loses one. **Logic
    /// thread only** (INV-2) — call once per tick, before building the frame.
    pub fn drain_ready(&mut self) -> Vec<AtlasUpload> {
        let mut uploads = Vec::new();

        // Mark acknowledged uploads first, so the resend pass below doesn't
        // re-attach one the render thread already confirmed this same tick.
        if let Some(rx) = &self.ack_receiver {
            let mut acked_ids = std::collections::HashSet::new();
            while let Ok(id) = rx.try_recv() {
                acked_ids.insert(id);
            }
            if !acked_ids.is_empty() {
                for entry in self.entries.values_mut() {
                    if let CacheEntry::Resident {
                        upload_id, acked, ..
                    } = entry
                    {
                        if acked_ids.contains(upload_id) {
                            *acked = true;
                        }
                    }
                }
            }
        }

        // Handles that just became resident this call already have their one
        // upload pushed below; skip them in the resend pass so they aren't
        // double-uploaded on their very first tick.
        let mut just_completed: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        while let Ok(msg) = self.receiver.try_recv() {
            match msg.result {
                Ok(glyph) => {
                    if let Some((x, y, layer)) = self.alloc_cell() {
                        #[allow(clippy::cast_precision_loss)]
                        let atlas_size = ATLAS_SIZE as f32;
                        #[allow(clippy::cast_precision_loss)]
                        let cell_size = GRID_SIZE as f32;
                        let uv_rect = Rect::new(
                            x as f32 / atlas_size,
                            y as f32 / atlas_size,
                            cell_size / atlas_size,
                            cell_size / atlas_size,
                        );
                        let upload_id = self.next_upload_id;
                        self.next_upload_id += 1;
                        let bytes: std::sync::Arc<[u8]> = glyph.bitmap.into();
                        uploads.push(AtlasUpload {
                            layer,
                            x,
                            y,
                            width: glyph.width,
                            height: glyph.height,
                            bytes: bytes.to_vec(),
                            id: upload_id,
                        });
                        eprintln!(
                            "vector: {:?} is now resident (layer {layer}, cell {x},{y}, {}x{} px)",
                            msg.handle, glyph.width, glyph.height
                        );
                        just_completed.insert(msg.handle.clone());
                        self.entries.insert(
                            msg.handle,
                            CacheEntry::Resident {
                                glyph: ResidentGlyph {
                                    uv_rect,
                                    layer,
                                    px_range: glyph.px_range,
                                },
                                bytes,
                                width: glyph.width,
                                height: glyph.height,
                                upload_id,
                                acked: false,
                            },
                        );
                    } else {
                        eprintln!(
                            "vector atlas is full; {} could not be placed (eviction is not yet implemented)",
                            msg.handle
                        );
                        self.entries.insert(msg.handle, CacheEntry::Failed);
                    }
                }
                Err(reason) => {
                    eprintln!(
                        "failed to generate an MSDF field for {}: {reason}",
                        msg.handle
                    );
                    self.entries.insert(msg.handle, CacheEntry::Failed);
                }
            }
        }
        // Re-attach every still-unacknowledged resident upload so a skipped
        // RenderFrame never loses it permanently — no arbitrary time or tick
        // limit, since generation/render pacing can't be bounded in general;
        // this simply stops once the render thread confirms receipt.
        for (handle, entry) in &mut self.entries {
            if just_completed.contains(handle) {
                continue;
            }
            if let CacheEntry::Resident {
                glyph,
                bytes,
                width,
                height,
                upload_id,
                acked,
            } = entry
            {
                if !*acked {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let x = (glyph.uv_rect.x * ATLAS_SIZE as f32).round() as u32;
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let y = (glyph.uv_rect.y * ATLAS_SIZE as f32).round() as u32;
                    uploads.push(AtlasUpload {
                        layer: glyph.layer,
                        x,
                        y,
                        width: *width,
                        height: *height,
                        bytes: bytes.to_vec(),
                        id: *upload_id,
                    });
                }
            }
        }
        uploads
    }

    fn alloc_cell(&mut self) -> Option<(u32, u32, u32)> {
        let idx = self.next_cell;
        let layer = idx / CELLS_PER_LAYER;
        if layer >= MAX_LAYERS {
            return None;
        }
        let local = idx % CELLS_PER_LAYER;
        let cx = local % CELLS_PER_ROW;
        let cy = local / CELLS_PER_ROW;
        self.next_cell += 1;
        Some((cx * GRID_SIZE, cy * GRID_SIZE, layer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_gear_fixture() -> String {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/svg/gear.svg").to_string();
        assert!(std::path::Path::new(&path).exists(), "fixture must exist");
        path
    }

    fn wait_for_drain(jit: &mut VectorJit) -> Vec<AtlasUpload> {
        // Generation is fast (32x32 grid); a short poll loop is enough and
        // keeps the test from being flaky under CI load.
        for _ in 0..200 {
            let uploads = jit.drain_ready();
            if !uploads.is_empty() {
                return uploads;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        Vec::new()
    }

    #[test]
    fn first_miss_returns_none_and_dispatches() {
        let path = write_gear_fixture();
        let mut jit = VectorJit::new();
        assert!(jit.lookup_or_dispatch(&path).is_none());
    }

    #[test]
    fn a_later_tick_resolves_the_same_handle_to_resident() {
        let path = write_gear_fixture();
        let mut jit = VectorJit::new();
        assert!(jit.lookup_or_dispatch(&path).is_none());
        let uploads = wait_for_drain(&mut jit);
        assert_eq!(
            uploads.len(),
            1,
            "exactly one glyph should have been generated"
        );
        let resident = jit
            .lookup_or_dispatch(&path)
            .expect("the handle must be resident after draining");
        assert_eq!(resident.layer, 0);
    }

    #[test]
    fn an_unacknowledged_upload_is_resent_on_subsequent_ticks() {
        // Regression: `RenderFrame`s are read latest-wins by the render
        // thread (RFC-0001 §5.2). A one-shot upload attached to exactly the
        // tick it completed on can land in a skipped frame and be lost
        // forever, even though the cache believes the glyph is resident. The
        // fix re-attaches the same upload every tick until acknowledged.
        let path = write_gear_fixture();
        let mut jit = VectorJit::new();
        assert!(jit.lookup_or_dispatch(&path).is_none());
        let first = wait_for_drain(&mut jit);
        assert_eq!(
            first.len(),
            1,
            "the completing tick emits exactly one upload"
        );

        let second = jit.drain_ready();
        assert_eq!(
            second.len(),
            1,
            "the very next tick must resend the same upload, not drop it"
        );
        assert_eq!(second[0].id, first[0].id);
        assert_eq!(second[0].layer, first[0].layer);
        assert_eq!(second[0].x, first[0].x);
        assert_eq!(second[0].y, first[0].y);
        assert_eq!(second[0].bytes, first[0].bytes);
    }

    #[test]
    fn resend_survives_a_burst_of_ticks_before_any_render_read() {
        // Regression (found via manual testing): a real dev session can burst
        // through hundreds of logic ticks before the render thread's very
        // first draw (e.g. a flurry of startup/resize input keeps the logic
        // thread off the idle-park path, RFC-0001 §5.1). Since there is no
        // time or tick limit — only "has it been acknowledged?" — no burst,
        // however large, can exhaust the resend.
        let path = write_gear_fixture();
        let mut jit = VectorJit::new();
        assert!(jit.lookup_or_dispatch(&path).is_none());
        let first = wait_for_drain(&mut jit);
        assert_eq!(first.len(), 1);

        for _ in 0..500 {
            jit.drain_ready();
        }

        let after_burst = jit.drain_ready();
        assert_eq!(
            after_burst.len(),
            1,
            "an unacknowledged upload must still be attached after hundreds of ticks"
        );
    }

    #[test]
    fn acknowledging_an_upload_stops_the_resend() {
        let path = write_gear_fixture();
        let mut jit = VectorJit::new();
        let (ack_tx, ack_rx) = crossbeam_channel::unbounded();
        jit.set_ack_receiver(ack_rx);

        assert!(jit.lookup_or_dispatch(&path).is_none());
        let first = wait_for_drain(&mut jit);
        assert_eq!(first.len(), 1);

        ack_tx.send(first[0].id).unwrap();
        let after_ack = jit.drain_ready();
        assert!(
            after_ack.is_empty(),
            "an acknowledged upload must not be resent"
        );
    }

    #[test]
    fn duplicate_misses_before_drain_generate_only_once() {
        let path = write_gear_fixture();
        let mut jit = VectorJit::new();
        assert!(jit.lookup_or_dispatch(&path).is_none());
        // A second miss on the same handle while it is still pending must not
        // spawn a second generation task.
        assert!(jit.lookup_or_dispatch(&path).is_none());
        let uploads = wait_for_drain(&mut jit);
        assert_eq!(
            uploads.len(),
            1,
            "the duplicate miss must not have generated twice"
        );
    }

    #[test]
    fn a_missing_file_fails_without_panicking() {
        let mut jit = VectorJit::new();
        assert!(jit.lookup_or_dispatch("/nonexistent/icon.svg").is_none());
        for _ in 0..100 {
            jit.drain_ready();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        // A second lookup on the now-`Failed` handle must stay `None`, not panic.
        assert!(jit.lookup_or_dispatch("/nonexistent/icon.svg").is_none());
    }
}
