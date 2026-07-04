//! Telemetry overlay formatting for `byard dev` (RFC-0013 §"Overlay format").
//!
//! Pure text formatting — no window, no `wgpu` device, fully unit-testable.
//! `byard dev`'s event loop (`commands/dev.rs`) drains and prints this
//! periodically on the render/main thread, combining two sources that live
//! on different threads:
//!
//! - the last published frame's CPU (`Interpreter`/`Native`) samples,
//!   drained on the **logic** thread at publish time
//!   ([`byard_core::Engine::latest_cpu_telemetry`]);
//! - this frame's GPU pass samples, pushed onto the **render/main** thread's
//!   own ring by `EncoderSubsystem` during `encode_frame_*` and drained here
//!   via [`byard_core::telemetry::drain_samples`] — they never cross into
//!   the `RenderFrame` itself, since GPU timing resolves asynchronously,
//!   independent of any particular tick.
//!
//! Flat per-scope list by default (RFC-0013 **P2**); no AOT projection is
//! ever included unless a caller explicitly opts in by calling
//! [`byard_core::telemetry::project_aot`] itself and appending its own line
//! (**P3**) — this module never calls it.

use std::fmt::Write as _;

use byard_core::telemetry::{Sample, SampleBlock, ScopeKind, scope_kind, scope_name};

/// Formats a flat, per-scope telemetry breakdown for one tick.
///
/// `cpu` is the logic thread's [`SampleBlock`] for the most recently
/// published frame; `gpu` is the calling thread's own ring, expected to hold
/// this frame's `Gpu`-tagged samples (drained separately — see the module
/// docs for why). `gpu_available` selects between listing `gpu`'s samples
/// and a "GPU timing unavailable" notice (RFC-0013 **P5**) — pass
/// [`byard_core::Engine::gpu_timing_available`].
#[must_use]
pub fn format_telemetry_overlay(
    cpu: &SampleBlock,
    gpu: &SampleBlock,
    gpu_available: bool,
) -> String {
    let cpu_total: u64 = cpu.samples.iter().map(Sample::duration_ns).sum();
    let gpu_total: u64 = gpu.samples.iter().map(Sample::duration_ns).sum();
    let interp_tax = cpu.interpreter_tax_ns();

    let mut out = format!(
        "measured total {}  (interp tax {})\n",
        fmt_ms(cpu_total + gpu_total),
        fmt_ms(interp_tax),
    );

    for sample in &cpu.samples {
        let name = scope_name(sample.scope).unwrap_or("<unknown scope>");
        let tag = matches!(scope_kind(sample.scope), Some(ScopeKind::Interpreter))
            .then_some("  [INTERPRETER — 0 in release]")
            .unwrap_or_default();
        let _ = writeln!(out, "  {name:<28} {}{tag}", fmt_ms(sample.duration_ns()));
    }

    if gpu_available {
        for sample in &gpu.samples {
            let name = scope_name(sample.scope).unwrap_or("<unknown scope>");
            let _ = writeln!(
                out,
                "  {name:<28} {}  (async, -2f)",
                fmt_ms(sample.duration_ns())
            );
        }
    } else {
        out.push_str("  GPU timing unavailable (device lacks TIMESTAMP_QUERY)\n");
    }

    let dropped = cpu.dropped + gpu.dropped;
    if dropped > 0 {
        let _ = writeln!(out, "  ({dropped} sample(s) dropped this tick — ring full)");
    }

    out
}

#[allow(clippy::cast_precision_loss)] // frame times never approach f64's precision limit
fn fmt_ms(ns: u64) -> String {
    format!("{:.2}ms", ns as f64 / 1_000_000.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use byard_core::telemetry::{ScopeId, scope_id_tagged};

    fn sample(scope: ScopeId, duration_ns: u64) -> Sample {
        Sample::gpu_duration(scope, duration_ns) // start=0, end=duration — fine for CPU too here
    }

    #[test]
    fn lists_every_cpu_scope_with_its_duration() {
        let native = scope_id_tagged("overlay.test.native_scope", ScopeKind::Native);
        let cpu = SampleBlock {
            samples: vec![sample(native, 1_500_000)],
            dropped: 0,
        };
        let out = format_telemetry_overlay(&cpu, &SampleBlock::default(), true);
        assert!(out.contains("overlay.test.native_scope"));
        assert!(out.contains("1.50ms"));
    }

    #[test]
    fn tags_interpreter_scopes_with_the_zero_in_release_note() {
        let interp = scope_id_tagged("overlay.test.interp_scope", ScopeKind::Interpreter);
        let cpu = SampleBlock {
            samples: vec![sample(interp, 2_000_000)],
            dropped: 0,
        };
        let out = format_telemetry_overlay(&cpu, &SampleBlock::default(), true);
        assert!(out.contains("[INTERPRETER — 0 in release]"));
        assert!(out.contains("interp tax 2.00ms"));
    }

    #[test]
    fn gpu_rows_are_marked_async_minus_two_frames() {
        let gpu_scope = scope_id_tagged("overlay.test.gpu_scope", ScopeKind::Gpu);
        let gpu = SampleBlock {
            samples: vec![sample(gpu_scope, 900_000)],
            dropped: 0,
        };
        let out = format_telemetry_overlay(&SampleBlock::default(), &gpu, true);
        assert!(out.contains("overlay.test.gpu_scope"));
        assert!(out.contains("(async, -2f)"));
    }

    #[test]
    fn shows_the_unavailable_notice_instead_of_fabricating_gpu_rows() {
        let gpu_scope = scope_id_tagged("overlay.test.gpu_scope_unavailable", ScopeKind::Gpu);
        let gpu = SampleBlock {
            samples: vec![sample(gpu_scope, 900_000)],
            dropped: 0,
        };
        let out = format_telemetry_overlay(&SampleBlock::default(), &gpu, false);
        assert!(out.contains("GPU timing unavailable"));
        assert!(
            !out.contains("overlay.test.gpu_scope_unavailable"),
            "an unavailable timer must never report fabricated GPU rows"
        );
    }

    #[test]
    fn reports_dropped_samples_from_either_ring() {
        let cpu = SampleBlock {
            samples: vec![],
            dropped: 3,
        };
        let gpu = SampleBlock {
            samples: vec![],
            dropped: 2,
        };
        let out = format_telemetry_overlay(&cpu, &gpu, true);
        assert!(out.contains("5 sample(s) dropped"));
    }

    #[test]
    fn never_includes_an_aot_projection_by_default() {
        // RFC-0013 P3: the projection is opt-in; this formatter never calls
        // `project_aot` itself, so its output must never claim a projected
        // number unless a caller appends one explicitly (out of scope here).
        let out = format_telemetry_overlay(&SampleBlock::default(), &SampleBlock::default(), true);
        assert!(!out.to_lowercase().contains("proj"));
    }
}
