//! `byard dev [file]` — the live-reload dev runner (RFC-0006 §3).
//!
//! Thread model:
//!   Main/winit thread → `on_resume` → `start_logic_from_view` (logic thread)
//!                     → `start_watcher` (OS notify thread)
//!   Logic thread: drain channel → `dispatch_events` → tick → render / error overlay
//!   Watcher thread: file change → re-parse → `LatestWins::publish`

use byard_compiler::CompileError;
use byard_compiler::interp::eval::{Interpreter, RenderNode};
use byard_compiler::interp::reload::{Gated, ViewReload};
use byard_compiler::interp::reload::{
    LatestWins, ParsedFile, ReloadKind, diff_program, gate, start_watcher,
};
use byard_compiler::parser::ast::ViewDecl;
use byard_compiler::parser::parse;
use byard_core::frame::{BoxInstance, RenderFrame, TargetId, TextLine};
use byard_core::{
    ByardError, Engine, LogicRuntime, PlatformHost, PointerButton, PointerState, WindowSize,
};
use byard_platform::WinitHost;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::manifest::Manifest;

pub fn run(file: Option<&Path>) -> Result<(), String> {
    let manifest = Manifest::discover(file)?;

    // Initial parse on the main thread: catch errors before opening the window.
    let src = std::fs::read_to_string(&manifest.entry)
        .map_err(|e| format!("{}: {e}", manifest.entry.display()))?;
    let parsed = parse(&src);

    let title = format!("Byard dev — {}", manifest.name);
    println!("  Byard 0.0.0 — dev mode");
    println!("  Entry: {}", manifest.entry.display());
    println!("  Watching for changes…");
    if parsed.errors.is_empty() {
        println!("  Loaded ({} views, 0 errors)", parsed.views.len());
    } else {
        println!(
            "  Loaded with {} error(s) — see overlay",
            parsed.errors.len()
        );
    }

    let initial_views = parsed.views;
    let initial_errors = parsed.errors;
    let entry_path = manifest.entry.clone();

    // Poll mode: the event loop spins continuously so file-change frames
    // appear without waiting for the next mouse event (RFC-0006 §3.2).
    let host = WinitHost::new(&title, 1280, 720).with_poll();
    host.run(App {
        engine: None,
        width_bits: None,
        height_bits: None,
        entry_path,
        initial_views,
        initial_errors,
        last_gpu_telemetry: byard_core::telemetry::SampleBlock::default(),
        last_telemetry_print: std::time::Instant::now(),
    })
    .map_err(|e| format!("event loop error: {e}"))
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() * 1000 + u64::from(d.subsec_millis()))
}

// ── Logic runtime ─────────────────────────────────────────────────────────────

struct ByldRuntime {
    interp: Interpreter,
    tree: Vec<RenderNode>,
    current_views: Vec<ViewDecl>,
    reload_channel: Arc<LatestWins<ParsedFile>>,
    /// A structure-incompatible reload held during an in-flight gesture (E5).
    pending_reload: Option<(Vec<ViewDecl>, ReloadKind)>,
    /// Parse errors from the last file save (drives error overlay, RFC-0006 §3.4).
    error_state: Option<Vec<CompileError>>,
    width_bits: Arc<AtomicU32>,
    height_bits: Arc<AtomicU32>,
}

impl ByldRuntime {
    fn apply_reload(&mut self, new_views: &[ViewDecl], _kind: ReloadKind) {
        // For a single-view project, reload the first matching view by name.
        // Multi-view support (D11 per-ViewDecl) is wired but only the first
        // View is tracked by the renderer at this stage.
        if let (Some(old), Some(new)) = (self.current_views.first(), new_views.first()) {
            let diff_kind = byard_compiler::interp::reload::diff_view(old, new);
            self.interp.reload(new, diff_kind);
            self.tree = self.interp.lower_view(new, &[]);
        }
        self.current_views = new_views.to_vec();
        self.error_state = None;
    }
}

impl LogicRuntime for ByldRuntime {
    fn evaluate_tick(
        &mut self,
        frame: &mut RenderFrame,
        input_events: &[byard_core::platform::InputEvent],
        _dirty: &[TargetId],
    ) {
        // ── Step 0: drain latest-wins reload channel (RFC-0006 §3.2 C3) ───────
        if let Some(parsed) = self.reload_channel.take() {
            if parsed.errors.is_empty() {
                let pointer_pressed = self.interp.router.is_pointer_pressed();
                // Classify the worst-case kind across all changed views.
                let diffs = diff_program(&self.current_views, &parsed.views);
                let worst =
                    diffs
                        .iter()
                        .fold(ReloadKind::ReactiveCompatible, |acc, (_, r)| match r {
                            ViewReload::Patch(ReloadKind::StructureIncompatible)
                            | ViewReload::Added
                            | ViewReload::Removed => ReloadKind::StructureIncompatible,
                            ViewReload::Patch(ReloadKind::ReactiveCompatible) => acc,
                        });
                match gate(worst, pointer_pressed) {
                    Gated::Apply => self.apply_reload(&parsed.views, worst),
                    Gated::Defer => {
                        self.pending_reload = Some((parsed.views, worst));
                    }
                }
            } else {
                self.error_state = Some(parsed.errors);
            }
        }

        // ── Step 0b: apply deferred reload once pointer released ───────────────
        if let Some((new_views, kind)) = self.pending_reload.take() {
            if self.interp.router.is_pointer_pressed() {
                self.pending_reload = Some((new_views, kind));
            } else {
                self.apply_reload(&new_views, kind);
            }
        }

        // ── Step 1: dispatch input events ─────────────────────────────────────
        self.interp.dispatch_events(input_events);

        // ── Step 2: reactive tick ─────────────────────────────────────────────
        self.interp.tick();

        // ── Step 3: render ────────────────────────────────────────────────────
        let w = f32::from_bits(self.width_bits.load(Ordering::Relaxed));
        let h = f32::from_bits(self.height_bits.load(Ordering::Relaxed));

        if let Some(errors) = &self.error_state {
            // Render *only* the overlay — deliberately NOT the last-good view
            // underneath it. Text is drawn in a single global pass after every
            // box (the flat 4-pass encoder order), so any app text painted here
            // would bleed *over* the overlay's scrim. Drawing the overlay alone
            // on an opaque background sidesteps that and reads as a dedicated
            // "fix your file" error screen (C4: overlay path is independent of
            // the interpreter).
            render_error_overlay(frame, errors, w, h);
        } else {
            self.interp.render(&self.tree, frame, w, h);
        }
    }
}

/// Max errors shown in the overlay before truncating (Phase 2 heuristic).
const OVERLAY_MAX_ERRORS: usize = 3;
/// Max chars per headline before adding "…" (avoids horizontal overflow).
const OVERLAY_MAX_HEADLINE_CHARS: usize = 60;

/// Renders a semi-transparent error overlay directly into `frame` without
/// going through the interpreter (RFC-0006 §3.4, decision C4).
///
/// Truncates to [`OVERLAY_MAX_ERRORS`] errors and [`OVERLAY_MAX_HEADLINE_CHARS`]
/// chars per headline to keep the overlay bounded without needing Taffy layout.
fn render_error_overlay(frame: &mut RenderFrame, errors: &[CompileError], w: f32, h: f32) {
    // Opaque dark background covering the full viewport. Opaque (not a scrim)
    // because the underlying view is intentionally not drawn while errors are
    // shown — see the call site in `App::render`.
    frame.push_instance(BoxInstance {
        rect: [0.0, 0.0, w, h],
        color: [0.09, 0.09, 0.11, 1.0],
        radii: [0.0; 4],
        transform: byard_core::frame::Transform::IDENTITY,
    });

    let padding = 32.0;
    let line_height = 22.0;
    let mut y = padding + line_height;

    let title = if errors.len() == 1 {
        "Parse error".to_string()
    } else {
        format!("Parse errors ({})", errors.len())
    };
    frame.push_text(TextLine {
        x: padding,
        y,
        text: title,
        font_size: 18.0,
        color: [1.0, 0.4, 0.4, 1.0],
        dirty: true,
    });
    y += line_height * 1.5;

    // Show at most OVERLAY_MAX_ERRORS errors; truncate each headline.
    let shown = errors.len().min(OVERLAY_MAX_ERRORS);
    for err in &errors[..shown] {
        let headline = truncate_str(&err.headline(), OVERLAY_MAX_HEADLINE_CHARS);
        frame.push_text(TextLine {
            x: padding,
            y,
            text: headline,
            font_size: 15.0,
            color: [1.0, 1.0, 1.0, 1.0],
            dirty: true,
        });
        y += line_height * 1.2;
    }

    if errors.len() > OVERLAY_MAX_ERRORS {
        y += line_height * 0.3;
        frame.push_text(TextLine {
            x: padding,
            y,
            text: format!("… and {} more error(s)", errors.len() - OVERLAY_MAX_ERRORS),
            font_size: 13.0,
            color: [0.6, 0.6, 0.6, 1.0],
            dirty: true,
        });
        y += line_height;
    }

    frame.push_text(TextLine {
        x: padding,
        y: y + line_height,
        text: "Fix the file and save to dismiss.".to_string(),
        font_size: 13.0,
        color: [0.5, 0.5, 0.5, 1.0],
        dirty: true,
    });
}

/// Truncates `s` to at most `max_chars` Unicode scalar values, appending "…"
/// if truncated. Operates on chars to avoid splitting multi-byte sequences.
fn truncate_str(s: &str, max_chars: usize) -> String {
    let mut chars = s.chars();
    let mut out = String::with_capacity(s.len().min(max_chars + 3));
    let mut count = 0;
    loop {
        match chars.next() {
            Some(c) if count < max_chars => {
                out.push(c);
                count += 1;
            }
            Some(_) => {
                out.push('…');
                break;
            }
            None => break,
        }
    }
    out
}

// ── Platform host (winit integration) ────────────────────────────────────────

struct App {
    engine: Option<Engine>,
    width_bits: Option<Arc<AtomicU32>>,
    height_bits: Option<Arc<AtomicU32>>,
    entry_path: std::path::PathBuf,
    initial_views: Vec<ViewDecl>,
    initial_errors: Vec<CompileError>,
    /// This (render/main) thread's GPU telemetry ring, drained on **every**
    /// redraw — not just when about to print — so it only ever holds
    /// samples produced since the *previous redraw*. `byard dev`'s Poll-mode
    /// loop redraws far more often than once a second; draining only at
    /// print time would let a whole second's worth of `gpu.ui_pass` samples
    /// pile up into one inflated, unreadable dump (RFC-0013's overlay is
    /// meant to be a per-frame snapshot, not an accumulator).
    last_gpu_telemetry: byard_core::telemetry::SampleBlock,
    /// Last time the telemetry overlay was printed (RFC-0013 "Overlay
    /// format"), throttled to roughly once a second so `byard dev` doesn't
    /// spam a line for every redraw. Printing is throttled; draining
    /// `last_gpu_telemetry` (above) is not.
    last_telemetry_print: std::time::Instant,
}

impl App {
    /// Prints the flat telemetry overlay (RFC-0013 "Overlay format") to
    /// stderr, throttled to roughly once a second. Combines the last
    /// published frame's CPU samples (drained on the logic thread at publish
    /// time) with `gpu`, this thread's most recent single-redraw GPU
    /// samples — see `telemetry_overlay`'s module docs for why the two live
    /// on separate rings, and [`App::last_gpu_telemetry`]'s doc comment for
    /// why `gpu` must be drained every redraw rather than only here.
    fn print_telemetry_overlay(
        engine: &Engine,
        gpu: &byard_core::telemetry::SampleBlock,
        last_print: &mut std::time::Instant,
    ) {
        if last_print.elapsed() < std::time::Duration::from_secs(1) {
            return;
        }
        *last_print = std::time::Instant::now();
        let cpu = engine.latest_cpu_telemetry().unwrap_or_default();
        let overlay = crate::telemetry_overlay::format_telemetry_overlay(
            &cpu,
            gpu,
            engine.gpu_timing_available(),
        );
        eprint!("{overlay}");
    }
}

impl PlatformHost for App {
    fn on_resume(
        &mut self,
        instance: &wgpu::Instance,
        surface: wgpu::Surface<'static>,
        size: WindowSize,
        waker: byard_core::relay::FrameWaker,
    ) -> Result<(), ByardError> {
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        let w = size.width as f32 / size.scale_factor as f32;
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        let h = size.height as f32 / size.scale_factor as f32;
        let width_bits = Arc::new(AtomicU32::new(w.to_bits()));
        let height_bits = Arc::new(AtomicU32::new(h.to_bits()));
        let w_clone = Arc::clone(&width_bits);
        let h_clone = Arc::clone(&height_bits);

        // Hot-reload channel (RFC-0006 §3.5, D10).
        let reload_channel = Arc::new(LatestWins::<ParsedFile>::new());
        // Watcher lifetime tied to App; Arc shared with logic thread (C5).
        let watcher_channel = Arc::clone(&reload_channel);
        // _watcher is held in the App (C5) — store via engine field workaround:
        // we drop the watcher when the engine drops. We keep it in a Box::leak
        // for now so the OS thread stays alive for the session.
        // TODO: store in App struct properly once Engine exposes a cleanup hook.
        let watcher = start_watcher(&self.entry_path, watcher_channel)
            .map_err(|e| ByardError::RenderSurface(format!("file watcher error: {e}")))?;
        // Keep the watcher alive for the entire process lifetime.
        // This is intentional: we want file watching to persist even if the
        // logic thread is restarted due to a structure-incompatible reload.
        std::mem::forget(watcher);

        let initial_views = self.initial_views.clone();
        let initial_errors = if self.initial_errors.is_empty() {
            None
        } else {
            Some(self.initial_errors.clone())
        };

        let mut engine = pollster::block_on(Engine::init(
            instance,
            surface,
            size.width,
            size.height,
            size.scale_factor,
        ))?;
        // `byard dev` runs in Poll mode (redraws every iteration for hot-reload),
        // so the waker is not strictly required — installing it is still correct
        // and makes input-driven redraws prompt if the mode ever changes.
        engine.set_frame_waker(waker);

        engine.start_logic_from_view(move |_arena| {
            let (interp, tree, current_views) = if initial_views.is_empty() {
                (Interpreter::new(), vec![], vec![])
            } else {
                let mut interp = Interpreter::new();
                let tree = interp.lower_view(&initial_views[0], &[]);
                interp.tick();
                (interp, tree, initial_views)
            };

            Box::new(ByldRuntime {
                interp,
                tree,
                current_views,
                reload_channel,
                pending_reload: None,
                error_state: initial_errors,
                width_bits: w_clone,
                height_bits: h_clone,
            })
        })?;

        self.engine = Some(engine);
        self.width_bits = Some(width_bits);
        self.height_bits = Some(height_bits);
        Ok(())
    }

    fn on_resize(&mut self, size: WindowSize) {
        if let Some(e) = self.engine.as_mut() {
            e.on_resize(size.width, size.height, size.scale_factor);
            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
            let w = size.width as f32 / size.scale_factor as f32;
            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
            let h = size.height as f32 / size.scale_factor as f32;
            if let Some(b) = &self.width_bits {
                b.store(w.to_bits(), Ordering::Relaxed);
            }
            if let Some(b) = &self.height_bits {
                b.store(h.to_bits(), Ordering::Relaxed);
            }
        }
    }

    fn on_redraw(&mut self) -> Result<(), ByardError> {
        if let Some(e) = self.engine.as_mut() {
            e.render_latest()?;
            // Drained every redraw (see `last_gpu_telemetry`'s doc comment),
            // independent of the print throttle below.
            self.last_gpu_telemetry = byard_core::telemetry::drain_samples();
            App::print_telemetry_overlay(
                e,
                &self.last_gpu_telemetry,
                &mut self.last_telemetry_print,
            );
        }
        Ok(())
    }

    fn on_pointer_input(&mut self, button: PointerButton, state: PointerState, x: f32, y: f32) {
        if let Some(engine) = &self.engine {
            let kind = match state {
                PointerState::Pressed => byard_core::platform::EventKind::PointerDown,
                PointerState::Released => byard_core::platform::EventKind::PointerUp,
            };
            // The router only consults this on `PointerDown` (RFC-0012
            // `secondary`): a right-button press flags the whole down→up
            // gesture as `secondary` instead of `tap`. Without it every
            // button reports as a plain left-click.
            let payload = (button == PointerButton::Right)
                .then_some(byard_core::platform::InputPayload::Bool(true));
            engine.push_input(byard_core::platform::InputEvent {
                kind,
                pos: (x, y),
                delta: (0.0, 0.0),
                payload,
                time_ms: now_ms(),
            });
        }
    }

    fn on_cursor_moved(&mut self, x: f32, y: f32) {
        if let Some(engine) = &self.engine {
            engine.push_input(byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::PointerMove,
                pos: (x, y),
                delta: (0.0, 0.0),
                payload: None,
                time_ms: now_ms(),
            });
        }
    }

    fn on_key(&mut self, key: &str, pressed: bool) {
        if let Some(engine) = &self.engine {
            let kind = if pressed {
                byard_core::platform::EventKind::KeyDown
            } else {
                byard_core::platform::EventKind::KeyUp
            };
            // The router keys `Tab` traversal (M18) and `Backspace`/edit
            // handling (M17) off this payload — dropping it here silently
            // breaks both, since every key would otherwise look identical.
            engine.push_input(byard_core::platform::InputEvent {
                kind,
                pos: (0.0, 0.0),
                delta: (0.0, 0.0),
                payload: Some(byard_core::platform::InputPayload::Key(key.to_string())),
                time_ms: now_ms(),
            });
        }
    }

    fn on_text(&mut self, text: &str) {
        if let Some(engine) = &self.engine {
            engine.push_input(byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::TextInput,
                pos: (0.0, 0.0),
                delta: (0.0, 0.0),
                payload: Some(byard_core::platform::InputPayload::Key(text.to_string())),
                time_ms: now_ms(),
            });
        }
    }

    fn on_scroll(&mut self, dx: f32, dy: f32, x: f32, y: f32) {
        if let Some(engine) = &self.engine {
            engine.push_input(byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::Scroll,
                pos: (x, y),
                delta: (dx, dy),
                payload: None,
                time_ms: now_ms(),
            });
        }
    }

    fn on_wheel(&mut self, dx: f32, dy: f32, x: f32, y: f32) {
        if let Some(engine) = &self.engine {
            engine.push_input(byard_core::platform::InputEvent {
                kind: byard_core::platform::EventKind::Wheel,
                pos: (x, y),
                delta: (dx, dy),
                payload: None,
                time_ms: now_ms(),
            });
        }
    }
}
