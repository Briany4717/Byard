//! # Relay
//!
//! Thread management and the double-buffered frame swap (RFC-0001 §5.1, §5.2).
//!
//! [`Relay`] is the single point of contact between the **logic thread**
//! (Evaluator + Atlas, produces frames) and the **render thread** (Encoder,
//! consumes frames). It owns three things:
//!
//! - a lock-free publish/subscribe slot for the latest [`RenderFrame`],
//! - a bounded recycle pool so steady-state operation reuses `RenderFrame`
//!   heap allocations instead of reallocating every frame, and
//! - the Tokio runtime backing the async I/O pool (file loads, network,
//!   timers — anything that must not block either the logic or render
//!   thread), plus the `tokio::sync::mpsc` channel that pool uses to hand
//!   completed results back to the logic thread (RFC-0001 §5.1, row 3).
//!
//! ## Why the I/O result channel carries `Box<dyn Any + Send>`
//!
//! RFC-0001 §5.1 says the Tokio pool "executes async I/O from Rust
//! controllers" and "sends results back to the logic thread via
//! `tokio::sync::mpsc`". No `#[byard_controller]` exists yet — that's a
//! `byld`-side feature for a later phase — so there is no concrete result
//! type to name today. Making `Relay` generic over that type would force
//! every call site (all 27 tests in this module included) to pin a type
//! parameter for a feature none of them use yet, which is exactly the kind
//! of speculative coupling this crate avoids. A type-erased channel keeps
//! `Relay` itself concrete and unchanged for existing callers, while still
//! giving the first real controller a working, tested delivery mechanism:
//! it sends `Box::new(value) as Box<dyn Any + Send>` and the logic thread
//! downcasts on receipt.
//!
//! ## Why `arc-swap`, not hand-rolled `unsafe`
//!
//! An earlier draft of this module used a raw `AtomicPtr<RenderFrame>` with
//! manual `Box::into_raw`/`Box::from_raw`. `CONTRIBUTING.md`'s bar for new
//! `unsafe` is: *"could this be done in safe code without significant cost
//! or correctness loss?"* Here the answer is yes — [`arc_swap::ArcSwapOption`]
//! is a published, audited, lock-free swap primitive with the same
//! single-instruction-exchange performance characteristics, so introducing
//! a new `#![allow(unsafe_code)]` module would have bought nothing. The
//! issue's own task list asks for `Arc<RenderFrame>` specifically, which is
//! exactly what `ArcSwapOption<RenderFrame>` stores.
//!
//! ## Why `Relay` does not own its logic thread's `JoinHandle`
//!
//! The logic thread closure must hold a strong `Arc<Relay>` to call back
//! into `acquire_recycled`/`publish`/`is_shutdown`. If `Relay` also stored
//! its own `JoinHandle` for that same thread, dropping the *last* external
//! `Arc<Relay>` would never actually run `Relay`'s drop glue — the thread's
//! own clone keeps the refcount above zero — so a join-on-drop inside
//! `Relay` itself would either never fire or, worse, fire from inside the
//! thread it's trying to join (a deadlock). [`Relay::spawn_logic_thread`]
//! therefore returns the [`JoinHandle`] to the caller, exactly as
//! `std::thread::spawn` does. The owner of that handle (today: a test;
//! eventually: [`crate::engine::Engine`]) is responsible for calling
//! [`Relay::request_shutdown`] and then joining before dropping its own
//! `Arc<Relay>`. This mirrors the issue's literal acceptance criterion
//! ("dropping the engine joins all threads cleanly") at the layer that can
//! actually guarantee it.
//!
//! ## Engine integration
//!
//! [`crate::engine::Engine`] owns an `Arc<Relay>`, spawns the logic thread
//! in [`Engine::start_logic`], and reads the latest frame via
//! [`Relay::current`] in [`Engine::render_latest`]. `Engine`'s `Drop` impl
//! calls [`Relay::request_shutdown`] and joins the logic thread before
//! releasing the last `Arc<Relay>`.
//!
//! [`Engine::start_logic`] does not use [`Relay::spawn_logic_thread`]
//! directly. The tick state it needs to capture — a `ReactiveLabel` — holds
//! a [`crate::evaluator::Signal`], which is intentionally `!Send` per
//! RFC-0001 §5.1 (signals are never accessed outside the logic thread; the
//! compiler enforces this). A `FnMut + Send` closure therefore cannot
//! capture `ReactiveLabel`. Instead, `start_logic` spawns via
//! `std::thread::Builder` and constructs the `ReactiveLabel` inside the
//! thread body, following the same `acquire_recycled → tick → publish →
//! yield_now → is_shutdown` loop that [`Relay::spawn_logic_thread`] would
//! have supplied.

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};
use std::thread::{self, JoinHandle};

use arc_swap::ArcSwapOption;
use crossbeam_channel::{Receiver, Sender, bounded};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

use crate::ByardError;
use crate::InputEvent;
use crate::LogicRuntime;
use crate::evaluator::ViewArena;
use crate::frame::RenderFrame;

/// A type-erased result delivered from the async I/O pool back to the
/// logic thread. See the module-level docs for why this is `Box<dyn Any +
/// Send>` rather than a generic parameter on [`Relay`].
pub type IoResult = Box<dyn Any + Send>;

/// Capacity of the frame recycle pool.
///
/// Two is the minimum that allows one frame to be "in flight" with the
/// render thread while the logic thread recycles another — the literal
/// "double" in double-buffered. Raising this trades a little memory for
/// more slack when the render thread holds a frame longer than usual.
const RECYCLE_POOL_SIZE: usize = 2;

// Compile-time invariant, not a runtime check: `bounded(0)` would make the
// recycle channel permanently full from construction, so every
// `acquire_recycled` would allocate and `publish` could never recycle.
// Asserting this on the constant itself (rather than as a `#[test]`) is
// clippy's own suggested form (`clippy::assertions_on_constants`) and fails
// the build immediately if anyone ever sets this to 0.
const _: () = assert!(RECYCLE_POOL_SIZE > 0);

/// How long an *idle* logic tick parks before re-checking for input. ~6 ms caps
/// idle CPU to a fraction of one core (vs a 100% `yield_now` spin) while keeping
/// the latency of the first input after an idle period imperceptible (well under
/// one 60 Hz frame). A waiting input is never delayed beyond this bound.
const IDLE_PARK: std::time::Duration = std::time::Duration::from_millis(6);

/// Owns the atomic frame swap, the frame recycle pool, and the async I/O
/// runtime described in RFC-0001 §5.
///
/// `Relay` is cheap to share: wrap it in `Arc<Relay>` and clone the `Arc`
/// into both the logic-thread closure and the render-thread owner (e.g.
/// `Engine`). All methods take `&self` and never block the caller for
/// longer than an atomic load/store or a non-blocking channel operation.
pub struct Relay {
    latest: ArcSwapOption<RenderFrame>,
    recycle_tx: Sender<RenderFrame>,
    recycle_rx: Receiver<RenderFrame>,
    shutdown: AtomicBool,
    io_runtime: tokio::runtime::Runtime,
    io_result_tx: UnboundedSender<IoResult>,
    // `tokio::sync::mpsc::UnboundedReceiver` only allows a single consumer
    // and needs `&mut self` to poll, so it is not `Sync` on its own. The
    // `Mutex` exists purely to grant `&self` access for the very short
    // `try_recv` in `Relay::try_recv_io_result` — it is never held across
    // an `.await` and never contended in practice (one logic thread polls
    // it once per tick), so it does not reintroduce blocking in any
    // meaningful sense.
    io_result_rx: Mutex<UnboundedReceiver<IoResult>>,
    input_tx: crossbeam_channel::Sender<InputEvent>,
    input_rx: crossbeam_channel::Receiver<InputEvent>,
    // Invoked by the logic thread after it publishes a frame that changed in
    // response to input, so a `Wait`-mode (event-driven) render loop knows to
    // wake and present it. `None` until a host installs one via
    // [`Relay::set_frame_waker`]; a continuously-redrawing (`Poll`) host can
    // leave it unset. Set rarely (once at startup), read once per published
    // frame — the mutex is never contended.
    frame_waker: Mutex<Option<FrameWaker>>,
}

/// A host-installed callback the logic thread fires after publishing a changed
/// frame (see [`Relay::set_frame_waker`]). The platform layer points it at its
/// event loop's wake primitive (e.g. a winit `EventLoopProxy`) so an
/// event-driven render thread redraws exactly when there is something new to
/// show — no busy polling, no stale frame after an input.
pub type FrameWaker = Arc<dyn Fn() + Send + Sync>;

impl Relay {
    /// Creates a new `Relay` with an empty frame slot, a seeded recycle
    /// pool, and a freshly started multi-threaded Tokio runtime for async
    /// I/O.
    ///
    /// # Errors
    ///
    /// Returns [`ByardError::RuntimeCreation`] if the OS refuses to start
    /// the Tokio runtime's worker threads (e.g. thread-creation resource
    /// limits).
    #[must_use = "ignoring the returned Relay drops it immediately, shutting down its I/O runtime"]
    pub fn new() -> Result<Self, ByardError> {
        let (recycle_tx, recycle_rx) = bounded(RECYCLE_POOL_SIZE);
        for _ in 0..RECYCLE_POOL_SIZE {
            // Channel was just created with this exact capacity — cannot be full.
            let _ = recycle_tx.try_send(RenderFrame::new());
        }

        // No `.enable_io()`/`.enable_time()`: those drivers need the `net`
        // and `time` Tokio features, which this crate does not currently
        // enable (nothing here uses sockets or timers yet — only spawned
        // compute futures). Add them, and the matching Cargo feature, the
        // day a real async I/O task needs them.
        let io_runtime = tokio::runtime::Builder::new_multi_thread()
            .thread_name("byard-io-worker")
            .build()
            .map_err(|e| ByardError::RuntimeCreation(e.to_string()))?;

        let (io_result_tx, io_result_rx) = mpsc::unbounded_channel();
        let (input_tx, input_rx) = crossbeam_channel::unbounded();

        Ok(Self {
            latest: ArcSwapOption::from(None),
            recycle_tx,
            recycle_rx,
            shutdown: AtomicBool::new(false),
            io_runtime,
            io_result_tx,
            io_result_rx: Mutex::new(io_result_rx),
            input_tx,
            input_rx,
            frame_waker: Mutex::new(None),
        })
    }

    /// Pushes an input event into the logic queue.
    pub fn push_input(&self, event: InputEvent) {
        let _ = self.input_tx.send(event);
    }

    /// Installs the callback fired after the logic thread publishes a frame
    /// that changed in response to input. See [`FrameWaker`].
    pub fn set_frame_waker(&self, waker: FrameWaker) {
        *self
            .frame_waker
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(waker);
    }

    /// Fires the installed [`FrameWaker`], if any. The `Arc` is cloned out of
    /// the lock first so the (host-supplied) callback never runs while the
    /// mutex is held.
    fn wake_renderer(&self) {
        let waker = self
            .frame_waker
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        if let Some(waker) = waker {
            waker();
        }
    }

    /// Returns a frame ready to be populated, preferring a recycled buffer
    /// over a fresh allocation.
    ///
    /// Never blocks: if the recycle pool is momentarily empty (the render
    /// thread is holding on to frames longer than usual), this allocates a
    /// new [`RenderFrame`] instead of waiting.
    #[must_use]
    pub fn acquire_recycled(&self) -> RenderFrame {
        let mut frame = self.recycle_rx.try_recv().unwrap_or_default();
        frame.clear();
        frame
    }

    /// Publishes `frame` as the new latest frame, atomically replacing
    /// whatever was previously visible to readers.
    ///
    /// This is the "single atomic pointer exchange" from RFC-0001 §5.2: the
    /// swap is one lock-free operation, so a concurrent [`Relay::current`]
    /// call always observes either the entire old frame or the entire new
    /// one — never a partial mix of both.
    ///
    /// If the previous frame is not referenced anywhere else (the render
    /// thread already dropped its clone, or never took one), its `Vec`
    /// allocation is returned to the recycle pool for reuse. If the pool is
    /// momentarily full, the frame is dropped normally — a missed recycle
    /// opportunity, not a correctness issue. This call never blocks.
    pub fn publish(&self, frame: RenderFrame) {
        let previous = self.latest.swap(Some(Arc::new(frame)));
        if let Some(arc) = previous {
            if let Ok(reclaimed) = Arc::try_unwrap(arc) {
                let _ = self.recycle_tx.try_send(reclaimed);
            }
            // else: a reader still holds a clone of the old Arc. It will be
            // deallocated normally once that reader drops it — we simply
            // don't get to recycle its buffer this time.
        }
    }

    /// Returns a clone of the current latest frame, or `None` if nothing
    /// has been published yet.
    ///
    /// Non-blocking and may be called concurrently from any number of
    /// threads, including while the logic thread is mid-[`Relay::publish`]
    /// — this is exactly the "render thread never blocks" guarantee.
    #[must_use]
    pub fn current(&self) -> Option<Arc<RenderFrame>> {
        self.latest.load_full()
    }

    /// Returns a sender that lets a consumer (e.g. the render thread, after
    /// it finishes drawing a frame) voluntarily return a `RenderFrame` to
    /// the recycle pool.
    ///
    /// Using this is optional — frames returned only via [`Relay::publish`]
    /// already keep the pool healthy in the common case where the render
    /// thread doesn't hold on to old frames.
    #[must_use]
    pub fn recycler(&self) -> Sender<RenderFrame> {
        self.recycle_tx.clone()
    }

    /// Returns a handle to the async I/O Tokio runtime.
    ///
    /// The handle is cheap to clone and can be used to spawn tasks from any
    /// thread, including from inside the logic thread's tick closure.
    #[must_use]
    pub fn io_handle(&self) -> tokio::runtime::Handle {
        self.io_runtime.handle().clone()
    }

    /// Returns a cloneable sender that tasks spawned on [`Relay::io_handle`]
    /// use to deliver a completed result back to the logic thread.
    ///
    /// Per RFC-0001 §5.1: "\[the Tokio pool\] sends results back to the
    /// logic thread via `tokio::sync::mpsc`." The payload is boxed and
    /// type-erased (see the module-level docs) since no concrete result
    /// type exists yet; the receiving side downcasts via
    /// [`Relay::try_recv_io_result`].
    #[must_use]
    pub fn io_result_sender(&self) -> UnboundedSender<IoResult> {
        self.io_result_tx.clone()
    }

    /// Non-blocking poll for the next completed I/O result, if any.
    ///
    /// Intended to be called once per logic-thread tick. Never blocks:
    /// returns `None` immediately if no result has arrived yet, mirroring
    /// every other `Relay` accessor's "never blocks" guarantee.
    #[must_use]
    pub fn try_recv_io_result(&self) -> Option<IoResult> {
        self.io_result_rx
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .try_recv()
            .ok()
    }

    /// Returns `true` once [`Relay::request_shutdown`] has been called.
    #[must_use]
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }

    /// Signals the logic thread (and any other cooperating loop) to stop.
    ///
    /// Idempotent — calling this more than once has no additional effect.
    /// Does not itself join any thread; pair with the [`JoinHandle`]
    /// returned by [`Relay::spawn_logic_thread`].
    pub fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    /// Spawns the logic thread: a loop that acquires a recycled frame, lets
    /// `tick` populate it, publishes it, and repeats until
    /// [`Relay::request_shutdown`] is called.
    ///
    /// `tick` is intentionally unpaced — it runs back-to-back with no
    /// sleep — because RFC-0001 does not yet specify a fixed tick rate.
    /// Callers that need pacing (vsync-driven redraw, a fixed-hz simulation
    /// step, etc.) should implement it inside `tick` itself, or wait on an
    /// external signal before returning. A future sub-issue may add a
    /// `Relay::run_at(hz, tick)` helper once that policy is decided.
    ///
    /// The caller owns the returned [`JoinHandle`] — see the module-level
    /// docs for why `Relay` cannot safely join its own logic thread.
    ///
    /// # Errors
    ///
    /// Returns [`ByardError::ThreadSpawn`] if the OS refuses to create the
    /// thread.
    pub fn spawn_logic_thread<F>(
        relay: &Arc<Relay>,
        mut tick: F,
    ) -> Result<JoinHandle<()>, ByardError>
    where
        F: FnMut(&mut RenderFrame) + Send + 'static,
    {
        let relay = Arc::clone(relay);
        thread::Builder::new()
            .name("byard-logic-thread".to_string())
            .spawn(move || {
                while !relay.is_shutdown() {
                    let mut frame = relay.acquire_recycled();
                    tick(&mut frame);
                    relay.publish(frame);
                    thread::yield_now();
                }
            })
            .map_err(|e| ByardError::ThreadSpawn(e.to_string()))
    }

    /// Spawns the logic thread from a `build` factory that constructs a
    /// [`LogicRuntime`] **inside** the thread, then drives it
    /// `acquire_recycled → evaluate_tick → publish` until shutdown
    /// (RFC-0002 §"Integration with Engine", RFC-0003 §8).
    ///
    /// This is the generalization of [`Relay::spawn_logic_thread`] for a
    /// stateful interpreter: the running runtime holds `!Send` data
    /// (`Signal`s, a `ViewArena`, a logic-thread-local reactive scope), so it
    /// can never cross a thread boundary. Only the **factory** is bounded
    /// `Send + 'static` (INV-6) — it closes over plain owned data (a
    /// `CompiledView`) and is moved into the thread, where it builds the arena
    /// and the borrowing runtime in place. The `for<'a>` HRTB ties the
    /// runtime's borrow to the thread-local arena's lifetime.
    ///
    /// # Errors
    ///
    /// Returns [`ByardError::ThreadSpawn`] if the OS refuses to create the
    /// thread.
    pub fn spawn_logic_from_view<F>(
        relay: &Arc<Relay>,
        build: F,
    ) -> Result<JoinHandle<()>, ByardError>
    where
        F: for<'a> FnOnce(&'a ViewArena) -> Box<dyn LogicRuntime + 'a> + Send + 'static,
    {
        let relay = Arc::clone(relay);
        thread::Builder::new()
            .name("byard-logic-thread".to_string())
            .spawn(move || {
                // The arena and the runtime that borrows it both live for the
                // thread body only; neither is ever observed off this thread.
                let arena = ViewArena::new();
                let mut runtime = build(&arena);
                while !relay.is_shutdown() {
                    let mut frame = relay.acquire_recycled();
                    let mut inputs = Vec::new();
                    while let Ok(ev) = relay.input_rx.try_recv() {
                        inputs.push(ev);
                    }
                    // The reactive interpreter computes its own dirty set; the
                    // engine-level dirty-target plumbing is wired in a later
                    // phase, so pass an empty slice for now.
                    let had_input = !inputs.is_empty();
                    runtime.evaluate_tick(&mut frame, &inputs, &[]);
                    relay.publish(frame);

                    // Idle throttle: a UI with no pending input re-publishes an
                    // identical frame every iteration, so a tight `yield_now`
                    // spin would peg a core at 100% (heat → thermal throttling →
                    // sluggish input). When input *is* waiting we loop at full
                    // speed so bursts drain immediately; only an idle tick parks
                    // briefly, capping idle CPU while keeping first-input latency
                    // under one short park. (RFC-0001 leaves pacing to the caller.)
                    if had_input {
                        // The frame just published reflects this input — wake an
                        // event-driven (`Wait`-mode) render thread so it presents
                        // the update now, rather than showing the pre-input frame
                        // until the next unrelated OS event.
                        relay.wake_renderer();
                        thread::yield_now();
                    } else {
                        thread::park_timeout(IDLE_PARK);
                    }
                }
            })
            .map_err(|e| ByardError::ThreadSpawn(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Rect;
    use std::sync::atomic::AtomicUsize;
    use std::time::{Duration, Instant};

    /// Static assertion: `Relay` must be safely shareable across threads
    /// behind an `Arc`, mirroring the `assert_send_sync`-style checks
    /// already used elsewhere in this crate (see `frame.rs`'s `TargetId`
    /// tests).
    #[test]
    fn relay_is_send_and_sync() {
        const fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Relay>();
    }

    #[test]
    fn new_relay_succeeds() {
        assert!(Relay::new().is_ok());
    }

    #[test]
    fn new_relay_has_no_latest_frame_initially() {
        let relay = Relay::new().unwrap();
        assert!(relay.current().is_none());
    }

    #[test]
    fn new_relay_is_not_shutdown_initially() {
        let relay = Relay::new().unwrap();
        assert!(!relay.is_shutdown());
    }

    #[test]
    fn acquire_recycled_returns_empty_frame() {
        let relay = Relay::new().unwrap();
        let frame = relay.acquire_recycled();
        assert!(frame.rects().is_empty());
    }

    #[test]
    fn current_returns_none_before_any_publish() {
        let relay = Relay::new().unwrap();
        assert!(relay.current().is_none());
    }

    #[test]
    fn publish_then_current_returns_published_content() {
        let relay = Relay::new().unwrap();
        let mut frame = relay.acquire_recycled();
        frame.push_rect(Rect::new(1.0, 2.0, 3.0, 4.0), false);
        relay.publish(frame);

        let observed = relay.current().expect("frame was published");
        assert_eq!(observed.rects().len(), 1);
        assert_eq!(observed.rects()[0], Rect::new(1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn multiple_publishes_overwrite_rather_than_merge() {
        let relay = Relay::new().unwrap();

        let mut a = relay.acquire_recycled();
        a.push_rect(Rect::new(0.0, 0.0, 1.0, 1.0), false);
        relay.publish(a);

        let mut b = relay.acquire_recycled();
        b.push_rect(Rect::new(9.0, 9.0, 9.0, 9.0), false);
        b.push_rect(Rect::new(8.0, 8.0, 8.0, 8.0), false);
        relay.publish(b);

        let observed = relay.current().unwrap();
        assert_eq!(observed.rects().len(), 2);
        assert_eq!(observed.rects()[0], Rect::new(9.0, 9.0, 9.0, 9.0));
    }

    #[test]
    fn current_can_be_called_repeatedly_without_consuming() {
        let relay = Relay::new().unwrap();
        let mut frame = relay.acquire_recycled();
        frame.push_rect(Rect::new(1.0, 1.0, 1.0, 1.0), false);
        relay.publish(frame);

        let first = relay.current().unwrap();
        let second = relay.current().unwrap();
        assert_eq!(first.rects(), second.rects());
    }

    #[test]
    fn holding_an_old_arc_keeps_its_content_unchanged_across_later_publishes() {
        let relay = Relay::new().unwrap();

        let mut a = relay.acquire_recycled();
        a.push_rect(Rect::new(1.0, 1.0, 1.0, 1.0), false);
        relay.publish(a);
        let held = relay.current().unwrap(); // render thread "holds" this Arc

        let mut b = relay.acquire_recycled();
        b.push_rect(Rect::new(2.0, 2.0, 2.0, 2.0), false);
        relay.publish(b);

        // The Arc the test is still holding must be unaffected by the swap.
        assert_eq!(held.rects()[0], Rect::new(1.0, 1.0, 1.0, 1.0));
        // But a fresh read sees the new frame.
        assert_eq!(
            relay.current().unwrap().rects()[0],
            Rect::new(2.0, 2.0, 2.0, 2.0)
        );
    }

    #[test]
    fn acquired_recycled_frame_is_always_empty_even_if_reused_buffer_had_content() {
        let relay = Relay::new().unwrap();

        // Publish a frame with content, then publish a second one so the
        // first (uncloned) Arc is reclaimed into the recycle pool.
        let mut a = relay.acquire_recycled();
        a.push_rect(Rect::new(5.0, 5.0, 5.0, 5.0), false);
        relay.publish(a);
        let b = relay.acquire_recycled();
        relay.publish(b);

        // Drain the pool looking for a previously-used buffer; every frame
        // handed back by acquire_recycled must be empty regardless of what
        // it held before.
        for _ in 0..RECYCLE_POOL_SIZE {
            let frame = relay.acquire_recycled();
            assert!(frame.rects().is_empty());
        }
    }

    #[test]
    fn acquire_recycled_falls_back_to_allocation_when_pool_is_empty() {
        let relay = Relay::new().unwrap();

        // Drain the pool completely without returning anything.
        for _ in 0..RECYCLE_POOL_SIZE {
            let _ = relay.acquire_recycled();
        }

        // One more acquire must still succeed (falls back to a fresh
        // allocation) rather than panicking or blocking.
        let frame = relay.acquire_recycled();
        assert!(frame.rects().is_empty());
    }

    #[test]
    fn publish_does_not_block_when_recycle_pool_is_already_full() {
        let relay = Relay::new().unwrap();

        // Saturate the pool directly (test is in the same module, so it can
        // see the private fields).
        for _ in 0..8 {
            let _ = relay.recycle_tx.try_send(RenderFrame::new());
        }

        let frame = relay.acquire_recycled();
        let start = Instant::now();
        relay.publish(frame); // must use try_send internally, never block
        assert!(start.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn recycler_handle_can_manually_return_a_frame() {
        let relay = Relay::new().unwrap();
        // `Relay::new()` seeds the bounded(RECYCLE_POOL_SIZE) channel to full
        // capacity, so a slot must be drained before a manual return has
        // room to land.
        let _ = relay.acquire_recycled();
        let recycler = relay.recycler();

        let mut frame = RenderFrame::new();
        frame.push_rect(Rect::new(1.0, 1.0, 1.0, 1.0), false);
        frame.clear();
        assert!(recycler.try_send(frame).is_ok());
    }

    #[test]
    fn request_shutdown_is_idempotent() {
        let relay = Relay::new().unwrap();
        relay.request_shutdown();
        relay.request_shutdown();
        assert!(relay.is_shutdown());
    }

    #[test]
    fn io_handle_can_run_a_future_to_completion() {
        let relay = Relay::new().unwrap();
        let result = relay.io_handle().block_on(async { 21 + 21 });
        assert_eq!(result, 42);
    }

    #[test]
    fn try_recv_io_result_returns_none_when_empty() {
        let relay = Relay::new().unwrap();
        assert!(relay.try_recv_io_result().is_none());
    }

    #[test]
    fn io_result_sent_then_received_round_trips_through_downcast() {
        let relay = Relay::new().unwrap();
        let tx = relay.io_result_sender();

        tx.send(Box::new(42_i32)).unwrap();

        let result = relay
            .try_recv_io_result()
            .expect("a result was sent and should be receivable");
        let value = result
            .downcast::<i32>()
            .expect("payload was sent as i32, should downcast back to i32");
        assert_eq!(*value, 42);
    }

    #[test]
    fn io_result_downcast_to_wrong_type_fails_without_panicking() {
        let relay = Relay::new().unwrap();
        let tx = relay.io_result_sender();

        tx.send(Box::new(42_i32)).unwrap();

        let result = relay.try_recv_io_result().unwrap();
        let failed = result.downcast::<String>();
        assert!(
            failed.is_err(),
            "downcasting to the wrong type must fail, not panic"
        );
    }

    #[test]
    fn multiple_io_results_are_received_in_fifo_order() {
        let relay = Relay::new().unwrap();
        let tx = relay.io_result_sender();

        tx.send(Box::new(1_i32)).unwrap();
        tx.send(Box::new(2_i32)).unwrap();
        tx.send(Box::new(3_i32)).unwrap();

        let mut observed = Vec::new();
        while let Some(result) = relay.try_recv_io_result() {
            observed.push(*result.downcast::<i32>().unwrap());
        }
        assert_eq!(observed, vec![1, 2, 3]);
    }

    #[test]
    fn io_result_sender_is_cloneable_and_both_clones_deliver_to_the_same_receiver() {
        let relay = Relay::new().unwrap();
        let tx_a = relay.io_result_sender();
        let tx_b = tx_a.clone();

        tx_a.send(Box::new("from-a".to_string())).unwrap();
        tx_b.send(Box::new("from-b".to_string())).unwrap();

        let first = *relay
            .try_recv_io_result()
            .unwrap()
            .downcast::<String>()
            .unwrap();
        let second = *relay
            .try_recv_io_result()
            .unwrap()
            .downcast::<String>()
            .unwrap();
        assert_eq!(first, "from-a");
        assert_eq!(second, "from-b");
    }

    #[test]
    fn io_result_sent_from_a_spawned_async_task_is_received_after_it_completes() {
        let relay = Relay::new().unwrap();
        let tx = relay.io_result_sender();

        let task = relay.io_handle().spawn(async move {
            tx.send(Box::new(99_i32)).unwrap();
        });
        relay.io_handle().block_on(task).unwrap();

        let result = relay
            .try_recv_io_result()
            .expect("spawned task should have sent a result");
        assert_eq!(*result.downcast::<i32>().unwrap(), 99);
    }

    #[test]
    fn real_image_decode_on_the_io_pool_is_received_after_it_completes() {
        // The M29 shape end-to-end at the relay level: a deliberately slow
        // (sleep + real decode) task on the I/O pool reports its result back
        // through the type-erased channel, exactly as `TextureCache::ensure`
        // does — proving a blocking `image` decode never touches the caller.
        use std::io::Cursor;

        let relay = Relay::new().unwrap();
        let tx = relay.io_result_sender();

        // A tiny PNG, encoded in-memory so the test needs no fixture file.
        let mut png = Vec::new();
        image::RgbaImage::from_pixel(4, 4, image::Rgba([1, 2, 3, 255]))
            .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();

        let task = relay.io_handle().spawn(async move {
            std::thread::sleep(std::time::Duration::from_millis(50));
            let decoded = image::load_from_memory(&png).unwrap().to_rgba8();
            let dims = (decoded.width(), decoded.height());
            tx.send(Box::new(dims)).unwrap();
        });
        relay.io_handle().block_on(task).unwrap();

        let result = relay
            .try_recv_io_result()
            .expect("decode task should have sent its result");
        assert_eq!(*result.downcast::<(u32, u32)>().unwrap(), (4, 4));
    }

    #[test]
    fn dropping_relay_with_unconsumed_io_results_does_not_panic() {
        let relay = Relay::new().unwrap();
        let tx = relay.io_result_sender();
        tx.send(Box::new(1_i32)).unwrap();
        drop(relay); // must not panic even with an undrained result queued
    }

    #[test]
    fn io_result_sender_outliving_the_relay_does_not_panic_on_send() {
        let relay = Relay::new().unwrap();
        let tx = relay.io_result_sender();
        drop(relay);
        // The receiver is gone now; sending into a closed channel must
        // return an error, not panic.
        assert!(tx.send(Box::new(1_i32)).is_err());
    }

    #[test]
    fn spawn_logic_thread_runs_tick_at_least_once() {
        let relay = Arc::new(Relay::new().unwrap());
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let handle = Relay::spawn_logic_thread(&relay, move |_frame| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("thread spawn should succeed in tests");

        // Give the thread a moment to run, then ask it to stop.
        thread::sleep(Duration::from_millis(20));
        relay.request_shutdown();
        handle.join().expect("logic thread must not panic");

        assert!(counter.load(Ordering::SeqCst) > 0);
    }

    #[test]
    fn spawn_logic_thread_uses_the_documented_thread_name() {
        let relay = Arc::new(Relay::new().unwrap());
        let (name_tx, name_rx) = bounded(1);

        let handle = Relay::spawn_logic_thread(&relay, move |_frame| {
            let _ = name_tx.try_send(thread::current().name().unwrap_or_default().to_string());
        })
        .unwrap();

        let name = name_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("tick should run and report a thread name");
        relay.request_shutdown();
        handle.join().unwrap();

        assert_eq!(name, "byard-logic-thread");
    }

    #[test]
    fn shutdown_then_join_pattern_completes_without_hanging() {
        let relay = Arc::new(Relay::new().unwrap());
        let handle = Relay::spawn_logic_thread(&relay, |_frame| {}).unwrap();

        relay.request_shutdown();
        // If this hangs, the test runner's own timeout will catch it — that
        // is the acceptable failure signal for "does not join cleanly".
        handle
            .join()
            .expect("logic thread must join after shutdown");
    }

    #[test]
    fn render_thread_never_blocks_while_logic_thread_publishes_continuously() {
        let relay = Arc::new(Relay::new().unwrap());
        let handle = Relay::spawn_logic_thread(&relay, |frame| {
            frame.push_rect(Rect::new(0.0, 0.0, 1.0, 1.0), false);
        })
        .unwrap();

        // Hammer `current()` from the "render thread" (this thread) while
        // the logic thread is publishing as fast as it can. The assertion
        // is on the *total* wall time for many calls, not on any single
        // call — a per-iteration millisecond bound is flaky under
        // scheduler jitter; a generous aggregate bound is not.
        let start = Instant::now();
        for _ in 0..2000 {
            let _ = relay.current();
        }
        let elapsed = start.elapsed();

        relay.request_shutdown();
        handle.join().unwrap();

        assert!(
            elapsed < Duration::from_secs(2),
            "2000 non-blocking reads took {elapsed:?} — render thread appears to be blocking"
        );
    }

    #[test]
    fn frame_content_is_never_torn_under_concurrent_publish_and_read() {
        let relay = Arc::new(Relay::new().unwrap());
        let generation = Arc::new(AtomicUsize::new(0));
        let generation_clone = Arc::clone(&generation);

        // Each published frame encodes a single monotonic "generation"
        // value into every rect it contains. If the swap were ever
        // non-atomic, a reader could observe a frame built from two
        // different generations — this test asserts that never happens.
        let handle = Relay::spawn_logic_thread(&relay, move |frame| {
            let generation_value = generation_clone.fetch_add(1, Ordering::SeqCst);
            #[allow(clippy::cast_precision_loss)]
            let generation_f = generation_value as f32;
            for _ in 0..4 {
                frame.push_rect(Rect::new(generation_f, generation_f, 1.0, 1.0), false);
            }
        })
        .unwrap();

        for _ in 0..500 {
            if let Some(observed) = relay.current() {
                let rects = observed.rects();
                if let Some(first) = rects.first() {
                    #[allow(clippy::float_cmp)]
                    let consistent = rects.iter().all(|r| r.x == first.x);
                    assert!(consistent, "observed a torn frame: {rects:?}");
                }
            }
        }

        relay.request_shutdown();
        handle.join().unwrap();
    }

    #[test]
    fn current_never_returns_none_once_something_has_been_published() {
        let relay = Arc::new(Relay::new().unwrap());
        let mut seed = relay.acquire_recycled();
        seed.push_rect(Rect::new(0.0, 0.0, 1.0, 1.0), false);
        relay.publish(seed);

        let mut readers = Vec::new();
        for _ in 0..8 {
            let relay = Arc::clone(&relay);
            readers.push(thread::spawn(move || {
                for _ in 0..100 {
                    assert!(relay.current().is_some());
                }
            }));
        }
        for reader in readers {
            reader.join().unwrap();
        }
    }

    #[test]
    fn stress_many_publish_acquire_cycles_without_panicking() {
        let relay = Relay::new().unwrap();
        for i in 0..10_000 {
            let mut frame = relay.acquire_recycled();
            #[allow(clippy::cast_precision_loss)]
            frame.push_rect(Rect::new(i as f32, 0.0, 1.0, 1.0), false);
            relay.publish(frame);
        }
        assert_eq!(relay.current().unwrap().rects().len(), 1);
    }

    #[test]
    fn dropping_relay_with_unconsumed_latest_frame_does_not_panic() {
        let relay = Relay::new().unwrap();
        let mut frame = relay.acquire_recycled();
        frame.push_rect(Rect::new(1.0, 1.0, 1.0, 1.0), false);
        relay.publish(frame);
        drop(relay); // must not panic
    }

    #[test]
    fn dropping_relay_after_clean_shutdown_and_join_does_not_panic() {
        let relay = Arc::new(Relay::new().unwrap());
        let handle = Relay::spawn_logic_thread(&relay, |_frame| {}).unwrap();
        relay.request_shutdown();
        handle.join().unwrap();
        drop(relay); // last Arc — runs Relay's drop glue, must not panic
    }

    #[test]
    fn two_relays_are_fully_independent() {
        let a = Relay::new().unwrap();
        let b = Relay::new().unwrap();

        let mut fa = a.acquire_recycled();
        fa.push_rect(Rect::new(1.0, 1.0, 1.0, 1.0), false);
        a.publish(fa);

        assert!(a.current().is_some());
        assert!(b.current().is_none());
    }

    // ── spawn_logic_from_view: the !Send-runtime / Send-factory contract ──

    /// A minimal `!Send` `LogicRuntime`: it holds a [`Signal`] (which is `!Send`
    /// by construction, RFC-0001 §5.1), proving a stateful interpreter can run
    /// on the logic thread without ever crossing a thread boundary.
    struct CounterRuntime<'a> {
        signal: crate::evaluator::Signal<'a, i64>,
        hits: Arc<AtomicUsize>,
    }

    impl LogicRuntime for CounterRuntime<'_> {
        fn evaluate_tick(
            &mut self,
            frame: &mut RenderFrame,
            _input_events: &[InputEvent],
            _dirty: &[crate::frame::TargetId],
        ) {
            self.signal.write(|v| *v += 1);
            #[allow(clippy::cast_sign_loss)]
            frame.set_version(self.signal.read(|v| *v as u64));
            self.hits.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn spawn_logic_from_view_builds_a_non_send_runtime_and_ticks() {
        let relay = Arc::new(Relay::new().unwrap());
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_factory = Arc::clone(&hits);

        // The factory is `Send` (it captures only an `Arc<AtomicUsize>`); the
        // runtime it builds borrows the thread-local arena and is `!Send`.
        let handle = Relay::spawn_logic_from_view(&relay, move |arena| {
            Box::new(CounterRuntime {
                signal: crate::evaluator::Signal::new_in(arena, 0_i64),
                hits: hits_factory,
            })
        })
        .expect("thread spawn should succeed in tests");

        // Wait until at least one tick has run.
        let start = Instant::now();
        while hits.load(Ordering::SeqCst) == 0 && start.elapsed() < Duration::from_secs(2) {
            thread::yield_now();
        }
        relay.request_shutdown();
        handle.join().expect("logic thread must not panic");

        assert!(
            hits.load(Ordering::SeqCst) >= 1,
            "the !Send runtime must have ticked at least once"
        );
        assert!(
            relay.current().is_some(),
            "ticking must have published at least one frame"
        );
    }

    #[test]
    fn frame_waker_fires_after_an_input_bearing_tick() {
        // The wake-on-publish contract (Wait-mode redraw): after the logic
        // thread processes input and publishes, it must fire the installed
        // waker so an event-driven render loop knows to present the update.
        let relay = Arc::new(Relay::new().unwrap());
        let woke = Arc::new(AtomicUsize::new(0));
        let woke_cb = Arc::clone(&woke);
        relay.set_frame_waker(Arc::new(move || {
            woke_cb.fetch_add(1, Ordering::SeqCst);
        }));

        let hits = Arc::new(AtomicUsize::new(0));
        let hits_factory = Arc::clone(&hits);
        let handle = Relay::spawn_logic_from_view(&relay, move |arena| {
            Box::new(CounterRuntime {
                signal: crate::evaluator::Signal::new_in(arena, 0_i64),
                hits: hits_factory,
            })
        })
        .expect("thread spawn should succeed in tests");

        // Feed an input event; the next (input-bearing) tick must wake.
        relay.push_input(InputEvent {
            kind: crate::platform::EventKind::PointerDown,
            pos: (1.0, 1.0),
            delta: (0.0, 0.0),
            payload: None,
            time_ms: 0,
        });

        let start = Instant::now();
        while woke.load(Ordering::SeqCst) == 0 && start.elapsed() < Duration::from_secs(2) {
            thread::yield_now();
        }
        relay.request_shutdown();
        handle.join().expect("logic thread must not panic");

        assert!(
            woke.load(Ordering::SeqCst) >= 1,
            "an input-bearing tick must fire the frame waker"
        );
    }
}
