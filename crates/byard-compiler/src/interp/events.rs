//! The input → mutation → screen event pipeline (RFC-0003 §1/§8; E1/E3/E4/E7).
//!
//! All handler execution is on the logic thread (RFC-0001 §5.1). The
//! [`EventRouter`] drains a tick's [`InputEvent`]s in FIFO order, coalescing
//! continuous events (E7), recognizing taps by threshold (E4, `pointer_up` then
//! `tap`), reflecting two-way writes back with value-dedup (E1), and stealing
//! keyboard focus (E3). Crucially, **dispatch only marks** — the single
//! [`ReactiveCtx::pull`] happens once, after all events settle (§8), so no
//! handler observes a half-updated view.
//!
//! Phase 2 hit-testing here is a flat topmost-wins scan over registered rects
//! (the spatial-hash grid lives in `byard-core`); there is no ancestor bubbling
//! (RFC-0003 §7).

use super::env::{SignalId, Value};
use super::intrinsics::Rect;
use super::reactive::ReactiveCtx;

/// The event kinds the Phase-2 router models.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EventKind {
    /// A pointer press.
    PointerDown,
    /// A pointer release.
    PointerUp,
    /// A qualifying tap (down+up within thresholds; E4).
    Tap,
    /// Continuous pointer movement.
    PointerMove,
    /// Pointer drag: synthesized by the router from PointerMove while the button
    /// is held (M16). Used by Slider to track the drag position.
    PointerDrag,
    /// Continuous scroll.
    Scroll,
    /// Continuous wheel.
    Wheel,
    /// A value change from a value-carrying intrinsic.
    Change,
    /// A keyboard key press; key name is in `InputEvent.value` (M17).
    KeyDown,
    /// A keyboard key release; key name is in `InputEvent.value` (M17).
    KeyUp,
    /// Printable text input; the text is in `InputEvent.value` (M17).
    TextInput,
    // ── M24: remaining event catalog ─────────────────────────────────────
    /// Cursor entered an element's rect (synthesized from PointerMove).
    PointerEnter,
    /// Cursor left an element's rect (synthesized from PointerMove).
    PointerExit,
    /// Hovering over an element (continuous; synthesized from PointerMove while
    /// the pointer is inside and no button is held).
    Hover,
    /// A press held for > 500 ms without moving more than `TAP_SLOP` px.
    LongPress,
    /// Two taps within `DOUBLE_TAP_MS` (E4 double-tap threshold).
    DoubleTap,
    /// A secondary (right) button tap.
    Secondary,
    // ── RFC-0012 §A: `focus =>`/`blur =>` sugar ──────────────────────────
    /// An element's focus state rose (RFC-0012 S2). **Internal-only**: never
    /// produced by a raw [`InputEvent`] — fired directly by
    /// [`EventRouter::steal_focus`] on the rising edge of `focused_sig`.
    Focus,
    /// An element's focus state fell (RFC-0012 S2). **Internal-only**: never
    /// produced by a raw [`InputEvent`] — fired directly by
    /// [`EventRouter::steal_focus`] on the falling edge of `focused_sig`.
    Blur,
}

impl EventKind {
    /// Whether this is a continuous (coalescible) event (E7).
    #[must_use]
    pub fn is_continuous(self) -> bool {
        matches!(self, Self::PointerMove | Self::Scroll | Self::Wheel)
    }
}

/// A normalized, `Send`-able input event produced by the platform thread.
#[derive(Clone, Debug)]
pub struct InputEvent {
    /// The event kind.
    pub kind: EventKind,
    /// Absolute cursor position (logical px).
    pub pos: (f32, f32),
    /// Incremental delta for continuous events.
    pub delta: (f32, f32),
    /// The new value for a `Change` event (write-back payload).
    pub value: Option<Value>,
    /// Event time in milliseconds (for the tap interval).
    pub time_ms: u64,
}

impl InputEvent {
    /// A discrete pointer event at `pos`.
    #[must_use]
    pub fn pointer(kind: EventKind, pos: (f32, f32), time_ms: u64) -> Self {
        Self {
            kind,
            pos,
            delta: (0.0, 0.0),
            value: None,
            time_ms,
        }
    }
}

/// Tap displacement threshold (logical px) — E4.
pub const TAP_SLOP: f32 = 8.0;
/// Tap interval upper bound (ms) — E4.
pub const TAP_MS: u64 = 500;
/// Long-press hold threshold (ms) — M24.
pub const LONG_PRESS_MS: u64 = 500;
/// Double-tap interval upper bound (ms) — M24 E4.
pub const DOUBLE_TAP_MS: u64 = 300;

thread_local! {
    /// The position of the event currently being dispatched, for use by
    /// handlers that need cursor position (e.g. Slider drag, M16).
    pub static CURRENT_EVENT_POS: std::cell::Cell<(f32, f32)> =
        const { std::cell::Cell::new((0.0, 0.0)) };
}

/// A handler's reactive action. Receives the context (for `var` mutation) and
/// an optional payload (the `Change` value).
pub type Action = Box<dyn FnMut(&mut ReactiveCtx, Option<&Value>)>;

struct Handler {
    elem: u32,
    rect: Rect,
    kind: EventKind,
    action: Action,
}

struct Focusable {
    elem: u32,
    rect: Rect,
    /// The `var` bound via `#[focused: …]`.
    focused_sig: SignalId,
}

struct DownState {
    elem: Option<u32>,
    pos: (f32, f32),
    time_ms: u64,
    /// True when the secondary (right) button initiated the press.
    secondary: bool,
}

/// The interaction-state mask of one element (RFC-0012 Part B): the engine-owned
/// booleans a stateful style layer (a future `on <state> {}`) resolves against,
/// reported by [`EventRouter::style_state`]. A small bit set rather than four
/// `bool`s so a whole state can be passed and matched at once.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct StyleState(u8);

impl StyleState {
    /// The pointer is over the element.
    pub const HOVER: StyleState = StyleState(1 << 0);
    /// The element is the target of the in-flight pointer press.
    pub const PRESSED: StyleState = StyleState(1 << 1);
    /// The element holds keyboard focus.
    pub const FOCUSED: StyleState = StyleState(1 << 2);
    /// The element's `disabled:` prop resolved true.
    pub const DISABLED: StyleState = StyleState(1 << 3);

    /// The empty state (no interaction flags).
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Whether every bit in `other` is set in `self`.
    #[must_use]
    pub const fn contains(self, other: StyleState) -> bool {
        self.0 & other.0 == other.0
    }

    /// The union of two states (every flag set in either).
    #[must_use]
    pub const fn union(self, other: StyleState) -> Self {
        Self(self.0 | other.0)
    }

    /// The raw bits (for a resolver's `(class, state)` keying).
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    fn insert(&mut self, other: StyleState) {
        self.0 |= other.0;
    }
}

/// Routes input events to registered handlers, on the logic thread.
#[derive(Default)]
pub struct EventRouter {
    handlers: Vec<Handler>,
    focusables: Vec<Focusable>,
    down: Option<DownState>,
    focused: Option<u32>,
    /// Element currently under the pointer (for enter/exit synthesis, M24).
    hovered: Option<u32>,
    /// Time and element of the most recent tap (for double-tap detection, M24).
    last_tap: Option<(u64, Option<u32>)>,
    /// Elements whose `disabled:` prop resolved true this tick (RFC-0012 S5).
    /// Rebuilt every render like the handler set; a disabled element reports the
    /// `DISABLED` state and never dispatches an action.
    disabled: std::collections::HashSet<u32>,
    /// Hit regions of elements that carry `on hover`/`on pressed` style blocks
    /// but register no event handler of their own (RFC-0016). They take part in
    /// hover/press hit-testing so [`style_state`](Self::style_state) reports the
    /// pointer state a purely-declarative interactive style depends on. Rebuilt
    /// every render like the handler set.
    hover_regions: Vec<(u32, Rect)>,
    /// RFC-0017 modality floor: a modal `Overlay` registers a full-viewport
    /// scrim and raises this to the scrim's handler index, so every handler
    /// registered *before* it (the main tree, and any lower overlay) is
    /// excluded from hit-testing. The scrim itself sits exactly at the floor
    /// and covers the viewport, so an outside tap always lands on it rather
    /// than falling through. `0` when no modal overlay is mounted (no effect).
    /// Rebuilt every render.
    modal_floor: usize,
    /// RFC-0017 focus-trap floor: the `focusables` index at which the topmost
    /// modal overlay's scope begins. `Tab` traversal is restricted to
    /// focusables at or above it, so focus cannot leave a modal overlay.
    focusable_floor: usize,
    /// Handler indices of the mounted modal scrims, in mount order (topmost
    /// last). `Escape` fires the topmost one's `dismiss` action (RFC-0017
    /// resolved-questions: accessibility). Rebuilt every render.
    modal_scrims: Vec<usize>,
}

impl EventRouter {
    /// Creates an empty router.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears the registered handlers and focusables (so they can be rebuilt
    /// from a fresh layout each tick) while **preserving** the transient gesture
    /// state — the in-flight `down` press and the current `focused` element.
    /// A `tap` spans two ticks (down then up) with a re-render between, so that
    /// state must survive the rebuild (RFC-0003 E4).
    pub fn clear_handlers(&mut self) {
        self.handlers.clear();
        self.focusables.clear();
        self.disabled.clear();
        self.hover_regions.clear();
        self.modal_floor = 0;
        self.focusable_floor = 0;
        self.modal_scrims.clear();
    }

    /// Registers a modal `Overlay`'s full-viewport scrim (RFC-0017 §Modality).
    ///
    /// Must be called *after* the main tree's handlers and *before* the
    /// overlay's own content handlers are registered, so the scrim sits below
    /// the content (content wins where it overlaps) but above the whole main
    /// tree. Raising [`modal_floor`](Self::modal_floor) to the scrim's index
    /// excludes every earlier handler from hit-testing — that is the modality:
    /// input can no longer reach anything beneath the overlay. `dismiss` is the
    /// action run when the scrim is tapped or `Escape` is pressed; a modal
    /// overlay with no `dismiss` still blocks input but cannot be dismissed by
    /// an outside tap (the caller relies on its own controls).
    pub fn push_modal_scrim(&mut self, elem: u32, rect: Rect, dismiss: Option<Action>) {
        self.modal_floor = self.handlers.len();
        self.focusable_floor = self.focusables.len();
        let idx = self.handlers.len();
        self.handlers.push(Handler {
            elem,
            rect,
            kind: EventKind::Tap,
            action: dismiss.unwrap_or_else(|| Box::new(|_, _| {})),
        });
        self.modal_scrims.push(idx);
    }

    /// Registers `elem`'s `rect` as a hover/press hit region (RFC-0016) so an
    /// element styled with `on hover`/`on pressed` but no event handler still
    /// reports those interaction states. Rebuilt every render.
    pub fn track_region(&mut self, elem: u32, rect: Rect) {
        self.hover_regions.push((elem, rect));
    }

    /// Marks `elem` disabled for this tick (RFC-0012 S5): it reports the
    /// `DISABLED` [`StyleState`] and its handlers never fire. Rebuilt every
    /// render, so an element re-enabled by a `var` flip simply isn't re-marked.
    pub fn set_disabled(&mut self, elem: u32) {
        self.disabled.insert(elem);
    }

    /// The interaction-state mask of `elem` (RFC-0012 Part B): `HOVER`/`PRESSED`/
    /// `FOCUSED` from the live gesture state, `DISABLED` from [`set_disabled`].
    ///
    /// [`set_disabled`]: Self::set_disabled
    #[must_use]
    pub fn style_state(&self, elem: u32) -> StyleState {
        let mut s = StyleState::empty();
        if self.hovered == Some(elem) {
            s.insert(StyleState::HOVER);
        }
        if self.down.as_ref().and_then(|d| d.elem) == Some(elem) {
            s.insert(StyleState::PRESSED);
        }
        if self.focused == Some(elem) {
            s.insert(StyleState::FOCUSED);
        }
        if self.disabled.contains(&elem) {
            s.insert(StyleState::DISABLED);
        }
        s
    }

    /// Registers a handler for `kind` on element `elem`'s `rect`.
    pub fn on(&mut self, elem: u32, rect: Rect, kind: EventKind, action: Action) {
        self.handlers.push(Handler {
            elem,
            rect,
            kind,
            action,
        });
    }

    /// Whether a pointer press at `pos` lands on an element that claims the
    /// gesture — one carrying a tap/press/drag/change handler. A `ScrollView`
    /// consults this to defer drag-to-scroll to interactive children (buttons,
    /// sliders) while still scrolling when the press lands on inert content
    /// (RFC-0005). Style-only hover/press regions do not claim the gesture.
    #[must_use]
    pub fn claims_pointer(&self, pos: (f32, f32)) -> bool {
        self.handlers.iter().any(|h| {
            contains(h.rect, pos)
                && matches!(
                    h.kind,
                    EventKind::Tap
                        | EventKind::PointerDown
                        | EventKind::PointerDrag
                        | EventKind::Change
                )
        })
    }

    /// Registers a focusable element bound to `focused_sig` (`#[focused: …]`).
    pub fn focusable(&mut self, elem: u32, rect: Rect, focused_sig: SignalId) {
        self.focusables.push(Focusable {
            elem,
            rect,
            focused_sig,
        });
    }

    /// Whether `elem` is the currently focused element (for focus-indicator
    /// visuals, M19). Reflects the focus state carried across the per-tick
    /// handler rebuild.
    #[must_use]
    pub fn is_focused(&self, elem: u32) -> bool {
        self.focused == Some(elem)
    }

    /// Sets the initially-focused element and its `var`.
    pub fn set_focus(&mut self, ctx: &mut ReactiveCtx, elem: u32) {
        self.steal_focus(ctx, Some(elem));
    }

    /// Drains one tick's events: coalesces continuous ones (E7), dispatches in
    /// order (E4 ordering, E1 write-back, E3 focus) — marking only. The caller
    /// runs the single pull afterwards (§8).
    pub fn dispatch_tick(
        &mut self,
        ctx: &mut ReactiveCtx,
        atlas: Option<&byard_core::atlas::layout::LayoutAtlas>,
        events: Vec<InputEvent>,
    ) {
        // E7 — coalesce continuous events per (kind, element); keep discrete in
        // FIFO order.
        let mut ordered: Vec<InputEvent> = Vec::new();
        // (kind, elem) → index into `ordered` for the coalesced event.
        let mut coalesced: Vec<((EventKind, Option<u32>), usize)> = Vec::new();
        for ev in events {
            if ev.kind.is_continuous() {
                let elem = self.hit_any(atlas, ev.pos);
                let key = (ev.kind, elem);
                if let Some((_, idx)) = coalesced.iter().find(|(k, _)| *k == key) {
                    let slot = &mut ordered[*idx];
                    slot.pos = ev.pos; // latest absolute position
                    slot.delta.0 += ev.delta.0; // summed deltas
                    slot.delta.1 += ev.delta.1;
                } else {
                    coalesced.push((key, ordered.len()));
                    ordered.push(ev);
                }
            } else {
                ordered.push(ev);
            }
        }

        for ev in ordered {
            self.dispatch(ctx, atlas, &ev);
        }
    }

    fn dispatch(
        &mut self,
        ctx: &mut ReactiveCtx,
        atlas: Option<&byard_core::atlas::layout::LayoutAtlas>,
        ev: &InputEvent,
    ) {
        // Expose position to handlers (e.g. Slider drag).
        CURRENT_EVENT_POS.with(|c| c.set(ev.pos));

        match ev.kind {
            EventKind::PointerDown => {
                let elem = self.hit_any(atlas, ev.pos);
                let secondary = matches!(ev.value, Some(Value::Bool(true)));
                self.down = Some(DownState {
                    elem,
                    pos: ev.pos,
                    time_ms: ev.time_ms,
                    secondary,
                });
                if secondary {
                    self.fire(ctx, atlas, EventKind::Secondary, ev.pos, None);
                } else {
                    self.fire(ctx, atlas, EventKind::PointerDown, ev.pos, None);
                }
                // A press on a focusable steals focus (E3).
                if let Some(f) = self.focusable_at(atlas, ev.pos) {
                    self.steal_focus(ctx, Some(f));
                }
            }
            EventKind::PointerUp => {
                // E4 precedence: pointer_up fires before tap.
                self.fire(ctx, atlas, EventKind::PointerUp, ev.pos, None);
                if let Some(down) = self.down.take() {
                    let up_elem = self.hit_any(atlas, ev.pos);
                    let dx = ev.pos.0 - down.pos.0;
                    let dy = ev.pos.1 - down.pos.1;
                    let moved = (dx * dx + dy * dy).sqrt();
                    let elapsed = ev.time_ms.saturating_sub(down.time_ms);
                    if down.elem.is_some()
                        && down.elem == up_elem
                        && moved < TAP_SLOP
                        && elapsed < TAP_MS
                        && !down.secondary
                    {
                        // Double-tap detection (M24).
                        let is_double = self.last_tap.is_some_and(|(t, elem)| {
                            ev.time_ms.saturating_sub(t) < DOUBLE_TAP_MS && elem == up_elem
                        });
                        if is_double {
                            self.fire(ctx, atlas, EventKind::DoubleTap, ev.pos, None);
                            self.last_tap = None;
                        } else {
                            self.fire(ctx, atlas, EventKind::Tap, ev.pos, None);
                            self.last_tap = Some((ev.time_ms, up_elem));
                        }
                    } else {
                        // Check long press (held > LONG_PRESS_MS without much movement).
                        if down.elem.is_some()
                            && down.elem == self.hit_any(atlas, ev.pos)
                            && moved < TAP_SLOP
                            && elapsed >= LONG_PRESS_MS
                        {
                            self.fire(ctx, atlas, EventKind::LongPress, ev.pos, None);
                        }
                        self.last_tap = None;
                    }
                }
            }
            EventKind::Change => {
                self.fire(ctx, atlas, EventKind::Change, ev.pos, ev.value.as_ref());
            }
            EventKind::PointerMove | EventKind::Scroll | EventKind::Wheel => {
                self.fire(ctx, atlas, ev.kind, ev.pos, None);
                if ev.kind == EventKind::PointerMove {
                    // Synthesize PointerDrag when the button is held (M16: Slider).
                    if self.down.is_some() {
                        self.fire(ctx, atlas, EventKind::PointerDrag, ev.pos, None);
                    } else {
                        // Enter / Exit / Hover (M24): compare new hovered elem to prev.
                        let new_hover = self.hit_any(atlas, ev.pos);
                        if new_hover != self.hovered {
                            if self.hovered.is_some() {
                                // Fire PointerExit on the element we left.
                                if let Some(old_pos) = self.hovered.map(|_| ev.pos) {
                                    self.fire_on_elem(
                                        ctx,
                                        self.hovered,
                                        EventKind::PointerExit,
                                        old_pos,
                                        None,
                                    );
                                }
                            }
                            self.hovered = new_hover;
                            if new_hover.is_some() {
                                self.fire_on_elem(
                                    ctx,
                                    new_hover,
                                    EventKind::PointerEnter,
                                    ev.pos,
                                    None,
                                );
                            }
                        }
                        // Hover fires every move while inside (like mousemove).
                        if new_hover.is_some() {
                            self.fire_on_elem(ctx, new_hover, EventKind::Hover, ev.pos, None);
                        }
                    }
                }
            }
            EventKind::Tap
            | EventKind::PointerDrag
            | EventKind::PointerEnter
            | EventKind::PointerExit
            | EventKind::Hover
            | EventKind::LongPress
            | EventKind::DoubleTap
            | EventKind::Secondary => {
                self.fire(ctx, atlas, ev.kind, ev.pos, None);
            }
            // Keyboard events are routed to the focused element (M17/M18).
            EventKind::KeyDown => {
                // Tab key cycles focus (M18).
                if let Some(Value::Str(key)) = &ev.value {
                    if key == "Tab" {
                        self.tab_focus(ctx, false);
                        return;
                    }
                    // RFC-0017: Escape dismisses the topmost modal overlay.
                    if key == "Escape" && !self.modal_scrims.is_empty() {
                        self.dismiss_topmost_modal(ctx);
                        return;
                    }
                }
                self.fire_focused(ctx, EventKind::KeyDown, ev.value.as_ref());
            }
            EventKind::KeyUp => {
                self.fire_focused(ctx, EventKind::KeyUp, ev.value.as_ref());
            }
            EventKind::TextInput => {
                self.fire_focused(ctx, EventKind::TextInput, ev.value.as_ref());
            }
            // Never produced as a raw `InputEvent` — `steal_focus` fires
            // these directly via `fire_on_elem`. Present only so this match
            // stays exhaustive as the enum grows.
            EventKind::Focus | EventKind::Blur => {}
        }
    }

    /// Fires a handler of `kind` on a specific element `elem_id` (for enter/exit/hover).
    fn fire_on_elem(
        &mut self,
        ctx: &mut ReactiveCtx,
        elem_id: Option<u32>,
        kind: EventKind,
        pos: (f32, f32),
        payload: Option<&Value>,
    ) {
        let Some(target) = elem_id else { return };
        // A disabled element dispatches nothing (RFC-0012 S5).
        if self.disabled.contains(&target) {
            return;
        }
        let i = self
            .handlers
            .iter()
            .enumerate()
            .rev()
            .find(|(_, h)| h.kind == kind && h.elem == target)
            .map(|(i, _)| i);
        let Some(i) = i else { return };
        let _ = pos;
        let mut action = std::mem::replace(&mut self.handlers[i].action, Box::new(|_, _| {}));
        action(ctx, payload);
        self.handlers[i].action = action;
    }

    /// Fires the topmost handler of `kind` covering `pos` (no bubbling; §7).
    fn fire(
        &mut self,
        ctx: &mut ReactiveCtx,
        atlas: Option<&byard_core::atlas::layout::LayoutAtlas>,
        kind: EventKind,
        pos: (f32, f32),
        payload: Option<&Value>,
    ) {
        let Some(i) = self.hit(atlas, pos, kind) else {
            return;
        };
        // A disabled element dispatches nothing (RFC-0012 S5).
        if self.disabled.contains(&self.handlers[i].elem) {
            return;
        }
        // Take the action out to avoid aliasing `self` while it runs.
        let mut action = std::mem::replace(&mut self.handlers[i].action, Box::new(|_, _| {}));
        action(ctx, payload);
        self.handlers[i].action = action;
    }

    /// Focus stealing (E3): blur the previous element's `var`, focus the new.
    fn steal_focus(&mut self, ctx: &mut ReactiveCtx, new: Option<u32>) {
        if self.focused == new {
            return;
        }
        if let Some(old) = self.focused {
            if let Some(f) = self.focusables.iter().find(|f| f.elem == old) {
                ctx.write_signal(f.focused_sig, Value::Bool(false));
            }
            // RFC-0012 S2: `blur =>` sugar fires on the falling edge, riding
            // the same `focused_sig` transition as the reflected prop above.
            self.fire_on_elem(ctx, Some(old), EventKind::Blur, (0.0, 0.0), None);
        }
        if let Some(n) = new {
            if let Some(f) = self.focusables.iter().find(|f| f.elem == n) {
                ctx.write_signal(f.focused_sig, Value::Bool(true));
            }
            // RFC-0012 S2: `focus =>` sugar fires on the rising edge.
            self.fire_on_elem(ctx, Some(n), EventKind::Focus, (0.0, 0.0), None);
        }
        self.focused = new;
    }

    /// Finds the topmost handler of `kind` whose registered (inflated, E8) hit
    /// rect contains `pos`. The inflated rects are the authoritative hit areas
    /// (RFC-0003 §4.2/E8) — there is **no** ancestor bubbling (§7) and no
    /// catch-all fallback, so a click outside every handler's rect fires
    /// nothing.
    fn hit(
        &self,
        _atlas: Option<&byard_core::atlas::layout::LayoutAtlas>,
        pos: (f32, f32),
        kind: EventKind,
    ) -> Option<usize> {
        self.handlers
            .iter()
            .enumerate()
            .rev()
            // RFC-0017: a mounted modal overlay raises `modal_floor`, hiding every
            // handler beneath its scrim so input can't reach the main tree.
            .find(|(i, h)| *i >= self.modal_floor && h.kind == kind && contains(h.rect, pos))
            .map(|(i, _)| i)
    }

    /// The element id of the topmost handler whose hit rect contains `pos`
    /// (used to match a tap's down/up element and to key event coalescing).
    fn hit_any(
        &self,
        _atlas: Option<&byard_core::atlas::layout::LayoutAtlas>,
        pos: (f32, f32),
    ) -> Option<u32> {
        self.handlers
            .iter()
            .enumerate()
            .rev()
            .find(|(i, h)| *i >= self.modal_floor && contains(h.rect, pos))
            .map(|(_, h)| h.elem)
            // Fall back to declarative hover/press regions (RFC-0016) for
            // elements that style interaction states but register no handler.
            // Suppressed under a modal overlay — nothing beneath the scrim may
            // report hover/press either.
            .or_else(|| {
                if self.modal_floor > 0 {
                    return None;
                }
                self.hover_regions
                    .iter()
                    .rev()
                    .find(|(_, rect)| contains(*rect, pos))
                    .map(|(elem, _)| *elem)
            })
    }

    fn focusable_at(
        &self,
        _atlas: Option<&byard_core::atlas::layout::LayoutAtlas>,
        pos: (f32, f32),
    ) -> Option<u32> {
        self.focusables
            .iter()
            .enumerate()
            .rev()
            // RFC-0017 focus trap: a modal overlay confines focus to its own
            // scope; a press below the focus floor cannot steal focus.
            .find(|(i, f)| *i >= self.focusable_floor && contains(f.rect, pos))
            .map(|(_, f)| f.elem)
    }

    /// Fires the handler of `kind` registered on the currently focused element,
    /// if any (M17/M18 keyboard routing).
    fn fire_focused(&mut self, ctx: &mut ReactiveCtx, kind: EventKind, payload: Option<&Value>) {
        let Some(focused) = self.focused else {
            return;
        };
        // A disabled element dispatches nothing (RFC-0012 S5).
        if self.disabled.contains(&focused) {
            return;
        }
        let i = self
            .handlers
            .iter()
            .enumerate()
            .rev()
            .find(|(_, h)| h.kind == kind && h.elem == focused)
            .map(|(i, _)| i);
        let Some(i) = i else {
            return;
        };
        let mut action = std::mem::replace(&mut self.handlers[i].action, Box::new(|_, _| {}));
        action(ctx, payload);
        self.handlers[i].action = action;
    }

    /// Advances keyboard focus to the next (or previous) focusable element
    /// (M18, Tab traversal).
    fn tab_focus(&mut self, ctx: &mut ReactiveCtx, reverse: bool) {
        // RFC-0017 focus trap: when a modal overlay is mounted, cycle only within
        // its scope (`focusables[focusable_floor..]`), wrapping last→first inside
        // it. Outside a modal overlay, `focusable_floor` is 0 — the whole set.
        let floor = self.focusable_floor;
        let scope = &self.focusables[floor..];
        if scope.is_empty() {
            return;
        }
        let n = scope.len();
        let cur_idx = self
            .focused
            .and_then(|elem| scope.iter().position(|f| f.elem == elem));
        let next_idx = match cur_idx {
            Some(i) => {
                if reverse {
                    (i + n - 1) % n
                } else {
                    (i + 1) % n
                }
            }
            None => {
                if reverse {
                    n - 1
                } else {
                    0
                }
            }
        };
        let next_elem = scope[next_idx].elem;
        self.steal_focus(ctx, Some(next_elem));
    }

    /// Fires the topmost mounted modal overlay's `dismiss` action (RFC-0017:
    /// `Escape` dismisses the topmost modal overlay). No-op when none is mounted.
    fn dismiss_topmost_modal(&mut self, ctx: &mut ReactiveCtx) {
        let Some(&i) = self.modal_scrims.last() else {
            return;
        };
        let mut action = std::mem::replace(&mut self.handlers[i].action, Box::new(|_, _| {}));
        action(ctx, None);
        self.handlers[i].action = action;
    }

    /// Test/inspection accessor: the `(elem, kind, rect)` of every registered
    /// handler, so a hit-test can be asserted against real bounds.
    #[must_use]
    pub fn handler_rects(&self) -> Vec<(u32, EventKind, Rect)> {
        self.handlers
            .iter()
            .map(|h| (h.elem, h.kind, h.rect))
            .collect()
    }

    /// Returns `true` while a pointer is held down (between `PointerDown` and
    /// `PointerUp`).  Used by the hot-reload gesture gate (RFC-0006 §3.2, E5).
    #[must_use]
    pub fn is_pointer_pressed(&self) -> bool {
        self.down.is_some()
    }
}

/// Reflected write-back with value-dedup (E1): builds a `Change` action that
/// writes `sig` only when the incoming value differs from the current one, so a
/// two-way binding loop terminates at length 1.
#[must_use]
pub fn write_back_action(sig: SignalId) -> Action {
    Box::new(move |ctx, payload| {
        if let Some(new) = payload {
            // Value dedup: equal ⇒ discard at zero cost.
            if ctx.peek_signal(sig) != *new {
                ctx.write_signal(sig, new.clone());
            }
        }
    })
}

fn contains(r: Rect, p: (f32, f32)) -> bool {
    p.0 >= r.x && p.0 <= r.x + r.w && p.1 >= r.y && p.1 <= r.y + r.h
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn rect() -> Rect {
        Rect::new(0.0, 0.0, 100.0, 100.0)
    }

    #[test]
    fn style_state_reports_disabled_and_is_a_proper_bit_set() {
        let mut r = EventRouter::new();
        // Nothing set → empty; a marked element reports DISABLED, others don't.
        assert_eq!(r.style_state(9), StyleState::empty());
        r.set_disabled(9);
        assert!(r.style_state(9).contains(StyleState::DISABLED));
        assert!(!r.style_state(9).contains(StyleState::HOVER));
        assert_eq!(r.style_state(10), StyleState::empty());
        // `clear_handlers` rebuilds the per-tick disabled set.
        r.clear_handlers();
        assert!(!r.style_state(9).contains(StyleState::DISABLED));

        // Bit-set semantics: `contains` is subset containment.
        let hover_focus = {
            let mut s = StyleState::HOVER;
            s.insert(StyleState::FOCUSED);
            s
        };
        assert!(hover_focus.contains(StyleState::HOVER));
        assert!(hover_focus.contains(StyleState::FOCUSED));
        assert!(!hover_focus.contains(StyleState::PRESSED));
    }

    /// Increments `sig` (the `count++` action).
    fn inc(sig: SignalId) -> Action {
        Box::new(move |ctx, _| {
            let n = ctx.peek_signal(sig).as_int().unwrap_or(0);
            ctx.write_signal(sig, Value::Int(n + 1));
        })
    }

    #[test]
    fn tap_mutates_a_var_and_projects_next_tick() {
        let mut ctx = ReactiveCtx::new();
        let count = ctx.create_signal(Value::Int(0));
        let bind = ctx.open_value_binding(super::super::reactive::FrameTarget(0), move |c| {
            c.read_signal(count)
        });
        let e = ctx.begin_tick();
        ctx.pull(e);

        let mut router = EventRouter::new();
        router.on(1, rect(), EventKind::Tap, inc(count));

        router.dispatch_tick(
            &mut ctx,
            None,
            vec![
                InputEvent::pointer(EventKind::PointerDown, (5.0, 5.0), 0),
                InputEvent::pointer(EventKind::PointerUp, (6.0, 6.0), 100),
            ],
        );
        let e = ctx.begin_tick();
        ctx.pull(e);
        assert_eq!(ctx.binding_value(bind), Some(Value::Int(1)));
    }

    #[test]
    fn pointer_up_fires_before_tap() {
        let mut ctx = ReactiveCtx::new();
        let log = Rc::new(RefCell::new(Vec::<&'static str>::new()));
        let mut router = EventRouter::new();
        let l1 = Rc::clone(&log);
        router.on(
            1,
            rect(),
            EventKind::PointerUp,
            Box::new(move |_, _| l1.borrow_mut().push("up")),
        );
        let l2 = Rc::clone(&log);
        router.on(
            1,
            rect(),
            EventKind::Tap,
            Box::new(move |_, _| l2.borrow_mut().push("tap")),
        );

        router.dispatch_tick(
            &mut ctx,
            None,
            vec![
                InputEvent::pointer(EventKind::PointerDown, (5.0, 5.0), 0),
                InputEvent::pointer(EventKind::PointerUp, (5.0, 5.0), 50),
            ],
        );
        assert_eq!(*log.borrow(), vec!["up", "tap"]);
    }

    #[test]
    fn continuous_moves_are_coalesced_into_one_call() {
        let mut ctx = ReactiveCtx::new();
        let calls = Rc::new(RefCell::new(0u32));
        let sum = ctx.create_signal(Value::Int(0));
        let mut router = EventRouter::new();
        let c = Rc::clone(&calls);
        router.on(
            1,
            rect(),
            EventKind::PointerMove,
            Box::new(move |_, _| *c.borrow_mut() += 1),
        );

        let moves: Vec<_> = (0..100)
            .map(|i| InputEvent {
                kind: EventKind::PointerMove,
                pos: (i as f32, 0.0),
                delta: (1.0, 0.0),
                value: None,
                time_ms: i,
            })
            .collect();
        router.dispatch_tick(&mut ctx, None, moves);
        assert_eq!(*calls.borrow(), 1, "100 moves coalesce to a single walk");
        let _ = sum;
    }

    #[test]
    fn write_back_round_trips_with_dedup_and_no_loop() {
        let mut ctx = ReactiveCtx::new();
        let query = ctx.create_signal(Value::Str(String::new()));
        let mut router = EventRouter::new();
        router.on(1, rect(), EventKind::Change, write_back_action(query));

        // Physical typing of "a".
        router.dispatch_tick(
            &mut ctx,
            None,
            vec![InputEvent {
                kind: EventKind::Change,
                pos: (5.0, 5.0),
                delta: (0.0, 0.0),
                value: Some(Value::Str("a".to_string())),
                time_ms: 0,
            }],
        );
        assert_eq!(ctx.peek_signal(query), Value::Str("a".to_string()));
        let version_after_first = ctx.signal_subscriber_count(query); // touch API

        // Re-delivering the SAME value is deduped: no further write (loop dies).
        let writes_before = peek_writes(&ctx, query);
        router.dispatch_tick(
            &mut ctx,
            None,
            vec![InputEvent {
                kind: EventKind::Change,
                pos: (5.0, 5.0),
                delta: (0.0, 0.0),
                value: Some(Value::Str("a".to_string())),
                time_ms: 1,
            }],
        );
        assert_eq!(
            peek_writes(&ctx, query),
            writes_before,
            "equal value deduped"
        );
        let _ = version_after_first;
    }

    /// Reads a signal's value-version via a fresh tick-free read (helper).
    fn peek_writes(ctx: &ReactiveCtx, sig: SignalId) -> Value {
        ctx.peek_signal(sig)
    }

    #[test]
    fn focus_steal_flips_both_vars_in_one_tick() {
        let mut ctx = ReactiveCtx::new();
        let fa = ctx.create_signal(Value::Bool(false));
        let fb = ctx.create_signal(Value::Bool(false));
        let mut router = EventRouter::new();
        router.focusable(1, Rect::new(0.0, 0.0, 50.0, 50.0), fa);
        router.focusable(2, Rect::new(50.0, 0.0, 50.0, 50.0), fb);
        // A pointer handler so hit_any sees element 2.
        router.on(
            2,
            Rect::new(50.0, 0.0, 50.0, 50.0),
            EventKind::PointerDown,
            Box::new(|_, _| {}),
        );

        router.set_focus(&mut ctx, 1);
        assert_eq!(ctx.peek_signal(fa), Value::Bool(true));

        // Press element B → A blurs, B focuses, same tick.
        router.dispatch_tick(
            &mut ctx,
            None,
            vec![InputEvent::pointer(EventKind::PointerDown, (60.0, 10.0), 0)],
        );
        assert_eq!(ctx.peek_signal(fa), Value::Bool(false), "A blurred");
        assert_eq!(ctx.peek_signal(fb), Value::Bool(true), "B focused");
    }

    #[test]
    fn far_or_slow_press_is_not_a_tap() {
        let mut ctx = ReactiveCtx::new();
        let count = ctx.create_signal(Value::Int(0));
        let mut router = EventRouter::new();
        router.on(1, rect(), EventKind::Tap, inc(count));

        // Moved > 8px between down and up: not a tap.
        router.dispatch_tick(
            &mut ctx,
            None,
            vec![
                InputEvent::pointer(EventKind::PointerDown, (5.0, 5.0), 0),
                InputEvent::pointer(EventKind::PointerUp, (40.0, 5.0), 100),
            ],
        );
        assert_eq!(
            ctx.peek_signal(count),
            Value::Int(0),
            "far drag is not a tap"
        );

        // Over 500 ms: not a tap.
        router.dispatch_tick(
            &mut ctx,
            None,
            vec![
                InputEvent::pointer(EventKind::PointerDown, (5.0, 5.0), 0),
                InputEvent::pointer(EventKind::PointerUp, (6.0, 5.0), 800),
            ],
        );
        assert_eq!(
            ctx.peek_signal(count),
            Value::Int(0),
            "slow press is not a tap"
        );
    }

    #[test]
    fn tap_bubbles_to_parent_handler_in_atlas() {
        use byard_core::atlas::layout::{ContainerStyle, LayoutAtlas, LeafSize};
        use byard_core::frame::Viewport;

        let mut ctx = ReactiveCtx::new();
        let count = ctx.create_signal(Value::Int(0));

        let mut atlas = LayoutAtlas::new();
        // Child: index 0 (leaf)
        let child = atlas.add_leaf(LeafSize::new(20.0, 20.0)).unwrap();
        // Parent: index 1 (container)
        let parent = atlas
            .add_container(ContainerStyle::new(Some(100.0), Some(100.0)), &[child])
            .unwrap();
        atlas.set_root(parent).unwrap();
        atlas.compute(Viewport::new(800.0, 600.0)).unwrap();

        let parent_idx = atlas.node_index(parent).unwrap();

        let mut router = EventRouter::new();
        // Register handler on the parent element (index 1)
        router.on(parent_idx, rect(), EventKind::Tap, inc(count));

        // Click directly on the child (which is at layout coords (0,0)-(20,20))
        router.dispatch_tick(
            &mut ctx,
            Some(&atlas),
            vec![
                InputEvent::pointer(EventKind::PointerDown, (5.0, 5.0), 0),
                InputEvent::pointer(EventKind::PointerUp, (5.0, 5.0), 100),
            ],
        );

        assert_eq!(ctx.peek_signal(count), Value::Int(1));
    }

    // ── M24: remaining event catalog ─────────────────────────────────────

    #[test]
    fn double_tap_fires_within_threshold_and_not_beyond() {
        let mut ctx = ReactiveCtx::new();
        let taps = ctx.create_signal(Value::Int(0));
        let doubles = ctx.create_signal(Value::Int(0));
        let mut router = EventRouter::new();
        router.on(1, rect(), EventKind::Tap, inc(taps));
        router.on(1, rect(), EventKind::DoubleTap, inc(doubles));

        // First tap.
        router.dispatch_tick(
            &mut ctx,
            None,
            vec![
                InputEvent::pointer(EventKind::PointerDown, (5.0, 5.0), 0),
                InputEvent::pointer(EventKind::PointerUp, (5.0, 5.0), 50),
            ],
        );
        assert_eq!(ctx.peek_signal(taps), Value::Int(1));
        assert_eq!(ctx.peek_signal(doubles), Value::Int(0));

        // Second tap within DOUBLE_TAP_MS → double.
        router.dispatch_tick(
            &mut ctx,
            None,
            vec![
                InputEvent::pointer(EventKind::PointerDown, (5.0, 5.0), 100),
                InputEvent::pointer(EventKind::PointerUp, (5.0, 5.0), 150),
            ],
        );
        assert_eq!(ctx.peek_signal(doubles), Value::Int(1), "double-tap fired");

        // Third tap — gap since last confirmed single tap, reset tracker.
        // last_tap was cleared after double, so next tap is a fresh single.
        router.dispatch_tick(
            &mut ctx,
            None,
            vec![
                InputEvent::pointer(EventKind::PointerDown, (5.0, 5.0), 600),
                InputEvent::pointer(EventKind::PointerUp, (5.0, 5.0), 650),
            ],
        );
        assert_eq!(
            ctx.peek_signal(taps),
            Value::Int(2),
            "single tap after reset"
        );
    }

    #[test]
    fn long_press_fires_after_hold_threshold() {
        let mut ctx = ReactiveCtx::new();
        let lp = ctx.create_signal(Value::Int(0));
        let mut router = EventRouter::new();
        router.on(1, rect(), EventKind::LongPress, inc(lp));

        // Hold for > LONG_PRESS_MS (500ms).
        router.dispatch_tick(
            &mut ctx,
            None,
            vec![
                InputEvent::pointer(EventKind::PointerDown, (5.0, 5.0), 0),
                InputEvent::pointer(EventKind::PointerUp, (5.0, 5.0), 600),
            ],
        );
        assert_eq!(ctx.peek_signal(lp), Value::Int(1), "long press fired");
    }

    #[test]
    fn long_press_does_not_fire_below_threshold() {
        let mut ctx = ReactiveCtx::new();
        let lp = ctx.create_signal(Value::Int(0));
        let mut router = EventRouter::new();
        router.on(1, rect(), EventKind::LongPress, inc(lp));

        router.dispatch_tick(
            &mut ctx,
            None,
            vec![
                InputEvent::pointer(EventKind::PointerDown, (5.0, 5.0), 0),
                InputEvent::pointer(EventKind::PointerUp, (5.0, 5.0), 200),
            ],
        );
        assert_eq!(ctx.peek_signal(lp), Value::Int(0), "long press not fired");
    }

    #[test]
    fn pointer_enter_and_exit_fire_on_crossing_boundary() {
        let mut ctx = ReactiveCtx::new();
        let enters = ctx.create_signal(Value::Int(0));
        let exits = ctx.create_signal(Value::Int(0));
        let mut router = EventRouter::new();
        router.on(1, rect(), EventKind::PointerEnter, inc(enters));
        router.on(1, rect(), EventKind::PointerExit, inc(exits));

        // Move into the rect.
        router.dispatch_tick(
            &mut ctx,
            None,
            vec![InputEvent::pointer(EventKind::PointerMove, (50.0, 50.0), 0)],
        );
        assert_eq!(ctx.peek_signal(enters), Value::Int(1), "entered");
        assert_eq!(ctx.peek_signal(exits), Value::Int(0));

        // Move to a different spot inside — no new enter/exit.
        router.dispatch_tick(
            &mut ctx,
            None,
            vec![InputEvent::pointer(
                EventKind::PointerMove,
                (60.0, 60.0),
                10,
            )],
        );
        assert_eq!(ctx.peek_signal(enters), Value::Int(1));

        // Move outside the rect.
        router.dispatch_tick(
            &mut ctx,
            None,
            vec![InputEvent::pointer(
                EventKind::PointerMove,
                (200.0, 200.0),
                20,
            )],
        );
        assert_eq!(ctx.peek_signal(exits), Value::Int(1), "exited");
    }

    #[test]
    fn secondary_fires_on_secondary_down_event() {
        let mut ctx = ReactiveCtx::new();
        let sec = ctx.create_signal(Value::Int(0));
        let mut router = EventRouter::new();
        router.on(1, rect(), EventKind::Secondary, inc(sec));

        // Secondary press: payload = Bool(true).
        router.dispatch_tick(
            &mut ctx,
            None,
            vec![InputEvent {
                kind: EventKind::PointerDown,
                pos: (5.0, 5.0),
                delta: (0.0, 0.0),
                value: Some(Value::Bool(true)), // marks secondary button
                time_ms: 0,
            }],
        );
        assert_eq!(ctx.peek_signal(sec), Value::Int(1), "secondary fired");
    }

    // ── RFC-0017: overlay modality & focus trap ──────────────────────────

    fn viewport() -> Rect {
        Rect::new(0.0, 0.0, 200.0, 200.0)
    }

    #[test]
    fn modal_scrim_blocks_lower_handlers_and_dismisses_outside_taps() {
        let mut ctx = ReactiveCtx::new();
        let behind = ctx.create_signal(Value::Int(0));
        let dismissed = ctx.create_signal(Value::Int(0));
        let content = ctx.create_signal(Value::Int(0));

        let mut router = EventRouter::new();
        // Main tree: a full-viewport tap handler behind the overlay.
        router.on(1, viewport(), EventKind::Tap, inc(behind));
        // Modal scrim, then the overlay's centred content handler.
        router.push_modal_scrim(2, viewport(), Some(inc(dismissed)));
        router.on(
            3,
            Rect::new(80.0, 80.0, 40.0, 40.0),
            EventKind::Tap,
            inc(content),
        );

        // Tap outside the content (top-left): the scrim dismisses, the button
        // behind never fires.
        router.dispatch_tick(
            &mut ctx,
            None,
            vec![
                InputEvent::pointer(EventKind::PointerDown, (5.0, 5.0), 0),
                InputEvent::pointer(EventKind::PointerUp, (5.0, 5.0), 20),
            ],
        );
        assert_eq!(ctx.peek_signal(dismissed), Value::Int(1), "scrim dismissed");
        assert_eq!(ctx.peek_signal(behind), Value::Int(0), "main tree blocked");

        // Tap on the content: it wins over the scrim (registered after it).
        router.dispatch_tick(
            &mut ctx,
            None,
            vec![
                InputEvent::pointer(EventKind::PointerDown, (100.0, 100.0), 100),
                InputEvent::pointer(EventKind::PointerUp, (100.0, 100.0), 120),
            ],
        );
        assert_eq!(ctx.peek_signal(content), Value::Int(1), "content fired");
        assert_eq!(
            ctx.peek_signal(dismissed),
            Value::Int(1),
            "scrim did not re-fire"
        );

        // Escape dismisses the topmost modal overlay.
        router.dispatch_tick(
            &mut ctx,
            None,
            vec![InputEvent {
                kind: EventKind::KeyDown,
                pos: (0.0, 0.0),
                delta: (0.0, 0.0),
                value: Some(Value::Str("Escape".to_string())),
                time_ms: 200,
            }],
        );
        assert_eq!(
            ctx.peek_signal(dismissed),
            Value::Int(2),
            "Escape dismissed"
        );

        // A rebuild clears the modal floor so the main tree is live again.
        router.clear_handlers();
        router.on(1, viewport(), EventKind::Tap, inc(behind));
        router.dispatch_tick(
            &mut ctx,
            None,
            vec![
                InputEvent::pointer(EventKind::PointerDown, (5.0, 5.0), 300),
                InputEvent::pointer(EventKind::PointerUp, (5.0, 5.0), 320),
            ],
        );
        assert_eq!(
            ctx.peek_signal(behind),
            Value::Int(1),
            "main tree live once the overlay unmounts"
        );
    }

    #[test]
    fn modal_focus_trap_confines_tab_to_the_overlay_scope() {
        let mut ctx = ReactiveCtx::new();
        let fa = ctx.create_signal(Value::Bool(false)); // main, below the floor
        let fb = ctx.create_signal(Value::Bool(false)); // overlay
        let fc = ctx.create_signal(Value::Bool(false)); // overlay

        let mut router = EventRouter::new();
        router.focusable(1, viewport(), fa);
        router.push_modal_scrim(9, viewport(), None);
        router.focusable(2, viewport(), fb);
        router.focusable(3, viewport(), fc);

        let tab = || InputEvent {
            kind: EventKind::KeyDown,
            pos: (0.0, 0.0),
            delta: (0.0, 0.0),
            value: Some(Value::Str("Tab".to_string())),
            time_ms: 0,
        };

        // First Tab → first focusable in the overlay scope (fb), not fa.
        router.dispatch_tick(&mut ctx, None, vec![tab()]);
        assert_eq!(ctx.peek_signal(fb), Value::Bool(true));
        assert_eq!(
            ctx.peek_signal(fa),
            Value::Bool(false),
            "focus never leaves the modal"
        );

        // Second Tab → fc.
        router.dispatch_tick(&mut ctx, None, vec![tab()]);
        assert_eq!(ctx.peek_signal(fc), Value::Bool(true));

        // Third Tab wraps back to fb, still never touching fa.
        router.dispatch_tick(&mut ctx, None, vec![tab()]);
        assert_eq!(ctx.peek_signal(fb), Value::Bool(true));
        assert_eq!(ctx.peek_signal(fa), Value::Bool(false));
    }
}
