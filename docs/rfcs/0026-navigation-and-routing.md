# RFC-0026: Navigation & Routing — stacks, transitions, deep linking

- **Status:** Draft
- **Author(s):** Briany4717
- **Created:** 2026-07-10
- **Last updated:** 2026-07-10
- **Depends on:** RFC-0001 (§2.1 ViewArena, §5.1 logic thread), RFC-0002 (`when`/`for` structural effects, `var`/`inject`), RFC-0003 (events, no widget references), RFC-0007 (user-view instantiation), RFC-0010 (animations — transition curves), RFC-0011 (transforms — slide/fade transitions), RFC-0017 (overlay system — modal navigation, bottom sheets), RFC-0019 (callback props — navigation action forwarding).
- **Enables:** Multi-screen apps with back navigation, tab-based layouts, deep linking from OS intents, shared-element (hero) transitions. Essential for any non-trivial Material or Cupertino application.

---

## Summary

Add a **declarative navigation model** with three primitives: `NavStack` (a push/
pop stack of Views with animated transitions), `NavHost` (a route-based container
that maps URL-like paths to Views), and navigation actions (`navigate`, `back`,
`replace`) exposed as functions callable from event handlers. Navigation state is
a reactive `var` — there are no imperative navigation controllers, no route
objects to hold, no widget references.

```byld
View App() {
    NavStack {
        route "/" { HomePage() }
        route "/detail/:id" { DetailPage(id: route.params.id) }
        route "/settings" { SettingsPage() }
    }
}
```

---

## Motivation

Every real-world mobile app has multiple screens. Today, a Byard app can only
show one View tree — there is no way to push a detail screen, go back, or
deep-link from an OS intent. `byard-material`'s `NavigationBar` and
`NavigationDrawer` render the chrome but can't actually navigate anywhere.

Both Material and Cupertino have deeply ingrained navigation patterns:

- **Material:** top-level navigation (NavigationBar/Drawer switches views),
  detail push (list → detail with slide transition), bottom-sheet navigation.
- **Cupertino:** UINavigationController (push/pop with right-to-left slide and
  swipe-back gesture), tab bar, modal presentation.

Without framework-level navigation, developers would need to build it from
`when` blocks and manual `var` management — error-prone, no transitions, no
deep linking, no back-button integration.

---

## Guide-level explanation

### `NavStack` — push/pop navigation

```byld
View App() {
    var navPath = "/"

    NavStack(path: navPath) #[transition: slide] {
        route "/" {
            HomePage(on_select: {|id| navPath = "/detail/{id}"})
        }
        route "/detail/:id" {|params|
            DetailPage(id: params.id, on_back: { navPath = "/" })
        }
        route "/settings" {
            SettingsPage()
        }
    }
}
```

`NavStack` maintains a **stack of routes**. Changing `navPath` pushes or pops:

- Setting `navPath = "/detail/42"` pushes `/detail/42` onto the stack, animating
  the new View in from the right (or per `transition`).
- Setting `navPath = "/"` pops back to the root, animating the current View out.
- The stack tracks history automatically: `/` → `/detail/42` → `/settings` is a
  3-deep stack. Setting `navPath = "/"` pops two levels.

### Route matching

Routes are matched top-to-bottom. `:param` segments are dynamic:

| Pattern | Matches | Params |
|---|---|---|
| `/` | exact root | (none) |
| `/detail/:id` | `/detail/42`, `/detail/abc` | `id = "42"` |
| `/user/:uid/post/:pid` | `/user/5/post/12` | `uid = "5", pid = "12"` |
| `*` | anything (catch-all) | (none) |

Unmatched paths are a runtime warning (the NavStack shows the last matched
route).

### Transitions

```byld
NavStack(path: navPath) #[transition: slide] { ... }   // iOS-style right-to-left
NavStack(path: navPath) #[transition: fade] { ... }     // cross-fade
NavStack(path: navPath) #[transition: none] { ... }     // instant swap
```

Built-in transitions:

| Transition | Description |
|---|---|
| `slide` | incoming slides from right, outgoing slides to left (push); reversed on pop |
| `slide_up` | incoming slides from bottom (for modal-style pushes) |
| `fade` | cross-fade between outgoing and incoming |
| `none` | instant swap |

Transitions use RFC-0010's animation system — the slide is a `translate` with
`anim.spring()`, the fade is `opacity` with `anim.linear(200ms)`. Custom
transitions are a future possibility.

### Tab-based navigation

```byld
View App() {
    var activeTab = "home"

    Column #[grow: 1] {
        NavHost(active: activeTab) {
            tab "home" { HomePage() }
            tab "search" { SearchPage() }
            tab "profile" { ProfilePage() }
        }

        NavigationBar {
            NavItem(label: "Home", glyph: "H",
                    active: activeTab == "home",
                    tap => activeTab = "home")
            NavItem(label: "Search", glyph: "S",
                    active: activeTab == "search",
                    tap => activeTab = "search")
            NavItem(label: "Profile", glyph: "P",
                    active: activeTab == "profile",
                    tap => activeTab = "profile")
        }
    }
}
```

`NavHost` shows one child at a time based on `active`. Unlike `NavStack`, there
is no stack history — switching tabs is a flat swap with a cross-fade transition.
Each tab's state is **preserved** (not unmounted) when switching away.

### Deep linking

```byld
// In the app's manifest or a dedicated route-table file
NavStack(path: navPath, deep_link: true) {
    route "/" { ... }
    route "/item/:id" { ... }
}
```

`deep_link: true` tells the engine to accept OS-level URL intents (Android
intents, iOS universal links, desktop URI schemes). When the OS delivers a URL,
the engine sets `navPath` to the URL path, which triggers normal route matching
and push animation.

### Swipe-back gesture (Cupertino)

```byld
NavStack(path: navPath) #[transition: slide, swipe_back: true] { ... }
```

`swipe_back: true` enables an edge-swipe gesture (from left) that interactively
drags the current screen to the right, revealing the previous screen underneath.
Releasing past 50% completes the pop; otherwise it snaps back. The gesture drives
the transition's `translate` in real-time (gesture-driven animation, built on
RFC-0010).

---

## Reference-level explanation

### 1. `NavStack` intrinsic

- **Content:** none. **Children:** `route` blocks only.
- **Props:** `path: Str` (reflected — current route path), `transition: Transition`
  (default `slide`), `swipe_back: Bool` (default `false`), `deep_link: Bool`
  (default `false`).
- **Events:** `route_change(e: ChangeEvent<Str>)` — fires after navigation settles.
- **Pipeline:** manages a stack of View subtrees, two of which are simultaneously
  alive during a transition (the outgoing and incoming views).

### 2. Route resolution

Routes are compiled into a trie at mount time:

```rust
struct RouteNode {
    segment: RouteSegment,    // Literal("detail") | Param("id") | Wildcard
    view_index: Option<usize>, // index into the route table
    children: Vec<RouteNode>,
}
```

Matching is O(depth of path) — path segments are walked against the trie.
Dynamic segments are extracted into a `Params` map.

### 3. Stack management

The `NavStack` maintains:

```rust
struct NavStackState {
    stack: Vec<RouteEntry>,     // the history stack
    current: usize,             // index of the visible route
    transitioning: bool,        // during a transition, two routes are alive
}

struct RouteEntry {
    path: String,
    params: HashMap<String, String>,
    view_tree: Option<RenderTree>,  // preserved state
    scroll_offsets: Vec<Vec2>,      // preserved scroll positions
}
```

On push: a new `RouteEntry` is appended, the incoming view is instantiated, and
the transition animation starts. On pop: the top entry's transition reverses,
and the entry is removed when the animation settles.

### 4. State preservation

Each route's View subtree is **preserved in memory** when covered by a push
(not unmounted). This means:

- `var` values are retained.
- Scroll positions are retained.
- Timers and controllers continue running.

When popped to, the preserved tree is re-displayed without re-mounting. Only the
`back` target is preserved (intermediate routes in a multi-pop are discarded).

Tab-based `NavHost` preserves all tabs permanently (none are unmounted when
switching away). This matches platform behavior (iOS tab switches don't destroy
tab state).

### 5. Transition rendering

During a transition, **two View trees are simultaneously rendered**:

- The outgoing tree with a `translate` and/or `opacity` animation moving it off.
- The incoming tree with a `translate` and/or `opacity` animation moving it on.

Both trees are composited in the same render pass (the incoming tree draws
above the outgoing one, consistent with the painter's algorithm). When the
transition animation settles (RFC-0010 active-set epsilon), the outgoing tree
is either preserved (push) or discarded (pop) and removed from rendering.

### 6. Platform back button

On Android, the hardware/gesture back button sets `navPath` to the previous
stack entry's path. On iOS, the swipe-back gesture (if enabled) does the same.
On desktop, there is no system back button — the app provides its own back
navigation through UI elements.

The back action is:

```rust
fn handle_back(stack: &mut NavStackState) {
    if stack.current > 0 {
        stack.current -= 1;
        // trigger pop transition
    }
    // At root: no-op (or fire an app_exit_request event)
}
```

### 7. `NavHost` (tab container)

Simpler than `NavStack` — no stack, no transitions by default:

- **Children:** `tab "name" { ... }` blocks.
- **Props:** `active: Str` (reflected — current tab name).
- **Behavior:** shows only the `active` tab's View tree. All tabs are
  instantiated on mount and preserved. Switching fades/slides between tabs.

---

## Drawbacks

- **Two new intrinsics** (`NavStack`, `NavHost`) and a `route`/`tab` sub-syntax.
  This is significant grammar expansion.
- **State preservation memory cost.** Keeping off-screen View trees alive means
  their `Signal`s, `RenderNode`s, and `var` values consume memory. For deep
  stacks (5+ levels), this adds up. Mitigation: a `preserve: false` option on
  routes that don't need state preservation.
- **Simultaneous rendering during transitions** doubles the render work for
  ~300ms. On low-end devices, this may cause frame drops. Mitigation: `none`
  transition for performance-critical paths.
- **Deep linking requires manifest integration.** The engine must register URL
  schemes with the OS — platform-specific code per target (Android manifest,
  iOS Info.plist, desktop URI handler).

---

## Rationale and alternatives

**Why `var` path strings, not a navigation object?** RFC-0003 forbids widget
references. A navigation "controller" would be a mutable reference — exactly
what Byard eliminates. A `var navPath` is a reactive string: setting it triggers
navigation. This is the SwiftUI `NavigationPath` model adapted to Byard's
reference-free world.

**Why route patterns, not conditional `when` blocks?** `when` blocks are
combinatorially explosive for multi-screen apps and don't provide transitions,
state preservation, or deep linking. Route patterns are the established solution
in every web/mobile framework.

**Why preserve off-screen state by default?** Users expect to scroll down a
list, push to a detail, pop back, and find the list where they left it. Not
preserving state is a UX regression that makes every push/pop feel like a page
reload.

**Why `NavHost` in addition to `NavStack`?** Tabs and stacks are fundamentally
different navigation models that coexist in the same app. Conflating them (a
tab switch as a stack push) breaks the mental model and the back-button contract.

---

## Prior art

- **SwiftUI `NavigationStack` + `NavigationPath`:** path-based push/pop with
  value-driven navigation. Direct inspiration.
- **Flutter `Navigator 2.0` / `GoRouter`:** declarative route matching. Byard's
  `route` syntax is simpler than GoRouter's builder pattern.
- **React Navigation (React Native):** stack/tab navigators with screen
  preservation. Validates the stack + tab dual model.
- **Jetpack Compose `NavHost` + `NavController`:** composable-based routing.
  Byard avoids the controller object.
- **UIKit `UINavigationController` + `UITabBarController`:** the platform
  originals. Byard's model is the declarative equivalent.

---

## Unresolved questions

- **Before merge:**
  - [ ] **Route parameter types.** Should `:id` always be `Str`, or can routes
    declare typed params (`:id(Int)`)? Recommendation: `Str` only in v1; the
    View's `Fn` params can parse as needed.
  - [ ] **Nested NavStacks.** A tab containing its own NavStack (common pattern:
    each tab has independent navigation). Recommendation: supported — each
    NavStack manages its own stack independently.
  - [ ] **Shared-element (hero) transitions.** Should this RFC include a
    `hero_tag` prop for cross-route element matching? Recommendation: deferred
    to a follow-up — the rendering is complex (element must animate between two
    layout positions during the transition).

- **During implementation:**
  - [ ] **Memory pressure.** How many preserved routes before the engine warns
    or evicts old state? Recommendation: warn at 10 stack depth; configurable.
  - [ ] **Route guards.** Can a route be conditionally accessible (auth check)?
    Recommendation: the route's `when` block handles this — no special guard API.

---

## Future possibilities

- **Shared-element transitions** (hero animations) between routes.
- **Route middleware** — interceptors that run before a route mounts (auth
  checks, analytics, data prefetching).
- **Animated route transitions API** — developer-defined transition animations
  (custom spring parameters, directional slides, circular reveals).
- **Browser history integration** — on web targets, sync `navPath` with
  `window.history` for browser back/forward.
- **Lazy route loading** — instantiate a route's View tree only when first
  navigated to, not at mount time.
