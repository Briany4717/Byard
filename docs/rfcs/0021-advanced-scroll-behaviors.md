# RFC-0021: Advanced Scroll Behaviors — snap, pull-to-refresh, collapsing headers, pagination

- **Status:** Draft
- **Author(s):** Briany4717
- **Created:** 2026-07-10
- **Last updated:** 2026-07-10
- **Depends on:** RFC-0001 (§4.1 Taffy layout, §4.2 spatial grid), RFC-0003 (scroll event, pointer drag), RFC-0005 (`ScrollView` intrinsic, `offset` reflected prop), RFC-0010 (spring animations for snap/overscroll), RFC-0011 (transforms).
- **Extends:** RFC-0005's `ScrollView` with new behavioral props.
- **Enables:** Material carousel, Cupertino page views, pull-to-refresh, large-title collapsing navigation bars, horizontal paging, infinite scroll patterns.

---

## Summary

Extend `ScrollView` (RFC-0005) with four behavioral capabilities delivered as
new props on the existing intrinsic — no new intrinsics needed:

1. **Snap scrolling** — content snaps to discrete positions (pages, items).
2. **Pull-to-refresh** — overscroll at the start triggers a refresh callback.
3. **Collapsing headers** — a header region that shrinks/expands as the user
   scrolls, with parallax and opacity effects.
4. **Pagination** — discrete page indicators and page-change events.

All behaviors are declared as props on `ScrollView`, animated through RFC-0010's
springs, and managed entirely by the engine — no imperative scroll controllers,
no widget references.

---

## Motivation

The byard-material gap analysis flagged **snap-scrolling** as a blocker for
carousels. But the real gap is broader: every polished mobile app relies on
scroll-linked behaviors that `ScrollView` cannot express today:

- **Material carousel** — horizontal snap to item boundaries.
- **Cupertino page view** — full-page horizontal snap with page dots.
- **Pull-to-refresh** — nearly universal on mobile; both Material and Cupertino
  have distinct visual treatments.
- **Large title navigation** (Cupertino) — the title collapses from large to
  inline as the user scrolls down. Material's `MediumTopAppBar` / `LargeTopAppBar`
  do the same.
- **Infinite scroll / load-more** — detecting scroll-to-bottom to trigger data
  fetching.

These are not niche features; they're table-stakes for any mobile UI framework.

---

## Guide-level explanation

### Snap scrolling

```byld
ScrollView #[axis: horizontal, snap: item, snap_align: center] {
    for card in cards {
        ElevatedCard #[width: 280, m: (0, 8)] {
            Text(card.title)
        }
    }
}
```

`snap: item` snaps to each direct child's boundary after a fling.
`snap: page` snaps to viewport-sized pages. `snap: none` (default) is free
scrolling. `snap_align` controls where the snapped item aligns within the
viewport: `start`, `center`, or `end`.

The snap physics use RFC-0010's spring with a fast settle — the content
decelerates from the fling velocity and springs to the nearest snap point.

### Pull-to-refresh

```byld
ScrollView #[pull_refresh: true,
             refreshing: isLoading,
             refresh => fetchData()] {
    // content
}
```

When the user overscrolls past the start (top for vertical), a pull indicator
appears. Releasing past the threshold fires the `refresh` event. The
`refreshing: Bool` reflected prop keeps the indicator spinning while the
controller loads data; setting it to `false` dismisses the indicator with a
spring animation.

### Collapsing header

```byld
ScrollView #[collapse_header: true] {
    // The first child is the collapsible header
    Box #[height: 200, collapse_min: 56, collapse_parallax: 0.5] {
        Image("hero.jpg") #[fit: cover, opacity: scroll_fraction]
        Text("My App") #[size: lerp(32, 18, scroll_fraction)]
    }

    // Rest of the content scrolls normally
    Column #[gap: 8] {
        for item in items {
            ListItem(headline: item.title)
        }
    }
}
```

`collapse_header: true` tells the engine that the first child is a collapsible
region. As the user scrolls down, the header shrinks from its natural height to
`collapse_min`. The `scroll_fraction` (0.0 at expanded, 1.0 at collapsed) is
exposed as a reactive value that header children can bind to for parallax,
opacity, and text-size interpolation.

### Pagination

```byld
ScrollView #[axis: horizontal, snap: page,
             page: currentPage,
             page_count: pages.len()] {
    for p in pages {
        PageContent(data: p) #[width: viewport]
    }
}
```

`page: Int` is a reflected prop — it reflects the current page index and can be
set programmatically to scroll to a page. `page_count` enables the engine to
provide page-indicator data (the actual dot UI is a user View reading `page` and
`page_count`).

---

## Reference-level explanation

### 1. New props on `ScrollView`

| Prop | Type | Default | Meaning |
|---|---|---|---|
| `snap` | `none\|item\|page` | `none` | snap behavior |
| `snap_align` | `start\|center\|end` | `start` | snap alignment within viewport |
| `pull_refresh` | `Bool` | `false` | enable pull-to-refresh |
| `refreshing` | `Bool` (reflected) | `false` | whether refresh is in progress |
| `collapse_header` | `Bool` | `false` | first child is collapsible |
| `page` | `Int` (reflected) | `0` | current page index |
| `page_count` | `Int` | `0` | total pages (for external indicators) |
| `on_end_reached` | `Fn()` | no-op | fires when scroll reaches the end (infinite scroll trigger) |
| `end_threshold` | `Float` | `0.8` | fraction of content at which `on_end_reached` fires |

New events:

| Event | Payload | When |
|---|---|---|
| `refresh` | none | pull-to-refresh threshold exceeded and released |
| `page_change(e: ChangeEvent<Int>)` | new page index | snap/page scroll settles on a new page |
| `scroll_end` | none | scroll animation settles (useful for snap completion) |

### 2. Snap physics

After a fling gesture ends, the engine:

1. Computes the projected resting position from current velocity + deceleration.
2. Finds the nearest snap point (child boundary for `item`, viewport multiple for
   `page`).
3. Redirects the scroll animation to a spring targeting that snap point.

The spring parameters are engine-managed (fast, critically damped) but can be
overridden via `snap_spring: spring(stiffness: N, damping: N)` if the developer
wants a bouncier or stiffer feel.

Implementation: the `ScrollView`'s `offset` (RFC-0005 reflected prop) is driven
by a `Motion` (RFC-0010) during the snap animation. When the spring settles
(active-set epsilon), the `offset` is clamped to the exact snap position.

### 3. Pull-to-refresh

The overscroll region (content pulled past `offset = 0`) is an elastic spring:
distance follows a diminishing-returns curve (e.g., `pull_distance^0.6`) to
create resistance. A circular progress indicator (RFC-0020 `Canvas` arc, themed
by the package) is mounted in the overscroll space.

When the user releases past the threshold:
1. Fire `refresh` event.
2. Set `refreshing` to `true` (the indicator spins).
3. The content settles at `offset = -indicator_height` (the indicator stays
   visible).
4. When the controller sets `refreshing = false`, the content springs back to
   `offset = 0`.

### 4. Collapsing header

The first child of a `collapse_header: true` `ScrollView` is measured at its
natural height (`h_max`) and at `collapse_min` (`h_min`). As `offset` increases
from 0:

- The header height interpolates from `h_max` to `h_min` over the first
  `h_max - h_min` pixels of scroll.
- `scroll_fraction` is `clamp(offset / (h_max - h_min), 0.0, 1.0)`.
- The header's children receive `scroll_fraction` as an injectable reactive value.
- `collapse_parallax: Float` (0.0–1.0) controls how fast the header content
  scrolls relative to the collapse (0.5 = half speed = parallax effect).

The layout is re-computed each frame during header collapse, but only the header's
Taffy node changes — the content below shifts by the height delta. This is O(1)
in the number of content items (Taffy incrementally relayouts changed subtrees).

### 5. `on_end_reached` (infinite scroll)

Each frame, if `offset / content_height > end_threshold`, the engine fires
`on_end_reached` once (debounced until offset decreases). The controller
appends items to the list; the `for` loop (RFC-0004) reactively adds nodes.

---

## Drawbacks

- **`ScrollView` prop surface grows significantly.** From 3 props (axis, offset,
  decorations) to 12+. Mitigation: all new props have sensible defaults; a plain
  `ScrollView` is unchanged.
- **Collapsing header requires layout-during-scroll.** This violates the "layout
  once" ideal, but the cost is bounded (one Taffy node changes per frame during
  collapse, not the entire tree).
- **Pull-to-refresh indicator.** The engine provides a default indicator, but
  design systems want custom indicators (Material's circular, Cupertino's spinner).
  This requires the indicator to be themeable or replaceable — a `refresh_indicator`
  slot or a composed approach.

---

## Rationale and alternatives

**Why props on `ScrollView`, not new intrinsics?** `CarouselView`,
`RefreshableScrollView`, `PagingView` would each be a near-duplicate of
`ScrollView` with one behavior toggled. Props compose better: a carousel with
pull-to-refresh is `ScrollView #[snap: item, pull_refresh: true]`, not
a `RefreshableCarouselView` that doesn't exist.

**Why engine-managed snap physics, not user-defined?** Snap physics are subtle
(velocity projection, spring targeting, edge clamping). Letting the developer
implement them from scroll events would produce janky, inconsistent behavior
across apps. The engine guarantees 60fps snap animations with correct physics.

**Why `scroll_fraction` for collapsing headers instead of a CSS `position: sticky`
model?** Sticky positioning is a layout concept that requires the element to
participate in two layout contexts simultaneously. `scroll_fraction` is a reactive
value that header children read — simpler, more flexible (any property can
interpolate), and avoids the layout complexity.

---

## Prior art

- **SwiftUI `ScrollView` + `.scrollTargetBehavior(.paging)`:** snap scrolling.
- **Flutter `PageView`, `RefreshIndicator`, `SliverAppBar`:** each a separate
  widget. Byard consolidates into `ScrollView` props.
- **Jetpack Compose `SnapFlingBehavior`, `pullRefresh`:** modifier-based.
- **CSS `scroll-snap-type` / `scroll-snap-align`:** declarative snap. Direct
  inspiration for `snap` + `snap_align` prop names.
- **UIKit `UIRefreshControl`, `UICollectionView` with paging:** imperative but
  battle-tested interaction patterns.

---

## Resolved questions

- **Before merge:**
  - [x] **Custom refresh indicator.** **Slot-based.** The `pull_refresh` prop
    accepts an optional content block:
    ```byld
    ScrollView #[pull_refresh: { MySpinner(progress: refresh_progress) }]
    ```
    If no content block is provided, the engine renders a default circular
    indicator (a simple arc, styled by the active theme). The slot receives
    an implicit `refresh_progress: Float` binding (0.0 = idle, 1.0 = threshold
    reached, >1.0 = overscroll). This gives design systems full control
    (Material uses a circular indicator, Cupertino uses a different spinner)
    while providing a sensible default.
  - [x] **Nested scroll coordination.** **Axis-based priority.** The inner
    ScrollView captures gestures along its scroll axis; the outer captures
    the orthogonal axis. A vertical outer + horizontal inner: horizontal
    swipes go to the inner, vertical swipes go to the outer. When axes match
    (vertical inside vertical), the inner ScrollView consumes the gesture
    until it hits its scroll extent, then the outer takes over (scroll
    chaining). This matches platform behavior (iOS nested UIScrollView,
    Android NestedScrollView). Formalized as: the innermost ScrollView whose
    axis matches the gesture direction gets priority; overflow chains upward.
  - [x] **`scroll_fraction` injection.** **Implicit binding.** Within a
    `collapse_header { ... }` child block, `scroll_fraction` is available
    as an implicit reactive binding (0.0 = fully expanded, 1.0 = fully
    collapsed). No explicit `var` declaration needed. This is consistent
    with the implicit `refresh_progress` in pull-to-refresh and with
    `route.params` in RFC-0026. The binding is scoped — it only exists
    inside `collapse_header`, not in the broader ScrollView scope.

- **During implementation:**
  - [x] **Fling velocity threshold.** **150 dp/s.** Below this velocity, snap
    springs to the nearest item. Above, the engine projects the fling and
    targets the item the velocity would reach, clamped to ±1 item (no
    multi-item skip on moderate fling). This threshold is tunable per
    platform via an engine constant. Derived from iOS's
    `UIScrollView.decelerationRate` behavior at the snap boundary.
  - [x] **Overscroll elasticity curve.** Platform-specific exponent:
    `offset = raw_offset * (1 / (1 + raw_offset / max_overscroll)^exp)`.
    iOS: `exp = 0.55` (bouncier, matches UIScrollView). Android: `exp = 0.75`
    (stiffer, matches EdgeEffect). Desktop: `exp = 1.0` (no rubber-banding,
    hard stop). These are engine constants, not developer-facing props.

---

## Future possibilities

- **Lazy/virtualized scroll** — only instantiate children within the viewport
  plus a buffer zone. Essential for long lists (thousands of items).
- **Scroll-linked animations** — arbitrary property interpolation driven by
  scroll position (parallax backgrounds, reveal animations).
- **Nested scroll coordination protocol** — formalize which scroll view captures
  a gesture when scrollviews are nested.
- **Horizontal paging dots** as a built-in `PageIndicator` intrinsic or as a
  composed View reading `page` + `page_count`.
