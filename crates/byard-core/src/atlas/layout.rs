//! Taffy-backed layout tree.
//!
//! See the module-level documentation in [`super`] for the design intent
//! and lifecycle contract.

use std::fmt;

use crate::frame::{Rect, RenderFrame, Viewport};
use taffy::prelude::FromLength;
use taffy::{AvailableSpace, Dimension, NodeId, Size, Style, TaffyError, TaffyTree};

/// Opaque identifier for a node owned by a [`LayoutAtlas`].
///
/// Wraps [`taffy::NodeId`] so the Atlas does not leak Taffy types into
/// the rest of the engine.
///
/// # Caveat: cross-atlas safety
///
/// `AtlasNodeId` is currently **not** scoped to the atlas that created it.
/// Passing an ID from one [`LayoutAtlas`] to another may return incorrect
/// geometry or hit a backend error rather than a clean rejection.
/// Callers must only use IDs with their originating atlas. A future
/// sub-issue will add a generation tag to enforce this at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AtlasNodeId(NodeId);

/// Explicit size for a leaf node.
#[derive(Debug, Clone, Copy)]
pub struct LeafSize {
    /// Width in logical pixels.
    pub width: f32,
    /// Height in logical pixels.
    pub height: f32,
}

impl LeafSize {
    /// Constructs a new leaf size.
    #[must_use]
    pub const fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }
}

/// Style for a container node.
///
/// # Layout defaults (Taffy 0.10)
///
/// When constructed via `Default::default()` or with only `width`/`height`
/// set, Taffy applies these defaults:
///
/// - `display: Flex`
/// - `flex_direction: Row` — children flow left-to-right
/// - `align_items: Stretch` — children stretch on the cross axis if their
///   size is not explicitly set, otherwise they keep their declared size
/// - `justify_content: FlexStart` — children packed against the start of
///   the main axis with no gap
///
/// These match CSS flexbox defaults. Phase 1 does not yet expose
/// `flex_direction`, `align_items`, etc. through `ContainerStyle` —
/// callers needing those must wait for the builder API sub-issue.
///
/// Marked `#[non_exhaustive]` so additional style fields can be added
/// without breaking downstream code.
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct ContainerStyle {
    /// Explicit width in logical pixels. `None` means "grow to fit children".
    pub width: Option<f32>,
    /// Explicit height in logical pixels. `None` means "grow to fit children".
    pub height: Option<f32>,
}

/// Errors produced by the [`LayoutAtlas`].
#[non_exhaustive]
#[derive(Debug)]
pub enum AtlasError {
    /// The layout backend returned an error during tree construction
    /// or layout computation.
    Backend(String),
}

impl AtlasError {
    pub(crate) fn from_taffy(e: &TaffyError) -> Self {
        Self::Backend(e.to_string())
    }
}

impl fmt::Display for AtlasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(message) => write!(f, "layout backend error: {message}"),
        }
    }
}

impl std::error::Error for AtlasError {}

impl From<TaffyError> for AtlasError {
    fn from(e: TaffyError) -> Self {
        Self::Backend(e.to_string())
    }
}

/// Two-phase lifecycle state of a [`LayoutAtlas`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AtlasState {
    /// Nodes can be added and modified. Querying resolved geometry panics.
    Building,
    /// Resolved geometry is accessible. Adding or modifying nodes panics.
    Computed,
}

/// Layout tree backed by Taffy.
///
/// See the module-level docs for the lifecycle contract and the RFC for
/// the architectural rationale.
pub struct LayoutAtlas {
    tree: TaffyTree<()>,
    root: Option<AtlasNodeId>,
    state: AtlasState,
    /// Reusable buffer for converting `AtlasNodeId` → `NodeId` during
    /// container creation. Kept on the struct so containers with
    /// children do not allocate after the first frame.
    children_scratch: Vec<NodeId>,
}

impl LayoutAtlas {
    /// Creates a new, empty atlas in the `Building` state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tree: TaffyTree::new(),
            root: None,
            state: AtlasState::Building,
            children_scratch: Vec::new(),
        }
    }

    /// Clears the tree but retains internal capacity.
    ///
    /// After the first frame, subsequent layouts pay zero allocation cost
    /// as long as node counts stay within the high-water mark. Transitions
    /// back to the `Building` state regardless of the current state.
    pub fn clear(&mut self) {
        self.tree.clear();
        self.root = None;
        self.state = AtlasState::Building;
    }

    /// Adds a leaf node with an explicit size.
    ///
    /// # Panics
    ///
    /// Panics if the atlas is in the `Computed` state. Call [`Self::clear`]
    /// before adding new nodes.
    ///
    /// # Errors
    ///
    /// Returns [`AtlasError::Backend`] if the underlying engine refuses the
    /// node (extremely rare; indicates resource exhaustion).
    pub fn add_leaf(&mut self, size: LeafSize) -> Result<AtlasNodeId, AtlasError> {
        self.assert_building("add_leaf");

        let style = Style {
            size: Size {
                width: Dimension::from_length(size.width),
                height: Dimension::from_length(size.height),
            },
            ..Default::default()
        };

        let node = self
            .tree
            .new_leaf(style)
            .map_err(|e| AtlasError::from_taffy(&e))?;
        Ok(AtlasNodeId(node))
    }

    /// Adds a container node that wraps the given children.
    ///
    /// # Panics
    ///
    /// Panics if the atlas is in the `Computed` state.
    ///
    /// # Errors
    ///
    /// Returns [`AtlasError::Backend`] if the underlying engine refuses the
    /// node, or if any child has already been attached to another parent.
    pub fn add_container(
        &mut self,
        style: ContainerStyle,
        children: &[AtlasNodeId],
    ) -> Result<AtlasNodeId, AtlasError> {
        self.assert_building("add_container");

        let taffy_style = Style {
            size: Size {
                width: style
                    .width
                    .map_or(Dimension::auto(), Dimension::from_length),
                height: style
                    .height
                    .map_or(Dimension::auto(), Dimension::from_length),
            },
            ..Default::default()
        };
        self.children_scratch.clear();
        self.children_scratch.extend(children.iter().map(|c| c.0));
        let node = self
            .tree
            .new_with_children(taffy_style, &self.children_scratch)
            .map_err(|e| AtlasError::from_taffy(&e))?;
        Ok(AtlasNodeId(node))
    }

    /// Sets the root node for layout computation.
    ///
    /// # Panics
    ///
    /// Panics if the atlas is in the `Computed` state.
    pub fn set_root(&mut self, root: AtlasNodeId) {
        self.assert_building("set_root");
        self.root = Some(root);
    }

    /// Computes layout against the given viewport size.
    ///
    /// Transitions the atlas from `Building` to `Computed`. After this
    /// call, [`Self::resolved_rect`] returns geometry; modifying nodes
    /// panics until [`Self::clear`] is called.
    ///
    /// # Panics
    ///
    /// Panics if the atlas is in the `Computed` state, or if no root has
    /// been set via [`Self::set_root`].
    ///
    /// # Errors
    ///
    /// Returns [`AtlasError::Backend`] if layout computation fails.
    pub fn compute(&mut self, viewport: Viewport) -> Result<(), AtlasError> {
        self.assert_building("compute");

        let root = self
            .root
            .expect("LayoutAtlas::compute called without a root node — call set_root first");

        let available = Size {
            width: AvailableSpace::Definite(viewport.width),
            height: AvailableSpace::Definite(viewport.height),
        };

        self.tree
            .compute_layout(root.0, available)
            .map_err(|e| AtlasError::from_taffy(&e))?;
        self.state = AtlasState::Computed;
        Ok(())
    }

    /// Returns the resolved rectangle for `node`.
    ///
    /// # Caveat: orphan nodes
    ///
    /// If `node` was added to the tree but never attached (directly or
    /// transitively) to the root configured via [`Self::set_root`], this
    /// returns the default `Rect` of all zeros rather than `None`. This
    /// matches Taffy's underlying behaviour. Callers should only query
    /// rects for nodes known to be reachable from the root.
    ///
    /// # Panics
    ///
    /// Panics if the atlas is in the `Building` state. Call
    /// [`Self::compute`] first.
    #[must_use]
    pub fn resolved_rect(&self, node: AtlasNodeId) -> Option<Rect> {
        assert_eq!(
            self.state,
            AtlasState::Computed,
            "LayoutAtlas::resolved_rect called before compute — geometry is not available yet"
        );

        let layout = self.tree.layout(node.0).ok()?;
        Some(Rect {
            x: layout.location.x,
            y: layout.location.y,
            width: layout.size.width,
            height: layout.size.height,
        })
    }

    /// Writes the resolved geometry of every node into `frame`.
    ///
    /// Walks the tree from the root in pre-order and appends each node's
    /// resolved [`Rect`] to the frame. This is how the Atlas hands geometry
    /// to the Encoder without either subsystem importing from the other —
    /// the frame is the shared boundary defined in RFC-0001 §9.
    ///
    /// The frame is **not** cleared before pushing; callers that want a
    /// fresh frame must call [`RenderFrame::clear`] first. This lets the
    /// orchestrator batch contributions from multiple subsystems into the
    /// same frame.
    ///
    /// # Panics
    ///
    /// Panics if the atlas is in the `Building` state. Call [`Self::compute`]
    /// first.
    #[track_caller]
    pub fn populate_frame(&self, frame: &mut RenderFrame) {
        assert_eq!(
            self.state,
            AtlasState::Computed,
            "LayoutAtlas::populate_frame called before compute — geometry is not available yet"
        );

        let root = self.root.expect(
            "LayoutAtlas::populate_frame reached Computed state without a root node — \
         this indicates an internal state-machine inconsistency",
        );

        self.walk_and_push(root, frame);
    }

    /// Recursively walks the tree from `node` in pre-order, pushing each
    /// resolved rectangle into `frame`.
    fn walk_and_push(&self, root: AtlasNodeId, frame: &mut RenderFrame) {
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if let Some(rect) = self.resolved_rect_internal(node) {
                frame.push_rect(rect);
            }
            if let Ok(children) = self.tree.children(node.0) {
                // Push in reverse so the leftmost child is popped first,
                // preserving pre-order traversal semantics.
                for child in children.iter().rev() {
                    stack.push(AtlasNodeId(*child));
                }
            }
        }
    }

    /// Same as `resolved_rect` but without the state assertion.
    ///
    /// Used internally during traversal where the state has already been
    /// validated by the entry point.
    fn resolved_rect_internal(&self, node: AtlasNodeId) -> Option<Rect> {
        let layout = self.tree.layout(node.0).ok()?;
        Some(Rect {
            x: layout.location.x,
            y: layout.location.y,
            width: layout.size.width,
            height: layout.size.height,
        })
    }

    /// Returns the current root node, if any.
    #[must_use]
    pub fn root(&self) -> Option<AtlasNodeId> {
        self.root
    }

    /// Returns the number of nodes currently in the tree.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.tree.total_node_count()
    }

    #[track_caller]
    fn assert_building(&self, method: &str) {
        assert_eq!(
            self.state,
            AtlasState::Building,
            "LayoutAtlas::{method} called while in Computed state — call clear() first"
        );
    }
}

impl Default for LayoutAtlas {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::ByardError;
    use super::*;

    /// Acceptance criterion: Atlas computes a valid layout for a single
    /// rectangle with a child.
    #[test]
    fn computes_layout_for_container_with_one_child() {
        let mut atlas = LayoutAtlas::new();

        let child = atlas
            .add_leaf(LeafSize::new(100.0, 50.0))
            .expect("add_leaf");
        let root = atlas
            .add_container(
                ContainerStyle {
                    width: Some(200.0),
                    height: Some(100.0),
                },
                &[child],
            )
            .expect("add_container");
        atlas.set_root(root);

        atlas.compute(Viewport::new(800.0, 600.0)).expect("compute");

        let root_rect = atlas.resolved_rect(root).expect("root rect");
        assert_f32_eq(root_rect.width, 200.0);
        assert_f32_eq(root_rect.height, 100.0);

        let child_rect = atlas.resolved_rect(child).expect("child rect");
        assert_f32_eq(child_rect.width, 100.0);
        assert_f32_eq(child_rect.height, 50.0);
    }

    #[test]
    fn empty_atlas_has_no_nodes() {
        let atlas = LayoutAtlas::new();
        assert_eq!(atlas.node_count(), 0);
        assert!(atlas.root().is_none());
    }

    #[test]
    fn clear_resets_to_building_and_allows_rebuild() {
        let mut atlas = LayoutAtlas::new();
        let child = atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();
        let root = atlas
            .add_container(ContainerStyle::default(), &[child])
            .unwrap();
        atlas.set_root(root);
        atlas.compute(Viewport::new(100.0, 100.0)).unwrap();

        atlas.clear();

        assert_eq!(atlas.node_count(), 0);
        assert!(atlas.root().is_none());
        // After clear, we should be able to build again without panic.
        let _ = atlas.add_leaf(LeafSize::new(5.0, 5.0)).unwrap();
    }

    #[test]
    #[should_panic(expected = "called while in Computed state")]
    fn add_leaf_after_compute_panics() {
        let mut atlas = LayoutAtlas::new();
        let leaf = atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();
        atlas.set_root(leaf);
        atlas.compute(Viewport::new(100.0, 100.0)).unwrap();

        // This must panic.
        let _ = atlas.add_leaf(LeafSize::new(20.0, 20.0));
    }

    #[test]
    #[should_panic(expected = "called while in Computed state")]
    fn add_container_after_compute_panics() {
        let mut atlas = LayoutAtlas::new();
        let leaf = atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();
        atlas.set_root(leaf);
        atlas.compute(Viewport::new(100.0, 100.0)).unwrap();

        let _ = atlas.add_container(ContainerStyle::default(), &[leaf]);
    }

    #[test]
    #[should_panic(expected = "called before compute")]
    fn resolved_rect_before_compute_panics() {
        let mut atlas = LayoutAtlas::new();
        let leaf = atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();
        atlas.set_root(leaf);

        let _ = atlas.resolved_rect(leaf);
    }

    #[test]
    #[should_panic(expected = "called without a root node")]
    fn compute_without_root_panics() {
        let mut atlas = LayoutAtlas::new();
        let _ = atlas.compute(Viewport::new(100.0, 100.0));
    }

    #[test]
    fn auto_sized_container_grows_to_fit_child() {
        let mut atlas = LayoutAtlas::new();
        let child = atlas.add_leaf(LeafSize::new(150.0, 75.0)).unwrap();
        let root = atlas
            .add_container(ContainerStyle::default(), &[child])
            .unwrap();
        atlas.set_root(root);
        atlas.compute(Viewport::new(800.0, 600.0)).unwrap();

        let root_rect = atlas.resolved_rect(root).unwrap();
        assert_f32_eq(root_rect.width, 150.0);
        assert_f32_eq(root_rect.height, 75.0);
    }

    #[test]
    fn atlas_node_id_is_copy() {
        const fn assert_copy<T: Copy>() {}
        assert_copy::<AtlasNodeId>();
    }

    /// Asserts two `f32` values are equal within the layout precision tolerance.
    ///
    /// Taffy produces deterministic exact values for simple layouts, but using
    /// `assert_eq!` on `f32` triggers `clippy::float_cmp`. This helper makes
    /// the tolerance explicit.
    #[track_caller]
    fn assert_f32_eq(actual: f32, expected: f32) {
        let diff = (actual - expected).abs();
        assert!(
            diff < 0.001,
            "expected {expected}, got {actual} (diff = {diff})",
        );
    }

    /// Acceptance criterion: resolved geometry is written into `RenderFrame`
    /// without crossing subsystem boundaries directly.
    ///
    /// The Atlas only touches `RenderFrame` via its public API. There is no
    /// import of `encoder` or any other subsystem.
    #[test]
    fn populate_frame_writes_resolved_geometry() {
        use crate::frame::RenderFrame;

        let mut atlas = LayoutAtlas::new();
        let child = atlas.add_leaf(LeafSize::new(100.0, 50.0)).unwrap();
        let root = atlas
            .add_container(
                ContainerStyle {
                    width: Some(200.0),
                    height: Some(100.0),
                },
                &[child],
            )
            .unwrap();
        atlas.set_root(root);
        atlas.compute(Viewport::new(800.0, 600.0)).unwrap();

        let mut frame = RenderFrame::new();
        atlas.populate_frame(&mut frame);

        assert_eq!(frame.rects().len(), 2, "root + child");

        // Pre-order: root first, then child.
        assert_f32_eq(frame.rects()[0].width, 200.0);
        assert_f32_eq(frame.rects()[0].height, 100.0);
        assert_f32_eq(frame.rects()[1].width, 100.0);
        assert_f32_eq(frame.rects()[1].height, 50.0);
    }

    #[test]
    fn populate_frame_appends_without_clearing() {
        use crate::frame::RenderFrame;

        let mut atlas = LayoutAtlas::new();
        let leaf = atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();
        atlas.set_root(leaf);
        atlas.compute(Viewport::new(100.0, 100.0)).unwrap();

        let mut frame = RenderFrame::new();
        atlas.populate_frame(&mut frame);
        atlas.populate_frame(&mut frame);

        assert_eq!(
            frame.rects().len(),
            2,
            "populate_frame appends; caller is responsible for clearing",
        );
    }

    #[test]
    #[should_panic(expected = "called before compute")]
    fn populate_frame_before_compute_panics() {
        use crate::frame::RenderFrame;

        let mut atlas = LayoutAtlas::new();
        let leaf = atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();
        atlas.set_root(leaf);

        let mut frame = RenderFrame::new();
        atlas.populate_frame(&mut frame);
    }

    #[test]
    #[should_panic(expected = "called while in Computed state")]
    fn set_root_after_compute_panics() {
        let mut atlas = LayoutAtlas::new();
        let leaf = atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();
        atlas.set_root(leaf);
        atlas.compute(Viewport::new(100.0, 100.0)).unwrap();

        atlas.set_root(leaf);
    }

    #[test]
    fn orphan_node_returns_zero_rect() {
        let mut atlas = LayoutAtlas::new();
        let root = atlas.add_leaf(LeafSize::new(50.0, 50.0)).unwrap();
        let orphan = atlas.add_leaf(LeafSize::new(999.0, 999.0)).unwrap();
        atlas.set_root(root);
        atlas.compute(Viewport::new(100.0, 100.0)).unwrap();

        let orphan_rect = atlas.resolved_rect(orphan).unwrap();
        assert_f32_eq(orphan_rect.width, 0.0);
        assert_f32_eq(orphan_rect.height, 0.0);
    }

    #[test]
    fn flex_row_layout_positions_children_at_known_offsets() {
        // Pixel-perfect layout contract for Phase 1.
        //
        // Taffy 0.10 with `Style::default()` uses Display::Flex and
        // FlexDirection::Row. A container of 200x200 with two 50x50 children
        // lays them out left-to-right at y=0:
        //
        //   child A → (x=0,  y=0, w=50, h=50)
        //   child B → (x=50, y=0, w=50, h=50)
        //
        // This validates the location field is correctly threaded from
        // taffy::Layout into our frame::Rect.
        let mut atlas = LayoutAtlas::new();

        let a = atlas.add_leaf(LeafSize::new(50.0, 50.0)).unwrap();
        let b = atlas.add_leaf(LeafSize::new(50.0, 50.0)).unwrap();
        let root = atlas
            .add_container(
                ContainerStyle {
                    width: Some(200.0),
                    height: Some(200.0),
                },
                &[a, b],
            )
            .unwrap();
        atlas.set_root(root);
        atlas.compute(Viewport::new(800.0, 600.0)).unwrap();

        let a_rect = atlas.resolved_rect(a).unwrap();
        let b_rect = atlas.resolved_rect(b).unwrap();

        // Child A: top-left corner of the container.
        assert_f32_eq(a_rect.x, 0.0);
        assert_f32_eq(a_rect.y, 0.0);
        assert_f32_eq(a_rect.width, 50.0);
        assert_f32_eq(a_rect.height, 50.0);

        // Child B: stacked immediately to the right of A on the main axis.
        assert_f32_eq(b_rect.x, 50.0);
        assert_f32_eq(b_rect.y, 0.0);
        assert_f32_eq(b_rect.width, 50.0);
        assert_f32_eq(b_rect.height, 50.0);
    }

    #[test]
    fn atlas_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AtlasError>();
        assert_send_sync::<ByardError>();
    }
}
