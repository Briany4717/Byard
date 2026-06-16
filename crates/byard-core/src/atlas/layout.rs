//! Taffy-backed layout tree.
//!
//! See the module-level documentation in [`super`] for the design intent
//! and lifecycle contract.

use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::frame::{Rect, RenderFrame, TargetId, TargetKind, Viewport};
use taffy::prelude::FromLength;
use taffy::{AvailableSpace, Dimension, NodeId, Size, Style, TaffyError, TaffyTree};

use super::spatial::SpatialGrid;

/// Opaque identifier for a node owned by a [`LayoutAtlas`].
///
/// Wraps [`taffy::NodeId`] so the Atlas does not leak Taffy types into
/// the rest of the engine.
///
/// # Cross-atlas safety
///
/// `AtlasNodeId` is scoped to the [`LayoutAtlas`] instance that created it.
/// Every atlas is assigned a unique `instance_id` at construction time
/// (see [`LayoutAtlas::next_instance_id`]), and that id travels with every
/// `AtlasNodeId` it produces. Passing an ID to a different atlas instance
/// is rejected with [`AtlasError::ForeignNode`] rather than returning
/// incorrect geometry or hitting an opaque backend error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AtlasNodeId {
    node_id: NodeId,
    atlas_id: u32,
}

// `AtlasNodeId` must stay cheap to pass by value — that's the entire point
// of scoping it with a plain `u32` rather than, say, an `Arc<str>` tag.
// `taffy::NodeId` is an 8-byte, 8-byte-aligned newtype over `u64`, so
// `node_id` (8) + `atlas_id` (4) rounds up to 16 bytes of struct alignment
// padding. This assertion guards that budget: if a future field pushes
// `AtlasNodeId` past 16 bytes, the build fails here instead of silently
// regressing pass-by-value performance.
const _: () = assert!(
    std::mem::size_of::<AtlasNodeId>() <= 16,
    "AtlasNodeId exceeded its 16-byte CPU register optimization budget!"
);

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

impl ContainerStyle {
    /// Constructs a `ContainerStyle` with the given explicit dimensions.
    ///
    /// Either dimension may be `None` to mean "grow to fit children".
    #[must_use]
    pub const fn new(width: Option<f32>, height: Option<f32>) -> Self {
        Self { width, height }
    }
}

/// Errors produced by the [`LayoutAtlas`].
#[non_exhaustive]
#[derive(Debug)]
pub enum AtlasError {
    /// The layout backend returned an error during tree construction
    /// or layout computation.
    Backend(String),

    /// An [`AtlasNodeId`] was used with a [`LayoutAtlas`] other than the
    /// one that created it.
    ///
    /// This is a misuse error, not a backend failure — without this check,
    /// passing an id from one atlas into a sibling atlas would silently
    /// read or mutate unrelated layout state (or panic deep inside Taffy),
    /// per the caveat this variant closes off.
    ForeignNode {
        /// The `instance_id` of the [`LayoutAtlas`] the id was used with.
        expected: u32,
        /// The `instance_id` of the [`LayoutAtlas`] that actually created
        /// the id.
        actual: u32,
    },
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
            Self::ForeignNode { expected, actual } => write!(
                f,
                "AtlasNodeId belongs to atlas instance {actual}, but was used with atlas instance {expected}"
            ),
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
    tree: TaffyTree<u32>,
    root: Option<AtlasNodeId>,
    state: AtlasState,
    children_scratch: Vec<NodeId>,
    grid: SpatialGrid,
    /// Reverse lookup from `TargetId.index` to the underlying node.
    /// Populated as nodes are added; reset on `clear()`.
    nodes_by_index: Vec<AtlasNodeId>,
    /// View generation. Incremented on `clear()` so `TargetId`s from
    /// previous views are silently rejected by `mark_dirty_all`.
    /// Wraps via `wrapping_add` after 65 535 clears — see the doc on
    /// `clear()` for the rationale.
    current_generation: u16,
    /// Unique id for this atlas instance, assigned at construction by
    /// [`Self::next_instance_id`]. Stamped onto every [`AtlasNodeId`] this
    /// atlas produces so a foreign id can be rejected in `O(1)`.
    instance_id: u32,
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
            grid: SpatialGrid::new(),
            nodes_by_index: Vec::new(),
            current_generation: 0,
            instance_id: Self::next_instance_id(),
        }
    }

    /// Returns this atlas's unique instance id.
    ///
    /// Every [`AtlasNodeId`] this atlas produces carries this id, so it can
    /// be rejected with [`AtlasError::ForeignNode`] if later used against a
    /// different `LayoutAtlas`.
    #[must_use]
    pub const fn instance_id(&self) -> u32 {
        self.instance_id
    }

    /// Allocates the next globally unique atlas instance id.
    ///
    /// Backed by a function-local `AtomicU32` rather than a module-level
    /// static — keeps the counter's existence scoped to the one place that
    /// uses it. `Relaxed` ordering is sufficient: callers only need a
    /// distinct value per atlas, not synchronization with any other memory
    /// access.
    fn next_instance_id() -> u32 {
        static NEXT_INSTANCE_ID: AtomicU32 = AtomicU32::new(0);
        NEXT_INSTANCE_ID.fetch_add(1, Ordering::Relaxed)
    }

    /// Returns `Ok(())` if `node` was created by this atlas instance, or
    /// [`AtlasError::ForeignNode`] otherwise.
    fn validate_node(&self, node: AtlasNodeId) -> Result<(), AtlasError> {
        if node.atlas_id == self.instance_id {
            Ok(())
        } else {
            Err(AtlasError::ForeignNode {
                expected: self.instance_id,
                actual: node.atlas_id,
            })
        }
    }

    /// Clears the tree but retains internal capacity.
    ///
    /// Increments the internal view generation, which causes any
    /// [`TargetId`]s produced before this call to be silently rejected
    /// by [`Self::mark_dirty_all`]. The generation wraps after `u16::MAX`
    /// clears — the collision probability with a stale `TargetId`
    /// surviving that long is statistically negligible (see project notes
    /// on `TargetId` packing).
    pub fn clear(&mut self) {
        self.tree.clear();
        self.root = None;
        self.state = AtlasState::Building;
        self.children_scratch.clear();
        self.grid.clear();
        self.nodes_by_index.clear();
        self.current_generation = self.current_generation.wrapping_add(1);
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

        let next_index = self.next_target_index();

        let node = self
            .tree
            .new_leaf_with_context(style, next_index)
            .map_err(|e| AtlasError::from_taffy(&e))?;
        let id = AtlasNodeId {
            node_id: node,
            atlas_id: self.instance_id,
        };
        self.nodes_by_index.push(id);
        Ok(id)
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
    /// Returns [`AtlasError::ForeignNode`] if any child was created by a
    /// different `LayoutAtlas` instance.
    pub fn add_container(
        &mut self,
        style: ContainerStyle,
        children: &[AtlasNodeId],
    ) -> Result<AtlasNodeId, AtlasError> {
        self.assert_building("add_container");

        for &child in children {
            self.validate_node(child)?;
        }

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
        self.children_scratch
            .extend(children.iter().map(|c| c.node_id));
        let next_index = self.next_target_index();
        let node = self
            .tree
            .new_with_children(taffy_style, &self.children_scratch)
            .map_err(|e| AtlasError::from_taffy(&e))?;
        self.tree
            .set_node_context(node, Some(next_index))
            .map_err(|e| AtlasError::from_taffy(&e))?;
        let id = AtlasNodeId {
            node_id: node,
            atlas_id: self.instance_id,
        };
        self.nodes_by_index.push(id);
        Ok(id)
    }

    /// Sets the root node for layout computation.
    ///
    /// # Panics
    ///
    /// Panics if the atlas is in the `Computed` state.
    ///
    /// # Errors
    ///
    /// Returns [`AtlasError::ForeignNode`] if `root` was created by a
    /// different `LayoutAtlas` instance.
    pub fn set_root(&mut self, root: AtlasNodeId) -> Result<(), AtlasError> {
        self.assert_building("set_root");
        self.validate_node(root)?;
        self.root = Some(root);
        Ok(())
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
            .compute_layout(root.node_id, available)
            .map_err(|e| AtlasError::from_taffy(&e))?;
        self.state = AtlasState::Computed;
        self.rebuild_grid();
        Ok(())
    }

    /// Returns the resolved rectangle for `node`.
    ///
    /// # Caveat: orphan nodes
    ///
    /// If `node` was added to the tree but never attached (directly or
    /// transitively) to the root configured via [`Self::set_root`], Taffy
    /// still resolves a default `Rect` of all zeros for it rather than
    /// failing — this returns `Ok(Some(zero_rect))`, not `Ok(None)`. See
    /// [`Self::resolved_rect_internal`] for the raw Taffy behaviour this
    /// wraps. Callers should only query rects for nodes known to be
    /// reachable from the root.
    ///
    /// # Panics
    ///
    /// Panics if the atlas is in the `Building` state. Call
    /// [`Self::compute`] first.
    ///
    /// # Errors
    ///
    /// Returns [`AtlasError::ForeignNode`] if `node` was created by a
    /// different `LayoutAtlas` instance.
    pub fn resolved_rect(&self, node: AtlasNodeId) -> Result<Option<Rect>, AtlasError> {
        assert_eq!(
            self.state,
            AtlasState::Computed,
            "LayoutAtlas::resolved_rect called before compute — geometry is not available yet"
        );
        self.validate_node(node)?;

        Ok(self.resolved_rect_internal(node))
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

    /// Performs spatial hit-testing to find the topmost node at the given screen coordinates.
    ///
    /// # Panics
    ///
    /// Panics if the atlas is in the `Building` state.
    #[must_use]
    pub fn hit_test(&self, x: f32, y: f32) -> Option<AtlasNodeId> {
        assert_eq!(
            self.state,
            AtlasState::Computed,
            "LayoutAtlas::hit_test called while in Building state — layout must be computed first"
        );

        if let Some(target) = self.grid.query(x, y) {
            if target.generation() == self.current_generation
                && (target.index() as usize) < self.nodes_by_index.len()
            {
                return Some(self.nodes_by_index[target.index() as usize]);
            }
        }
        None
    }

    /// Recursively walks the tree from `node` in pre-order, pushing each
    /// resolved rectangle into `frame`.
    fn walk_and_push(&self, root: AtlasNodeId, frame: &mut RenderFrame) {
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if let Some(rect) = self.resolved_rect_internal(node) {
                frame.push_rect(rect);
            }
            if let Ok(children) = self.tree.children(node.node_id) {
                // Push in reverse so the leftmost child is popped first,
                // preserving pre-order traversal semantics.
                for child in children.iter().rev() {
                    stack.push(AtlasNodeId {
                        node_id: *child,
                        atlas_id: self.instance_id,
                    });
                }
            }
        }
    }

    /// Rebuilds the hit-testing spatial grid from the current layout.
    fn rebuild_grid(&mut self) {
        self.grid.clear();
        let Some(root) = self.root else {
            return;
        };

        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            let index = *self.tree.get_node_context(node.node_id).unwrap();
            let target =
                TargetId::new(index, self.current_generation, TargetKind::AtlasNode as u16);
            if let Some(rect) = self.resolved_rect_internal(node) {
                self.grid.insert(rect, target);
            }
            if let Ok(children) = self.tree.children(node.node_id) {
                // Push in reverse so the leftmost child is popped first,
                // preserving pre-order traversal semantics.
                for child in children.iter().rev() {
                    stack.push(AtlasNodeId {
                        node_id: *child,
                        atlas_id: self.instance_id,
                    });
                }
            }
        }
    }

    /// Same as `resolved_rect` but without the state assertion or the
    /// cross-atlas check — used internally during traversal, where every
    /// `node` is already known to belong to this atlas (it was fetched
    /// from `self.tree` or `self.root`, never supplied by an external
    /// caller).
    fn resolved_rect_internal(&self, node: AtlasNodeId) -> Option<Rect> {
        let layout = self.tree.layout(node.node_id).ok()?;
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

    /// Returns the index that the next added node will receive.
    ///
    /// Useful when constructing a [`TargetId`] before the node is created
    /// (e.g. when registering a `Signal` that points to a yet-to-be-built
    /// layout target).
    ///
    /// # Truncation
    ///
    /// The returned value is `u32` to match the [`TargetId`] bit layout.
    /// In the theoretical case of an atlas containing more than `u32::MAX`
    /// (≈ 4.3 billion) nodes, the cast truncates and subsequent `TargetId`s
    /// will alias earlier ones. A `debug_assert!` catches this in debug
    /// builds; in release the bug would surface as ghost dirty marks on
    /// the wrong nodes.
    ///
    /// In practice this limit is unreachable — 4 billion nodes at ~100
    /// bytes each would require ~400 GB of RAM for the tree alone.
    #[must_use]
    pub fn next_target_index(&self) -> u32 {
        let len = self.nodes_by_index.len();
        debug_assert!(
            u32::try_from(len).is_ok(),
            "LayoutAtlas exceeded u32::MAX nodes — TargetId indexing will alias",
        );
        #[allow(clippy::cast_possible_truncation)]
        {
            len as u32
        }
    }

    /// Returns the current view generation.
    ///
    /// Embed this into [`TargetId::new`] when registering a `Signal`
    /// against an Atlas node; future `mark_dirty_all` calls will then
    /// validate it against the Atlas's current generation.
    #[must_use]
    pub fn current_generation(&self) -> u16 {
        self.current_generation
    }

    /// Marks every target in `targets` that belongs to this atlas as dirty.
    ///
    /// Targets are filtered by [`TargetKind::AtlasNode`] and by matching
    /// generation, so callers can safely pass the full batch produced by
    /// [`EvaluatorTick::collect_dirty`](crate::evaluator::EvaluatorTick::collect_dirty).
    /// Foreign or stale targets are silently ignored — this is the
    /// broadcast/event-bus pattern documented in RFC-0001 §4.1.
    ///
    /// # Panics
    ///
    /// Panics if the atlas is in the `Building` state. Call [`Self::compute`]
    /// first.
    pub fn mark_dirty_all(&mut self, targets: &[TargetId]) {
        assert_eq!(
            self.state,
            AtlasState::Computed,
            "LayoutAtlas::mark_dirty_all called before compute — \
         no geometry exists to mark dirty yet"
        );

        for target in targets {
            if target.kind() != TargetKind::AtlasNode as u16 {
                continue;
            }
            if target.generation() != self.current_generation {
                continue;
            }
            let index = target.index() as usize;
            if let Some(&node) = self.nodes_by_index.get(index) {
                // If Taffy refuses (very rare — would indicate the node
                // was somehow removed from the tree), skip silently. The
                // next recompute will produce a layout that reflects the
                // tree as it actually is.
                let _ = self.tree.mark_dirty(node.node_id);
            }
        }
    }

    /// Recomputes layout for the subtrees marked dirty since the last
    /// `compute` or `recompute_dirty`.
    ///
    /// Taffy caches geometry that has not changed, so this is typically
    /// much cheaper than a full [`Self::compute`]. After this call, the
    /// atlas remains in `Computed` state.
    ///
    /// # Panics
    ///
    /// Panics if the atlas is in the `Building` state, or if no root has
    /// been set.
    ///
    /// # Errors
    ///
    /// Returns [`AtlasError::Backend`] if layout computation fails.
    pub fn recompute_dirty(&mut self, viewport: Viewport) -> Result<(), AtlasError> {
        assert_eq!(
            self.state,
            AtlasState::Computed,
            "LayoutAtlas::recompute_dirty called before compute — \
         the initial layout pass must run via compute() first"
        );

        let root = self.root.expect(
            "LayoutAtlas::recompute_dirty reached Computed state without a root node — \
         this indicates an internal state-machine inconsistency",
        );

        let available = Size {
            width: AvailableSpace::Definite(viewport.width),
            height: AvailableSpace::Definite(viewport.height),
        };

        self.tree
            .compute_layout(root.node_id, available)
            .map_err(|e| AtlasError::from_taffy(&e))?;
        self.rebuild_grid();
        Ok(())
    }
}

impl Default for LayoutAtlas {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ByardError;

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
        atlas.set_root(root).unwrap();

        atlas.compute(Viewport::new(800.0, 600.0)).expect("compute");

        let root_rect = atlas.resolved_rect(root).unwrap().expect("root rect");
        assert_f32_eq(root_rect.width, 200.0);
        assert_f32_eq(root_rect.height, 100.0);

        let child_rect = atlas.resolved_rect(child).unwrap().expect("child rect");
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
        atlas.set_root(root).unwrap();
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
        atlas.set_root(leaf).unwrap();
        atlas.compute(Viewport::new(100.0, 100.0)).unwrap();

        // This must panic.
        let _ = atlas.add_leaf(LeafSize::new(20.0, 20.0));
    }

    #[test]
    #[should_panic(expected = "called while in Computed state")]
    fn add_container_after_compute_panics() {
        let mut atlas = LayoutAtlas::new();
        let leaf = atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();
        atlas.set_root(leaf).unwrap();
        atlas.compute(Viewport::new(100.0, 100.0)).unwrap();

        let _ = atlas.add_container(ContainerStyle::default(), &[leaf]);
    }

    #[test]
    #[should_panic(expected = "called before compute")]
    fn resolved_rect_before_compute_panics() {
        let mut atlas = LayoutAtlas::new();
        let leaf = atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();
        atlas.set_root(leaf).unwrap();

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
        atlas.set_root(root).unwrap();
        atlas.compute(Viewport::new(800.0, 600.0)).unwrap();

        let root_rect = atlas.resolved_rect(root).unwrap().unwrap();
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
        atlas.set_root(root).unwrap();
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
        atlas.set_root(leaf).unwrap();
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
        atlas.set_root(leaf).unwrap();

        let mut frame = RenderFrame::new();
        atlas.populate_frame(&mut frame);
    }

    #[test]
    #[should_panic(expected = "called while in Computed state")]
    fn set_root_after_compute_panics() {
        let mut atlas = LayoutAtlas::new();
        let leaf = atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();
        atlas.set_root(leaf).unwrap();
        atlas.compute(Viewport::new(100.0, 100.0)).unwrap();

        atlas.set_root(leaf).unwrap();
    }

    #[test]
    fn orphan_node_returns_zero_rect() {
        let mut atlas = LayoutAtlas::new();
        let root = atlas.add_leaf(LeafSize::new(50.0, 50.0)).unwrap();
        let orphan = atlas.add_leaf(LeafSize::new(999.0, 999.0)).unwrap();
        atlas.set_root(root).unwrap();
        atlas.compute(Viewport::new(100.0, 100.0)).unwrap();

        let orphan_rect = atlas.resolved_rect(orphan).unwrap().unwrap();
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
        atlas.set_root(root).unwrap();
        atlas.compute(Viewport::new(800.0, 600.0)).unwrap();

        let a_rect = atlas.resolved_rect(a).unwrap().unwrap();
        let b_rect = atlas.resolved_rect(b).unwrap().unwrap();

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

    #[test]
    fn mark_dirty_filters_foreign_kinds() {
        let mut atlas = LayoutAtlas::new();
        let leaf = atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();
        atlas.set_root(leaf).unwrap();
        atlas.compute(Viewport::new(100.0, 100.0)).unwrap();

        let foreign_kind: u16 = 999;
        let target = TargetId::new(0, atlas.current_generation(), foreign_kind);

        // Should not panic, should not affect anything.
        atlas.mark_dirty_all(&[target]);

        // Recompute must still succeed (no spurious dirty propagation).
        atlas.recompute_dirty(Viewport::new(100.0, 100.0)).unwrap();
    }

    #[test]
    fn mark_dirty_filters_stale_generation() {
        let mut atlas = LayoutAtlas::new();
        let leaf = atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();
        atlas.set_root(leaf).unwrap();
        atlas.compute(Viewport::new(100.0, 100.0)).unwrap();

        let stale_generation = atlas.current_generation().wrapping_sub(1);
        let target = TargetId::new(0, stale_generation, TargetKind::AtlasNode as u16);

        atlas.mark_dirty_all(&[target]);
        atlas.recompute_dirty(Viewport::new(100.0, 100.0)).unwrap();
    }

    #[test]
    fn mark_dirty_accepts_matching_kind_and_generation() {
        let mut atlas = LayoutAtlas::new();
        let leaf = atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();
        atlas.set_root(leaf).unwrap();
        atlas.compute(Viewport::new(100.0, 100.0)).unwrap();

        let target = TargetId::new(0, atlas.current_generation(), TargetKind::AtlasNode as u16);
        atlas.mark_dirty_all(&[target]);

        atlas.recompute_dirty(Viewport::new(200.0, 200.0)).unwrap();

        // Re-fetched rect reflects the new viewport. Container with no
        // explicit width takes viewport-driven size only if it has flex_grow
        // — here it's a leaf with fixed size, so the size stays at 10x10.
        let rect = atlas.resolved_rect(leaf).unwrap().unwrap();
        assert_f32_eq(rect.width, 10.0);
        assert_f32_eq(rect.height, 10.0);
    }

    #[test]
    fn clear_invalidates_previous_generation_targets() {
        let mut atlas = LayoutAtlas::new();
        let leaf = atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();
        atlas.set_root(leaf).unwrap();
        atlas.compute(Viewport::new(100.0, 100.0)).unwrap();

        let old_target = TargetId::new(0, atlas.current_generation(), TargetKind::AtlasNode as u16);

        atlas.clear();
        let new_leaf = atlas.add_leaf(LeafSize::new(20.0, 20.0)).unwrap();
        atlas.set_root(new_leaf).unwrap();
        atlas.compute(Viewport::new(100.0, 100.0)).unwrap();

        // Old target points at index 0, but its generation no longer
        // matches — must be silently ignored.
        atlas.mark_dirty_all(&[old_target]);
        atlas.recompute_dirty(Viewport::new(100.0, 100.0)).unwrap();
    }

    #[test]
    #[should_panic(expected = "called before compute")]
    fn recompute_dirty_before_compute_panics() {
        let mut atlas = LayoutAtlas::new();
        let leaf = atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();
        atlas.set_root(leaf).unwrap();

        let _ = atlas.recompute_dirty(Viewport::new(100.0, 100.0));
    }

    #[test]
    #[should_panic(expected = "called before compute")]
    fn mark_dirty_before_compute_panics() {
        let mut atlas = LayoutAtlas::new();
        let _ = atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();

        atlas.mark_dirty_all(&[]);
    }

    #[test]
    fn next_target_index_returns_consecutive_values() {
        let mut atlas = LayoutAtlas::new();
        assert_eq!(atlas.next_target_index(), 0);
        let _ = atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();
        assert_eq!(atlas.next_target_index(), 1);
        let _ = atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();
        assert_eq!(atlas.next_target_index(), 2);
    }

    #[test]
    fn current_generation_increments_on_clear() {
        let mut atlas = LayoutAtlas::new();
        assert_eq!(atlas.current_generation(), 0);
        atlas.clear();
        assert_eq!(atlas.current_generation(), 1);
        atlas.clear();
        assert_eq!(atlas.current_generation(), 2);
    }

    /// Acceptance criterion: a signal that mutates one leaf produces
    /// exactly one `TargetId` in the tick, which the atlas processes as
    /// exactly one `mark_dirty` call.
    ///
    /// This is the end-to-end validation of the Evaluator → Atlas flow
    /// described in RFC-0001 §2.2 and §4.1.
    #[test]
    fn signal_mutation_propagates_to_atlas_via_target_id() {
        use crate::evaluator::{EvaluatorTick, Signal, ViewArena};

        // ── Setup the Atlas ──────────────────────────────────────────────
        let mut atlas = LayoutAtlas::new();
        let leaf = atlas.add_leaf(LeafSize::new(50.0, 50.0)).unwrap();
        atlas.set_root(leaf).unwrap();
        atlas.compute(Viewport::new(200.0, 200.0)).unwrap();

        // The leaf is registered as TargetId index 0 in the atlas.
        let leaf_target = TargetId::new(
            atlas.next_target_index().wrapping_sub(1),
            atlas.current_generation(),
            TargetKind::AtlasNode as u16,
        );

        // ── Setup an Evaluator signal subscribed to the leaf ─────────────
        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 0_u32);
        signal.subscribe(leaf_target);

        let mut tick = EvaluatorTick::new();
        tick.register(signal);

        // First tick: no writes, no dirty targets.
        let dirty = tick.collect_dirty();
        assert!(
            dirty.is_empty(),
            "no writes should produce no dirty targets"
        );

        // ── Mutate the signal ────────────────────────────────────────────
        signal.write(|v| *v = 42);

        // The tick must collect exactly one TargetId pointing at our leaf.
        let dirty = tick.collect_dirty();
        assert_eq!(dirty.len(), 1, "one mutation → one dirty target");
        assert_eq!(dirty[0], leaf_target);

        // ── Atlas processes the dirty set ────────────────────────────────
        atlas.mark_dirty_all(&dirty);

        // Recompute completes successfully (Taffy has the leaf marked dirty
        // and re-runs layout for the affected subtree).
        atlas.recompute_dirty(Viewport::new(200.0, 200.0)).unwrap();

        // Geometry is still queryable post-recompute.
        let rect = atlas.resolved_rect(leaf).unwrap().unwrap();
        assert_f32_eq(rect.width, 50.0);
        assert_f32_eq(rect.height, 50.0);
    }

    #[test]
    fn container_style_constructor_round_trips() {
        let s = ContainerStyle::new(Some(100.0), Some(200.0));
        assert_eq!(s.width, Some(100.0));
        assert_eq!(s.height, Some(200.0));
    }

    #[test]
    fn hit_test_pure_success_and_miss() {
        let mut atlas = LayoutAtlas::new();
        let leaf = atlas.add_leaf(LeafSize::new(100.0, 100.0)).unwrap();
        atlas.set_root(leaf).unwrap();
        atlas.compute(Viewport::new(800.0, 600.0)).unwrap();

        // Hit within leaf
        assert_eq!(atlas.hit_test(50.0, 50.0), Some(leaf));

        // Miss (empty area)
        assert_eq!(atlas.hit_test(150.0, 150.0), None);
    }

    #[test]
    fn hit_test_z_order_implicit() {
        let mut atlas = LayoutAtlas::new();
        // A child that overlaps with its parent.
        // Let's create a container of size 200x200, and a child of size 100x100.
        // Taffy flexbox layout will position the child at (0, 0) relative to the container.
        let child = atlas.add_leaf(LeafSize::new(100.0, 100.0)).unwrap();
        let parent = atlas
            .add_container(
                ContainerStyle {
                    width: Some(200.0),
                    height: Some(200.0),
                },
                &[child],
            )
            .unwrap();
        atlas.set_root(parent).unwrap();
        atlas.compute(Viewport::new(800.0, 600.0)).unwrap();

        // The intersection is at (0, 0) to (100, 100).
        // Since child is traversed later (pre-order: parent then child),
        // query should return the child node.
        assert_eq!(atlas.hit_test(50.0, 50.0), Some(child));

        // Outside child, but inside parent
        assert_eq!(atlas.hit_test(150.0, 150.0), Some(parent));
    }

    #[test]
    fn hit_test_negative_coordinates() {
        let mut atlas = LayoutAtlas::new();
        let leaf = atlas.add_leaf(LeafSize::new(100.0, 100.0)).unwrap();
        atlas.set_root(leaf).unwrap();
        atlas.compute(Viewport::new(800.0, 600.0)).unwrap();

        // Should return None safely without panic
        assert_eq!(atlas.hit_test(-50.0, -50.0), None);
    }

    #[test]
    fn hit_test_invalidation_cycle() {
        let mut atlas = LayoutAtlas::new();
        let leaf = atlas.add_leaf(LeafSize::new(100.0, 100.0)).unwrap();
        atlas.set_root(leaf).unwrap();
        atlas.compute(Viewport::new(800.0, 600.0)).unwrap();

        // Verify hit-test works initially
        assert_eq!(atlas.hit_test(50.0, 50.0), Some(leaf));

        // Clear and construct a new view
        atlas.clear();

        let new_leaf = atlas.add_leaf(LeafSize::new(50.0, 50.0)).unwrap();
        atlas.set_root(new_leaf).unwrap();
        atlas.compute(Viewport::new(800.0, 600.0)).unwrap();

        // The old node is no longer valid, and coordinates outside the new leaf (e.g. 75, 75)
        // should return None, even though they were inside the old leaf.
        assert_eq!(atlas.hit_test(75.0, 75.0), None);
        // Inside the new leaf, it should return new_leaf
        assert_eq!(atlas.hit_test(25.0, 25.0), Some(new_leaf));
    }

    #[test]
    #[should_panic(expected = "called while in Building state")]
    fn hit_test_in_building_state_panics() {
        let mut atlas = LayoutAtlas::new();
        let leaf = atlas.add_leaf(LeafSize::new(100.0, 100.0)).unwrap();
        atlas.set_root(leaf).unwrap();

        // This must panic because compute has not been called.
        let _ = atlas.hit_test(50.0, 50.0);
    }
}
