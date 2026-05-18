//! Taffy-backed layout tree.
//!
//! See the module-level documentation in [`super`] for the design intent
//! and lifecycle contract.

use std::fmt;

use crate::frame::{Rect, Viewport};
use taffy::prelude::FromLength;
use taffy::{AvailableSpace, Dimension, NodeId, Size, Style, TaffyError, TaffyTree};

/// Opaque identifier for a node owned by a [`LayoutAtlas`].
///
/// Wraps [`taffy::NodeId`] so the Atlas does not leak Taffy types into the
/// rest of the engine.
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
/// Phase 1 exposes only the minimum needed for the acceptance criteria
/// (one container with a child). Further style fields will be added as
/// the `bylang` DSL grows.
#[derive(Debug, Clone, Copy, Default)]
pub struct ContainerStyle {
    /// Explicit width, if any. `None` means "grow to fit children".
    pub width: Option<f32>,
    /// Explicit height, if any. `None` means "grow to fit children".
    pub height: Option<f32>,
}

/// Errors produced by the [`LayoutAtlas`].
#[derive(Debug)]
pub enum AtlasError {
    /// The underlying Taffy engine returned an error during tree
    /// construction or layout computation.
    Taffy(TaffyError),
}

impl fmt::Display for AtlasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Taffy(e) => write!(f, "taffy error: {e}"),
        }
    }
}

impl std::error::Error for AtlasError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Taffy(e) => Some(e),
        }
    }
}

impl From<TaffyError> for AtlasError {
    fn from(e: TaffyError) -> Self {
        Self::Taffy(e)
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
}

impl LayoutAtlas {
    /// Creates a new, empty atlas in the `Building` state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tree: TaffyTree::new(),
            root: None,
            state: AtlasState::Building,
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
    /// Returns [`AtlasError::Taffy`] if the underlying engine refuses the
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

        let node = self.tree.new_leaf(style)?;
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
    /// Returns [`AtlasError::Taffy`] if the underlying engine refuses the
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

        // Convert our newtype IDs back to Taffy IDs for the call.
        let taffy_children: Vec<NodeId> = children.iter().map(|c| c.0).collect();
        let node = self.tree.new_with_children(taffy_style, &taffy_children)?;
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
    /// Returns [`AtlasError::Taffy`] if layout computation fails.
    pub fn compute(&mut self, viewport: Viewport) -> Result<(), AtlasError> {
        self.assert_building("compute");

        let root = self
            .root
            .expect("LayoutAtlas::compute called without a root node — call set_root first");

        let available = Size {
            width: AvailableSpace::Definite(viewport.width),
            height: AvailableSpace::Definite(viewport.height),
        };

        self.tree.compute_layout(root.0, available)?;
        self.state = AtlasState::Computed;
        Ok(())
    }

    /// Returns the resolved rectangle for `node`.
    ///
    /// # Panics
    ///
    /// Panics if the atlas is in the `Building` state. Call
    /// [`Self::compute`] first.
    #[must_use]
    pub fn resolved_rect(&self, node: AtlasNodeId) -> Option<Rect> {
        assert!(
            self.state == AtlasState::Computed,
            "LayoutAtlas::resolved_rect called before compute — geometry is not available yet",
        );

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

    fn assert_building(&self, method: &str) {
        assert!(
            self.state == AtlasState::Building,
            "LayoutAtlas::{method} called while in Computed state — call clear() first",
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
    fn clear_resets_to_building_and_keeps_capacity() {
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
}
