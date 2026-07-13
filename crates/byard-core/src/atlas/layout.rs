//! Taffy-backed layout tree.
//!
//! See the module-level documentation in [`super`] for the design intent
//! and lifecycle contract.

use std::collections::HashSet;
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::frame::{Rect, RenderFrame, TargetId, TargetKind, Viewport};
use taffy::style_helpers::{FromFr, FromLength, TaffyAuto, TaffyGridLine, TaffyGridSpan};
use taffy::{
    AlignItems, AvailableSpace, Dimension, Display, FlexDirection, GridPlacement,
    GridTemplateComponent, JustifyContent, LengthPercentage, LengthPercentageAuto, Line, NodeId,
    Overflow, Point, Position, Rect as TaffyRect, Size, Style, TaffyError, TaffyTree,
};

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

/// Granular spacing for padding or margin (top, right, bottom, left).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Spacing {
    /// Top spacing.
    pub top: f32,
    /// Right spacing.
    pub right: f32,
    /// Bottom spacing.
    pub bottom: f32,
    /// Left spacing.
    pub left: f32,
}

impl Spacing {
    /// Creates a Spacing with all sides set to the same value.
    #[must_use]
    pub const fn all(val: f32) -> Self {
        Self {
            top: val,
            right: val,
            bottom: val,
            left: val,
        }
    }

    /// Creates a Spacing with specific vertical and horizontal values.
    #[must_use]
    pub const fn symmetric(vertical: f32, horizontal: f32) -> Self {
        Self {
            top: vertical,
            right: horizontal,
            bottom: vertical,
            left: horizontal,
        }
    }
}

/// One track in a `Grid` template (RFC-0018): a flexible fraction (`1fr`), a
/// fixed length in logical px (`100`), or an auto-sized track (`auto`). Maps to
/// a Taffy `GridTemplateComponent`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GridTrack {
    /// A flexible fraction of the leftover space (`Nfr`).
    Fr(f32),
    /// A fixed length in logical pixels.
    Px(f32),
    /// An auto-sized track (fits its content).
    Auto,
}

/// Where a grid child sits in the grid (RFC-0018). `col_start`/`row_start` are
/// 1-based grid lines (CSS convention; negative counts from the end); `None`
/// leaves the axis to Taffy's auto-placement. Spans default to 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GridItemPlacement {
    /// 1-based column line to start at, or `None` for auto-placement.
    pub col_start: Option<i16>,
    /// Number of columns to span (≥ 1).
    pub col_span: u16,
    /// 1-based row line to start at, or `None` for auto-placement.
    pub row_start: Option<i16>,
    /// Number of rows to span (≥ 1).
    pub row_span: u16,
}

/// Main-axis direction of a flex container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlexDir {
    /// Children flow left-to-right.
    #[default]
    Row,
    /// Children flow top-to-bottom.
    Column,
}

/// Cross-axis alignment of a flex container's children.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Align {
    /// Pack against the cross-axis start.
    Start,
    /// Center on the cross axis.
    Center,
    /// Pack against the cross-axis end.
    End,
    /// Stretch to fill the cross axis (the default).
    #[default]
    Stretch,
}

/// Main-axis distribution of a flex container's children.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Justify {
    /// Pack against the main-axis start (the default).
    #[default]
    Start,
    /// Center on the main axis.
    Center,
    /// Pack against the main-axis end.
    End,
    /// Even space between children.
    Between,
    /// Even space around children.
    Around,
    /// Even space including the ends.
    Evenly,
}

/// Style for a container node, mapped onto a Taffy flex `Style`.
///
/// Marked `#[non_exhaustive]`; construct with [`ContainerStyle::new`] /
/// [`ContainerStyle::default`] and the `with_*` builders.
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct ContainerStyle {
    /// Explicit width in logical pixels. `None` means "grow to fit children".
    pub width: Option<f32>,
    /// Explicit height in logical pixels. `None` means "grow to fit children".
    pub height: Option<f32>,
    /// Main-axis direction.
    pub direction: FlexDir,
    /// Space between children, in logical pixels.
    pub gap: f32,
    /// Granular padding, in logical pixels.
    pub padding: Spacing,
    /// Granular margin, in logical pixels.
    pub margin: Spacing,
    /// Cross-axis alignment of children.
    pub align: Align,
    /// Main-axis distribution of children.
    pub justify: Justify,
    /// Flex-grow factor (how much this node expands to fill its parent's main
    /// axis).
    pub grow: f32,
    /// Whether this is a **horizontal scroll container** (RFC-0005 `ScrollView`
    /// `axis: horizontal|both`): content is measured at natural width and
    /// overflows the fixed viewport on the inline axis (Taffy `overflow.x =
    /// Scroll`), rather than being shrunk to fit. The renderer clips and scrolls
    /// the overflow.
    pub scroll_x: bool,
    /// Whether this is a **vertical scroll container** (RFC-0005 `ScrollView`,
    /// the default `axis: vertical`): content overflows on the block axis
    /// (Taffy `overflow.y = Scroll`). Clipped and scrolled by the renderer.
    pub scroll_y: bool,
    /// Whether this node is **absolutely positioned** and pinned to its
    /// containing block's edges (Taffy `position: Absolute`, `inset: 0`)
    /// — RFC-0017 overlay layer. An absolute node is removed from its parent's
    /// flex flow (it neither displaces siblings nor is displaced by them) and
    /// stretched to fill the containing block, so several can stack over the
    /// same viewport rect independently. The overlay compositor uses this to
    /// float each overlay above the main tree without perturbing its layout.
    pub absolute: bool,
}

impl ContainerStyle {
    /// Constructs a `ContainerStyle` with the given explicit dimensions and
    /// flex defaults (row, stretch, start, no gap/padding/grow).
    #[must_use]
    pub fn new(width: Option<f32>, height: Option<f32>) -> Self {
        Self {
            width,
            height,
            ..Default::default()
        }
    }

    /// Sets the main-axis direction.
    #[must_use]
    pub fn with_direction(mut self, direction: FlexDir) -> Self {
        self.direction = direction;
        self
    }

    /// Sets the inter-child gap (logical px).
    #[must_use]
    pub fn with_gap(mut self, gap: f32) -> Self {
        self.gap = gap;
        self
    }

    /// Sets the padding (logical px).
    #[must_use]
    pub fn with_padding(mut self, padding: Spacing) -> Self {
        self.padding = padding;
        self
    }

    /// Sets the margin (logical px).
    #[must_use]
    pub fn with_margin(mut self, margin: Spacing) -> Self {
        self.margin = margin;
        self
    }

    /// Sets the cross-axis alignment.
    #[must_use]
    pub fn with_align(mut self, align: Align) -> Self {
        self.align = align;
        self
    }

    /// Sets the main-axis distribution.
    #[must_use]
    pub fn with_justify(mut self, justify: Justify) -> Self {
        self.justify = justify;
        self
    }

    /// Sets the flex-grow factor.
    #[must_use]
    pub fn with_grow(mut self, grow: f32) -> Self {
        self.grow = grow;
        self
    }

    /// Marks this a scroll container on the given axes (RFC-0005 `ScrollView`):
    /// content overflows the viewport where enabled instead of shrinking to fit.
    #[must_use]
    pub fn with_scroll_axes(mut self, scroll_x: bool, scroll_y: bool) -> Self {
        self.scroll_x = scroll_x;
        self.scroll_y = scroll_y;
        self
    }

    /// Marks this node absolutely positioned, pinned to fill its containing
    /// block (RFC-0017 overlay layer). See [`absolute`](Self::absolute).
    #[must_use]
    pub fn with_absolute(mut self, absolute: bool) -> Self {
        self.absolute = absolute;
        self
    }

    /// Builds the Taffy `Style` this container maps to.
    fn to_taffy(self) -> Style {
        Style {
            size: Size {
                width: self.width.map_or(Dimension::auto(), Dimension::from_length),
                height: self
                    .height
                    .map_or(Dimension::auto(), Dimension::from_length),
            },
            flex_direction: match self.direction {
                FlexDir::Row => FlexDirection::Row,
                FlexDir::Column => FlexDirection::Column,
            },
            gap: Size {
                width: LengthPercentage::from_length(self.gap),
                height: LengthPercentage::from_length(self.gap),
            },
            padding: TaffyRect {
                left: LengthPercentage::from_length(self.padding.left),
                right: LengthPercentage::from_length(self.padding.right),
                top: LengthPercentage::from_length(self.padding.top),
                bottom: LengthPercentage::from_length(self.padding.bottom),
            },
            margin: TaffyRect {
                left: LengthPercentageAuto::length(self.margin.left),
                right: LengthPercentageAuto::length(self.margin.right),
                top: LengthPercentageAuto::length(self.margin.top),
                bottom: LengthPercentageAuto::length(self.margin.bottom),
            },
            align_items: Some(match self.align {
                Align::Start => AlignItems::FlexStart,
                Align::Center => AlignItems::Center,
                Align::End => AlignItems::FlexEnd,
                Align::Stretch => AlignItems::Stretch,
            }),
            justify_content: Some(match self.justify {
                Justify::Start => JustifyContent::FlexStart,
                Justify::Center => JustifyContent::Center,
                Justify::End => JustifyContent::FlexEnd,
                Justify::Between => JustifyContent::SpaceBetween,
                Justify::Around => JustifyContent::SpaceAround,
                Justify::Evenly => JustifyContent::SpaceEvenly,
            }),
            flex_grow: self.grow,
            // RFC-0017 overlay layer: an absolute node leaves its parent's flex
            // flow and is pinned to the containing block's edges (inset 0), so it
            // stretches to fill the viewport and stacks over siblings without
            // displacing them. A relative node keeps Taffy's `auto` insets.
            position: if self.absolute {
                Position::Absolute
            } else {
                Position::Relative
            },
            inset: if self.absolute {
                TaffyRect {
                    left: LengthPercentageAuto::length(0.0),
                    right: LengthPercentageAuto::length(0.0),
                    top: LengthPercentageAuto::length(0.0),
                    bottom: LengthPercentageAuto::length(0.0),
                }
            } else {
                TaffyRect::auto()
            },
            // RFC-0005 `ScrollView`: a scroll container measures its content at
            // natural size and lets it overflow the fixed viewport on the
            // scrolling axes, instead of flex-shrinking children to fit. The
            // renderer clips and scrolls the overflow.
            overflow: Point {
                x: if self.scroll_x {
                    Overflow::Scroll
                } else {
                    Overflow::Visible
                },
                y: if self.scroll_y {
                    Overflow::Scroll
                } else {
                    Overflow::Visible
                },
            },
            ..Default::default()
        }
    }
}

/// Declarative description of a layout (sub)tree.
///
/// `AtlasNodeSpec` values are plain data — building one never touches
/// Taffy or any [`LayoutAtlas`], so describing a tree can never fail.
/// They're produced with [`LayoutAtlasBuilder::leaf`] /
/// [`LayoutAtlasBuilder::container`] and committed to a real atlas in one
/// recursive pass via [`LayoutAtlas::build`] or [`LayoutAtlas::build_root`].
///
/// This is the fluent construction API — it sits on top
/// of [`LayoutAtlas::add_leaf`] / [`LayoutAtlas::add_container`] /
/// [`LayoutAtlas::set_root`] and calls them in the exact same depth-first,
/// children-before-parent order a hand-written imperative sequence would,
/// so it produces identical [`AtlasNodeId`]s. The low-level methods are
/// the tested foundation (PR #14) and are unchanged by this type.
#[derive(Debug, Clone)]
pub enum AtlasNodeSpec {
    /// Describes a leaf, mirroring [`LayoutAtlas::add_leaf`].
    Leaf(LeafSize),
    /// Describes a container and its children, mirroring
    /// [`LayoutAtlas::add_container`]. Children are built and attached in
    /// iteration order.
    Container(ContainerStyle, Vec<AtlasNodeSpec>),
}

/// Entry point for building [`AtlasNodeSpec`] trees fluently.
///
/// `LayoutAtlasBuilder` does not wrap a [`LayoutAtlas`] — it has no state
/// of its own. It's a pair of associated functions that produce
/// [`AtlasNodeSpec`] values, which `LayoutAtlas::build`/`build_root` then
/// commit. Nesting `container` calls lets a multi-level tree of mixed
/// leaves and containers be expressed as a single chained expression:
///
/// ```
/// use byard_core::atlas::{ContainerStyle, LayoutAtlas, LayoutAtlasBuilder as B, LeafSize};
///
/// let mut atlas = LayoutAtlas::new();
/// let root = atlas.build_root(
///     B::container(ContainerStyle::new(Some(300.0), Some(200.0)), [
///         B::leaf(LeafSize::new(50.0, 50.0)),
///         B::container(ContainerStyle::default(), [
///             B::leaf(LeafSize::new(20.0, 20.0)),
///         ]),
///     ]),
/// ).unwrap();
/// # let _ = root;
/// ```
pub struct LayoutAtlasBuilder;

impl LayoutAtlasBuilder {
    /// Describes a leaf node with the given size.
    #[must_use]
    pub const fn leaf(size: LeafSize) -> AtlasNodeSpec {
        AtlasNodeSpec::Leaf(size)
    }

    /// Describes a container node wrapping `children`, built in order.
    #[must_use]
    pub fn container(
        style: ContainerStyle,
        children: impl IntoIterator<Item = AtlasNodeSpec>,
    ) -> AtlasNodeSpec {
        AtlasNodeSpec::Container(style, children.into_iter().collect())
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
    parents: rustc_hash::FxHashMap<NodeId, NodeId>,
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
            parents: rustc_hash::FxHashMap::default(),
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
        self.parents.clear();
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

        let taffy_style = style.to_taffy();
        self.children_scratch.clear();
        self.children_scratch
            .extend(children.iter().map(|c| c.node_id));
        let next_index = self.next_target_index();
        let node = self
            .tree
            .new_with_children(taffy_style, &self.children_scratch)
            .map_err(|e| AtlasError::from_taffy(&e))?;
        for &child in children {
            self.parents.insert(child.node_id, node);
        }
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

    /// Adds a **CSS-grid** container (RFC-0018 `Grid`) wrapping `children`.
    ///
    /// The `style` supplies the shared box properties (size, padding, margin,
    /// grow, alignment); this method overrides the display mode to grid and sets
    /// the column/row track templates and the per-axis gaps. Children are
    /// auto-placed left-to-right, top-to-bottom by default; call
    /// [`set_grid_item`](Self::set_grid_item) on a child to place it explicitly.
    ///
    /// # Panics
    ///
    /// Panics if the atlas is in the `Computed` state.
    ///
    /// # Errors
    ///
    /// Returns [`AtlasError::Backend`] if the backend refuses the node, or
    /// [`AtlasError::ForeignNode`] if any child came from another atlas.
    pub fn add_grid_container(
        &mut self,
        style: ContainerStyle,
        columns: &[GridTrack],
        rows: &[GridTrack],
        col_gap: f32,
        row_gap: f32,
        children: &[AtlasNodeId],
    ) -> Result<AtlasNodeId, AtlasError> {
        self.assert_building("add_grid_container");

        for &child in children {
            self.validate_node(child)?;
        }

        let mut taffy_style = style.to_taffy();
        taffy_style.display = Display::Grid;
        // Inline the track mapping at each `collect` so the destination field's
        // type constrains `GridTemplateComponent`'s string generic (a shared
        // closure couldn't infer it).
        taffy_style.grid_template_columns = columns
            .iter()
            .map(|t| match t {
                GridTrack::Fr(f) => GridTemplateComponent::from_fr(*f),
                GridTrack::Px(p) => GridTemplateComponent::from_length(*p),
                GridTrack::Auto => GridTemplateComponent::AUTO,
            })
            .collect();
        taffy_style.grid_template_rows = rows
            .iter()
            .map(|t| match t {
                GridTrack::Fr(f) => GridTemplateComponent::from_fr(*f),
                GridTrack::Px(p) => GridTemplateComponent::from_length(*p),
                GridTrack::Auto => GridTemplateComponent::AUTO,
            })
            .collect();
        // Grid gaps are per-axis: `gap.width` is the column gap, `gap.height` the
        // row gap (Taffy/CSS convention).
        taffy_style.gap = Size {
            width: LengthPercentage::from_length(col_gap),
            height: LengthPercentage::from_length(row_gap),
        };

        self.children_scratch.clear();
        self.children_scratch
            .extend(children.iter().map(|c| c.node_id));
        let next_index = self.next_target_index();
        let node = self
            .tree
            .new_with_children(taffy_style, &self.children_scratch)
            .map_err(|e| AtlasError::from_taffy(&e))?;
        for &child in children {
            self.parents.insert(child.node_id, node);
        }
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

    /// Places an already-created grid child explicitly (RFC-0018): sets its
    /// `grid_column`/`grid_row` from `placement`. A `None` start leaves that
    /// axis to auto-placement; the span (≥ 1) always applies. A no-op-equivalent
    /// placement (auto start, span 1) still resolves to the same auto flow.
    ///
    /// # Panics
    ///
    /// Panics if the atlas is in the `Computed` state.
    ///
    /// # Errors
    ///
    /// Returns [`AtlasError::ForeignNode`] if `node` came from another atlas, or
    /// [`AtlasError::Backend`] if the backend rejects the restyle.
    pub fn set_grid_item(
        &mut self,
        node: AtlasNodeId,
        placement: GridItemPlacement,
    ) -> Result<(), AtlasError> {
        self.assert_building("set_grid_item");
        self.validate_node(node)?;

        let mut style = self
            .tree
            .style(node.node_id)
            .map_err(|e| AtlasError::from_taffy(&e))?
            .clone();
        // Assigned directly onto the `Line<GridPlacement<S>>` fields so the field
        // type constrains the string generic `S` (a shared helper couldn't infer
        // it). A `None` start stays auto; the span (≥ 1) always applies.
        let col_span = placement.col_span.max(1);
        style.grid_column = match placement.col_start {
            Some(s) => Line {
                start: GridPlacement::from_line_index(s),
                end: GridPlacement::from_span(col_span),
            },
            None => Line {
                start: GridPlacement::Auto,
                end: GridPlacement::from_span(col_span),
            },
        };
        let row_span = placement.row_span.max(1);
        style.grid_row = match placement.row_start {
            Some(s) => Line {
                start: GridPlacement::from_line_index(s),
                end: GridPlacement::from_span(row_span),
            },
            None => Line {
                start: GridPlacement::Auto,
                end: GridPlacement::from_span(row_span),
            },
        };
        self.tree
            .set_style(node.node_id, style)
            .map_err(|e| AtlasError::from_taffy(&e))?;
        Ok(())
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

    /// Commits an [`AtlasNodeSpec`] tree built via [`LayoutAtlasBuilder`].
    ///
    /// Walks `spec` depth-first, building every child before its parent —
    /// the exact same order and resulting [`AtlasNodeId`]s a hand-written
    /// call sequence of [`Self::add_leaf`] / [`Self::add_container`] would
    /// produce. Does not set the result as the root; use
    /// [`Self::build_root`] for that, or call [`Self::set_root`] yourself
    /// on the returned id.
    ///
    /// # Panics
    ///
    /// Panics if the atlas is in the `Computed` state (same contract as
    /// [`Self::add_leaf`] / [`Self::add_container`]).
    ///
    /// # Errors
    ///
    /// Returns [`AtlasError::Backend`] if the underlying engine refuses a
    /// node. [`AtlasError::ForeignNode`] cannot occur here — every
    /// `AtlasNodeId` `build` consumes is one it just created itself while
    /// walking `spec`, never one supplied by the caller.
    pub fn build(&mut self, spec: AtlasNodeSpec) -> Result<AtlasNodeId, AtlasError> {
        match spec {
            AtlasNodeSpec::Leaf(size) => self.add_leaf(size),
            AtlasNodeSpec::Container(style, children) => {
                let mut built = Vec::with_capacity(children.len());
                for child in children {
                    built.push(self.build(child)?);
                }
                self.add_container(style, &built)
            }
        }
    }

    /// Like [`Self::build`], but also installs the result as the root via
    /// [`Self::set_root`] — the common case when `spec` describes a whole
    /// view rather than a fragment to be attached elsewhere.
    ///
    /// # Panics
    ///
    /// See [`Self::build`].
    ///
    /// # Errors
    ///
    /// See [`Self::build`].
    pub fn build_root(&mut self, spec: AtlasNodeSpec) -> Result<AtlasNodeId, AtlasError> {
        let root = self.build(spec)?;
        self.set_root(root)?;
        Ok(root)
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

    /// The `(width, height)` of a node's **content** — the extent of its
    /// children, which for a `ScrollView` (Taffy `overflow: Scroll`) exceeds the
    /// node's own box. Subtracting the viewport size gives the maximum scroll
    /// distance (RFC-0005). `None` if the node is unknown or not yet computed.
    ///
    /// # Errors
    ///
    /// Returns [`AtlasError::ForeignNode`] if `node` belongs to another atlas.
    pub fn content_size(&self, node: AtlasNodeId) -> Result<Option<(f32, f32)>, AtlasError> {
        self.validate_node(node)?;
        if self.state != AtlasState::Computed {
            return Ok(None);
        }
        Ok(self
            .tree
            .layout(node.node_id)
            .ok()
            .map(|l| (l.content_size.width, l.content_size.height)))
    }

    /// Writes the resolved geometry of every node into `frame`, marking each
    /// entry dirty if its `TargetId` appears in `dirty_targets`.
    ///
    /// Walks the tree from the root in pre-order and appends each node's
    /// resolved [`Rect`] to the frame. This is how the Atlas hands geometry
    /// to the Encoder without either subsystem importing from the other —
    /// the frame is the shared boundary defined in RFC-0001 §9.
    ///
    /// `dirty_targets` is typically the output of
    /// [`EvaluatorTick::collect_dirty`](crate::evaluator::EvaluatorTick::collect_dirty)
    /// for this tick. Each node's own `TargetId` (kind, generation, index)
    /// is reconstructed using the same scheme as [`Self::rebuild_grid`] and
    /// [`Self::mark_dirty_all`], so stale-generation targets are excluded
    /// for free — they simply will not match any live node's id.
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
    pub fn populate_frame(&self, frame: &mut RenderFrame, dirty_targets: &[TargetId]) {
        assert_eq!(
            self.state,
            AtlasState::Computed,
            "LayoutAtlas::populate_frame called before compute — geometry is not available yet"
        );

        let root = self.root.expect(
            "LayoutAtlas::populate_frame reached Computed state without a root node — \
         this indicates an internal state-machine inconsistency",
        );

        let dirty: HashSet<u64> = dirty_targets.iter().map(|t| t.as_raw()).collect();

        self.walk_and_push(root, frame, &dirty);
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

    /// Returns the context index of the given node, if it belongs to this atlas.
    #[must_use]
    pub fn node_index(&self, node: AtlasNodeId) -> Option<u32> {
        if node.atlas_id == self.instance_id {
            self.tree.get_node_context(node.node_id).copied()
        } else {
            None
        }
    }

    /// Returns the parent node of the given node, if it belongs to this atlas and has a parent.
    #[must_use]
    pub fn parent_node(&self, node: AtlasNodeId) -> Option<AtlasNodeId> {
        if node.atlas_id == self.instance_id {
            let parent_id = self.parents.get(&node.node_id)?;
            Some(AtlasNodeId {
                node_id: *parent_id,
                atlas_id: self.instance_id,
            })
        } else {
            None
        }
    }

    /// Recursively walks the tree from `node` in pre-order, pushing each
    /// resolved rectangle — and its dirty state — into `frame`.
    ///
    /// `dirty` holds the raw (`TargetId::as_raw`) bits of every dirty
    /// target for this tick. Each node's own `TargetId` is reconstructed
    /// from its Taffy node-context index, the atlas's current generation,
    /// and `TargetKind::AtlasNode` — the same triple `rebuild_grid` uses —
    /// so the lookup is exact and stale generations never match.
    fn walk_and_push(&self, root: AtlasNodeId, frame: &mut RenderFrame, dirty: &HashSet<u64>) {
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if let Some(rect) = self.resolved_rect_internal(node) {
                let index = *self.tree.get_node_context(node.node_id).unwrap();
                let target =
                    TargetId::new(index, self.current_generation, TargetKind::AtlasNode as u16);
                let is_dirty = dirty.contains(&target.as_raw());
                frame.push_rect(rect, is_dirty);
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
    ///
    /// This is a full `clear()` + root-to-leaf walk on every
    /// `compute`/`recompute_dirty`, regardless of how many nodes were marked
    /// dirty. That was measured (M28) on a 200-leaf tree (the high end
    /// of `EvaluatorTick`'s expected per-tick target count): the whole
    /// `recompute_dirty` — layout + this grid rebuild — costs ~24 µs with one
    /// dirty leaf and ~111 µs with every node dirty, i.e. ≲0.7% of a 60 Hz
    /// frame even in the worst case. A partial grid update would have to track
    /// nodes whose rect shifted only as a *side effect* of a sibling's flex
    /// reflow, risking dangling (stale-but-queryable) hit rects — a correctness
    /// hazard strictly worse than a redundant walk this cheap. The full walk is
    /// therefore kept deliberately; see the `atlas` bench for the numbers.
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
        let mut x = layout.location.x;
        let mut y = layout.location.y;

        let mut current = node.node_id;
        while let Some(&parent) = self.parents.get(&current) {
            if let Ok(p_layout) = self.tree.layout(parent) {
                x += p_layout.location.x;
                y += p_layout.location.y;
                current = parent;
            } else {
                break;
            }
        }

        Some(Rect {
            x,
            y,
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
                    ..Default::default()
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
                    ..Default::default()
                },
                &[child],
            )
            .unwrap();
        atlas.set_root(root).unwrap();
        atlas.compute(Viewport::new(800.0, 600.0)).unwrap();

        let mut frame = RenderFrame::new();
        atlas.populate_frame(&mut frame, &[]);

        assert_eq!(frame.rects().len(), 2, "root + child");

        // Pre-order: root first, then child.
        assert_f32_eq(frame.rects()[0].width, 200.0);
        assert_f32_eq(frame.rects()[0].height, 100.0);
        assert_f32_eq(frame.rects()[1].width, 100.0);
        assert_f32_eq(frame.rects()[1].height, 50.0);

        // No dirty targets were passed in, so nothing is marked dirty.
        assert_eq!(frame.dirty(), &[false, false]);
    }

    #[test]
    fn populate_frame_appends_without_clearing() {
        use crate::frame::RenderFrame;

        let mut atlas = LayoutAtlas::new();
        let leaf = atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();
        atlas.set_root(leaf).unwrap();
        atlas.compute(Viewport::new(100.0, 100.0)).unwrap();

        let mut frame = RenderFrame::new();
        atlas.populate_frame(&mut frame, &[]);
        atlas.populate_frame(&mut frame, &[]);

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
        atlas.populate_frame(&mut frame, &[]);
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
                    ..Default::default()
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
    fn grid_two_columns_positions_children_side_by_side() {
        // RFC-0018: a 200×100 grid with two `1fr` columns (100px each) and two
        // auto-placed children lays them one per column — A at x=0, B at x=100.
        let mut atlas = LayoutAtlas::new();
        let a = atlas.add_leaf(LeafSize::new(0.0, 40.0)).unwrap();
        let b = atlas.add_leaf(LeafSize::new(0.0, 40.0)).unwrap();
        let grid = atlas
            .add_grid_container(
                ContainerStyle::new(Some(200.0), Some(100.0)),
                &[GridTrack::Fr(1.0), GridTrack::Fr(1.0)],
                &[],
                0.0,
                0.0,
                &[a, b],
            )
            .unwrap();
        atlas.set_root(grid).unwrap();
        atlas.compute(Viewport::new(800.0, 600.0)).unwrap();

        let a_rect = atlas.resolved_rect(a).unwrap().unwrap();
        let b_rect = atlas.resolved_rect(b).unwrap().unwrap();
        let mut xs = [a_rect.x, b_rect.x];
        xs.sort_by(f32::total_cmp);
        assert_f32_eq(xs[0], 0.0);
        assert_f32_eq(xs[1], 100.0);
    }

    #[test]
    fn grid_explicit_placement_pins_child_to_second_column() {
        // RFC-0018: `set_grid_item` with `col_start = 2` pins the child to the
        // second 1fr column (x = 100) even though it is the only/first child.
        let mut atlas = LayoutAtlas::new();
        let a = atlas.add_leaf(LeafSize::new(0.0, 40.0)).unwrap();
        let grid = atlas
            .add_grid_container(
                ContainerStyle::new(Some(200.0), Some(100.0)),
                &[GridTrack::Fr(1.0), GridTrack::Fr(1.0)],
                &[],
                0.0,
                0.0,
                &[a],
            )
            .unwrap();
        atlas
            .set_grid_item(
                a,
                GridItemPlacement {
                    col_start: Some(2),
                    col_span: 1,
                    row_start: None,
                    row_span: 1,
                },
            )
            .unwrap();
        atlas.set_root(grid).unwrap();
        atlas.compute(Viewport::new(800.0, 600.0)).unwrap();

        let a_rect = atlas.resolved_rect(a).unwrap().unwrap();
        assert_f32_eq(a_rect.x, 100.0);
    }

    #[test]
    fn absolute_node_fills_containing_block_and_ignores_flow() {
        // RFC-0017: two absolute children pinned inset-0 both fill the viewport
        // and neither displaces the other (nor a flowing sibling).
        let mut atlas = LayoutAtlas::new();
        let flow = atlas.add_leaf(LeafSize::new(40.0, 40.0)).unwrap();
        let overlay_a = atlas
            .add_container(ContainerStyle::default().with_absolute(true), &[])
            .unwrap();
        let overlay_b = atlas
            .add_container(ContainerStyle::default().with_absolute(true), &[])
            .unwrap();
        let root = atlas
            .add_container(
                ContainerStyle::new(Some(300.0), Some(200.0)),
                &[flow, overlay_a, overlay_b],
            )
            .unwrap();
        atlas.set_root(root).unwrap();
        atlas.compute(Viewport::new(300.0, 200.0)).unwrap();

        // The flowing child keeps its natural size at the origin — absolute
        // siblings did not push it.
        let flow_rect = atlas.resolved_rect(flow).unwrap().unwrap();
        assert_f32_eq(flow_rect.x, 0.0);
        assert_f32_eq(flow_rect.width, 40.0);

        for ov in [overlay_a, overlay_b] {
            let r = atlas.resolved_rect(ov).unwrap().unwrap();
            assert_f32_eq(r.x, 0.0);
            assert_f32_eq(r.y, 0.0);
            assert_f32_eq(r.width, 300.0);
            assert_f32_eq(r.height, 200.0);
        }
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

    /// Acceptance criterion: a `Signal` mutation results in
    /// only the affected entries being marked dirty in `RenderFrame`.
    ///
    /// Builds a two-leaf tree, subscribes a signal to only one leaf, and
    /// verifies that after mutating it and ticking, `populate_frame` marks
    /// exactly that leaf's `RenderFrame` entry dirty — the sibling and the
    /// root stay clean.
    #[test]
    fn evaluator_tick_marks_only_affected_render_frame_entries_dirty() {
        use crate::evaluator::{EvaluatorTick, Signal, ViewArena};
        use crate::frame::RenderFrame;

        let mut atlas = LayoutAtlas::new();
        let a = atlas.add_leaf(LeafSize::new(50.0, 50.0)).unwrap();
        let b = atlas.add_leaf(LeafSize::new(50.0, 50.0)).unwrap();
        let root = atlas
            .add_container(
                ContainerStyle {
                    width: Some(200.0),
                    height: Some(200.0),
                    ..Default::default()
                },
                &[a, b],
            )
            .unwrap();
        atlas.set_root(root).unwrap();
        atlas.compute(Viewport::new(200.0, 200.0)).unwrap();

        // `b` was the second leaf added, so its TargetId index is 1.
        let b_target = TargetId::new(1, atlas.current_generation(), TargetKind::AtlasNode as u16);

        let arena = ViewArena::new();
        let signal = Signal::new_in(&arena, 0_u32);
        signal.subscribe(b_target);

        let mut tick = EvaluatorTick::new();
        tick.register(signal);

        // Mutate only the signal subscribed to `b`.
        signal.write(|v| *v = 1);
        let dirty_targets = tick.collect_dirty();
        assert_eq!(dirty_targets, vec![b_target]);

        let mut frame = RenderFrame::new();
        atlas.populate_frame(&mut frame, &dirty_targets);

        // Pre-order: root, then a, then b.
        assert_eq!(frame.rects().len(), 3, "root + a + b");
        assert_eq!(
            frame.dirty(),
            &[false, false, true],
            "only b's entry is dirty — root and a are untouched"
        );
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
                    ..Default::default()
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

    // --- Cross-atlas AtlasNodeId scoping ---------------------
    //
    // These tests exercise the actual hazard this closes off: an
    // `AtlasNodeId` produced by one `LayoutAtlas` must never be silently
    // accepted by a different instance. Every entry point that takes an
    // `AtlasNodeId` from a caller must return `Err(AtlasError::ForeignNode)`
    // — never panic, never silently produce wrong geometry.

    #[test]
    fn two_atlases_have_distinct_instance_ids() {
        let atlas_a = LayoutAtlas::new();
        let atlas_b = LayoutAtlas::new();
        assert_ne!(atlas_a.instance_id(), atlas_b.instance_id());
    }

    #[test]
    fn set_root_rejects_foreign_node() {
        let mut atlas_a = LayoutAtlas::new();
        let foreign_leaf = atlas_a.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();

        let mut atlas_b = LayoutAtlas::new();
        let err = atlas_b.set_root(foreign_leaf).unwrap_err();

        match err {
            AtlasError::ForeignNode { expected, actual } => {
                assert_eq!(expected, atlas_b.instance_id());
                assert_eq!(actual, atlas_a.instance_id());
            }
            other => panic!("expected AtlasError::ForeignNode, got {other:?}"),
        }
    }

    #[test]
    fn add_container_rejects_foreign_child() {
        let mut atlas_a = LayoutAtlas::new();
        let foreign_leaf = atlas_a.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();

        let mut atlas_b = LayoutAtlas::new();
        let local_leaf = atlas_b.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();

        // Mixing one local and one foreign child must still be rejected —
        // validation can't be satisfied by having at least one valid id.
        let err = atlas_b
            .add_container(
                ContainerStyle::new(Some(100.0), Some(100.0)),
                &[local_leaf, foreign_leaf],
            )
            .unwrap_err();

        match err {
            AtlasError::ForeignNode { expected, actual } => {
                assert_eq!(expected, atlas_b.instance_id());
                assert_eq!(actual, atlas_a.instance_id());
            }
            other => panic!("expected AtlasError::ForeignNode, got {other:?}"),
        }
    }

    #[test]
    fn resolved_rect_rejects_foreign_node() {
        let mut atlas_a = LayoutAtlas::new();
        let leaf_a = atlas_a.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();
        atlas_a.set_root(leaf_a).unwrap();
        atlas_a.compute(Viewport::new(800.0, 600.0)).unwrap();

        let mut atlas_b = LayoutAtlas::new();
        let leaf_b = atlas_b.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();
        atlas_b.set_root(leaf_b).unwrap();
        atlas_b.compute(Viewport::new(800.0, 600.0)).unwrap();

        // atlas_b is Computed, so the state assertion passes — the
        // cross-atlas check must still catch the foreign id from atlas_a.
        let err = atlas_b.resolved_rect(leaf_a).unwrap_err();

        match err {
            AtlasError::ForeignNode { expected, actual } => {
                assert_eq!(expected, atlas_b.instance_id());
                assert_eq!(actual, atlas_a.instance_id());
            }
            other => panic!("expected AtlasError::ForeignNode, got {other:?}"),
        }
    }

    #[test]
    fn foreign_node_error_bridges_to_byard_error() {
        let mut atlas_a = LayoutAtlas::new();
        let foreign_leaf = atlas_a.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();

        let mut atlas_b = LayoutAtlas::new();
        let atlas_err = atlas_b.set_root(foreign_leaf).unwrap_err();
        let byard_err: ByardError = atlas_err.into();

        assert!(byard_err.to_string().contains("AtlasNodeId belongs to"));
    }

    // --- Builder API -----------------------------------------
    //
    // These tests exercise the acceptance criteria: a
    // multi-level tree expressed as a single chained expression, identical
    // `AtlasNodeId`s to the equivalent imperative sequence, and that the
    // low-level API (PR #14) is untouched by the addition.

    /// Acceptance criterion: a 3-level hierarchy mixing leaves and
    /// containers can be expressed as a single chained expression.
    #[test]
    fn builder_expresses_three_level_mixed_hierarchy() {
        use LayoutAtlasBuilder as B;

        let mut atlas = LayoutAtlas::new();

        let root = atlas
            .build_root(B::container(
                ContainerStyle::new(Some(300.0), Some(200.0)),
                [
                    B::leaf(LeafSize::new(50.0, 50.0)),
                    B::container(
                        ContainerStyle::default(),
                        [
                            B::leaf(LeafSize::new(20.0, 20.0)),
                            B::container(
                                ContainerStyle::default(),
                                [B::leaf(LeafSize::new(10.0, 10.0))],
                            ),
                        ],
                    ),
                ],
            ))
            .unwrap();

        atlas.compute(Viewport::new(800.0, 600.0)).unwrap();

        // 1 outer container + 1 leaf + 1 inner container + 1 leaf +
        // 1 innermost container + 1 leaf = 6 nodes.
        assert_eq!(atlas.node_count(), 6);
        let root_rect = atlas.resolved_rect(root).unwrap().expect("root rect");
        assert_f32_eq(root_rect.width, 300.0);
        assert_f32_eq(root_rect.height, 200.0);
    }

    /// Acceptance criterion: the builder produces the same `AtlasNodeId`s
    /// as the equivalent imperative sequence.
    ///
    /// Each sequence runs on its own *fresh* atlas. We compare only the
    /// Taffy-level `node_id` (not the full `AtlasNodeId`): `atlas_id` is
    /// instance-specific by design, so two different atlases can never
    /// produce an equal `AtlasNodeId`, regardless of how their trees were
    /// built. A fresh Taffy tree's internal slot allocation depends only
    /// on insertion order — there are no removals involved here to make
    /// slot reuse a confounding factor — so identical `node_id`s on two
    /// fresh atlases is exactly the signal that `build` issues
    /// `add_leaf`/`add_container` calls in the same order the imperative
    /// version does.
    #[test]
    fn build_produces_same_ids_as_imperative_sequence() {
        use LayoutAtlasBuilder as B;

        // Imperative sequence: children before parents, leaf before the
        // sibling container, matching the builder's depth-first order.
        let mut imperative = LayoutAtlas::new();
        let leaf_a = imperative.add_leaf(LeafSize::new(50.0, 50.0)).unwrap();
        let leaf_b = imperative.add_leaf(LeafSize::new(20.0, 20.0)).unwrap();
        let inner = imperative
            .add_container(ContainerStyle::default(), &[leaf_b])
            .unwrap();
        let root_imperative = imperative
            .add_container(
                ContainerStyle::new(Some(300.0), Some(200.0)),
                &[leaf_a, inner],
            )
            .unwrap();

        // Equivalent builder sequence, on a separate fresh atlas.
        let mut built = LayoutAtlas::new();
        let root_builder = built
            .build(B::container(
                ContainerStyle::new(Some(300.0), Some(200.0)),
                [
                    B::leaf(LeafSize::new(50.0, 50.0)),
                    B::container(
                        ContainerStyle::default(),
                        [B::leaf(LeafSize::new(20.0, 20.0))],
                    ),
                ],
            ))
            .unwrap();

        assert_eq!(root_builder.node_id, root_imperative.node_id);
    }

    /// Acceptance criterion (paraphrased): the low-level API from PR #14
    /// is unchanged — `add_leaf`/`add_container`/`set_root` still work
    /// exactly as before, with no signature or behavior change introduced
    /// by adding the builder.
    #[test]
    fn low_level_api_unchanged_alongside_builder() {
        let mut atlas = LayoutAtlas::new();
        let leaf = atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();
        let root = atlas
            .add_container(ContainerStyle::default(), &[leaf])
            .unwrap();
        atlas.set_root(root).unwrap();
        atlas.compute(Viewport::new(100.0, 100.0)).unwrap();

        let rect = atlas.resolved_rect(leaf).unwrap().unwrap();
        assert_f32_eq(rect.width, 10.0);
        assert_f32_eq(rect.height, 10.0);
    }

    /// `build` rejects nodes the same way `add_leaf`/`add_container` do
    /// once the atlas has moved to `Computed` — the panic contract is
    /// inherited, not bypassed by going through the builder.
    #[test]
    #[should_panic(expected = "called while in Computed state")]
    fn build_after_compute_panics() {
        use LayoutAtlasBuilder as B;

        let mut atlas = LayoutAtlas::new();
        let leaf = atlas.add_leaf(LeafSize::new(10.0, 10.0)).unwrap();
        atlas.set_root(leaf).unwrap();
        atlas.compute(Viewport::new(100.0, 100.0)).unwrap();

        let _ = atlas.build(B::leaf(LeafSize::new(5.0, 5.0)));
    }

    /// A single `build_root` call replaces the imperative
    /// add-then-set_root pair and leaves the atlas ready to compute.
    #[test]
    fn build_root_sets_root_and_allows_compute() {
        use LayoutAtlasBuilder as B;

        let mut atlas = LayoutAtlas::new();
        let root = atlas
            .build_root(B::leaf(LeafSize::new(42.0, 24.0)))
            .unwrap();

        assert_eq!(atlas.root(), Some(root));
        atlas.compute(Viewport::new(100.0, 100.0)).unwrap();

        let rect = atlas.resolved_rect(root).unwrap().unwrap();
        assert_f32_eq(rect.width, 42.0);
        assert_f32_eq(rect.height, 24.0);
    }

    /// Deeply-nested trees (beyond the 3-level acceptance criterion) build
    /// correctly — verifies the recursive `build` walk doesn't have a
    /// depth assumption baked in.
    #[test]
    fn builder_handles_deeply_nested_tree() {
        use LayoutAtlasBuilder as B;

        let mut atlas = LayoutAtlas::new();

        // 5 levels of single-child containers, terminating in a leaf.
        let spec = B::container(
            ContainerStyle::default(),
            [B::container(
                ContainerStyle::default(),
                [B::container(
                    ContainerStyle::default(),
                    [B::container(
                        ContainerStyle::default(),
                        [B::leaf(LeafSize::new(15.0, 15.0))],
                    )],
                )],
            )],
        );

        let root = atlas.build_root(spec).unwrap();
        atlas.compute(Viewport::new(100.0, 100.0)).unwrap();

        // 4 containers + 1 leaf.
        assert_eq!(atlas.node_count(), 5);
        let root_rect = atlas.resolved_rect(root).unwrap().unwrap();
        assert_f32_eq(root_rect.width, 15.0);
        assert_f32_eq(root_rect.height, 15.0);
    }

    #[test]
    fn column_direction_gap_and_padding_lay_children_vertically() {
        let mut atlas = LayoutAtlas::new();
        let a = atlas.add_leaf(LeafSize::new(40.0, 20.0)).unwrap();
        let b = atlas.add_leaf(LeafSize::new(40.0, 20.0)).unwrap();
        let col = atlas
            .add_container(
                ContainerStyle::new(Some(200.0), Some(200.0))
                    .with_direction(FlexDir::Column)
                    .with_gap(10.0)
                    .with_padding(Spacing::all(8.0)),
                &[a, b],
            )
            .unwrap();
        atlas.set_root(col).unwrap();
        atlas.compute(Viewport::new(200.0, 200.0)).unwrap();

        let ra = atlas.resolved_rect(a).unwrap().unwrap();
        let rb = atlas.resolved_rect(b).unwrap().unwrap();
        // Padding offsets the first child; the gap separates the two.
        assert_f32_eq(ra.x, 8.0);
        assert_f32_eq(ra.y, 8.0);
        assert_f32_eq(rb.y, 8.0 + 20.0 + 10.0); // padding + first height + gap
    }

    #[test]
    fn grow_distributes_main_axis_space() {
        let mut atlas = LayoutAtlas::new();
        let spacer = atlas
            .add_container(ContainerStyle::default().with_grow(1.0), &[])
            .unwrap();
        let fixed = atlas.add_leaf(LeafSize::new(40.0, 20.0)).unwrap();
        let row = atlas
            .add_container(
                ContainerStyle::new(Some(200.0), Some(50.0)).with_direction(FlexDir::Row),
                &[spacer, fixed],
            )
            .unwrap();
        atlas.set_root(row).unwrap();
        atlas.compute(Viewport::new(200.0, 50.0)).unwrap();

        let rs = atlas.resolved_rect(spacer).unwrap().unwrap();
        let rf = atlas.resolved_rect(fixed).unwrap().unwrap();
        // The grow:1 spacer eats the slack, pushing the fixed leaf to the end.
        assert_f32_eq(rs.width, 160.0); // 200 - 40 fixed
        assert_f32_eq(rf.x, 160.0);
    }
}
