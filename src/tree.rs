//! A single decision tree produced by gradient boosting.
//!
//! A tree is a flat, ordered list of `Node`s (node 0 is the root), where each
//! node is either a `Split` on one feature bin or a `Leaf` holding a value.
//! Rows are routed from the root to a leaf via [`Tree::find_leaf`].
//!
//! Structures:
//! - `Tree`: the flat list of nodes; node 0 is the root.
//! - `Node`: a split or leaf, its depth, and its coverage.
//! - `NodeType`: `Split(NodeSplit)` or `Leaf(f32)`.
//! - `NodeSplit`: one split condition on a feature bin plus children and gain.

use crate::dataset::Feature;

/// A decision tree: a flat, ordered list of [`Node`]s, with node 0 as root.
#[derive(Clone, Debug)]
pub struct Tree {
    /// All nodes, parent before children (preorder).
    pub nodes: Vec<Node>,
}

/// A tree node: either an internal split or a leaf value.
#[derive(Clone, Debug)]
pub struct Node {
    /// Stable node identifier (matches its index in `Tree::nodes`).
    pub id: u32,
    /// Whether the node is a split or a leaf, and its payload.
    pub type_: NodeType,
    /// Distance from the root (root is depth 0).
    pub depth: u32,
    /// Coverage of this node: the hessian sum of the rows reaching it.
    pub cover: f32,
}

/// The kind of payload a [`Node`] carries.
#[derive(Clone, Debug)]
pub enum NodeType {
    /// Internal node: branch on a feature bin.
    Split(NodeSplit),
    /// Leaf node: the predicted value.
    Leaf(f32),
}

/// A split condition: route a row left or right based on its bin for one feature.
#[derive(Clone, Debug)]
pub struct NodeSplit {
    /// Which feature column this split tests.
    pub feature_id: usize,
    /// Target bin the row must match, 1-indexed (boundary is
    /// `Feature::bins[bin_id - 1]`). `0` denotes testing a *missing* value.
    pub bin_id: usize,
    /// `true` for a linear/range split (`bin <= bin_id`).
    /// `false` for an exact one-hot match (`bin == bin_id`).
    pub is_range: bool,
    /// If the row's value is missing, route it to the positive branch.
    pub with_missing: bool,
    /// Child node taken when the split matches.
    pub positive_id: usize,
    /// Child node taken when the split does not match.
    pub negative_id: usize,
    /// Improvement in the objective gained by this split.
    pub gain: f32,
}

impl Tree {
    /// Route one row from the root to its leaf.
    ///
    /// `get_bin` resolves the row's bin for a feature column: `get_bin(feature_id)`
    /// returns the 1-indexed bin, where 0 denotes a missing value (routed by
    /// `NodeSplit::with_missing`). Follows splits until reaching a leaf.
    pub fn find_leaf<F>(&self, get_bin: F) -> &Node
    where
        F: Fn(usize) -> usize,
    {
        assert!(!self.nodes.is_empty());
        let mut node = &self.nodes[0];
        while let NodeType::Split(split) = &node.type_ {
            let bin = get_bin(split.feature_id);
            let next_id = if split.matches(bin) {
                split.positive_id
            } else {
                split.negative_id
            };
            node = &self.nodes[next_id];
        }
        node
    }

    /// Human-readable tree representation, one node per line.
    ///
    /// Split lines show the feature predicate (using `Feature::name` and the
    /// boundary value when `features` is provided), children, gain and cover;
    /// leaves show their value and cover.
    pub fn to_string(&self, features: Option<&[Feature]>) -> String {
        let mut stack = Vec::new();
        if !self.nodes.is_empty() {
            stack.push(0);
        }
        let mut result = Vec::<String>::new();
        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx];
            let indent = "    ".repeat(node.depth as usize);
            result.push(match &node.type_ {
                NodeType::Split(split) => {
                    // Push negative first so the positive child prints first.
                    stack.push(split.negative_id);
                    stack.push(split.positive_id);
                    format!(
                        "{}{}:[{}] yes={},no={},gain={},cover={}",
                        indent,
                        node.id,
                        split.to_string(features.and_then(|x| x.get(split.feature_id))),
                        split.positive_id,
                        split.negative_id,
                        split.gain,
                        node.cover
                    )
                }
                NodeType::Leaf(value) => {
                    format!("{}{}:value={},cover={}", indent, node.id, value, node.cover)
                }
            });
        }
        result.join("\n")
    }
}

impl Node {
    /// The leaf value, or `f32::NAN` when the node is a split.
    #[inline]
    pub fn get_value(&self) -> f32 {
        match &self.type_ {
            NodeType::Leaf(value) => *value,
            _ => f32::NAN,
        }
    }
}

impl NodeSplit {
    /// Whether `bin` (the row's 1-indexed bin for this feature) matches the split;
    /// `bin == 0` is a missing value and matches only if [`Self::with_missing`] is set.
    #[inline]
    pub fn matches(&self, bin: usize) -> bool {
        if bin != 0 {
            if self.is_range {
                bin <= self.bin_id
            } else {
                bin == self.bin_id
            }
        } else {
            self.with_missing
        }
    }

    /// The split predicate as text, e.g. `age <= 30 | null`.
    ///
    /// Uses the feature's name and threshold when `feature` is provided,
    /// otherwise falls back to the raw feature/bin ids.
    pub fn to_string(&self, feature: Option<&Feature>) -> String {
        if self.bin_id != 0 {
            let pred = if self.is_range { "<=" } else { "==" };
            let null = if self.with_missing { " | null" } else { "" };
            match feature {
                Some(info) => format!(
                    "{} {pred} {}{null}",
                    info.name,
                    if self.bin_id <= info.bins.len() {
                        info.bins[self.bin_id - 1]
                    } else {
                        f32::NAN
                    }
                ),
                None => format!("{} {pred} {}{null}", self.feature_id, self.bin_id),
            }
        } else {
            "is null".into()
        }
    }
}

impl std::fmt::Display for Tree {
    /// Format the tree using raw feature/bin ids (no `Feature` descriptors).
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.to_string(None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small two-level tree: one split at the root with two leaf children.
    fn build_tree(is_range: bool, with_missing: bool) -> Tree {
        let root = Node {
            id: 0,
            type_: NodeType::Split(NodeSplit {
                feature_id: 0,
                bin_id: 2,
                is_range,
                with_missing,
                positive_id: 1,
                negative_id: 2,
                gain: 0.0,
            }),
            depth: 0,
            cover: 5.0,
        };
        let left = Node {
            id: 1,
            type_: NodeType::Leaf(0.5),
            depth: 1,
            cover: 2.0,
        };
        let right = Node {
            id: 2,
            type_: NodeType::Leaf(-0.5),
            depth: 1,
            cover: 3.0,
        };
        Tree {
            nodes: vec![root, left, right],
        }
    }

    #[test]
    fn search_onehot() {
        let tree = build_tree(/*is_range*/ false, /*with_missing*/ true);
        assert_eq!(tree.find_leaf(|_| 0).id, 1);
        assert_eq!(tree.find_leaf(|_| 1).id, 2);
        assert_eq!(tree.find_leaf(|_| 2).id, 1);
        assert_eq!(tree.find_leaf(|_| 3).id, 2);
    }

    #[test]
    fn search_linear() {
        let tree = build_tree(/*is_range*/ true, /*with_missing*/ false);
        assert_eq!(tree.find_leaf(|_| 0).id, 2);
        assert_eq!(tree.find_leaf(|_| 1).id, 1);
        assert_eq!(tree.find_leaf(|_| 2).id, 1);
        assert_eq!(tree.find_leaf(|_| 3).id, 2);
    }

    #[test]
    fn tree_format() {
        let tree = build_tree(/*is_range*/ true, /*with_missing*/ true);
        assert_eq!(
            tree.to_string(None),
            "0:[0 <= 2 | null] yes=1,no=2,gain=0,cover=5\n    \
                1:value=0.5,cover=2\n    \
                2:value=-0.5,cover=3"
        );
    }
}
