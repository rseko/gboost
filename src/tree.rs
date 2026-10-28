//! A single decision tree produced by gradient boosting.
//!
//! A tree is a flat, ordered list of `Node`s (node 0 is the root), where each
//! node is either a `Split` on one feature bin or a `Leaf` holding a value.
//! Rows are routed from the root to a leaf via [`Tree::find_leaf`].
//!
//! Structures:
//! - `Tree`: the flat list of nodes; node 0 is the root.
//! - `Node`: a split or leaf, its depth, and its coverage.
//! - `NodeKind`: `Split(NodeSplit)` or `Leaf(f32)`.
//! - `NodeSplit`: one split condition on a feature bin plus children and gain.

use crate::dataset::Feature;
use anyhow::{Result, bail};
use std::collections::HashSet;

/// A decision tree: a flat, ordered list of [`Node`]s, with node 0 as root.
#[derive(Clone, Debug)]
pub struct Tree {
    /// All nodes, parent before children (preorder).
    nodes: Vec<Node>,
}

/// A tree node: either an internal split or a leaf value.
#[derive(Clone, Debug)]
pub struct Node {
    /// Whether the node is a split or a leaf, and its payload.
    pub kind: NodeKind,
    /// Distance from the root (root is depth 0).
    pub depth: u32,
    /// Coverage of this node: the hessian sum of the rows reaching it.
    pub cover: f32,
}

/// The kind of payload a [`Node`] carries.
#[derive(Clone, Debug)]
pub enum NodeKind {
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
    /// A split with `bin_id == 0` and `with_missing == false` is invalid.
    pub bin_id: usize,
    /// `true` for a linear/range split (`bin <= bin_id`).
    /// `false` for an exact one-hot match (`bin == bin_id`).
    pub is_range: bool,
    /// If the row's value is missing, route it to the positive branch.
    pub with_missing: bool,
    /// Improvement in the objective gained by this split.
    pub gain: f32,
    /// Child node taken when the split matches.
    pub positive_id: usize,
    /// Child node taken when the split does not match.
    pub negative_id: usize,
}

impl Tree {
    /// Construct and validate the tree.
    pub fn new(nodes: Vec<Node>) -> Result<Self> {
        let tree = Tree { nodes };
        if tree.nodes.is_empty() {
            bail!("tree has no nodes");
        }
        if !tree.is_valid() {
            bail!("tree structure is invalid");
        }
        Ok(tree)
    }

    /// Get a tree node by index.
    pub fn get(&self, index: usize) -> &Node {
        &self.nodes[index]
    }

    /// Route one row from the root to its leaf.
    ///
    /// `get_bin` resolves the row's bin for a feature column: `get_bin(feature_id)`
    /// returns the 1-indexed bin, where 0 denotes a missing value (routed by
    /// `NodeSplit::with_missing`). Follows splits until reaching a leaf.
    pub fn find_leaf<F>(&self, get_bin: F) -> &Node
    where
        F: Fn(usize) -> usize,
    {
        let mut node = &self.nodes[0];
        while let NodeKind::Split(split) = &node.kind {
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
        let mut stack = Vec::from([0]);
        let mut result = Vec::<String>::new();
        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx];
            let indent = "    ".repeat(node.depth as usize);
            result.push(match &node.kind {
                NodeKind::Split(split) => {
                    // Push negative first so the positive child prints first.
                    stack.push(split.negative_id);
                    stack.push(split.positive_id);
                    format!(
                        "{}{}:[{}] yes={},no={},gain={},cover={}",
                        indent,
                        idx,
                        split.to_string(features.and_then(|x| x.get(split.feature_id))),
                        split.positive_id,
                        split.negative_id,
                        split.gain,
                        node.cover
                    )
                }
                NodeKind::Leaf(value) => {
                    format!("{}{}:value={},cover={}", indent, idx, value, node.cover)
                }
            });
        }
        result.join("\n")
    }

    /// Verify that the tree is valid (not empty, in-bound indices, no loops).
    fn is_valid(&self) -> bool {
        let mut visited = HashSet::new();
        let mut stack = Vec::from([0]);
        while let Some(idx) = stack.pop() {
            if !visited.insert(idx) || idx >= self.nodes.len() {
                return false;
            }
            if let NodeKind::Split(split) = &self.nodes[idx].kind {
                for next_id in [split.positive_id, split.negative_id] {
                    stack.push(next_id);
                }
            }
        }
        visited.len() == self.nodes.len()
    }
}

impl Node {
    /// The leaf value, or `f32::NAN` when the node is a split.
    #[inline]
    pub fn get_value(&self) -> f32 {
        match &self.kind {
            NodeKind::Leaf(value) => *value,
            _ => f32::NAN,
        }
    }

    /// Default split node (used in tests).
    pub fn default_split(split: NodeSplit) -> Self {
        Self {
            kind: NodeKind::Split(split),
            depth: 0,
            cover: 0.0,
        }
    }

    /// Default leaf node (used in tests).
    pub fn default_leaf(value: f32) -> Self {
        Self {
            kind: NodeKind::Leaf(value),
            depth: 1,
            cover: 0.0,
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
                    info.name(),
                    if self.bin_id <= info.bins().len() {
                        info.bins()[self.bin_id - 1]
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
        let root = Node::default_split(NodeSplit {
            feature_id: 0,
            bin_id: 2,
            is_range,
            with_missing,
            positive_id: 1,
            negative_id: 2,
            gain: 9.9,
        });
        let left = Node::default_leaf(1.0);
        let right = Node::default_leaf(2.0);
        Tree::new(vec![root, left, right]).unwrap()
    }

    #[test]
    fn create_tree() {
        let tree = build_tree(false, false);
        assert!(tree.get(0).get_value().is_nan());

        let single_node_tree = Tree::new(vec![Node::default_leaf(0.0)]).unwrap();
        assert_eq!(single_node_tree.get(0).get_value(), 0.0);
    }

    #[test]
    fn invalid_tree() {
        let make_split = |l, r| {
            Node::default_split(NodeSplit {
                feature_id: 0,
                bin_id: 1,
                is_range: true,
                with_missing: false,
                positive_id: l,
                negative_id: r,
                gain: 0.0,
            })
        };

        let empty_tree = Tree::new(vec![]);
        assert_eq!(empty_tree.unwrap_err().to_string(), "tree has no nodes");

        let bad_tree_index = Tree::new(vec![make_split(1, 1)]);
        assert!(bad_tree_index.is_err());

        let bad_tree_loop = Tree::new(vec![make_split(0, 1), Node::default_leaf(0.0)]);
        assert!(bad_tree_loop.is_err());

        let bad_tree_extra = Tree::new(vec![Node::default_leaf(0.0); 2]);
        assert!(bad_tree_extra.is_err());
    }

    #[test]
    fn search_onehot() {
        let tree = build_tree(/*is_range*/ false, /*with_missing*/ true);
        assert_eq!(tree.find_leaf(|_| 0).get_value(), 1.);
        assert_eq!(tree.find_leaf(|_| 1).get_value(), 2.);
        assert_eq!(tree.find_leaf(|_| 2).get_value(), 1.);
        assert_eq!(tree.find_leaf(|_| 3).get_value(), 2.);
    }

    #[test]
    fn search_linear() {
        let tree = build_tree(/*is_range*/ true, /*with_missing*/ false);
        assert_eq!(tree.find_leaf(|_| 0).get_value(), 2.);
        assert_eq!(tree.find_leaf(|_| 1).get_value(), 1.);
        assert_eq!(tree.find_leaf(|_| 2).get_value(), 1.);
        assert_eq!(tree.find_leaf(|_| 3).get_value(), 2.);
    }

    #[test]
    fn tree_format() {
        let mut tree = build_tree(/*is_range*/ true, /*with_missing*/ true);
        assert_eq!(
            tree.to_string(None),
            "0:[0 <= 2 | null] yes=1,no=2,gain=9.9,cover=0\n    \
                1:value=1,cover=0\n    \
                2:value=2,cover=0"
        );
        if let NodeKind::Split(split) = &mut tree.nodes[0].kind {
            split.bin_id = 0;
        }
        assert!(format!("{tree}").starts_with("0:[is null]"));
    }

    #[test]
    fn tree_format_with_context() {
        let tree = build_tree(/*is_range*/ false, /*with_missing*/ false);
        let feature = Feature::onehot("name".into(), vec![2.0, 4.0]);
        assert_eq!(
            tree.to_string(Some(&[feature])),
            "0:[name == 4] yes=1,no=2,gain=9.9,cover=0\n    \
                1:value=1,cover=0\n    \
                2:value=2,cover=0"
        );
    }
}
