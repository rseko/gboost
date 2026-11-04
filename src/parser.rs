//! Text parser for a serialized gradient-boosting model.
//!
//! Functions:
//! - `parse_model`: split the text into features and trees.
//! - `parse_features` / `parse_feature`: read one feature header line.
//! - `parse_trees` / `parse_tree`: read one tree's node list.
//! - `parse_node` / `parse_node_split` / `parse_node_leaf`: build a single node.
//! - `parse_predicate`: turn a `name <= value` predicate into feature/bin ids.
//! - `parse_map`: turn `k=v, k=v` values into a `HashMap`.

use crate::dataset::{Feature, FeatureKind};
use crate::tree::{Node, NodeKind, NodeSplit, Tree};
use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;

/// Parse a serialized model into its feature descriptors and the list of trees.
pub fn parse_model(text: &str) -> Result<(Vec<Feature>, Vec<Tree>)> {
    let (features_block, trees_block) = text
        .split_once("\n\n")
        .ok_or_else(|| anyhow!("invalid model"))?;
    let trees_offset = features_block.lines().count() + 2;

    let features = parse_features(features_block, 1)?;
    let trees = parse_trees(trees_block, &features, trees_offset)?;
    Ok((features, trees))
}

/// Parse the feature header block: one line per feature, `//` comments skipped.
pub fn parse_features(text: &str, offset: usize) -> Result<Vec<Feature>> {
    text.lines()
        .enumerate()
        // 1-indexed line numbers so errors point at the right source line.
        .map(|(i, s)| (offset + i, s.trim()))
        .filter(|(_, s)| !s.starts_with("//"))
        .map(|(line, s)| parse_feature(s).map_err(|e| e.context(line)))
        .collect()
}

/// Parse one feature header line of the form `name kind [bins]`.
fn parse_feature(text: &str) -> Result<Feature> {
    let parts: Vec<&str> = text.splitn(3, ' ').collect();
    if parts.len() != 3 {
        bail!("invalid feature");
    }

    let name = Some(parts[0].to_string());
    let kind = match parts[1] {
        "linear" => FeatureKind::Linear,
        "onehot" => FeatureKind::OneHot,
        _ => bail!("invalid feature type"),
    };
    let bins: Vec<f32> = parts[2]
        .trim_matches(['[', ']'])
        .split(',')
        .map(|s| s.trim().parse::<f32>().map_err(|e| anyhow!(e)))
        .collect::<Result<_>>()?;

    Ok(match kind {
        FeatureKind::Linear => Feature::linear(name, bins),
        FeatureKind::OneHot => Feature::onehot(name, bins),
    })
}

/// Parse every blank-line-separated tree in the trees block.
pub fn parse_trees(text: &str, features: &[Feature], offset: usize) -> Result<Vec<Tree>> {
    if text.is_empty() {
        return Ok(vec![]);
    }
    text.split("\n\n")
        // Track a running line counter so each tree reports its own start line.
        .scan(offset, |line, chunk| {
            let start = *line;
            *line += chunk.lines().count() + 1;
            Some((chunk, start))
        })
        .map(|(chunk, start)| parse_tree(chunk, features, start))
        .collect()
}

/// Parse one tree's node list into a flat, index-addressed `Tree`.
fn parse_tree(text: &str, features: &[Feature], offset: usize) -> Result<Tree> {
    let lines = text
        .lines()
        .enumerate()
        // Absolute line numbers (against the model text) for error context.
        .map(|(i, s)| (offset + i, s.trim_end()))
        .filter(|(_, s)| !s.starts_with("//"))
        .collect::<Vec<_>>();

    // Nodes may appear out of order; index them once, then verify all slots filled.
    let mut nodes: Vec<Option<Node>> = vec![None; lines.len()];
    for (line, node_text) in lines {
        let (index, node) = parse_node(node_text, features).map_err(|e| e.context(line))?;
        if index >= nodes.len() || nodes[index].is_some() {
            return Err(anyhow!("invalid node id: {index}").context(line));
        }
        nodes[index] = Some(node);
    }

    let nodes = nodes.into_iter().map(|x| x.unwrap()).collect();
    Tree::new(nodes)
}

/// Parse one node line of the form `<indent>[idx]:body` into its id and node.
fn parse_node(text: &str, features: &[Feature]) -> Result<(usize, Node)> {
    let (head, tail) = text
        .split_once(':')
        .ok_or_else(|| anyhow!("invalid node"))?;
    let node_id = head.trim_start();
    let index = node_id.parse::<usize>()?;

    let mut node = if tail.starts_with('[') {
        parse_node_split(tail, features)?
    } else {
        parse_node_leaf(tail)?
    };
    // The leading indentation (4 spaces per level) before the id encodes depth.
    node.depth = (head.len() - node_id.len()) as u32 / 4;
    Ok((index, node))
}

/// Parse a split node's bracketed predicate and its `yes/no/gain/cover` fields.
fn parse_node_split(text: &str, features: &[Feature]) -> Result<Node> {
    let (head, tail) = text
        .trim_start_matches('[')
        .split_once(']')
        .ok_or_else(|| anyhow!("invalid split"))?;

    // An optional trailing ` | null` makes the split also route missing values.
    let (pred, with_missing) = head
        .strip_suffix(" | null")
        .map(|s| (s, true))
        .unwrap_or((head, false));
    let (feature_id, bin_id) = parse_predicate(pred, features)?;
    let fields = parse_map(tail)?;

    Ok(Node {
        kind: NodeKind::Split(NodeSplit {
            feature_id,
            bin_id,
            is_range: features[feature_id].kind() == FeatureKind::Linear,
            // `bin_id == 0` is itself the "missing" sentinel, so it forces with_missing.
            with_missing: with_missing || bin_id == 0,
            gain: fields.get("gain").unwrap_or(&"").parse::<f32>()?,
            positive_id: fields.get("yes").unwrap_or(&"").parse::<usize>()?,
            negative_id: fields.get("no").unwrap_or(&"").parse::<usize>()?,
        }),
        depth: 0,
        cover: fields.get("cover").unwrap_or(&"0").parse::<f32>()?,
    })
}

/// Parse a leaf node's `value=...,cover=...` fields.
fn parse_node_leaf(text: &str) -> Result<Node> {
    let fields = parse_map(text)?;
    let value = fields.get("value").unwrap_or(&"").parse::<f32>()?;

    Ok(Node {
        kind: NodeKind::Leaf(value),
        depth: 0,
        cover: fields.get("cover").unwrap_or(&"0").parse::<f32>()?,
    })
}

/// Resolve a `name <op> value` predicate to a (feature id, 1-indexed bin id).
fn parse_predicate(text: &str, features: &[Feature]) -> Result<(usize, usize)> {
    let parts: Vec<&str> = text.splitn(3, ' ').collect();
    if parts.len() != 3 {
        bail!("invalid predicate");
    }

    let feature_id = features
        .iter()
        .position(|f| f.name() == parts[0])
        .ok_or_else(|| anyhow!("invalid predicate name"))?;
    // `name is null` tests for a missing value (bin id 0).
    if parts[1..] == ["is", "null"] {
        return Ok((feature_id, 0));
    }

    let is_valid = match features[feature_id].kind() {
        FeatureKind::Linear => parts[1] == "<=",
        FeatureKind::OneHot => parts[1] == "==",
    };
    if !is_valid {
        bail!("invalid predicate type");
    }

    let value = parts[2].parse::<f32>()?;
    let bin_id = features[feature_id]
        .bins()
        .iter()
        .position(|x| *x == value)
        .ok_or_else(|| anyhow!("invalid predicate value"))?;
    // position() is 0-based; the rest of the crate expects a 1-indexed bin id.
    Ok((feature_id, bin_id + 1))
}

/// Parse `k=v, k=v, ...` values into a `HashMap`.
fn parse_map(text: &str) -> Result<HashMap<&str, &str>> {
    text.split(',')
        .map(|part| {
            let (k, v) = part
                .split_once('=')
                .ok_or_else(|| anyhow!("invalid key/value"))?;
            Ok((k.trim(), v.trim()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example(pred: &str) -> String {
        format!(
            concat!(
                "// [features]\n",
                "name onehot [1, 2, 3]\n",
                "age linear [10.0, 20.0, 30.0]\n\n",
                "// [trees]\n",
                "0:[{}] yes=1,no=2,gain=0.1,cover=5\n",
                "    1:value=0.5,cover=2\n",
                "    2:value=-0.5,cover=3\n\n",
                "0:value=1.0,cover=5"
            ),
            pred
        )
    }

    fn example_root(feature_id: usize, bin_id: usize, with_missing: bool) -> Node {
        Node {
            kind: NodeKind::Split(NodeSplit {
                feature_id,
                bin_id,
                is_range: feature_id == 1,
                with_missing,
                gain: 0.1,
                positive_id: 1,
                negative_id: 2,
            }),
            depth: 0,
            cover: 5.0,
        }
    }

    #[test]
    fn parse_valid() {
        let (features, trees) = parse_model(&example("age <= 20")).unwrap();
        assert_eq!(
            features,
            vec![
                Feature::onehot(Some("name".into()), vec![1.0, 2.0, 3.0]),
                Feature::linear(Some("age".into()), vec![10.0, 20.0, 30.0]),
            ]
        );
        assert_eq!(trees.len(), 2);
        assert_eq!(trees[0].get(0), &example_root(1, 2, false));

        let example_leaf = |value: f32, cover: f32, depth: u32| Node {
            kind: NodeKind::Leaf(value),
            depth,
            cover,
        };
        assert_eq!(trees[0].get(1), &example_leaf(0.5, 2.0, 1));
        assert_eq!(trees[0].get(2), &example_leaf(-0.5, 3.0, 1));
        assert_eq!(trees[1].get(0), &example_leaf(1.0, 5.0, 0));

        let (_, trees) = parse_model(&example("name == 3")).unwrap();
        assert_eq!(trees[0].get(0), &example_root(0, 3, false));
    }

    #[test]
    fn parse_with_missing() {
        let (_, trees) = parse_model(&example("age is null")).unwrap();
        assert_eq!(trees[0].get(0), &example_root(1, 0, true));

        let (_, trees) = parse_model(&example("name == 1 | null")).unwrap();
        assert_eq!(trees[0].get(0), &example_root(0, 1, true));
    }

    #[test]
    fn parse_no_trees() {
        let content = example("").split_once("// [trees]").unwrap().0.to_owned();
        let (_, trees) = parse_model(&content).unwrap();
        assert!(trees.is_empty());
    }

    #[test]
    fn parse_invalid_features() {
        let test_err =
            |s: &str, e: &str| assert_eq!(format!("{:#}", parse_model(s).unwrap_err()), e);
        test_err("", "invalid model");
        test_err("name\n\n", "1: invalid feature");
        test_err("name type []\n\n", "1: invalid feature type");
        test_err("name linear [x]\n\n", "1: invalid float literal");
    }

    #[test]
    fn parse_invalid_trees() {
        let test_err = |s: &str, e: &str| {
            let err = parse_model(&format!("a linear [1]\nb onehot [2]\n\n{s}")).unwrap_err();
            assert_eq!(format!("{err:#}",), e)
        };
        test_err("1:value=0.1,cover=1", "4: invalid node id: 1");
        test_err(
            "0:value=0.1,cover=1\n0:value=0.2,cover=2",
            "5: invalid node id: 0",
        );
        test_err("0", "4: invalid node");
        test_err("?:", "4: invalid digit found in string");
        test_err("0:value", "4: invalid key/value");
        test_err("0:value=?", "4: invalid float literal");
        test_err("0:value=1,cover=?", "4: invalid float literal");

        test_err("0:[", "4: invalid split");
        test_err("0:[a]", "4: invalid predicate");
        test_err("0:[c <= 1]", "4: invalid predicate name");
        test_err("0:[b <= 1]", "4: invalid predicate type");
        test_err("0:[a <= 0]", "4: invalid predicate value");
        test_err("0:[a <= ?]", "4: invalid float literal");

        test_err("0:[a <= 1] yes,no", "4: invalid key/value");
        test_err("0:[a <= 1] yes=0,no=0,gain=?", "4: invalid float literal");
        test_err(
            "0:[a <= 1] yes=?,no=0,gain=1",
            "4: invalid digit found in string",
        );
        test_err(
            "0:[a <= 1] yes=0,no=?,gain=1",
            "4: invalid digit found in string",
        );
        test_err(
            "0:[a <= 1] yes=0,no=0,gain=1,cover=?",
            "4: invalid float literal",
        );
    }
}
