//! Aggregated training options and their builder-style setters.
//!
//! `Options` groups the per-booster settings (`BoosterOptions`), the
//! per-tree settings (`BoostTreeOptions`) and the per-node settings
//! (`BoostNodeOptions`). Every field is exposed through a self-consuming
//! `set_*` method so callers can build a fully configured instance in a
//! single chain starting from [`Options::default`].

use crate::boost_node::BoostNodeOptions;
use crate::boost_tree::BoostTreeOptions;
use std::path::PathBuf;

/// Top-level options: the booster, tree and node settings together.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Options {
    /// Booster-level settings (objective, scoring, offload, parallelism).
    pub boost_options: BoosterOptions,
    /// Tree-level settings (subsampling, depth/leaf limits, pool policy).
    pub tree_options: BoostTreeOptions,
    /// Node-level settings (regularization, split thresholds).
    pub node_options: BoostNodeOptions,
}

/// Booster-level options shared across every tree in a run.
#[derive(Clone, Debug, PartialEq)]
pub struct BoosterOptions {
    /// The objective the booster optimizes.
    pub objective: ObjectiveKind,
    /// The initial constant prediction before any tree is added.
    pub base_score: f32,
    /// Where to persist large-dataset artifacts, if offloading is used.
    pub persist_path: Option<PathBuf>,
    /// Whether the large (on-disk) update path is used.
    pub use_offload: bool,
    /// Whether the parallel execution paths are enabled.
    pub parallel: bool,
    /// Seed for all deterministically reproducible random draws.
    pub seed: u64,
}

impl Default for BoosterOptions {
    fn default() -> Self {
        Self {
            objective: ObjectiveKind::SquaredLoss,
            base_score: 0.0,
            persist_path: None,
            use_offload: false,
            parallel: true,
            seed: 0,
        }
    }
}

/// The objective a booster minimizes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ObjectiveKind {
    /// Binary cross-entropy (classification).
    LogLoss,
    /// Squared error (regression).
    SquaredLoss,
}

impl Options {
    // ----- `Options` setters ---------------------------------------------------------------------

    /// Replace the whole booster-level options block.
    pub fn set_boost_options(mut self, boost_options: BoosterOptions) -> Self {
        self.boost_options = boost_options;
        self
    }

    /// Replace the whole tree-level options block.
    pub fn set_tree_options(mut self, tree_options: BoostTreeOptions) -> Self {
        self.tree_options = tree_options;
        self
    }

    /// Replace the whole node-level options block.
    pub fn set_node_options(mut self, node_options: BoostNodeOptions) -> Self {
        self.node_options = node_options;
        self
    }

    // ----- `BoosterOptions` setters --------------------------------------------------------------

    /// Set the objective (squared error or binary cross-entropy).
    pub fn set_objective(mut self, objective: ObjectiveKind) -> Self {
        self.boost_options.objective = objective;
        self
    }

    /// Set the initial constant prediction (base score).
    pub fn set_base_score(mut self, base_score: f32) -> Self {
        self.boost_options.base_score = base_score;
        self
    }

    /// Set the on-disk artifact path (or `None` to disable persistence).
    pub fn set_persist_path(mut self, persist_path: Option<PathBuf>) -> Self {
        self.boost_options.persist_path = persist_path;
        self
    }

    /// Set whether the large on-disk update path is used.
    pub fn set_use_offload(mut self, use_offload: bool) -> Self {
        self.boost_options.use_offload = use_offload;
        self
    }

    /// Set whether the parallel execution paths are enabled.
    pub fn set_parallel(mut self, parallel: bool) -> Self {
        self.boost_options.parallel = parallel;
        self
    }

    /// Set the seed for reproducible random draws.
    pub fn set_seed(mut self, seed: u64) -> Self {
        self.boost_options.seed = seed;
        self
    }

    // ----- `BoostTreeOptions` setters ------------------------------------------------------------

    /// Set the learning rate (per-tree update step size).
    pub fn set_learning_rate(mut self, learning_rate: f64) -> Self {
        self.tree_options.learning_rate = learning_rate;
        self
    }

    /// Set the row subsampling fraction (`< 1.0` samples a subset).
    pub fn set_subsample(mut self, subsample: f64) -> Self {
        self.tree_options.subsample = subsample;
        self
    }

    /// Set the per-tree feature subsampling fraction.
    pub fn set_colsample_bytree(mut self, colsample_bytree: f64) -> Self {
        self.tree_options.colsample_bytree = colsample_bytree;
        self
    }

    /// Set the per-level feature subsampling fraction.
    pub fn set_colsample_bylevel(mut self, colsample_bylevel: f64) -> Self {
        self.tree_options.colsample_bylevel = colsample_bylevel;
        self
    }

    /// Set the per-node feature subsampling fraction.
    pub fn set_colsample_bynode(mut self, colsample_bynode: f64) -> Self {
        self.tree_options.colsample_bynode = colsample_bynode;
        self
    }

    /// Set the maximum tree depth (`0` for no limit).
    pub fn set_max_depth(mut self, max_depth: usize) -> Self {
        self.tree_options.max_depth = max_depth;
        self
    }

    /// Set the maximum number of leaves (`0` for no limit).
    pub fn set_max_leaves(mut self, max_leaves: usize) -> Self {
        self.tree_options.max_leaves = max_leaves;
        self
    }

    /// Set whether the pool pops highest-gain nodes first.
    pub fn set_loss_guide(mut self, loss_guide: bool) -> Self {
        self.tree_options.loss_guide = loss_guide;
        self
    }

    // ----- `BoostNodeOptions` setters ------------------------------------------------------------

    /// Set the L1 (alpha) regularizer on leaf weights.
    pub fn set_reg_alpha(mut self, reg_alpha: f64) -> Self {
        self.node_options.reg_alpha = reg_alpha;
        self
    }

    /// Set the L2 (lambda) regularizer on leaf weights.
    pub fn set_reg_lambda(mut self, reg_lambda: f64) -> Self {
        self.node_options.reg_lambda = reg_lambda;
        self
    }

    /// Set the minimum gain required to take a split.
    pub fn set_min_split_gain(mut self, min_split_gain: f64) -> Self {
        self.node_options.min_split_gain = min_split_gain;
        self
    }

    /// Set the minimum hessian sum allowed in one leaf.
    pub fn set_min_child_weight(mut self, min_child_weight: f64) -> Self {
        self.node_options.min_child_weight = min_child_weight;
        self
    }

    /// Set whether missing values may drive their own split.
    pub fn set_allow_missing_splits(mut self, allow_missing_splits: bool) -> Self {
        self.node_options.allow_missing_splits = allow_missing_splits;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_all_options() {
        let path = std::path::PathBuf::from("/tmp/gboost");
        let boost_options = BoosterOptions {
            objective: ObjectiveKind::LogLoss,
            base_score: 1.0,
            persist_path: Some(path.clone()),
            use_offload: true,
            parallel: false,
            seed: 42,
        };
        let tree_options = BoostTreeOptions {
            learning_rate: 0.1,
            subsample: 0.5,
            colsample_bytree: 0.9,
            colsample_bylevel: 0.8,
            colsample_bynode: 0.7,
            max_depth: 3,
            max_leaves: 8,
            loss_guide: true,
        };
        let node_options = BoostNodeOptions {
            reg_alpha: 0.3,
            reg_lambda: 1.2,
            min_split_gain: 0.4,
            min_child_weight: 5.0,
            allow_missing_splits: true,
        };

        let actual = Options::default()
            .set_objective(ObjectiveKind::LogLoss)
            .set_base_score(1.0)
            .set_persist_path(Some(path.clone()))
            .set_use_offload(true)
            .set_parallel(false)
            .set_seed(42)
            .set_learning_rate(0.1)
            .set_subsample(0.5)
            .set_colsample_bytree(0.9)
            .set_colsample_bylevel(0.8)
            .set_colsample_bynode(0.7)
            .set_max_depth(3)
            .set_max_leaves(8)
            .set_loss_guide(true)
            .set_reg_alpha(0.3)
            .set_reg_lambda(1.2)
            .set_min_split_gain(0.4)
            .set_min_child_weight(5.0)
            .set_allow_missing_splits(true);
        assert_eq!(
            actual,
            Options {
                boost_options: boost_options.clone(),
                tree_options: tree_options.clone(),
                node_options: node_options.clone(),
            }
        );

        let actual_alt = Options::default()
            .set_boost_options(boost_options)
            .set_tree_options(tree_options)
            .set_node_options(node_options);
        assert_eq!(actual_alt, actual);
    }
}
