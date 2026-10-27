//! Tree construction for one boosting iteration: build the root's histogram
//! and stats, then repeatedly pull a batch of nodes from the pool, find their
//! best splits, record the resulting split/leaf nodes, and push the produced
//! children back into the pool until it empties.
//!
//! Structures:
//! - `BoostTreeOptions`: subsampling, depth/leaf limits, pool policy.
//! - `TreeBuilder`: the stateful builder. `execute` runs the build to
//!   completion in one call; `process` + `finish` give batch-by-batch control.

use crate::boost_node::{
    BoostNodeOptions, SplitCandidate, SplitStats, Splittable, apply_l1, calc_gain, find_all_splits,
};
use crate::dataset::{Dataset, FeatureType, IndexT, Objective, RowIndex};
use crate::histogram::Histogram;
use crate::tree::{Node, NodeSplit, NodeType, Tree};
use rand::{rngs::StdRng, seq::SliceRandom};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Options controlling how a tree is built.
#[derive(Clone, Debug)]
pub struct BoostTreeOptions {
    /// Learning rate defines the update step size.
    pub learning_rate: f64,
    /// Select a subset of rows for new trees.
    pub subsample: f64,
    /// Select a subset of features for new trees.
    pub colsample_bytree: f64,
    /// Select a subset of features for each level.
    pub colsample_bylevel: f64,
    /// Select a subset of features for new splits.
    pub colsample_bynode: f64,
    /// Maximum depth of a tree (`0` means no limit).
    pub max_depth: usize,
    /// Maximum number of leaves in a tree (`0` means no limit).
    pub max_leaves: usize,
    /// Pick next node to split using gain value.
    pub loss_guide: bool,
}

impl Default for BoostTreeOptions {
    fn default() -> Self {
        Self {
            learning_rate: 0.3,
            subsample: 1.0,
            colsample_bytree: 1.0,
            colsample_bylevel: 1.0,
            colsample_bynode: 1.0,
            max_depth: 5,
            max_leaves: 0,
            loss_guide: false,
        }
    }
}

/// A stateful builder for one boosting tree.
///
/// Owns the work queue of unsplit nodes (`pool`), accumulates the finished
/// nodes in `result_nodes` keyed by node index, and applies the subsampling
/// and leaf/depth budgets from `tree_options`.
pub struct TreeBuilder<'a, T: Copy + Into<usize> + Sync, I: IndexT, O: Objective> {
    /// Queue of nodes still waiting for a split attempt.
    pool: BuilderPool<'a, I, O>,
    /// The dataset the tree is grown on.
    dataset: &'a Dataset<T>,
    /// Tree-level building options.
    tree_options: &'a BoostTreeOptions,
    /// Split-level options (used for finding the best split).
    node_options: &'a BoostNodeOptions,
    /// RNG for all subsampling draws.
    rng: &'a mut StdRng,

    /// Per-depth feature sets when `colsample_bylevel < 1.0`; entry `d` is
    /// the feature set sampled for nodes at depth `d` (level 0 is seeded in
    /// `new`; deeper levels are appended in `process`).
    columns_bylevel: Vec<Vec<usize>>,
    /// Finished nodes keyed by their node index; node 0 is the root and
    /// indices are allocated in breadth/preorder fashion by `next_index`.
    result_nodes: HashMap<usize, Node>,
    /// Remaining split budget; `max_leaves.overflowing_sub(1)` so that a
    /// limit of `k` leaves allows at most `k - 1` splits.
    leaf_budget: usize,
    /// Next free node index to assign.
    next_index: usize,
    /// Whether the split search and partitioning run on a thread pool.
    parallel: bool,
}

impl<'a, T, I, O> TreeBuilder<'a, T, I, O>
where
    T: Copy + Into<usize> + Sync,
    I: IndexT,
    O: Objective,
{
    /// Create a builder: subsample rows and features, build the root
    /// histogram and stats, and seed the pool with the root node.
    ///
    /// `index` is the per-row gradient/hessian for this iteration; the
    /// builder takes a mutable borrow of it, repartitioning it as nodes
    /// split. `parallel` controls the split-search and partition
    /// parallelism.
    pub fn new(
        mut index: &'a mut [RowIndex<I, O>],
        dataset: &'a Dataset<T>,
        tree_options: &'a BoostTreeOptions,
        node_options: &'a BoostNodeOptions,
        rng: &'a mut StdRng,
        parallel: bool,
    ) -> Self {
        assert!(dataset.is_valid());
        assert_eq!(index.len(), dataset.rows());

        // Row subsampling: keep only the first `sample_size` rows of a
        // shuffled `index` (their relative order in the slice no longer
        // matches row positions, hence `initial = false` below).
        let subsample = tree_options.subsample < 1.0;
        if subsample {
            let sample_size = get_sample_size(index.len(), tree_options.subsample);
            (index, _) = index.partial_shuffle(rng, sample_size);
        }

        // Feature subsampling for the whole tree.
        let mut columns = (0..dataset.columns()).collect::<Vec<_>>();
        if tree_options.colsample_bytree < 1.0 {
            let sample_size = get_sample_size(columns.len(), tree_options.colsample_bytree);
            columns = columns.partial_shuffle(rng, sample_size).0.to_vec();
        };

        // Seed level 0's feature set, used by `select_subsample` to derive
        // a fresh sampled set for each subsequent depth.
        let mut columns_bylevel = Vec::new();
        if tree_options.colsample_bylevel < 1.0 {
            columns_bylevel.push(columns.clone());
        }

        // The root's histogram over the sampled rows and features.
        let columns_set = HashSet::from_iter(columns);
        let histogram = Histogram::build(dataset, Some(&columns_set), index, !subsample, parallel);

        // Root stats: the histogram is uniform per feature, so summing any
        // one feature's bins yields the totals over all rows.
        let (gradient, hessian) = histogram
            .get(*columns_set.iter().next().unwrap())
            .iter()
            .fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
        let (l1, l2) = (node_options.reg_alpha, node_options.reg_lambda);
        let gain = calc_gain(gradient, hessian, l1, l2);

        let root = IndexedSplittable {
            index: 0,
            depth: 0,
            value: Splittable {
                index,
                histogram,
                stats: SplitStats::new(gradient, hessian, gain),
            },
        };
        // Pick the pool policy: best-gain-first vs. level-order.
        let pool = if tree_options.loss_guide {
            BuilderPool::LossGuide(BinaryHeap::from([root]))
        } else {
            BuilderPool::DepthWise(Vec::from([root]))
        };

        Self {
            pool,
            dataset,
            tree_options,
            node_options,
            rng,
            columns_bylevel,
            result_nodes: HashMap::new(),
            // `max_leaves = 0` wraps to `usize::MAX` (no limit).
            leaf_budget: tree_options.max_leaves.overflowing_sub(1).0,
            next_index: 1,
            parallel,
        }
    }

    /// Build the whole tree and return it, driving `process` batches
    /// until the work queue is empty and then consuming the builder to
    /// assemble the final `Tree`.
    ///
    /// This is the one-shot entry point for callers that do not need to
    /// observe or pause the build between batches; manual batch control is
    /// available via `process` followed by `finish`.
    pub fn execute(mut self) -> Tree {
        while self.process() {}
        self.finish()
    }

    /// Consume the builder and return the assembled tree.
    ///
    /// `result_nodes` must contain exactly the indices `0..next_index`
    /// (guaranteed after the pool has been drained, i.e. `process` has
    /// returned `false`).
    pub fn finish(mut self) -> Tree {
        let mut nodes = Vec::with_capacity(self.next_index);
        for idx in 0..self.next_index {
            nodes.push(self.result_nodes.remove(&idx).unwrap());
        }
        Tree { nodes }
    }

    /// Process one batch of queued nodes: find their best splits, record
    /// split and leaf nodes, split the rows, and push the children back
    /// into the pool.
    ///
    /// Returns `true` while queued nodes remain to process, `false` when
    /// the pool is exhausted (the tree is complete).
    pub fn process(&mut self) -> bool {
        let batch = self.get_next_batch();
        if batch.is_empty() {
            return false;
        }

        let depth_limit = self.tree_options.max_depth.overflowing_sub(1).0;
        let (can_split, cannot_split) = self.partition_batch(batch, depth_limit);
        self.leaf_budget -= can_split.len();

        // Record one split node per node that is being split. Each node's
        // two children receive the next two free indices, so `base_id` is
        // stable because `perform_splits` emits exactly two children per
        // entry of `can_split` in order.
        for (idx, (item, split)) in can_split.iter().enumerate() {
            let base_id = self.next_index + idx * 2;
            let feature_type = self.dataset.features[split.feature_id].type_;

            self.result_nodes.insert(
                item.index,
                Node {
                    type_: NodeType::Split(NodeSplit {
                        feature_id: split.feature_id,
                        bin_id: split.bin_id,
                        is_range: feature_type == FeatureType::Linear,
                        with_missing: split.with_missing,
                        positive_id: base_id,
                        negative_id: base_id + 1,
                        gain: item.value.stats.gain as f32,
                    }),
                    depth: item.depth as u32,
                    cover: item.value.stats.hessian as f32,
                },
            );
        }

        // Record one leaf per node that is not split: the L1-soft-thresholded
        // best weight scaled by the learning rate.
        for item in cannot_split {
            let (grad, hess) = (item.value.stats.gradient, item.value.stats.hessian);
            let (l1, l2) = (self.node_options.reg_alpha, self.node_options.reg_lambda);
            let weight = self.tree_options.learning_rate * apply_l1(grad, l1) / (hess + l2);

            self.result_nodes.insert(
                item.index,
                Node {
                    type_: NodeType::Leaf(-weight as f32),
                    depth: item.depth as u32,
                    cover: hess as f32,
                },
            );
        }

        // Split the rows of the split nodes and queue the children. Guarded
        // so a batch with zero split candidates (budget exhausted, depth
        // capped, or nothing splittable) terminates the build.
        if !can_split.is_empty() {
            let next_depth = can_split[0].0.depth + 1;
            for splittable in self.perform_splits(can_split, depth_limit) {
                let mut item = IndexedSplittable {
                    index: self.next_index,
                    depth: next_depth,
                    value: splittable,
                };
                self.select_subsample(&mut item);
                self.add_to_pool(item);
                self.next_index += 1;
            }
        }
        true
    }

    /// Partition a non-empty `batch` into nodes that found a split
    /// candidate and those that did not, enforcing the depth limit and the
    /// remaining leaf budget.
    ///
    /// If nothing can be split (budget exhausted or depth limit reached),
    /// the whole batch is returned as `(vec![], batch)`. Among the
    /// splittable nodes, the ones with the lowest gain are demoted to
    /// leaves when they exceed `leaf_budget`.
    fn partition_batch(
        &self,
        batch: Vec<IndexedSplittable<'a, I, O>>,
        depth_limit: usize,
    ) -> PartitionedResult<'a, I, O> {
        if self.leaf_budget == 0 || batch[0].depth > depth_limit {
            return (vec![], batch);
        }

        let (mut can_split, mut cannot_split) = (vec![], vec![]);
        let splittables = batch.iter().map(|x| &x.value).collect::<Vec<_>>();
        let results = find_all_splits(
            splittables,
            &self.dataset.features,
            self.node_options,
            self.parallel,
        );

        for (splittable, result) in batch.into_iter().zip(results) {
            if let Some(candidate) = result {
                can_split.push((splittable, candidate));
            } else {
                cannot_split.push(splittable);
            }
        }

        // Enforce the budget by keeping the highest-gain splits and
        // demoting the rest to leaves.
        can_split.sort_by(|a, b| a.1.gain.partial_cmp(&b.1.gain).unwrap());
        if can_split.len() > self.leaf_budget {
            let tail = can_split.split_off(self.leaf_budget);
            cannot_split.extend(tail.into_iter().map(|x| x.0));
        }

        (can_split, cannot_split)
    }

    /// Apply each node's chosen split, returning one child splittable per
    /// side (positive side first), in the order of `group`.
    ///
    /// Children of a node that may not be split again are returned with
    /// empty histograms to save memory - they will become leaves.
    fn perform_splits(
        &self,
        group: Vec<PreparedSplittable<'a, I, O>>,
        depth_limit: usize,
    ) -> Vec<Splittable<'a, I, O>> {
        group
            .into_iter()
            .flat_map(|(item, split)| {
                let final_split = self.leaf_budget == 0 || item.depth == depth_limit;
                let (left, right) =
                    item.value
                        .split(split, self.dataset, final_split, self.parallel);
                [left, right]
            })
            .collect()
    }

    /// Apply the level/node feature subsampling to a child node's
    /// histogram before it is queued.
    ///
    /// Per level: derive a freshly sampled feature set when a new depth is
    /// first reached, then restrict the histogram to that set. Per node:
    /// additionally restrict the histogram to a freshly sampled subset of
    /// whatever is left.
    fn select_subsample(&mut self, item: &mut IndexedSplittable<'a, I, O>) {
        if self.tree_options.colsample_bylevel < 1.0 {
            // First time this depth is reached: sample from the previous
            // level's set and record it.
            if item.depth == self.columns_bylevel.len() {
                let mut columns = self.columns_bylevel[item.depth - 1].clone();
                let sample_size =
                    get_sample_size(columns.len(), self.tree_options.colsample_bylevel);
                columns = columns.partial_shuffle(self.rng, sample_size).0.to_vec();
                self.columns_bylevel.push(columns);
            }

            let columns_set = HashSet::from_iter(self.columns_bylevel[item.depth].iter().cloned());
            item.value.histogram.compress(&columns_set);
        }

        if self.tree_options.colsample_bynode < 1.0 {
            let mut columns = Vec::from_iter(item.value.histogram.iter());
            let sample_size = get_sample_size(columns.len(), self.tree_options.colsample_bynode);
            let slice = columns.partial_shuffle(self.rng, sample_size).0;

            let columns_set = HashSet::from_iter(slice.iter().copied());
            item.value.histogram.compress(&columns_set);
        }
    }

    /// Push a node into the work queue.
    fn add_to_pool(&mut self, item: IndexedSplittable<'a, I, O>) {
        match &mut self.pool {
            BuilderPool::DepthWise(array) => array.push(item),
            BuilderPool::LossGuide(heap) => heap.push(item),
        }
    }

    /// Take the next batch of nodes from the work queue.
    ///
    /// The depth-wise queue is drained in full (one level at a time); the
    /// loss-guided queue pops its single highest-gain node.
    fn get_next_batch(&mut self) -> Vec<IndexedSplittable<'a, I, O>> {
        match &mut self.pool {
            BuilderPool::DepthWise(array) => std::mem::take(array),
            BuilderPool::LossGuide(heap) => match heap.pop() {
                Some(item) => vec![item],
                None => vec![],
            },
        }
    }
}

/// Number of items to keep for a subsample ratio, clamped to at least one.
fn get_sample_size(total: usize, ratio: f64) -> usize {
    ((total as f64 * ratio) as usize).max(1)
}

/// One entry in the work queue: a node's state plus its bookkeeping.
struct IndexedSplittable<'a, I: IndexT, O: Objective> {
    /// The node's index in the finished tree (allocated by the builder).
    index: usize,
    /// Distance of the node from the root.
    depth: usize,
    /// The node's rows, histogram, and stats.
    value: Splittable<'a, I, O>,
}

/// A node paired with the split candidate that will be applied to it.
type PreparedSplittable<'a, I, O> = (IndexedSplittable<'a, I, O>, SplitCandidate);

/// The outcome of partitioning a batch: the nodes to split followed by the
/// nodes that become leaves.
type PartitionedResult<'a, I, O> = (
    Vec<PreparedSplittable<'a, I, O>>,
    Vec<IndexedSplittable<'a, I, O>>,
);

/// The work queue policy.
enum BuilderPool<'a, I: IndexT, O: Objective> {
    /// Level-order queue: each batch is a whole level.
    DepthWise(Vec<IndexedSplittable<'a, I, O>>),
    /// Best-gain-first heap: each batch is a single node.
    LossGuide(BinaryHeap<IndexedSplittable<'a, I, O>>),
}

// Heap ordering so that `pop` returns the highest-gain node first.
impl<'a, I: IndexT, O: Objective> PartialOrd for IndexedSplittable<'a, I, O> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a, I: IndexT, O: Objective> Ord for IndexedSplittable<'a, I, O> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.value.stats.gain.total_cmp(&self.value.stats.gain)
    }
}

impl<'a, I: IndexT, O: Objective> PartialEq for IndexedSplittable<'a, I, O> {
    fn eq(&self, other: &Self) -> bool {
        self.value.stats.gain == other.value.stats.gain
    }
}

impl<'a, I: IndexT, O: Objective> Eq for IndexedSplittable<'a, I, O> {}
