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
use crate::dataset::{Dataset, FeatureKind, IndexT, Objective, RowIndex};
use crate::histogram::Histogram;
use crate::tree::{Node, NodeKind, NodeSplit, Tree};
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
    split_budget: usize,
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
        assert_eq!(index.len(), dataset.rows());

        // Row subsampling: keep only the first `sample_size` rows of a
        // shuffled `index` (their relative order in the slice no longer
        // matches row positions, hence `initial = false` below).
        let subsample = tree_options.subsample < 1.0;
        if subsample {
            let sample_size = get_sample_size(index.len(), tree_options.subsample);
            index = index.partial_shuffle(rng, sample_size).0;
        }

        // Feature subsampling for the whole tree.
        let mut columns = Vec::from_iter(0..dataset.columns());
        if tree_options.colsample_bytree < 1.0 {
            let sample_size = get_sample_size(columns.len(), tree_options.colsample_bytree);
            columns = columns.partial_shuffle(rng, sample_size).0.to_vec();
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

        // Create the builder instance.
        let mut builder = Self {
            pool: if tree_options.loss_guide {
                BuilderPool::LossGuide(BinaryHeap::new())
            } else {
                BuilderPool::DepthWise(Vec::new())
            },
            dataset,
            tree_options,
            node_options,
            rng,
            columns_bylevel: Vec::new(),
            result_nodes: HashMap::new(),
            // `max_leaves = 0` wraps to `usize::MAX` (no limit).
            split_budget: tree_options.max_leaves.overflowing_sub(1).0,
            next_index: 1,
            parallel,
        };

        // Add root node, apply subsampling.
        builder.add_to_pool(IndexedSplittable {
            index: 0,
            depth: 0,
            value: Box::new(Splittable {
                index,
                histogram,
                stats: SplitStats::new(gradient, hessian, gain),
                columns: None,
            }),
        });
        builder
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
        Tree::new(nodes).unwrap()
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
        self.split_budget -= can_split.len();

        // Record one split node per node that is being split. Each node's
        // two children receive the next two free indices, so `base_id` is
        // stable because `perform_splits` emits exactly two children per
        // entry of `can_split` in order.
        for (idx, (item, split)) in can_split.iter().enumerate() {
            let base_id = self.next_index + idx * 2;
            let feature_kind = self.dataset.features()[split.feature_id].kind();

            self.result_nodes.insert(
                item.index,
                Node {
                    kind: NodeKind::Split(NodeSplit {
                        feature_id: split.feature_id,
                        bin_id: split.bin_id,
                        is_range: feature_kind == FeatureKind::Linear,
                        with_missing: split.with_missing,
                        positive_id: base_id,
                        negative_id: base_id + 1,
                        gain: split.gain as f32,
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
                    kind: NodeKind::Leaf(-weight as f32),
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
                self.add_to_pool(IndexedSplittable {
                    index: self.next_index,
                    depth: next_depth,
                    value: Box::new(splittable),
                });
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
        if self.split_budget == 0 || batch[0].depth > depth_limit {
            return (vec![], batch);
        }

        let (mut can_split, mut cannot_split) = (vec![], vec![]);
        let splittables = Vec::from_iter(batch.iter().map(|x| x.value.as_ref()));
        let results = find_all_splits(
            splittables,
            self.dataset.features(),
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
        can_split.sort_by(|a, b| b.1.gain.partial_cmp(&a.1.gain).unwrap());
        if can_split.len() > self.split_budget {
            let tail = can_split.split_off(self.split_budget);
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
                let final_split = self.split_budget == 0 || item.depth == depth_limit;
                let (left, right) =
                    item.value
                        .split(split, self.dataset, final_split, self.parallel);
                [left, right]
            })
            .collect()
    }

    /// Apply column sampling to a node's candidate features: `colsample_bylevel`
    /// samples a set once per tree depth, then `colsample_bynode` samples from
    /// that node's columns; the result is stored in `item.value.columns`.
    fn select_subsample(&mut self, item: &mut IndexedSplittable<'a, I, O>) {
        let get_tree_columns = || Vec::from_iter(item.value.histogram.iter());

        if self.tree_options.colsample_bylevel < 1.0 {
            // First time this depth is reached: sample from the root
            // level's set and record it.
            if item.depth == self.columns_bylevel.len() {
                let mut columns = get_tree_columns();
                let sample_size =
                    get_sample_size(columns.len(), self.tree_options.colsample_bylevel);
                columns = columns.partial_shuffle(self.rng, sample_size).0.to_vec();
                self.columns_bylevel.push(columns);
            }
            item.value.columns = Some(self.columns_bylevel[item.depth].clone());
        }

        if self.tree_options.colsample_bynode < 1.0 {
            let mut columns = match &mut item.value.columns {
                Some(cols) => std::mem::take(cols),
                None => get_tree_columns(),
            };
            let sample_size = get_sample_size(columns.len(), self.tree_options.colsample_bynode);
            columns = columns.partial_shuffle(self.rng, sample_size).0.to_vec();
            item.value.columns = Some(columns);
        }
    }

    /// Push a node into the work queue.
    fn add_to_pool(&mut self, mut item: IndexedSplittable<'a, I, O>) {
        if !item.value.histogram.data().is_empty() {
            self.select_subsample(&mut item);
        }
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
    ((total as f64 * ratio).round() as usize).max(1)
}

/// One entry in the work queue: a node's state plus its bookkeeping.
struct IndexedSplittable<'a, I: IndexT, O: Objective> {
    /// The node's index in the finished tree (allocated by the builder).
    index: usize,
    /// Distance of the node from the root.
    depth: usize,
    /// The node's rows, histogram, and stats.
    value: Box<Splittable<'a, I, O>>,
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
        self.value.stats.gain.total_cmp(&other.value.stats.gain)
    }
}

impl<'a, I: IndexT, O: Objective> PartialEq for IndexedSplittable<'a, I, O> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<'a, I: IndexT, O: Objective> Eq for IndexedSplittable<'a, I, O> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{SquaredLoss, create_test_dataset};
    use rand::{RngExt, SeedableRng};

    // Create random dataset and index structures.
    fn create_data_and_index(rng: &mut StdRng) -> (Dataset<u16>, Vec<RowIndex<u32, SquaredLoss>>) {
        let mut data =
            create_test_dataset(/*columns*/ 16, /*rows*/ 1000, /*bins*/ 8);
        let mut index = RowIndex::<u32, SquaredLoss>::new_array(data.rows());

        for slice in data.slices_mut() {
            for value in slice {
                *value = rng.random_range(0..=8);
            }
        }
        for (pos, item) in index.iter_mut().enumerate() {
            let y = (pos % 2) as f32;
            let y_pred = rng.random_range(0.0..=1.0);
            *item = RowIndex::new(pos, SquaredLoss::derive(y, y_pred));
        }
        (data, index)
    }

    #[test]
    fn random_sample() {
        let mut rng = StdRng::seed_from_u64(0);
        let (data, mut index) = create_data_and_index(&mut rng);

        let tree_options = BoostTreeOptions {
            subsample: 0.1,
            colsample_bytree: 0.75,
            colsample_bylevel: 0.65,
            colsample_bynode: 0.5,
            max_depth: 3,
            max_leaves: 6,
            ..Default::default()
        };
        let node_options = BoostNodeOptions::default();

        // Run the builder with subsampling enabled.
        let mut builder = TreeBuilder::new(
            &mut index,
            &data,
            &tree_options,
            &node_options,
            &mut rng,
            false,
        );

        // Verify that the index is random before running.
        let root = builder.get_next_batch().pop().unwrap();
        let columns = Vec::from_iter(root.value.histogram.iter());
        let mut rows = Vec::from_iter(root.value.index.iter().map(|x| x.as_usize()));
        assert!(root == root);
        builder.add_to_pool(root);

        rows.sort();
        let adjacent = rows.array_windows().filter(|[a, b]| *a + 1 == *b).count();
        assert_eq!(rows.len(), 100);
        assert_eq!(adjacent, 16);

        // At each tree level, verify the histogram columns.
        while builder.process() {
            for item in builder.get_next_batch() {
                if item.depth != tree_options.max_depth {
                    let pool = &builder.columns_bylevel[item.depth];
                    let columns = item.value.columns.as_ref().unwrap();
                    assert_eq!(columns.len(), 4); // 16 columns * 0.75 * 0.65 * 0.5
                    assert!(columns.iter().all(|c| pool.contains(c)));
                }
                builder.add_to_pool(item);
            }
        }

        // Verify the columns selected for each tree level.
        assert_eq!(columns, &[0, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14]);
        assert_eq!(
            builder.columns_bylevel,
            vec![
                vec![5, 7, 0, 13, 3, 14, 6, 11],
                vec![0, 6, 13, 5, 14, 7, 4, 8],
                vec![6, 14, 2, 13, 4, 7, 5, 3],
            ]
        );

        // At least one column must be selected.
        let tree_options = BoostTreeOptions {
            colsample_bynode: 0.01,
            max_depth: 3,
            ..Default::default()
        };
        let tree = TreeBuilder::new(
            &mut index,
            &data,
            &tree_options,
            &node_options,
            &mut rng,
            false,
        )
        .execute();
        assert_eq!(tree.iter().count(), 15);
    }

    #[test]
    fn loss_guide() {
        let mut rng = StdRng::seed_from_u64(0);
        let (data, mut index) = create_data_and_index(&mut rng);

        let tree_options = BoostTreeOptions {
            loss_guide: true,
            max_depth: 0,
            max_leaves: 6,
            ..Default::default()
        };
        let node_options = BoostNodeOptions::default();

        // Construct and run the tree builder.
        let tree = TreeBuilder::new(
            &mut index,
            &data,
            &tree_options,
            &node_options,
            &mut rng,
            true,
        )
        .execute();

        // Verify the resulting tree.
        assert_eq!(
            tree.to_string(None),
            concat!(
                "0:[14 <= 5] yes=1,no=2,gain=1.8308164,cover=1000\n",
                "    1:value=-0.011305538,cover=575\n",
                "    2:[5 <= 2] yes=3,no=4,gain=1.399197,cover=425\n",
                "        3:value=-0.015970366,cover=102\n",
                "        4:[13 <= 7] yes=5,no=6,gain=1.6376567,cover=323\n",
                "            5:[5 <= 6] yes=7,no=8,gain=2.3093631,cover=261\n",
                "                7:[15 <= 3 | null] yes=9,no=10,gain=2.380442,cover=140\n",
                "                    9:value=0.110347375,cover=54\n",
                "                    10:value=0.029982187,cover=86\n",
                "                8:value=0.0043913317,cover=121\n",
                "            6:value=-0.019494265,cover=62",
            )
        );
    }
}
