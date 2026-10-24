//! One boosting iteration's data update: accumulate the new tree's leaf values
//! into the running predictions, then recompute the per-row gradient/hessian pairs.
//!
//! Functions:
//! - `update_small`: optional tree update plus gradient/hessian update, optionally
//!   parallelized over row chunks.

use crate::dataset::{Dataset, IndexT, Objective, RowIndex};
use crate::tree::Tree;
use rayon::prelude::*;

/// Update a dataset for one boosting round.
///
/// With `tree` given, each row's running prediction is increased by the
/// value of the leaf the row lands in. Then each row's `RowIndex` is
/// rewritten with its position and the objective's gradient/hessian at
/// (`target`, `predict`).
///
/// `parallel` runs both updates over `TASK_SIZE`-row chunks via rayon; rows
/// within and across chunks are still processed independently, so results
/// are identical to the sequential path.
pub fn update_small<T: Copy + Into<usize> + Sync, I: IndexT, O: Objective>(
    data: &mut Dataset<T>,
    index: &mut [RowIndex<I, O>],
    tree: Option<&Tree>,
    parallel: bool,
) {
    assert!(data.is_valid());
    assert_eq!(index.len(), data.rows());

    if parallel {
        // Rows are partitioned into `TASK_SIZE` chunks processed concurrently;
        // each chunk gets its base offset so global row positions stay correct.
        const TASK_SIZE: usize = 1024;
        let target_chunks = data.target.par_chunks(TASK_SIZE);
        let predict_chunks = data.predict.par_chunks_mut(TASK_SIZE);
        let index_chunks = index.par_chunks_mut(TASK_SIZE);

        index_chunks
            .zip(target_chunks.zip(predict_chunks))
            .enumerate()
            .for_each(|(idx, (dst, (y, y_pred)))| {
                let offset = idx * TASK_SIZE;
                if let Some(tree) = tree {
                    update_predict_slice(y_pred, tree, &data.inputs, offset);
                }
                update_index_slice::<I, O>(dst, y, y_pred, offset);
            });
    } else {
        if let Some(tree) = tree {
            update_predict_slice(&mut data.predict, tree, &data.inputs, 0);
        }
        update_index_slice::<I, O>(index, &data.target, &data.predict, 0);
    }
}

/// Add the tree's leaf value to each row in `predict`.
fn update_predict_slice<T: Copy + Into<usize>>(
    predict: &mut [f32],
    tree: &Tree,
    inputs: &[Vec<T>],
    offset: usize,
) {
    for (idx, y_pred) in predict.iter_mut().enumerate() {
        let row = offset + idx;
        let get_bin = |col: usize| inputs[col][row].into();
        *y_pred += tree.find_leaf(get_bin).get_value();
    }
}

/// Rewrite each row's [`RowIndex`] with its global position and the
/// objective's first/second loss derivatives at (`target`, `predict`).
///
/// `offset` is the dataset-wide index of the first row in this slice, so
/// parallel chunks keep correct global row ids.
fn update_index_slice<I: IndexT, O: Objective>(
    index: &mut [RowIndex<I, O>],
    target: &[f32],
    predict: &[f32],
    offset: usize,
) {
    for idx in 0..index.len() {
        let deriv = O::derive(target[idx], predict[idx]);
        index[idx] = RowIndex::new(offset + idx, deriv);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Feature, LogLoss, SquaredLoss};
    use crate::tree::Node;

    /// A small dataset with a single feature column.
    fn build_data() -> Dataset<u8> {
        const NUM_ROWS: usize = 2048;
        Dataset::new(
            vec![Feature::default()],
            vec![vec![0; NUM_ROWS]],
            vec![1.0; NUM_ROWS],
            0.4,
        )
    }

    /// A degenerate tree that predicts a constant 0.1 per row.
    fn build_tree() -> Tree {
        let root = Node::default_leaf(0.1);
        Tree { nodes: vec![root] }
    }

    #[test]
    fn update_simple() {
        let (mut data, tree) = (build_data(), build_tree());
        let mut index = RowIndex::<u32, SquaredLoss>::new_array(data.rows());
        update_small(&mut data, &mut index, Some(&tree), /*parallel*/ false);
        assert!(data.predict.iter().all(|x| *x == 0.5));
        index
            .iter()
            .for_each(|x| assert_eq!(x.deriv.grad_hess(), (-0.5, 1.0)));
    }

    #[test]
    fn update_parallel() {
        let (mut data, tree) = (build_data(), build_tree());
        let mut index = RowIndex::<u32, LogLoss>::new_array(data.rows());
        update_small(&mut data, &mut index, Some(&tree), /*parallel*/ true);
        assert!(data.predict.iter().all(|x| *x == 0.5));
        index
            .iter()
            .for_each(|x| assert_eq!(x.deriv.grad_hess(), (-0.37754065, 0.23500371)));
    }
}
