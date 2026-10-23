//! One boosting iteration's data update: accumulate the new tree's leaf values
//! into the running predictions, then recompute the per-row gradient/hessian pairs.
//!
//! Functions:
//! - `update_small`: optional tree update plus gradient/hessian update, optionally
//!   parallelized over row chunks.

use crate::dataset::{Dataset, Objective, RowIndex, RowIndexT};
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
pub fn update_small<O: Objective, T: Copy + Into<usize> + Sync, I: RowIndexT>(
    data: &mut Dataset<T, I>,
    tree: Option<&Tree>,
    parallel: bool,
) {
    assert!(data.is_valid());

    if parallel {
        // Rows are partitioned into `TASK_SIZE` chunks processed concurrently;
        // each chunk gets its base offset so global row positions stay correct.
        const TASK_SIZE: usize = 1024;
        let target_chunks = data.target.par_chunks(TASK_SIZE);
        let predict_chunks = data.predict.par_chunks_mut(TASK_SIZE);
        let index_chunks = data.index.par_chunks_mut(TASK_SIZE);

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
        update_index_slice::<I, O>(&mut data.index, &data.target, &data.predict, 0);
    }
}

/// Add this row's new tree leaf value to its running prediction, for each
/// row in `predict` chunk-by-chunk against `tree`.
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
fn update_index_slice<I: RowIndexT, O: Objective>(
    row_index: &mut [RowIndex<I>],
    target: &[f32],
    predict: &[f32],
    offset: usize,
) {
    row_index
        .iter_mut()
        .zip(target.iter().zip(predict))
        .enumerate()
        .for_each(|(idx, (dst, (y, y_pred)))| {
            let (gradient, hessian) = O::grad_hess(*y, *y_pred);
            *dst = RowIndex::new(offset + idx, gradient, hessian);
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Feature, LogLoss, SquaredLoss};
    use crate::tree::{Node, NodeType};

    /// A small dataset with a single feature column.
    fn build_data() -> Dataset<u8, u32> {
        const NUM_ROWS: usize = 2048;
        Dataset {
            features: vec![Feature::default()],
            inputs: vec![vec![0; NUM_ROWS]],
            target: vec![1.0; NUM_ROWS],
            predict: vec![0.4; NUM_ROWS],
            index: vec![RowIndex::new(0, 0.0, 0.0); NUM_ROWS],
        }
    }

    /// A degenerate tree that predicts a constant 0.1 per row.
    fn build_tree() -> Tree {
        let root = Node {
            id: 0,
            type_: NodeType::Leaf(0.1),
            depth: 0,
            cover: 1.0,
        };
        Tree { nodes: vec![root] }
    }

    #[test]
    fn update_simple() {
        let (mut data, tree) = (build_data(), build_tree());
        update_small::<SquaredLoss, _, _>(&mut data, Some(&tree), false);
        assert!(data.predict.iter().all(|x| *x == 0.5));
        assert!(data.index.iter().all(|x| x.gradient == -0.5));
    }

    #[test]
    fn update_parallel() {
        let (mut data, tree) = (build_data(), build_tree());
        update_small::<LogLoss, _, _>(&mut data, Some(&tree), true);
        assert!(data.predict.iter().all(|x| *x == 0.5));
        assert!(data.index.iter().all(|x| x.hessian == 0.23500371));
    }
}
