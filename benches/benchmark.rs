use criterion::{Criterion, criterion_group, criterion_main};
use half::f16;
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::time::Duration;

use gboost::boost_node::BoostNodeOptions;
use gboost::boost_tree::{BoostTreeOptions, TreeBuilder};
use gboost::dataset::{Dataset, Feature, IndexT, Objective, RowIndex, SquaredLoss as Loss};
use gboost::histogram::Histogram;
use gboost::quantize::{QuantFloat, bins_from, quantize_all};
use gboost::tree::{Node, NodeKind, NodeSplit, Tree};
use gboost::update_small::update_small;

const NUM_ROWS: usize = 1_000_000;
const NUM_COLS: usize = 128;
const NUM_BINS: usize = 255;

fn create_dataset(rng: &mut StdRng) -> Dataset<u8> {
    let feature = Feature::linear(None, vec![0.0; NUM_BINS]);
    let mut new_row = || {
        let mut row = vec![0; NUM_ROWS];
        rng.fill_bytes(&mut row);
        row
    };
    Dataset::new(
        vec![feature; NUM_COLS],
        (0..NUM_COLS).map(|_| new_row()).collect(),
        vec![0.0; NUM_ROWS],
    )
    .unwrap()
}

fn create_tree(rng: &mut StdRng, depth: u32) -> Tree {
    let mut nodes = vec![];
    let leaves = (1 << depth) / 2;
    for idx in 1..leaves {
        nodes.push(Node {
            kind: NodeKind::Split(NodeSplit {
                feature_id: rng.random_range(0..NUM_COLS),
                bin_id: rng.random_range(0..NUM_BINS),
                is_range: true,
                with_missing: true,
                positive_id: idx * 2 - 1,
                negative_id: idx * 2,
                gain: 0.1 * idx as f32,
            }),
            depth: idx.ilog2(),
            cover: 0.0,
        });
    }
    for idx in 1..=leaves {
        nodes.push(Node {
            kind: NodeKind::Leaf(0.1 * idx as f32),
            depth: depth - 1,
            cover: 0.0,
        });
    }
    Tree::new(nodes).unwrap()
}

fn bench_main<T: IndexT>(c: &mut Criterion, prefix: &str) {
    let name = |s: &str| String::from(prefix) + s;
    let mut rng = StdRng::seed_from_u64(0);

    let data = create_dataset(&mut rng);
    let tree = create_tree(&mut rng, /*depth*/ 5);
    let mut predict = vec![0.0; NUM_ROWS];
    let mut index = RowIndex::<T, Loss>::new_array(NUM_ROWS);
    index.shuffle(&mut rng);

    c.bench_function(&name("histogram:1"), |b| {
        b.iter(|| Histogram::build(&data, None, &index[..NUM_ROWS / 10], false, false));
    });
    c.bench_function(&name("histogram:N"), |b| {
        b.iter(|| Histogram::build(&data, None, &index[..NUM_ROWS / 10], false, true));
    });

    c.bench_function(&name("update:1"), |b| {
        b.iter(|| update_small(&mut predict, &mut index, &data, Some(&tree), false));
    });
    c.bench_function(&name("update:N"), |b| {
        b.iter(|| update_small(&mut predict, &mut index, &data, Some(&tree), true));
    });

    let tree_options = BoostTreeOptions::default();
    let node_options = BoostNodeOptions::default();
    let mut build_tree = |parallel: bool| {
        let mut index = (0..NUM_ROWS)
            .map(|i| RowIndex::<T, Loss>::new(i, Loss::derive(1.0, rng.random_range(0.0..=1.0))))
            .collect::<Vec<_>>();
        TreeBuilder::new(
            &mut index,
            &data,
            &tree_options,
            &node_options,
            &mut rng,
            parallel,
        )
        .execute()
    };

    c.bench_function(&name("tree:1"), |b| {
        b.iter(|| build_tree(false));
    });
    c.bench_function(&name("tree:N"), |b| {
        b.iter(|| build_tree(true));
    });
}

fn bench_index_types(c: &mut Criterion) {
    bench_main::<u32>(c, "u32/");
    #[cfg(feature = "large-index")]
    bench_main::<u64>(c, "u64/");
}

fn bench_quant<T: QuantFloat>(c: &mut Criterion, prefix: &str) {
    let name = |s: &str| String::from(prefix) + s;
    let mut rng = StdRng::seed_from_u64(0);

    let data = (0..NUM_ROWS)
        .map(|_| T::from(rng.random_range(-1.0..1.0)).unwrap())
        .collect::<Vec<T>>();

    c.bench_function(&name("bins:1"), |b| {
        b.iter(|| bins_from(&data, NUM_BINS));
    });

    let bins = bins_from(&data, NUM_BINS);
    c.bench_function(&name("quantize:1"), |b| {
        b.iter(|| quantize_all::<T, u8>(&[(&data, &bins)], false));
    });
}

fn bench_float_types(c: &mut Criterion) {
    bench_quant::<f16>(c, "f16/");
    bench_quant::<f32>(c, "f32/");
}

fn bench_config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(2))
        .warm_up_time(Duration::from_millis(100))
}

criterion_group! {
    name = benches;
    config = bench_config();
    targets = bench_index_types, bench_float_types
}
criterion_main!(benches);
