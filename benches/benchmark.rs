use criterion::{Criterion, criterion_group, criterion_main};
use std::time::Duration;

use gboost::dataset::{Dataset, Feature, RowIndex, SquaredLoss};
use gboost::update_small::update_small;

const NUM_ROWS: usize = 1_000_000;
const NUM_COLS: usize = 128;

fn create_dataset() -> Dataset<u8, u32> {
    Dataset {
        features: vec![Feature::default(); NUM_COLS],
        inputs: vec![vec![0; NUM_ROWS]; NUM_COLS],
        target: vec![0.0; NUM_ROWS],
        predict: vec![0.0; NUM_ROWS],
        index: (0..NUM_ROWS).map(|i| RowIndex::new(i, 0.0, 0.0)).collect(),
    }
}

fn bench_update(c: &mut Criterion) {
    let mut data = create_dataset();
    c.bench_function("update/simple", |b| {
        b.iter(|| update_small::<SquaredLoss, _, _>(&mut data, None, false));
    });
    c.bench_function("update/parallel", |b| {
        b.iter(|| update_small::<SquaredLoss, _, _>(&mut data, None, true));
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(2))
        .warm_up_time(Duration::from_millis(100));
    targets = bench_update
}
criterion_main!(benches);
