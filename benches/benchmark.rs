use criterion::{Criterion, criterion_group, criterion_main};
use std::time::Duration;

use gboost::dataset::{Dataset, Feature, RowIndex, SquaredLoss};
use gboost::update_small::update_small;

const NUM_ROWS: usize = 1_000_000;
const NUM_COLS: usize = 128;

fn create_dataset() -> Dataset<u8> {
    Dataset::new(
        vec![Feature::default(); NUM_COLS],
        vec![vec![0; NUM_ROWS]; NUM_COLS],
        vec![0.0; NUM_ROWS],
        0.0,
    )
}

fn bench_update(c: &mut Criterion) {
    let mut data = create_dataset();

    let mut index_u32 = RowIndex::<u32, SquaredLoss>::new_array(data.rows());
    c.bench_function("update/u32:1", |b| {
        b.iter(|| update_small(&mut data, &mut index_u32, None, false));
    });
    c.bench_function("update/u32:N", |b| {
        b.iter(|| update_small(&mut data, &mut index_u32, None, true));
    });

    let mut index_u64 = RowIndex::<u64, SquaredLoss>::new_array(data.rows());
    c.bench_function("update/u64:1", |b| {
        b.iter(|| update_small(&mut data, &mut index_u64, None, false));
    });
    c.bench_function("update/u64:N", |b| {
        b.iter(|| update_small(&mut data, &mut index_u64, None, true));
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
