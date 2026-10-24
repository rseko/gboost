use criterion::{Criterion, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::time::Duration;

use gboost::dataset::{Dataset, Feature, IndexT, RowIndex, SquaredLoss as Loss};
use gboost::histogram::Histogram;
use gboost::update_small::update_small;

const NUM_ROWS: usize = 1_000_000;
const NUM_COLS: usize = 128;
const NUM_BINS: usize = 255;

fn create_dataset(rng: &mut StdRng) -> Dataset<u8> {
    let feature = Feature::linear("".into(), vec![0.0; NUM_BINS]);
    let mut new_row = || {
        let mut row = vec![0; NUM_ROWS];
        rng.fill_bytes(&mut row);
        row
    };
    Dataset::new(
        vec![feature; NUM_COLS],
        (0..NUM_COLS).map(|_| new_row()).collect(),
        vec![0.0; NUM_ROWS],
        0.0,
    )
}

fn bench_main<T: IndexT>(c: &mut Criterion, prefix: &str) {
    let name = |s: &str| String::from(prefix) + s;

    let mut rng = StdRng::seed_from_u64(0);
    let mut data = create_dataset(&mut rng);
    let mut index = RowIndex::<T, Loss>::new_array(data.rows());
    index.shuffle(&mut rng);

    c.bench_function(&name("histogram:1"), |b| {
        b.iter(|| Histogram::build(&data, &index[..NUM_ROWS / 10], false, false));
    });
    c.bench_function(&name("histogram:N"), |b| {
        b.iter(|| Histogram::build(&data, &index[..NUM_ROWS / 10], false, true));
    });

    c.bench_function(&name("update:1"), |b| {
        b.iter(|| update_small(&mut data, &mut index, None, false));
    });
    c.bench_function(&name("update:N"), |b| {
        b.iter(|| update_small(&mut data, &mut index, None, true));
    });
}

fn bench_index32(c: &mut Criterion) {
    bench_main::<u32>(c, "u32/");
}

fn bench_index64(c: &mut Criterion) {
    bench_main::<u64>(c, "u64/");
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(2))
        .warm_up_time(Duration::from_millis(100));
    targets = bench_index32, bench_index64
}
criterion_main!(benches);
