//! `YOLO_BENCH_MODELS` lists ONNX models separated by commas; each is timed on
//! `examples/yolo-cli/data/baseball.jpg`. Without it, only preprocessing is timed.

use std::{env, hint::black_box, path::Path};

use criterion::{Criterion, SamplingMode, criterion_group, criterion_main};
use yolo_rs::{image_to_yolo_input_tensor, inference, model::YoloModelSession};

fn benches(c: &mut Criterion) {
    let image_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/yolo-cli/data/baseball.jpg");
    let image = image::open(&image_path).expect("failed to open the benchmark image");

    c.bench_function("preprocess", |b| {
        b.iter(|| image_to_yolo_input_tensor(black_box(&image)))
    });

    let input = image_to_yolo_input_tensor(&image);
    let models = env::var("YOLO_BENCH_MODELS").unwrap_or_default();

    let mut group = c.benchmark_group("inference");
    // An x model takes about a second per run on a CPU: time each run once.
    group.sampling_mode(SamplingMode::Flat).sample_size(10);
    for path in models.split(',').filter(|path| !path.is_empty()) {
        let mut model = YoloModelSession::from_filename_v8(path)
            .unwrap_or_else(|error| panic!("failed to load {path}: {error}"));
        let name = Path::new(path).file_stem().map_or_else(
            || path.to_owned(),
            |stem| stem.to_string_lossy().into_owned(),
        );

        group.bench_function(name, |b| {
            b.iter(|| inference(&mut model, black_box(input.view())).expect("inference failed"))
        });
    }
    group.finish();
}

criterion_group!(inference_benches, benches);
criterion_main!(inference_benches);
