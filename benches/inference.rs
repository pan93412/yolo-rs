//! `YOLO_BENCH_MODELS` lists ONNX models separated by commas; each is timed on
//! `examples/yolo-cli/data/baseball.jpg`. Without it, only preprocessing is timed.

use std::env;
use std::error::Error;
use std::hint::black_box;
use std::path::Path;

use criterion::{Criterion, SamplingMode};
use image::DynamicImage;
use yolo_rs::model::YoloModelSession;
use yolo_rs::{YoloInput, image_to_yolo_input_tensor, inference};

// Our own `main` instead of `criterion_main!`, so a failure is returned rather than a panic.
fn main() -> Result<(), Box<dyn Error>> {
    let image_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/yolo-cli/data/baseball.jpg");
    let image = image::open(&image_path)
        .map_err(|error| format!("failed to open {}: {error}", image_path.display()))?;
    let input = image_to_yolo_input_tensor(&image);
    let models = load_models(&input)?;

    time(&image, &input, models);
    Ok(())
}

fn time(image: &DynamicImage, input: &YoloInput, models: Vec<(String, YoloModelSession)>) {
    let mut criterion = Criterion::default().configure_from_args();
    criterion.bench_function("preprocess", |b| {
        b.iter(|| image_to_yolo_input_tensor(black_box(image)));
    });
    time_inference(&mut criterion, input, models);
    criterion.final_summary();
}

fn time_inference(
    criterion: &mut Criterion,
    input: &YoloInput,
    models: Vec<(String, YoloModelSession)>,
) {
    let mut group = criterion.benchmark_group("inference");
    // An x model takes about a second per run on a CPU: time each run once.
    group.sampling_mode(SamplingMode::Flat).sample_size(10);
    for (name, mut model) in models {
        group.bench_function(name, |b| {
            b.iter(|| inference(&mut model, black_box(input.view())));
        });
    }
    group.finish();
}

/// Loads every model in `YOLO_BENCH_MODELS` and runs it once, so a broken one fails before any
/// timing starts; inside the timing loop the result only goes to `black_box`.
fn load_models(input: &YoloInput) -> Result<Vec<(String, YoloModelSession)>, Box<dyn Error>> {
    let models = env::var("YOLO_BENCH_MODELS").unwrap_or_default();
    models
        .split(',')
        .filter(|path| !path.is_empty())
        .map(|path| {
            let mut model = YoloModelSession::from_filename_v8(path)
                .map_err(|error| format!("failed to load {path}: {error}"))?;
            inference(&mut model, input.view())
                .map_err(|error| format!("inference with {path} failed: {error}"))?;
            let name = Path::new(path).file_stem().map_or_else(
                || path.to_owned(),
                |stem| stem.to_string_lossy().into_owned(),
            );
            Ok((name, model))
        })
        .collect()
}
