//! Runs the ONNX model in `YOLO_E2E_MODEL` on `examples/yolo-cli/data/baseball.jpg` and checks
//! that it finds what YOLOv8x, YOLO11x, YOLO12x and YOLO26x all find there at the default
//! threshold. The Models workflow sets it; without it the test does nothing.

use std::{env, path::Path};

use yolo_rs::{BoundingBox, image_to_yolo_input_tensor, inference, model::YoloModelSession};

/// Each found by all four models with a confidence of 0.74 or more, in 640×480 image pixels.
const EXPECTED: [(&str, [f32; 4]); 4] = [
    ("person", [24., 326., 128., 480.]),
    ("person", [107., 373., 239., 480.]),
    ("person", [268., 285., 350., 480.]),
    ("baseball glove", [212., 409., 240., 453.]),
];

fn iou(a: &BoundingBox, [x1, y1, x2, y2]: [f32; 4]) -> f32 {
    let width = (a.x2.min(x2) - a.x1.max(x1)).max(0.);
    let height = (a.y2.min(y2) - a.y1.max(y1)).max(0.);
    let intersection = width * height;
    let union = (a.x2 - a.x1) * (a.y2 - a.y1) + (x2 - x1) * (y2 - y1) - intersection;
    intersection / union
}

#[test]
fn finds_the_players_on_baseball_jpg() {
    let Ok(model_path) = env::var("YOLO_E2E_MODEL") else {
        eprintln!("YOLO_E2E_MODEL is not set; nothing to check");
        return;
    };

    let image_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/yolo-cli/data/baseball.jpg");
    let image = image::open(&image_path).expect("failed to open baseball.jpg");
    let input = image_to_yolo_input_tensor(&image);
    let mut model = YoloModelSession::from_filename_v8(&model_path)
        .unwrap_or_else(|error| panic!("failed to load {model_path}: {error}"));
    let found = inference(&mut model, input.view()).expect("inference failed");

    for (label, expected) in EXPECTED {
        let best = found
            .iter()
            .filter(|entity| entity.label.as_str() == label)
            .map(|entity| iou(&entity.bounding_box, expected))
            .fold(0., f32::max);
        assert!(
            best >= 0.5,
            "{model_path}: no {label} overlaps {expected:?} by IoU 0.5 or more (best {best:.2}); found {found:#?}"
        );
    }
}
