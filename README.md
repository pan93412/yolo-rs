# yolo-rs: YOLO object detection in Rust

yolo-rs runs [Ultralytics YOLO](https://docs.ultralytics.com/) object detection models in Rust with [ONNX Runtime](https://onnxruntime.ai/) (through [`ort`](https://ort.pyke.io/)). Give it an image and an ONNX export of YOLOv8, YOLO11, YOLO12 or YOLO26, and it returns each detected object's bounding box, label and confidence score.

See docs.rs for the [latest documentation](https://docs.rs/yolo-rs).

## Supported models

CI exports each model below from Ultralytics and checks that yolo-rs finds the same people and baseball glove in [the test picture](examples/yolo-cli/data/baseball.jpg) with every one of them ([`tests/e2e.rs`](tests/e2e.rs)). It runs on every change and weekly against the newest Ultralytics release.

| Model | Tested weights | Export |
| --- | --- | --- |
| YOLOv8 | `yolov8x` | `yolo export model=yolov8x.pt format=onnx` |
| YOLO11 | `yolo11x` | `yolo export model=yolo11x.pt format=onnx` |
| YOLO12 | `yolo12x` | `yolo export model=yolo12x.pt format=onnx` |
| YOLO26 | `yolo26x` | `yolo export model=yolo26x.pt format=onnx end2end=False` |

The smaller sizes of each version (`n`, `s`, `m`, `l`) have the same inputs and outputs. YOLO26 needs `end2end=False` for now: its default end-to-end export has a different output ([#90](https://github.com/pan93412/yolo-rs/issues/90)).

## Features

- **Object detection**: bounding boxes, labels and confidence scores, with non-maximum suppression.
- **COCO labels built in**: the 80 classes the official YOLO detection models are trained on; bring your own labels for custom models.
- **ONNX Runtime**: runs on the CPU or any execution provider `ort` supports, such as CoreML or CUDA.
- **Pure Rust API**: takes an [`image`](https://docs.rs/image) `DynamicImage`.

## Performance

CI benchmarks every tested model on a GitHub-hosted Arm runner (`ubuntu-26.04-arm`, CPU only) with [criterion](https://github.com/bheisler/criterion.rs) (`cargo bench --bench inference`):

| Model | Inference (640×640) |
| --- | --- |
| YOLOv8x | 1.37 s |
| YOLO11x | 1.09 s |
| YOLO12x | 1.24 s |
| YOLO26x | 1.08 s |

Preparing the input (resizing and normalising the image) takes about 11 ms.

On a MacBook Pro (2024) with M3 Max, the median of 20 runs after 5 warm-up runs:

| Model | CoreML | CPU |
| --- | --- | --- |
| YOLOv8x | 28 ms | 374 ms |
| YOLO11x | 45 ms | 322 ms |
| YOLO12x | 164 ms | 373 ms |
| YOLO26x | 47 ms | 345 ms |

Times cover `inference` (the model and non-maximum suppression) on [the test picture](examples/yolo-cli/data/baseball.jpg), with ONNX Runtime from `ort` 2.0.0-rc.12 and models exported by Ultralytics 8.4.51.

## Building on macOS

ONNX Runtime's Apple build calls `__isPlatformVersionAtLeast`, which lives in the compiler runtime that Rust does not link by default. Without it, linking a binary, test or benchmark that uses yolo-rs fails with `Undefined symbols for architecture arm64: "___isPlatformVersionAtLeast"`. Add this to your project's `.cargo/config.toml`:

```toml
[target.'cfg(target_vendor = "apple")']
rustflags = ["-Clink-arg=-fapple-link-rtlib"]
```

Or emit it from your binary's build script: `println!("cargo:rustc-link-arg=-fapple-link-rtlib");`.

### CoreML by default

`ort` lets ONNX Runtime pick a device for each session, and on macOS it always adds CoreML, even when you register only the CPU execution provider. `from_filename_v8` therefore runs on CoreML on a Mac. To run on the CPU, build the session yourself with `AutoDevicePolicy::PreferCPU` and hand it to `new_v8`:

```rust
use ort::session::Session;
use ort::session::builder::AutoDevicePolicy;
use yolo_rs::model::YoloModelSession;

let environment = ort::init().build()?;
let session = Session::builder(&environment)?
    .with_auto_device(AutoDevicePolicy::PreferCPU)?
    .commit_from_file("yolo11x.onnx")?;
let model = YoloModelSession::new_v8(session);
```

## Examples

- [**yolo-cli**](examples/yolo-cli): Command-line interface for running YOLO inference on images.

## Acknowledgements

This project is inspired by the [YOLOv8 example](https://github.com/pykeio/ort/tree/main/examples/yolov8) from the `ort` repository.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
