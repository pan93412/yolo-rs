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

Preparing the input (resizing and normalising the image) takes about 11 ms. On a MacBook Pro (2024) with M3 Max and CoreML, YOLO11x takes about **57 ms** per image.

<details>
    <summary>yolo-cli inference logs on MacBook Pro (2024) with M3 Max</summary>

    2025-07-11T13:15:07.344683Z  INFO example_yolo_gui: Inference took 56.944167ms
    2025-07-11T13:15:07.344765Z  INFO example_yolo_gui: Found entity "person" with confidence 0.91 at (23.53, 325.31) - (127.34, 480.19)
    2025-07-11T13:15:07.344813Z  INFO example_yolo_gui: Found entity "person" with confidence 0.91 at (268.38, 285.00) - (349.62, 480.00)
    2025-07-11T13:15:07.344843Z  INFO example_yolo_gui: Found entity "person" with confidence 0.89 at (106.31, 373.31) - (238.69, 480.19)
    2025-07-11T13:15:07.344865Z  INFO example_yolo_gui: Found entity "baseball glove" with confidence 0.75 at (210.94, 409.50) - (240.06, 453.00)
    2025-07-11T13:15:07.344875Z  INFO example_yolo_gui: Found entity "person" with confidence 0.66 at (20.17, 276.28) - (64.45, 364.97)
    2025-07-11T13:15:07.344890Z  INFO example_yolo_gui: Found entity "baseball bat" with confidence 0.50 at (222.94, 372.84) - (275.81, 381.66)
</details>

## Examples

- [**yolo-cli**](examples/yolo-cli): Command-line interface for running YOLO inference on images.

## Acknowledgements

This project is inspired by the [YOLOv8 example](https://github.com/pykeio/ort/tree/main/examples/yolov8) from the `ort` repository.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
