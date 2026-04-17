# YOLO inference in Rust

This project provides a Rust implementation of YOLO-style ONNX object detection, enabling inference on images to identify objects along with their bounding boxes, labels, and confidence scores. It works out of the box with YOLO v8/v11 COCO exports and can also run exported closed-set YOLOE ONNX models when you provide the matching labels. The implementation is inspired by the [YOLOv8 example](https://github.com/pykeio/ort/tree/main/examples/yolov8) from the `ort` repository.

See docs.rs for the [latest documentation](https://docs.rs/yolo-rs).

## Features

- **Object Detection**: Detects objects within an image and provides their bounding boxes, labels, and confidence scores.
- **YOLO-style ONNX Integration**: Supports default YOLO v8/v11 exports and exported closed-set YOLOE models.
- **Rust Implementation**: Written entirely in Rust, ensuring performance and safety.
- **ONNX Runtime**: Utilizes the `ort` library for executing the ONNX model.

## Performance

On a MacBook Pro (2024) with M3 Max, it tooks about **57ms** to inferring an image with the YOLO11x model.

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

## YOLOE support

This crate can run exported YOLOE ONNX models when they behave like regular fixed-class detectors. In practice that means exporting from Ultralytics after configuring the classes you want to detect, then passing the same labels into `yolo-rs`.

Example export flow in Python:

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-26s-seg.pt")
model.set_classes(["person", "bus"])
model.export(format="onnx")
```

Example CLI usage:

```bash
cargo run --release -p example-yolo-gui -- exported-yoloe.onnx image.jpg --label person --label bus
```

Current scope:

- Closed-set YOLOE detection exported to ONNX is supported.
- Detection heads with extra mask coefficients can be decoded for boxes and classes when you provide the label list.
- Open-vocabulary text prompts, visual prompts, offline prompt embeddings, and segmentation mask decoding are not implemented in this crate yet.

## Acknowledgements

This project is inspired by the [YOLOv8 example](https://github.com/pykeio/ort/tree/main/examples/yolov8) from the `ort` repository.

## License

This project is dual-licensed under the MIT License and Apache-2.0 LICENSE. See the [LICENSE](LICENSE) file for details.
