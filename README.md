# YOLO inference in Rust

This project provides a Rust implementation of the YOLO v11 object detection model, enabling inference on images to identify objects along with their bounding boxes, labels, and confidence scores. It utilizes the YOLO v11 model in ONNX format and leverages the `ort` library for ONNX Runtime integration. The implementation is inspired by the [YOLOv8 example](https://github.com/pykeio/ort/tree/main/examples/yolov8) from the `ort` repository.

See docs.rs for the [latest documentation](https://docs.rs/yolo-rs).

## Features

- **Object Detection**: Detects objects within an image and provides their bounding boxes, labels, and confidence scores.
- **ONNX Model Integration**: Employs the YOLO v11 model in ONNX format for efficient inference.
- **Rust Implementation**: Written entirely in Rust, ensuring performance and safety.
- **ONNX Runtime**: Utilizes the `ort` library for executing the ONNX model.

## Performance

On a MacBook Pro (2024) with M3 Max, it tooks about **86ms** to inferring an image with the YOLO11x model.

<details>
    <summary>yolo-cli inference logs on MacBook Pro (2024) with M3 Max</summary>

    2024-12-18T13:17:31.565158Z  INFO example_yolo_gui: Running inference…
    2024-12-18T13:17:31.651758Z  INFO example_yolo_gui: Inference took 86.589625ms
    2024-12-18T13:17:31.651861Z  INFO example_yolo_gui: Found entity "person" with confidence 0.91 at (23.53, 325.31) - (127.34, 480.19)
    2024-12-18T13:17:31.651902Z  INFO example_yolo_gui: Found entity "person" with confidence 0.91 at (268.38, 285.00) - (349.62, 480.00)
    2024-12-18T13:17:31.651933Z  INFO example_yolo_gui: Found entity "person" with confidence 0.89 at (106.31, 373.31) - (238.69, 480.19)
    2024-12-18T13:17:31.651956Z  INFO example_yolo_gui: Found entity "baseball glove" with confidence 0.75 at (210.94, 409.50) - (240.06, 453.00)
    2024-12-18T13:17:31.651967Z  INFO example_yolo_gui: Found entity "person" with confidence 0.66 at (20.17, 276.28) - (64.45, 364.97)
    2024-12-18T13:17:31.651982Z  INFO example_yolo_gui: Found entity "baseball bat" with confidence 0.50 at (222.94, 372.84) - (275.81, 381.66)
</details>

## Examples

- [**yolo-cli**](examples/yolo-cli): Command-line interface for running YOLO inference on images.

## Acknowledgements

This project is inspired by the [YOLOv8 example](https://github.com/pykeio/ort/tree/main/examples/yolov8) from the `ort` repository.

## License

This project is dual-licensed under the MIT License and Apache-2.0 LICENSE. See the [LICENSE](LICENSE) file for details.
