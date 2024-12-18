# YOLO inference in Rust

This project provides a Rust implementation of the YOLO v11 object detection model, enabling inference on images to identify objects along with their bounding boxes, labels, and confidence scores. It utilizes the YOLO v11 model in ONNX format and leverages the `ort` library for ONNX Runtime integration. The implementation is inspired by the [YOLOv8 example](https://github.com/pykeio/ort/tree/main/examples/yolov8) from the `ort` repository.

See docs.rs for the [latest documentation](https://docs.rs/yolo).

## Features

- **Object Detection**: Detects objects within an image and provides their bounding boxes, labels, and confidence scores.
- **ONNX Model Integration**: Employs the YOLO v11 model in ONNX format for efficient inference.
- **Rust Implementation**: Written entirely in Rust, ensuring performance and safety.
- **ONNX Runtime**: Utilizes the `ort` library for executing the ONNX model.

## Acknowledgements

This project is inspired by the [YOLOv8 example](https://github.com/pykeio/ort/tree/main/examples/yolov8) from the `ort` repository.

## License

This project is dual-licensed under the MIT License and Apache-2.0 LICENSE. See the [LICENSE](LICENSE) file for details.
