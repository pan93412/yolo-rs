# YOLO inference in Rust

This project provides a Rust implementation of YOLO-style ONNX inference, enabling detection and segmentation on images with bounding boxes, labels, confidence scores, and decoded masks. It works out of the box with YOLO v8/v11 COCO exports and can also run exported closed-set YOLOE ONNX models when you provide the matching labels. The implementation is inspired by the [YOLOv8 example](https://github.com/pykeio/ort/tree/main/examples/yolov8) from the `ort` repository.

See docs.rs for the [latest documentation](https://docs.rs/yolo-rs).

## Features

- **Object Detection and Segmentation**: Detects objects within an image and provides their bounding boxes, labels, confidence scores, and segmentation masks when the model exports them.
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

Validated export flow in Python:

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-v8s-seg.pt")
embeddings = model.get_text_pe(["person", "baseball bat", "baseball glove"])
model.set_classes(["person", "baseball bat", "baseball glove"], embeddings)
model.export(format="onnx")
```

That export produces a standard segmentation-style ONNX graph with:

- input `images` shaped `[1, 3, 640, 640]`
- output `output0` shaped `[1, 39, 8400]`
- output `output1` shaped `[1, 32, 160, 160]`

`yolo-rs` follows the same broad exported ONNX contract as Ultralytics' YOLO segmentation examples: it letterboxes the input image, decodes boxes from `output0`, and reconstructs instance masks from the trailing mask coefficients in `output0` plus the prototype tensor in `output1`.

Example CLI usage:

```bash
cargo run --release -p example-yolo-gui -- yoloe-v8s-seg.onnx examples/yolo-cli/data/baseball.jpg --labels-file target/yoloe_inspect_v839/yoloe.labels.txt --no-display
```

Current scope:

- Closed-set YOLOE detection exported to ONNX is supported.
- Detection heads with extra mask coefficients can be decoded for boxes and classes when you provide the label list.
- Segmentation-style exports can decode instance masks from `output1` and the mask coefficients stored in `output0`.
- Preprocessing and box rescaling now follow YOLO-style letterboxing instead of stretching the image to `640x640`.
- Runtime open-vocabulary text prompts and visual prompts are not implemented for ONNX in this crate. Ultralytics' exported ONNX graphs expose only the `images` input, so prompts must be fused into the model before export via `set_classes(...)`.

## Acknowledgements

This project is inspired by the [YOLOv8 example](https://github.com/pykeio/ort/tree/main/examples/yolov8) from the `ort` repository.

## License

This project is dual-licensed under the MIT License and Apache-2.0 LICENSE. See the [LICENSE](LICENSE) file for details.
