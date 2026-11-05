# yolo-cli

Detects objects in a picture and saves a copy with a box around each one. Mostly referenced from <https://github.com/pykeio/ort/tree/main/examples/yolov8>.

## Usage

From a [release](https://github.com/pan93412/yolo-rs/releases/latest) archive:

```bash
./yolo-cli models/yolo11x.onnx data/baseball.jpg --output detected.png
```

or from this directory:

```bash
cargo run --release -- models/yolo11x.onnx data/baseball.jpg --output detected.png
```

where:

- `models/yolo11x.onnx` is the YOLO model exported to ONNX. Any version in the [supported models](https://github.com/pan93412/yolo-rs#supported-models) table works.
- `data/baseball.jpg` is the picture to detect objects in (JPEG or PNG).
- `--output` (`-o`) is where the picture with the boxes goes; `detected.png` by default.

The detections are also logged, with their labels, confidence and coordinates. `--probability-threshold` and `--iou-threshold` change the thresholds.

## Exporting a model

With [Ultralytics](https://docs.ultralytics.com/integrations/onnx/) installed:

```shell
mkdir models
yolo export model=yolo11x.pt format=onnx
mv yolo11x.onnx models/
```
