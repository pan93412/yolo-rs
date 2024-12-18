# yolo-cli

Mostly referenced from <https://github.com/pykeio/ort/tree/main/examples/yolov8>.

## Usage

```bash
cargo run --release models/yolo11x.onnx data/baseball.jpg
```

where:

- `models/yolo11x.onnx` is the path to the ONNX model file.
    - You can export the model according to [Ultralytics' manual](https://docs.ultralytics.com/integrations/onnx/).
- `data/baseball.jpg` is the path to the image file.
