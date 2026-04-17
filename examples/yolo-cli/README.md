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

For exported closed-set YOLOE models, pass the matching labels when you run the CLI:

```bash
cargo run --release exported-yoloe.onnx image.jpg --label person --label bus
```

You can also load labels from a file:

```bash
cargo run --release exported-yoloe.onnx image.jpg --labels-file labels.txt
```

## Downloading models

```shell
mkdir models
wget -O models/yolo11x.onnx https://huggingface.co/pan93412/yolo-v11-onnx/resolve/main/yolo11x.onnx
```
