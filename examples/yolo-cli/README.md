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

For non-interactive validation, skip the GUI window:

```bash
cargo run --release exported-yoloe.onnx image.jpg --labels-file labels.txt --no-display
```

For segmentation-style exports, the CLI will also decode and draw instance masks when the model exposes a second ONNX output with mask prototypes.

Open-vocabulary runtime prompting is available only for custom promptable YOLOE ONNX exports. Ultralytics' default ONNX export still produces a single-input fixed-class graph.

For true runtime prompting with a promptable two-input YOLOE ONNX graph, pass prompts together with a prompt encoder and tokenizer:

```bash
cargo run --release promptable-yoloe.onnx image.jpg \
    --prompt person \
    --prompt bus \
    --prompt-encoder-model yoloe-prompt-encoder.onnx \
    --prompt-tokenizer tokenizer.json
```

This path expects a custom promptable export rather than Ultralytics' default single-input ONNX export. See [scripts/export_promptable_yoloe.py](/Users/vincent/Work/yolo-rs/scripts/export_promptable_yoloe.py) for a helper script.

You can also load labels from a file:

```bash
cargo run --release exported-yoloe.onnx image.jpg --labels-file labels.txt
```

## Downloading models

```shell
mkdir models
wget -O models/yolo11x.onnx https://huggingface.co/pan93412/yolo-v11-onnx/resolve/main/yolo11x.onnx
```
