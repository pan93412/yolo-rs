from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLOE
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.text_model import build_text_model


class PromptableDetector(torch.nn.Module):
    def __init__(self, checkpoint: str) -> None:
        super().__init__()
        wrapper = YOLOE(checkpoint)
        self.model = wrapper.model
        self.model.eval().float()

        for module in self.model.modules():
            if isinstance(module, Detect):
                module.dynamic = False
                module.export = True
                module.format = "onnx"
                module.max_det = 300

    def forward(self, images: torch.Tensor, prompt_embeddings: torch.Tensor):
        return self.model.predict(images, tpe=prompt_embeddings)


class PromptEncoder(torch.nn.Module):
    def __init__(self, checkpoint: str) -> None:
        super().__init__()
        wrapper = YOLOE(checkpoint)
        self.model = wrapper.model
        self.model.eval().float()
        self.text_model = build_text_model(self.model.args.get("text_model"), device=torch.device("cpu"))
        self.text_model.eval()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        text_features = self.text_model.encode_text(input_ids, dtype=torch.float32)
        text_features = text_features.reshape(1, input_ids.shape[0], text_features.shape[-1])
        head = self.model.model[-1]
        return head.get_tpe(text_features)


def save_tokenizer(encoder: PromptEncoder, output_path: Path) -> None:
    tokenizer = encoder.text_model.tokenizer
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(tokenizer, "save"):
        tokenizer.save(str(output_path))
        return

    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(output_path.parent))
        return

    raise RuntimeError("Tokenizer does not support save() or save_pretrained(); export it manually.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to a YOLOE checkpoint such as yoloe-v8s-seg.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("exported-promptable-yoloe"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    detector = PromptableDetector(args.checkpoint)
    prompt_encoder = PromptEncoder(args.checkpoint)

    sample_image = torch.zeros((1, 3, args.height, args.width), dtype=torch.float32)
    sample_tokens = prompt_encoder.text_model.tokenize(["object"]).to(torch.long)
    sample_prompt_embeddings = prompt_encoder(sample_tokens)

    detector_path = args.output_dir / "yoloe-promptable.onnx"
    prompt_encoder_path = args.output_dir / "yoloe-prompt-encoder.onnx"
    tokenizer_path = args.output_dir / "tokenizer.json"

    detector_output_names = ["output0", "output1"] if "seg" in Path(args.checkpoint).stem else ["output0"]

    torch.onnx.export(
        detector,
        (sample_image, sample_prompt_embeddings),
        detector_path,
        opset_version=args.opset,
        input_names=["images", "prompt_embeddings"],
        output_names=detector_output_names,
        dynamic_axes={
            "images": {0: "batch"},
            "prompt_embeddings": {1: "prompt_count"},
            "output0": {0: "batch", 2: "anchors"},
            **({"output1": {0: "batch", 2: "mask_height", 3: "mask_width"}} if len(detector_output_names) > 1 else {}),
        },
    )

    torch.onnx.export(
        prompt_encoder,
        sample_tokens,
        prompt_encoder_path,
        opset_version=args.opset,
        input_names=["input_ids"],
        output_names=["prompt_embeddings"],
        dynamic_axes={
            "input_ids": {0: "prompt_count", 1: "token_count"},
            "prompt_embeddings": {1: "prompt_count"},
        },
    )

    save_tokenizer(prompt_encoder, tokenizer_path)

    print(f"Wrote detector ONNX to {detector_path}")
    print(f"Wrote prompt encoder ONNX to {prompt_encoder_path}")
    print(f"Wrote tokenizer to {tokenizer_path}")


if __name__ == "__main__":
    main()