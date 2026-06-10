"""Export teacher ViT model from PyTorch checkpoint to ONNX format.

Usage:
    python -m src.export_teacher_onnx \
        --checkpoint checkpoints/teacher_vit_best_model.pt \
        --output_onnx models/vit_smartcampus.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.models import load_teacher_checkpoint
from src.utils import file_size_mb


def parse_args():
    parser = argparse.ArgumentParser(description="Export teacher ViT model to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to teacher .pt checkpoint.")
    parser.add_argument("--model_name", type=str, default="vit_b_16", help="ViT variant name.")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of output classes.")
    parser.add_argument("--output_onnx", type=str, required=True, help="Output ONNX path.")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size.")
    parser.add_argument("--opset_version", type=int, default=17, help="ONNX opset version.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading teacher model from: {args.checkpoint}")
    model = load_teacher_checkpoint(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        model_name=args.model_name,
    )
    model.eval()

    dummy_input = torch.randn(1, 3, args.img_size, args.img_size)
    Path(args.output_onnx).parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to ONNX: {args.output_onnx}")
    torch.onnx.export(
        model,
        dummy_input,
        args.output_onnx,
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    size_mb = file_size_mb(args.output_onnx)
    print(f"Teacher ONNX exported: {args.output_onnx} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
