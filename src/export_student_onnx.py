"""Export a trained student model from PyTorch checkpoint to ONNX format.

This enables fair benchmarking of the student model using the same ONNX pipeline
as the baseline and quantized models.

Usage:
    python -m src.export_student_onnx \
        --checkpoint outputs/kd_student/student_best.pt \
        --student_model mobilenet_v2 \
        --output_onnx models/student_mobilenet_v2.onnx
"""
from __future__ import annotations

import argparse

from src.models import export_student_to_onnx
from src.utils import file_size_mb


def parse_args():
    parser = argparse.ArgumentParser(description="Export student model to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to student .pt checkpoint.")
    parser.add_argument("--student_model", type=str, default="mobilenet_v2",
                        choices=["mobilenet_v2", "resnet18"], help="Student architecture name.")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of output classes.")
    parser.add_argument("--output_onnx", type=str, required=True, help="Output ONNX path.")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size.")
    parser.add_argument("--opset_version", type=int, default=17, help="ONNX opset version.")
    return parser.parse_args()


def main():
    args = parse_args()
    export_student_to_onnx(
        checkpoint_path=args.checkpoint,
        student_model=args.student_model,
        num_classes=args.num_classes,
        output_path=args.output_onnx,
        img_size=args.img_size,
        opset_version=args.opset_version,
    )
    size_mb = file_size_mb(args.output_onnx)
    print(f"ONNX model size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
