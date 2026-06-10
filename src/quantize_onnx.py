"""Apply ONNX dynamic quantization to reduce model size.

Usage:
    python -m src.quantize_onnx \
        --input_onnx models/vit_smartcampus.onnx \
        --output_onnx models/vit_smartcampus_dynamic_int8.onnx \
        --mode dynamic

This converts FP32 weights to INT8, significantly reducing model size.
Accuracy should be re-evaluated after quantization.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

from src.utils import file_size_mb, format_size, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Apply dynamic quantization to ONNX model.")
    parser.add_argument("--input_onnx", type=str, required=True, help="Path to input ONNX model.")
    parser.add_argument("--output_onnx", type=str, required=True, help="Path for quantized ONNX model.")
    parser.add_argument("--mode", type=str, default="dynamic", choices=["dynamic"],
                        help="Quantization mode (currently only dynamic).")
    parser.add_argument("--weight_type", type=str, default="QInt8", choices=["QInt8", "QUInt8"],
                        help="Weight data type after quantization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_onnx)
    output_path = Path(args.output_onnx)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input ONNX not found: {input_path}")

    weight_type = QuantType.QInt8 if args.weight_type == "QInt8" else QuantType.QUInt8

    print(f"\n{'='*60}")
    print(f"ONNX Dynamic Quantization")
    print(f"{'='*60}")
    print(f"  Input:       {input_path}")
    print(f"  Output:      {output_path}")
    print(f"  Weight type: {args.weight_type}")

    baseline_size = file_size_mb(input_path)
    print(f"  Input size:  {format_size(baseline_size)}")
    print(f"  Quantizing...")

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=weight_type,
    )

    compressed_size = file_size_mb(output_path)
    reduction_pct = (1 - compressed_size / baseline_size) * 100 if baseline_size > 0 else 0

    report = {
        "method": "onnx_dynamic_quantization",
        "input_onnx": str(input_path),
        "output_onnx": str(output_path),
        "weight_type": args.weight_type,
        "baseline_size_mb": round(baseline_size, 2),
        "compressed_size_mb": round(compressed_size, 2),
        "size_reduction_percent": round(reduction_pct, 2),
    }

    report_path = output_path.parent / "quantization_report.json"
    save_json(report, report_path)

    print(f"  Output size: {format_size(compressed_size)}")
    print(f"  Reduction:   {reduction_pct:.1f}%")
    print(f"  Report:      {report_path}")
    print(f"{'='*60}")
    print(f"\n  IMPORTANT: Re-evaluate accuracy after quantization!")
    print(f"  Run: python -m src.evaluate_onnx --onnx_path {output_path} ...")
    print()


if __name__ == "__main__":
    main()
