"""Benchmark one or more ONNX models for latency, throughput, and model size.

Usage:
    python -m src.benchmark \
        --onnx_paths models/vit_smartcampus.onnx models/vit_smartcampus_dynamic_int8.onnx \
        --names baseline_onnx quantized_int8 \
        --batch_sizes 1 4 8 \
        --warmup 10 \
        --repeat 50 \
        --output_csv outputs/benchmark_quantization.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.runtime import create_onnx_session, run_onnx
from src.utils import file_size_mb, format_size


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark one or more ONNX models.")
    parser.add_argument("--onnx_paths", nargs="+", required=True, help="Paths to ONNX models.")
    parser.add_argument("--names", nargs="+", required=True, help="Names for each model (same order as paths).")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 4, 8],
                        help="Batch sizes to benchmark.")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size.")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations.")
    parser.add_argument("--repeat", type=int, default=50, help="Number of timed iterations.")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV path.")
    return parser.parse_args()


def measure(session, batch_np, warmup: int, repeat: int):
    """Run warmup + timed iterations and return latency measurements in ms."""
    for _ in range(warmup):
        _ = run_onnx(session, batch_np)

    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        _ = run_onnx(session, batch_np)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    return times


def main():
    args = parse_args()
    if len(args.onnx_paths) != len(args.names):
        raise ValueError("--onnx_paths and --names must have the same length.")

    print(f"\n{'='*80}")
    print(f"ONNX Model Benchmark")
    print(f"{'='*80}")
    print(f"  Warmup:  {args.warmup} iterations")
    print(f"  Repeat:  {args.repeat} iterations")
    print(f"  Batches: {args.batch_sizes}")
    print(f"{'='*80}\n")

    rows = []
    for name, onnx_path in zip(args.names, args.onnx_paths):
        print(f"  Benchmarking: {name} ({onnx_path})")
        session = create_onnx_session(onnx_path)
        size_mb = file_size_mb(onnx_path)

        for bs in args.batch_sizes:
            batch = torch.randn(bs, 3, args.img_size, args.img_size).numpy().astype(np.float32)
            times = measure(session, batch, args.warmup, args.repeat)
            mean_latency = float(np.mean(times))
            std_latency = float(np.std(times))
            rows.append({
                "model": name,
                "onnx_path": onnx_path,
                "batch_size": bs,
                "mean_latency_ms": round(mean_latency, 2),
                "median_latency_ms": round(float(np.median(times)), 2),
                "std_latency_ms": round(std_latency, 2),
                "min_latency_ms": round(float(np.min(times)), 2),
                "max_latency_ms": round(float(np.max(times)), 2),
                "p95_latency_ms": round(float(np.percentile(times, 95)), 2),
                "p99_latency_ms": round(float(np.percentile(times, 99)), 2),
                "throughput_img_per_sec": round(float(bs / (mean_latency / 1000.0)), 2),
                "model_size_mb": round(size_mb, 2),
            })
            print(f"    bs={bs}: mean={mean_latency:.2f}ms, p95={np.percentile(times, 95):.2f}ms, "
                  f"throughput={bs / (mean_latency / 1000.0):.1f} img/s")

    df = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"\n{'-'*80}")
    print(df.to_string(index=False))
    print(f"{'-'*80}")
    print(f"\nResults saved to: {args.output_csv}\n")


if __name__ == "__main__":
    main()
