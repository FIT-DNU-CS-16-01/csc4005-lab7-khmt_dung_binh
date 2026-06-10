"""Generate a trade-off comparison table between baseline and compressed models.

Usage (Quantization):
    python -m src.make_tradeoff_table \
        --baseline_eval outputs/eval_baseline_onnx.json \
        --compressed_eval outputs/eval_quantized_onnx.json \
        --baseline_benchmark outputs/benchmark_quantization.csv \
        --compressed_benchmark outputs/benchmark_quantization.csv \
        --baseline_name baseline_onnx \
        --compressed_name quantized_int8 \
        --output_csv outputs/tradeoff_table.csv \
        --output_md outputs/tradeoff_table.md

Usage (KD):
    python -m src.make_tradeoff_table \
        --baseline_eval outputs/eval_baseline_onnx.json \
        --compressed_eval outputs/eval_kd_student.json \
        --baseline_benchmark outputs/benchmark_quantization.csv \
        --compressed_benchmark outputs/benchmark_kd_student.csv \
        --baseline_name baseline_onnx \
        --compressed_name kd_student \
        --output_csv outputs/tradeoff_table_kd.csv \
        --output_md outputs/tradeoff_table_kd.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils import load_json, percent_change


def parse_args():
    parser = argparse.ArgumentParser(description="Create trade-off comparison table.")
    parser.add_argument("--baseline_eval", type=str, required=True, help="Baseline evaluation JSON.")
    parser.add_argument("--compressed_eval", type=str, required=True, help="Compressed model evaluation JSON.")
    parser.add_argument("--baseline_benchmark", type=str, required=True, help="Baseline benchmark CSV.")
    parser.add_argument("--compressed_benchmark", type=str, required=True, help="Compressed model benchmark CSV.")
    parser.add_argument("--baseline_name", type=str, default="baseline_onnx",
                        help="Name of baseline model in benchmark CSV.")
    parser.add_argument("--compressed_name", type=str, default="quantized_int8",
                        help="Name of compressed model in benchmark CSV.")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV path.")
    parser.add_argument("--output_md", type=str, required=True, help="Output Markdown path.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for trade-off comparison.")
    return parser.parse_args()


def pick_benchmark_row(csv_path: str, model_name: str, batch_size: int):
    """Extract benchmark data for a specific model and batch size.

    If the CSV contains multiple models, filter by model_name.
    If it contains only one model, use the first matching batch_size.
    """
    df = pd.read_csv(csv_path)

    # Try filtering by model name first
    if "model" in df.columns:
        filtered = df[(df["model"] == model_name) & (df["batch_size"] == batch_size)]
        if not filtered.empty:
            row = filtered.iloc[0]
            return float(row["mean_latency_ms"]), float(row["throughput_img_per_sec"]), float(row["model_size_mb"])

    # Fallback: filter by batch_size only
    filtered = df[df["batch_size"] == batch_size]
    if filtered.empty:
        raise ValueError(f"No benchmark row found for model={model_name}, batch_size={batch_size} in {csv_path}")
    row = filtered.iloc[0]
    return float(row["mean_latency_ms"]), float(row["throughput_img_per_sec"]), float(row["model_size_mb"])


def main():
    args = parse_args()
    base_eval = load_json(args.baseline_eval)
    comp_eval = load_json(args.compressed_eval)

    base_latency, base_thr, base_size = pick_benchmark_row(
        args.baseline_benchmark, args.baseline_name, args.batch_size)
    comp_latency, comp_thr, comp_size = pick_benchmark_row(
        args.compressed_benchmark, args.compressed_name, args.batch_size)

    rows = [
        {
            "model": "baseline",
            "accuracy": round(base_eval.get("accuracy", 0), 4),
            "macro_f1": round(base_eval.get("macro_f1", 0), 4),
            "mean_latency_ms": round(base_latency, 2),
            "throughput_img_per_sec": round(base_thr, 2),
            "model_size_mb": round(base_size, 2),
        },
        {
            "model": "compressed",
            "accuracy": round(comp_eval.get("accuracy", 0), 4),
            "macro_f1": round(comp_eval.get("macro_f1", 0), 4),
            "mean_latency_ms": round(comp_latency, 2),
            "throughput_img_per_sec": round(comp_thr, 2),
            "model_size_mb": round(comp_size, 2),
        },
    ]

    df = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    # Compute change summary
    changes = {
        "accuracy_change_pp": round(rows[1]["accuracy"] - rows[0]["accuracy"], 4),
        "macro_f1_change_pp": round(rows[1]["macro_f1"] - rows[0]["macro_f1"], 4),
        "accuracy_change_percent": round(percent_change(rows[0]["accuracy"], rows[1]["accuracy"]), 2),
        "macro_f1_change_percent": round(percent_change(rows[0]["macro_f1"], rows[1]["macro_f1"]), 2),
        "latency_change_percent": round(percent_change(base_latency, comp_latency), 2),
        "throughput_change_percent": round(percent_change(base_thr, comp_thr), 2),
        "size_change_percent": round(percent_change(base_size, comp_size), 2),
        "size_reduction_mb": round(base_size - comp_size, 2),
    }

    # Generate Markdown report
    md = f"# Trade-off Analysis: Baseline vs Compressed\n\n"
    md += f"Batch size for latency comparison: **{args.batch_size}**\n\n"
    md += "## Comparison Table\n\n"
    md += df.to_markdown(index=False)

    md += "\n\n## Change Summary\n\n"
    md += f"| Metric | Change |\n|---|---:|\n"
    md += f"| Accuracy | {changes['accuracy_change_pp']:+.4f} ({changes['accuracy_change_percent']:+.2f}%) |\n"
    md += f"| Macro-F1 | {changes['macro_f1_change_pp']:+.4f} ({changes['macro_f1_change_percent']:+.2f}%) |\n"
    md += f"| Mean Latency | {changes['latency_change_percent']:+.2f}% |\n"
    md += f"| Throughput | {changes['throughput_change_percent']:+.2f}% |\n"
    md += f"| Model Size | {changes['size_change_percent']:+.2f}% ({changes['size_reduction_mb']:+.2f} MB) |\n"

    md += "\n## Interpretation\n\n"
    # Auto-generate interpretation
    size_reduction_pct = abs(changes["size_change_percent"])
    acc_drop = abs(changes["accuracy_change_pp"])
    f1_drop = abs(changes["macro_f1_change_pp"])
    latency_change = changes["latency_change_percent"]

    if size_reduction_pct > 50:
        md += f"- **Model size** giảm đáng kể ({size_reduction_pct:.1f}%), từ {base_size:.1f} MB xuống {comp_size:.1f} MB.\n"
    elif size_reduction_pct > 20:
        md += f"- **Model size** giảm vừa phải ({size_reduction_pct:.1f}%), từ {base_size:.1f} MB xuống {comp_size:.1f} MB.\n"
    else:
        md += f"- **Model size** thay đổi không nhiều ({size_reduction_pct:.1f}%).\n"

    if acc_drop < 0.01:
        md += f"- **Accuracy** gần như không đổi (chênh {acc_drop:.4f}).\n"
    elif acc_drop < 0.03:
        md += f"- **Accuracy** giảm nhẹ ({acc_drop:.4f} điểm), có thể chấp nhận được.\n"
    else:
        md += f"- **Accuracy** giảm đáng kể ({acc_drop:.4f} điểm), cần cân nhắc.\n"

    if latency_change < -10:
        md += f"- **Latency** cải thiện rõ rệt ({latency_change:.1f}%).\n"
    elif latency_change < 0:
        md += f"- **Latency** cải thiện nhẹ ({latency_change:.1f}%).\n"
    else:
        md += f"- **Latency** không cải thiện hoặc tăng ({latency_change:+.1f}%).\n"

    Path(args.output_md).write_text(md, encoding="utf-8")
    print(md)
    print(f"\nCSV saved to: {args.output_csv}")
    print(f"Markdown saved to: {args.output_md}\n")


if __name__ == "__main__":
    main()
