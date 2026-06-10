"""Evaluate an ONNX model on the Smart Campus dataset.

Usage:
    python -m src.evaluate_onnx \
        --onnx_path models/vit_smartcampus.onnx \
        --data_dir data/mit_indoor_smartcampus_5 \
        --output_json outputs/eval_baseline_onnx.json
"""
from __future__ import annotations

import argparse

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SmartCampusDataset
from src.metrics import compute_classification_metrics, compute_per_class_metrics
from src.runtime import create_onnx_session, run_onnx
from src.utils import file_size_mb, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ONNX model on Smart Campus dataset.")
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to ONNX model file.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory.")
    parser.add_argument("--classes", nargs="+", default=["classroom", "computerroom", "library", "corridor", "office"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for quick test.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save evaluation results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = SmartCampusDataset(
        data_dir=args.data_dir,
        classes=args.classes,
        img_size=args.img_size,
        augment=False,
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    session = create_onnx_session(args.onnx_path)

    y_true, y_pred = [], []
    for images, labels in tqdm(loader, desc="Evaluating ONNX"):
        logits = run_onnx(session, images.numpy().astype(np.float32))
        preds = np.argmax(logits, axis=1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

    metrics = compute_classification_metrics(y_true, y_pred)
    per_class = compute_per_class_metrics(y_true, y_pred, class_names=dataset.classes)

    metrics.update({
        "onnx_path": args.onnx_path,
        "model_size_mb": round(file_size_mb(args.onnx_path), 2),
        "num_samples": len(dataset),
        "classes": dataset.classes,
        "per_class_report": per_class["classification_report"],
        "confusion_matrix": per_class["confusion_matrix"],
    })
    save_json(metrics, args.output_json)

    print(f"\n{'='*50}")
    print(f"ONNX Evaluation Results: {args.onnx_path}")
    print(f"{'='*50}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Macro-F1:    {metrics['macro_f1']:.4f}")
    print(f"  Model size:  {metrics['model_size_mb']:.2f} MB")
    print(f"  Num samples: {metrics['num_samples']}")
    print(f"  Saved to:    {args.output_json}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
