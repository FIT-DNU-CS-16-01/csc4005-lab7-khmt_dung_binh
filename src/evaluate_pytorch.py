"""Evaluate a PyTorch student model on the Smart Campus dataset.

Usage:
    python -m src.evaluate_pytorch \
        --checkpoint outputs/kd_student/student_best.pt \
        --student_model mobilenet_v2 \
        --data_dir data/mit_indoor_smartcampus_5 \
        --output_json outputs/eval_kd_student.json
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SmartCampusDataset
from src.metrics import compute_classification_metrics, compute_per_class_metrics
from src.models import build_student, count_parameters
from src.utils import file_size_mb, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch student model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to student checkpoint.")
    parser.add_argument("--student_model", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "resnet18"])
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory.")
    parser.add_argument("--classes", nargs="+", default=["classroom", "computerroom", "library", "corridor", "office"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_json", type=str, required=True, help="Path to save evaluation results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SmartCampusDataset(data_dir=args.data_dir, classes=args.classes, img_size=args.img_size, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_student(num_classes=len(args.classes), student_model=args.student_model, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    num_params = count_parameters(model)

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating PyTorch"):
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu()
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    metrics = compute_classification_metrics(y_true, y_pred)
    per_class = compute_per_class_metrics(y_true, y_pred, class_names=dataset.classes)

    metrics.update({
        "checkpoint": args.checkpoint,
        "model_size_mb": round(file_size_mb(args.checkpoint), 2),
        "num_samples": len(dataset),
        "student_model": args.student_model,
        "num_parameters": num_params,
        "per_class_report": per_class["classification_report"],
        "confusion_matrix": per_class["confusion_matrix"],
    })
    save_json(metrics, args.output_json)

    print(f"\n{'='*50}")
    print(f"PyTorch Evaluation Results: {args.student_model}")
    print(f"{'='*50}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Macro-F1:    {metrics['macro_f1']:.4f}")
    print(f"  Model size:  {metrics['model_size_mb']:.2f} MB")
    print(f"  Parameters:  {num_params:,}")
    print(f"  Num samples: {metrics['num_samples']}")
    print(f"  Saved to:    {args.output_json}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
