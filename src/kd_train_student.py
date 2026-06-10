"""Knowledge Distillation: Train a student model guided by a teacher.

Usage:
    python -m src.kd_train_student \
        --teacher_checkpoint checkpoints/teacher_vit_best_model.pt \
        --data_dir data/mit_indoor_smartcampus_5 \
        --student_model mobilenet_v2 \
        --alpha 0.5 \
        --temperature 4.0 \
        --epochs 10 \
        --batch_size 16 \
        --use_wandb

Key concepts:
    - alpha: balances hard label loss (CE) vs soft knowledge transfer (KD).
      alpha=1.0 means only CE (no distillation), alpha=0.0 means only KD.
      Typical values: 0.3-0.7.

    - temperature (T): softens the probability distributions from both teacher
      and student. Higher T produces softer distributions, revealing more
      inter-class relationships. Typical values: 2.0-10.0.

    - KD loss: KL divergence between softened student and teacher distributions,
      scaled by T^2 to maintain gradient magnitude when T > 1.

Loss formula:
    loss = alpha * CE(student_logits, labels)
         + (1 - alpha) * KL(softmax(student/T), softmax(teacher/T)) * T^2
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.dataset import SmartCampusDataset
from src.models import build_student, count_parameters, export_student_to_onnx, load_teacher_checkpoint
from src.metrics import compute_classification_metrics
from src.utils import ensure_dir, file_size_mb, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge Distillation: train student from teacher.")
    parser.add_argument("--teacher_checkpoint", type=str, required=True, help="Path to teacher .pt checkpoint.")
    parser.add_argument("--teacher_model", type=str, default="vit_b_16", help="Teacher architecture name.")
    parser.add_argument("--student_model", type=str, default="mobilenet_v2",
                        choices=["mobilenet_v2", "resnet18"], help="Student architecture name.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory.")
    parser.add_argument("--classes", nargs="+", default=["classroom", "computerroom", "library", "corridor", "office"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Balance between CE loss (alpha) and KD loss (1-alpha). "
                             "alpha=1.0 means pure CE, alpha=0.0 means pure KD.")
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="Temperature for softening logits. Higher = softer distributions.")
    parser.add_argument("--val_split", type=float, default=0.15, help="Fraction of data for validation.")
    parser.add_argument("--project", type=str, default="csc4005-lab7-compression")
    parser.add_argument("--run_name", type=str, default="kd_student")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging.")
    parser.add_argument("--export_onnx", action="store_true", help="Export best student model to ONNX.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def kd_loss(student_logits, teacher_logits, labels, alpha: float, temperature: float):
    """Compute the Knowledge Distillation loss.

    The loss combines two terms:
    1. Hard label loss: standard cross-entropy between student predictions and true labels.
       This ensures the student learns to classify correctly.

    2. Soft knowledge loss: KL divergence between the softened probability distributions
       of teacher and student. The temperature T > 1 softens the distributions, making
       them more informative about inter-class relationships.
       The T^2 scaling factor compensates for the reduced gradient magnitude when T > 1.

    Args:
        student_logits: Raw logits from the student model.
        teacher_logits: Raw logits from the teacher model (no gradient needed).
        labels: True class labels.
        alpha: Weight for the CE loss (1-alpha for KD loss).
        temperature: Temperature for softening distributions.

    Returns:
        Tuple of (total_loss, ce_value, kd_value) where ce and kd are detached for logging.
    """
    # Hard label loss: standard cross-entropy with true labels
    ce = F.cross_entropy(student_logits, labels)

    # Soft knowledge loss: KL divergence between teacher and student soft distributions
    # We use log_softmax for the student (input) and softmax for the teacher (target)
    # as required by F.kl_div(input=log_probs, target=probs)
    # The T^2 factor ensures gradients remain at the correct scale
    kd = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)

    # Combined loss
    total = alpha * ce + (1 - alpha) * kd
    return total, ce.detach(), kd.detach()


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on a data loader and return classification metrics."""
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        preds = logits.argmax(dim=1).cpu()
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())
    return compute_classification_metrics(y_true, y_pred)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ensure_dir(Path("outputs") / args.run_name)

    print(f"\n{'='*60}")
    print(f"Knowledge Distillation Training")
    print(f"{'='*60}")
    print(f"  Device:       {device}")
    print(f"  Teacher:      {args.teacher_model}")
    print(f"  Student:      {args.student_model}")
    print(f"  Alpha:        {args.alpha}")
    print(f"  Temperature:  {args.temperature}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  LR:           {args.lr}")
    print(f"  Val split:    {args.val_split}")
    print(f"{'='*60}\n")

    # ─── Dataset ────────────────────────────────────────────────
    dataset = SmartCampusDataset(
        data_dir=args.data_dir,
        classes=args.classes,
        img_size=args.img_size,
        augment=True,
    )

    n_total = len(dataset)
    n_val = max(1, int(args.val_split * n_total))
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"  Dataset:  {n_total} total, {n_train} train, {n_val} val")

    # ─── Teacher ────────────────────────────────────────────────
    teacher = load_teacher_checkpoint(
        checkpoint_path=args.teacher_checkpoint,
        num_classes=len(args.classes),
        model_name=args.teacher_model,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher params: {teacher_params:,}")

    # Evaluate teacher for reference
    teacher_metrics = evaluate(teacher, val_loader, device)
    print(f"  Teacher val accuracy: {teacher_metrics['accuracy']:.4f}")
    print(f"  Teacher val macro-F1: {teacher_metrics['macro_f1']:.4f}")

    # ─── Student ────────────────────────────────────────────────
    student = build_student(
        num_classes=len(args.classes),
        student_model=args.student_model,
        pretrained=True,
    ).to(device)

    student_params = count_parameters(student)
    print(f"  Student params: {student_params:,}")
    print(f"  Compression ratio (params): {teacher_params / student_params:.1f}x")

    # ─── Optimizer & Scheduler ──────────────────────────────────
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ─── W&B ────────────────────────────────────────────────────
    wandb_run = None
    if args.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.project,
            name=args.run_name,
            config={
                **vars(args),
                "teacher_params": teacher_params,
                "student_params": student_params,
                "n_train": n_train,
                "n_val": n_val,
                "teacher_val_accuracy": teacher_metrics["accuracy"],
                "teacher_val_macro_f1": teacher_metrics["macro_f1"],
            },
        )

    # ─── Training loop ──────────────────────────────────────────
    best_val_f1 = -1.0
    best_path = output_dir / "student_best.pt"
    history = []
    start_time = time.time()

    print(f"\n{'─'*60}")
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Val Acc':>8} {'Val F1':>8} {'LR':>10} {'Best':>5}")
    print(f"{'─'*60}")

    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0
        n_batches = 0

        for images, labels in tqdm(train_loader, desc=f"KD Epoch {epoch}/{args.epochs}", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)
            loss, ce_value, kd_value = kd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                alpha=args.alpha,
                temperature=args.temperature,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_ce += ce_value.item() * images.size(0)
            total_kd += kd_value.item() * images.size(0)
            n_batches += 1

        scheduler.step()

        avg_loss = total_loss / n_train
        avg_ce = total_ce / n_train
        avg_kd = total_kd / n_train
        current_lr = scheduler.get_last_lr()[0]

        val_metrics = evaluate(student, val_loader, device)

        is_best = val_metrics["macro_f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": student.state_dict(),
                    "student_model": args.student_model,
                    "classes": args.classes,
                    "epoch": epoch,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "args": vars(args),
                },
                best_path,
            )

        row = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "train_ce_loss": round(avg_ce, 4),
            "train_kd_loss": round(avg_kd, 4),
            "val_acc": round(val_metrics["accuracy"], 4),
            "val_macro_f1": round(val_metrics["macro_f1"], 4),
            "lr": current_lr,
            "is_best": is_best,
        }
        history.append(row)

        best_marker = "  *" if is_best else ""
        print(f"{epoch:>6d} {avg_loss:>11.4f} {val_metrics['accuracy']:>8.4f} "
              f"{val_metrics['macro_f1']:>8.4f} {current_lr:>10.6f}{best_marker}")

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train/loss": avg_loss,
                "train/ce_loss": avg_ce,
                "train/kd_loss": avg_kd,
                "val/accuracy": val_metrics["accuracy"],
                "val/macro_f1": val_metrics["macro_f1"],
                "lr": current_lr,
            })

    elapsed = time.time() - start_time
    print(f"{'─'*60}")
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Best val macro-F1: {best_val_f1:.4f}")

    # ─── Summary ────────────────────────────────────────────────
    summary = {
        "student_checkpoint": str(best_path),
        "student_model": args.student_model,
        "student_params": student_params,
        "teacher_checkpoint": args.teacher_checkpoint,
        "teacher_model": args.teacher_model,
        "teacher_params": teacher_params,
        "teacher_val_accuracy": round(teacher_metrics["accuracy"], 4),
        "teacher_val_macro_f1": round(teacher_metrics["macro_f1"], 4),
        "best_val_macro_f1": round(best_val_f1, 4),
        "alpha": args.alpha,
        "temperature": args.temperature,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "training_time_seconds": round(elapsed, 1),
        "history": history,
    }
    save_json(summary, output_dir / "kd_summary.json")

    # ─── Export student to ONNX ─────────────────────────────────
    if args.export_onnx:
        onnx_path = str(output_dir / f"student_{args.student_model}.onnx")
        export_student_to_onnx(
            checkpoint_path=str(best_path),
            student_model=args.student_model,
            num_classes=len(args.classes),
            output_path=onnx_path,
            img_size=args.img_size,
        )
        summary["student_onnx_path"] = onnx_path
        summary["student_onnx_size_mb"] = round(file_size_mb(onnx_path), 2)
        save_json(summary, output_dir / "kd_summary.json")

    if wandb_run is not None:
        wandb_run.log({"best_val_macro_f1": best_val_f1})
        wandb_run.finish()

    print(f"\nSummary saved to: {output_dir / 'kd_summary.json'}")
    print(f"Best student checkpoint: {best_path}\n")


if __name__ == "__main__":
    main()
