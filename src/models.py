from __future__ import annotations

import torch
from torch import nn


def build_vit_teacher(num_classes: int = 5, model_name: str = "vit_b_16", dropout: float = 0.2):
    """Build a Vision Transformer teacher model.

    Args:
        num_classes: Number of output classes.
        model_name: Which ViT variant to use ('vit_b_16' or 'vit_b_32').
        dropout: Dropout rate before the classification head.

    Returns:
        A ViT model with the classification head replaced.
    """
    from torchvision.models import vit_b_16, vit_b_32

    if model_name == "vit_b_16":
        model = vit_b_16(weights=None)
    elif model_name == "vit_b_32":
        model = vit_b_32(weights=None)
    else:
        raise ValueError(f"Unsupported teacher model: {model_name}")

    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def build_student(num_classes: int = 5, student_model: str = "mobilenet_v2", pretrained: bool = True):
    """Build a student model for Knowledge Distillation.

    Supported architectures:
        - mobilenet_v2: ~3.4M params, very lightweight
        - resnet18: ~11.7M params, good balance of speed and accuracy

    Args:
        num_classes: Number of output classes.
        student_model: Which student architecture to use.
        pretrained: Whether to use ImageNet pretrained weights.

    Returns:
        A student model with the classification head replaced.
    """
    from torchvision.models import MobileNet_V2_Weights, ResNet18_Weights, mobilenet_v2, resnet18

    if student_model == "mobilenet_v2":
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if student_model == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported student model: {student_model}")


def load_teacher_checkpoint(checkpoint_path: str, num_classes: int = 5, model_name: str = "vit_b_16"):
    """Load a teacher model from a checkpoint file.

    Handles both raw state_dict and dict-wrapped checkpoints.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        num_classes: Number of output classes.
        model_name: Which ViT variant was used.

    Returns:
        Teacher model with loaded weights.
    """
    model = build_vit_teacher(num_classes=num_classes, model_name=model_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_student_to_onnx(
    checkpoint_path: str,
    student_model: str,
    num_classes: int,
    output_path: str,
    img_size: int = 224,
    opset_version: int = 17,
):
    """Export a trained student model to ONNX format.

    This enables benchmarking the student model with the same ONNX pipeline
    as the baseline and quantized models for fair comparison.

    Args:
        checkpoint_path: Path to student .pt checkpoint.
        student_model: Student architecture name.
        num_classes: Number of output classes.
        output_path: Path for the output .onnx file.
        img_size: Input image size.
        opset_version: ONNX opset version.
    """
    from pathlib import Path
    model = build_student(num_classes=num_classes, student_model=student_model, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Student model exported to ONNX: {output_path}")
