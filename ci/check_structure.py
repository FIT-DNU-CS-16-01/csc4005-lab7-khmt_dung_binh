from pathlib import Path

REQUIRED_PATHS = [
    "README.md",
    "REPORT_TEMPLATE.md",
    "RUBRIC_LAB7.md",
    "requirements.txt",
    "configs/quantization_dynamic.json",
    "configs/kd_student_mobilenet.json",
    "configs/tradeoff_eval.json",
    "docs/LAB_GUIDE_LAB7_COMPRESSION.md",
    "docs/QUANTIZATION_GUIDE.md",
    "docs/KD_GUIDE.md",
    "docs/TRADEOFF_ANALYSIS_GUIDE.md",
    "docs/GITHUB_CLASSROOM_GUIDE.md",
    "src/__init__.py",
    "src/dataset.py",
    "src/models.py",
    "src/metrics.py",
    "src/runtime.py",
    "src/evaluate_pytorch.py",
    "src/evaluate_onnx.py",
    "src/quantize_onnx.py",
    "src/kd_train_student.py",
    "src/benchmark.py",
    "src/make_tradeoff_table.py",
    "src/utils.py",
    "src/export_student_onnx.py",
    "ci/smoke_imports.py",
]

def main():
    missing = [p for p in REQUIRED_PATHS if not Path(p).exists()]
    if missing:
        print("Missing required files:")
        for p in missing:
            print(f"- {p}")
        raise SystemExit(1)
    print("Structure check passed.")

if __name__ == "__main__":
    main()
