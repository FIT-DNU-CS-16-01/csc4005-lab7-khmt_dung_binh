[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/NBAUjP4C)
# CSC4005 Lab 7 – Compression: KD + Quantization Trade-offs

Đây là **repo scaffold** cho **Lab 7 – Compression: Knowledge Distillation + Quantization Trade-offs** của học phần **CSC4005 – Học sâu**.

> Lab này nối tiếp trực tiếp Lab 6 – Export ONNX + Consistency Test + Benchmark. Sinh viên sử dụng checkpoint PyTorch hoặc file ONNX từ lab trước, sau đó chọn một kỹ thuật nén mô hình: **Quantization** hoặc **Knowledge Distillation (KD)**. Mục tiêu không chỉ là “nén được”, mà phải phân tích trade-off giữa **accuracy – latency – model size**.

## 1. Case study

Case study tiếp tục là:

```text
Smart Campus Scene Classification
```

Mô hình phân loại ảnh không gian trong trường học vào 5 lớp:

```text
classroom
computerroom
library
corridor
office
```

Dataset gợi ý: **MIT Indoor Scenes 67 subset 5 lớp**.

## 2. Mục tiêu

Sau lab này, sinh viên cần:

1. hiểu vì sao cần model compression;
2. chọn và triển khai ít nhất một kỹ thuật nén:
   - **Option A:** ONNX Dynamic Quantization;
   - **Option B:** Knowledge Distillation;
3. đo lại accuracy/macro-F1 sau khi nén;
4. benchmark latency, throughput và model size trước/sau khi nén;
5. lập bảng trade-off;
6. giải thích khi nào nên chọn KD, khi nào nên chọn quantization.

## 3. Vì sao đây là scaffold?

Repo này không phải lời giải hoàn chỉnh. Một số file có các vùng:

```python
# TODO: student implementation
```

Sinh viên cần đọc hướng dẫn, hoàn thiện code và chạy thí nghiệm. Giảng viên có thể dùng repo này làm GitHub Classroom starter, sau đó chấm dựa trên artefact và báo cáo.

## 4. Cấu trúc repo

```text
csc4005_lab7_compression_kd_quantization_scaffold/
├── README.md
├── REPORT_TEMPLATE.md
├── RUBRIC_LAB7.md
├── requirements.txt
├── configs/
│   ├── quantization_dynamic.json
│   ├── kd_student_mobilenet.json
│   └── tradeoff_eval.json
├── docs/
│   ├── LAB_GUIDE_LAB7_COMPRESSION.md
│   ├── QUANTIZATION_GUIDE.md
│   ├── KD_GUIDE.md
│   ├── TRADEOFF_ANALYSIS_GUIDE.md
│   └── GITHUB_CLASSROOM_GUIDE.md
├── notebooks/
│   └── lab7_compression_demo.ipynb
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── models.py
│   ├── metrics.py
│   ├── runtime.py
│   ├── evaluate_pytorch.py
│   ├── evaluate_onnx.py
│   ├── quantize_onnx.py
│   ├── kd_train_student.py
│   ├── benchmark.py
│   ├── make_tradeoff_table.py
│   └── utils.py
├── ci/
│   ├── check_structure.py
│   └── smoke_imports.py
├── checkpoints/
├── models/
├── outputs/
└── .github/workflows/
    └── ci.yml
```

## 5. Đầu vào cần chuẩn bị

Sinh viên cần có ít nhất một trong hai loại model từ lab trước:

### Trường hợp dùng Quantization

```text
models/vit_smartcampus.onnx
```

File này là ONNX model đã export ở Lab 6.

### Trường hợp dùng Knowledge Distillation

```text
checkpoints/teacher_vit_best_model.pt
```

File này là checkpoint PyTorch của teacher model, ví dụ ViT đã fine-tune ở lab trước.

Dữ liệu:

```text
data/mit_indoor_smartcampus_5/
├── classroom/
├── computerroom/
├── library/
├── corridor/
└── office/
```

## 6. Option A – ONNX Dynamic Quantization

Dynamic quantization là đường đi nhẹ nhất cho lab này. Sinh viên dùng file ONNX từ Lab 6 và tạo ra một file ONNX đã quantize.

Chạy:

```bash
python -m src.quantize_onnx \
  --input_onnx models/vit_smartcampus.onnx \
  --output_onnx models/vit_smartcampus_dynamic_int8.onnx \
  --mode dynamic
```

Sau đó benchmark:

```bash
python -m src.benchmark \
  --onnx_paths models/vit_smartcampus.onnx models/vit_smartcampus_dynamic_int8.onnx \
  --names baseline_onnx quantized_int8 \
  --batch_sizes 1 4 8 \
  --warmup 10 \
  --repeat 50 \
  --output_csv outputs/benchmark_quantization.csv
```

Đánh giá accuracy/macro-F1:

```bash
python -m src.evaluate_onnx \
  --onnx_path models/vit_smartcampus_dynamic_int8.onnx \
  --data_dir data/mit_indoor_smartcampus_5 \
  --output_json outputs/eval_quantized_onnx.json
```

## 7. Option B – Knowledge Distillation

KD dùng teacher model lớn để huấn luyện student model nhỏ hơn.

Ví dụ:

```text
Teacher: ViT-B/16
Student: MobileNetV2 hoặc ResNet18
```

Chạy scaffold KD:

```bash
python -m src.kd_train_student \
  --teacher_checkpoint checkpoints/teacher_vit_best_model.pt \
  --data_dir data/mit_indoor_smartcampus_5 \
  --student_model mobilenet_v2 \
  --alpha 0.5 \
  --temperature 4.0 \
  --epochs 10 \
  --batch_size 16 \
  --use_wandb
```

Sinh viên cần hoàn thiện các phần `TODO` trong `src/kd_train_student.py`.

## 8. Tạo bảng trade-off

Sau khi có kết quả baseline và compressed model, tạo bảng tổng hợp:

```bash
python -m src.make_tradeoff_table \
  --baseline_eval outputs/eval_baseline_onnx.json \
  --compressed_eval outputs/eval_quantized_onnx.json \
  --baseline_benchmark outputs/benchmark_baseline.csv \
  --compressed_benchmark outputs/benchmark_quantization.csv \
  --output_csv outputs/tradeoff_table.csv \
  --output_md outputs/tradeoff_table.md
```

Nếu sinh viên chọn KD, đổi file đầu vào tương ứng với kết quả student model.

## 9. Artefact cần nộp

Tối thiểu cần có:

```text
outputs/eval_baseline_*.json
outputs/eval_compressed_*.json
outputs/benchmark_*.csv
outputs/tradeoff_table.csv
outputs/tradeoff_table.md
REPORT_TEMPLATE.md đã điền
Link W&B nếu chọn KD
```

Nếu model file quá lớn, không commit lên GitHub. Ghi link lưu trữ trong báo cáo.

## 10. Checklist nhanh

- [ ] Có baseline model từ Lab 6
- [ ] Có ít nhất một compressed model
- [ ] Có accuracy/macro-F1 trước và sau nén
- [ ] Có latency/throughput/model size trước và sau nén
- [ ] Có bảng trade-off
- [ ] Có nhận xét khi nào chọn KD, khi nào chọn quantization
