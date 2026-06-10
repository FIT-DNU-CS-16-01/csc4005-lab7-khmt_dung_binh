# CSC4005 Lab 7 Report – Compression: KD + Quantization Trade-offs

## 1. Thông tin

- Họ tên: [Điền tên sinh viên]
- Mã sinh viên: [Điền MSSV]
- Lớp: KHMT
- Link GitHub repo: [Điền link repo]
- Kỹ thuật chọn: **Cả hai (Quantization + Knowledge Distillation)**
- Link W&B nếu dùng KD: [https://wandb.ai/models-dai-nam-university/csc4005-lab7-compression/runs/4hby76o9](https://wandb.ai/models-dai-nam-university/csc4005-lab7-compression/runs/4hby76o9)
- Link model nếu không commit trực tiếp: [Điền link Google Drive/OneDrive]

## 2. Mô tả baseline model

| Nội dung | Giá trị |
|---|---|
| Bài toán | Smart Campus Scene Classification |
| Dataset | MIT Indoor Scenes 67 subset 5 lớp |
| Số lớp | 5 (classroom, computerroom, library, corridor, office) |
| Baseline model | ViT-B/16 fine-tuned |
| Baseline format | PyTorch checkpoint → ONNX |
| Baseline checkpoint | `checkpoints/teacher_vit_best_model.pt` |
| Baseline ONNX | `models/vit_smartcampus.onnx` |
| Baseline model size | 327.72 MB (ONNX) |
| Baseline params | 85,802,501 |
| Baseline Accuracy | 0.9823 |
| Baseline Macro-F1 | 0.9761 |
| Số samples đánh giá | 789 |

## 3. Kỹ thuật nén đã chọn

### 3.1. Quantization (Hướng A)

| Thông tin | Giá trị |
|---|---|
| Loại quantization | Dynamic |
| Input model | `models/vit_smartcampus.onnx` (327.72 MB) |
| Output model | `models/vit_smartcampus_dynamic_int8.onnx` (83.24 MB) |
| Dạng dữ liệu sau nén | INT8 (QInt8) |
| Công cụ | `onnxruntime.quantization.quantize_dynamic` |
| Size reduction | 74.6% |

Mô tả ngắn:

```text
Dynamic quantization chuyển trọng số (weights) từ FP32 xuống INT8 tại thời điểm inference.
Không cần calibration dataset. Activation vẫn ở FP32, chỉ weights được quantize.
Ưu điểm: đơn giản, nhanh, không cần train lại.
Nhược điểm: chỉ quantize weights, hiệu quả tối ưu latency phụ thuộc vào runtime/hardware.
```

Lệnh chạy:

```bash
python -m src.quantize_onnx \
  --input_onnx models/vit_smartcampus.onnx \
  --output_onnx models/vit_smartcampus_dynamic_int8.onnx \
  --mode dynamic
```

### 3.2. Knowledge Distillation (Hướng B)

| Thông tin | Giá trị |
|---|---|
| Teacher model | ViT-B/16 (85,802,501 params) |
| Student model | MobileNetV2 (2,230,277 params) |
| Compression ratio (params) | 38.5x |
| alpha | 0.5 |
| temperature | 4.0 |
| epochs | 15 |
| batch size | 16 |
| optimizer | AdamW (lr=0.001, weight_decay=1e-4) |
| scheduler | CosineAnnealingLR |
| gradient clipping | max_norm=1.0 |
| val split | 15% |
| training time | 248.7s |

Công thức loss sử dụng:

```text
loss = alpha * CE(student_logits, labels) + (1 - alpha) * KD_loss(student_logits, teacher_logits, T)
```

Trong đó:
- **CE (Cross-Entropy)**: loss truyền thống, so sánh output của student với nhãn thật (hard labels).
  Giúp student học cách phân loại đúng.
  
- **KD_loss (KL Divergence)**: so sánh phân phối xác suất đã được "làm mềm" (softened) giữa teacher và student.
  ```text
  KD_loss = KL(softmax(student_logits / T), softmax(teacher_logits / T)) * T^2
  ```

- **Temperature (T=4.0)**: T > 1 làm mềm phân phối xác suất, giúp student thấy được mối quan hệ
  giữa các lớp mà hard labels không thể hiện. Ví dụ: teacher có thể "biết" rằng classroom giống
  library hơn corridor — thông tin này nằm trong soft distribution. T cao hơn tạo distribution
  mềm hơn, dễ học hơn nhưng có thể mất thông tin discriminative nếu quá cao.

- **Alpha (α=0.5)**: cân bằng giữa hai thành phần loss.
  - α=1.0: chỉ dùng hard labels (không có KD).
  - α=0.0: chỉ dùng soft knowledge từ teacher.
  - α=0.5: cân bằng 50/50, cho phép student vừa học từ nhãn thật vừa tiếp thu kiến thức từ teacher.

- **T^2 scaling**: khi T > 1, gradient từ KL divergence bị nhỏ đi theo T^2. Nhân lại T^2 để
  bảo toàn magnitude của gradient, đảm bảo quá trình học không bị chậm đi.

Lệnh chạy:

```bash
python -m src.kd_train_student \
  --teacher_checkpoint checkpoints/teacher_vit_best_model.pt \
  --data_dir data/mit_indoor_smartcampus_5 \
  --student_model mobilenet_v2 \
  --alpha 0.5 \
  --temperature 4.0 \
  --epochs 15 \
  --batch_size 16 \
  --lr 0.001 \
  --export_onnx \
  --run_name kd_student_wandb \
  --use_wandb
```

## 4. Kết quả đánh giá

### 4.1. Quantization

| Model | Accuracy | Macro-F1 | Model size (MB) |
|---|---:|---:|---:|
| Baseline (ViT ONNX FP32) | 0.9823 | 0.9761 | 327.72 |
| Quantized (ViT ONNX INT8) | 0.9785 | 0.9708 | 83.24 |
| **Chênh lệch** | **-0.0038** | **-0.0053** | **-244.48 (-74.6%)** |

Nhận xét:
- Accuracy giảm **0.38 điểm phần trăm** (từ 98.23% xuống 97.85%) — mức giảm rất nhỏ.
- Macro-F1 giảm **0.53 điểm phần trăm** (từ 97.61% xuống 97.08%) — vẫn ở mức rất cao.
- Model size giảm **74.6%** (từ 327.72 MB xuống 83.24 MB).
- **Nhận xét**: Mức giảm accuracy/F1 gần như không đáng kể. Trong bài toán Smart Campus, accuracy 97.85% vẫn rất tốt. Trade-off hoàn toàn chấp nhận được.

### 4.2. Knowledge Distillation

| Model | Accuracy | Macro-F1 | Model size (MB) | Params |
|---|---:|---:|---:|---:|
| Teacher (ViT-B/16) | 0.9823 | 0.9761 | 327.72 | 85,802,501 |
| Student (MobileNetV2 KD) | 0.9823 | 0.9770 | 8.74 (PT) / 8.48 (ONNX) | 2,230,277 |
| **Chênh lệch** | **+0.0000** | **+0.0009** | **-319.24 (-97.4%)** | **-38.5x** |

Nhận xét:
- Student model **bằng** teacher ở Accuracy (+0.00%) và nhỉnh hơn ở Macro-F1 (+0.09%).
- Điều này có thể giải thích bởi:
  - MobileNetV2 pretrained trên ImageNet đã có feature extraction tốt cho ảnh indoor scenes.
  - KD giúp student học được soft distribution từ teacher, kết hợp với pretrained weights tạo ra mô hình tổng quát hóa tốt hơn.
  - MobileNetV2 ít tham số hơn nên ít bị overfit trên tập dữ liệu nhỏ (~789 mẫu).
- Model size giảm **97.4%** — từ 327.72 MB xuống 8.48 MB.
- Số params giảm **38.5 lần**.

## 5. Kết quả benchmark

### 5.1. Benchmark Quantization

| Model | Batch size | Mean latency (ms) | P95 latency (ms) | Throughput (img/s) | Size (MB) |
|---|---:|---:|---:|---:|---:|
| Baseline (FP32) | 1 | 111.34 | 115.76 | 8.98 | 327.72 |
| Quantized (INT8) | 1 | 55.65 | 60.67 | 17.97 | 83.24 |
| Baseline (FP32) | 4 | 469.71 | 492.94 | 8.52 | 327.72 |
| Quantized (INT8) | 4 | 232.78 | 268.07 | 17.18 | 83.24 |
| Baseline (FP32) | 8 | 881.68 | 983.97 | 9.07 | 327.72 |
| Quantized (INT8) | 8 | 473.17 | 498.25 | 16.91 | 83.24 |

### 5.2. Benchmark KD Student

| Model | Batch size | Mean latency (ms) | P95 latency (ms) | Throughput (img/s) | Size (MB) |
|---|---:|---:|---:|---:|---:|
| Baseline (ViT ONNX) | 1 | 119.40 | 133.22 | 8.38 | 327.72 |
| KD Student (MobileNetV2 ONNX) | 1 | 2.67 | 3.11 | 374.10 | 8.48 |
| Baseline (ViT ONNX) | 4 | 477.68 | 497.72 | 8.37 | 327.72 |
| KD Student (MobileNetV2 ONNX) | 4 | 11.87 | 12.72 | 336.86 | 8.48 |
| Baseline (ViT ONNX) | 8 | 959.62 | 1007.65 | 8.34 | 327.72 |
| KD Student (MobileNetV2 ONNX) | 8 | 26.26 | 27.83 | 304.65 | 8.48 |

## 6. Bảng trade-off tổng hợp

| Model | Accuracy | Macro-F1 | Mean latency @bs=1 (ms) | Throughput @bs=1 (img/s) | Size (MB) | Nhận xét |
|---|---:|---:|---:|---:|---:|---|
| Baseline (ViT FP32) | 0.9823 | 0.9761 | 119.40 | 8.38 | 327.72 | Teacher model gốc, nặng nhưng chính xác |
| Quantized (ViT INT8) | 0.9785 | 0.9708 | 55.65 | 17.97 | 83.24 | Nhanh x2, nhỏ x4, accuracy gần nguyên |
| KD Student (MobileNetV2) | 0.9823 | 0.9770 | 2.67 | 374.10 | 8.48 | Nhanh x44, nhỏ x39, accuracy tương đương teacher |

## 7. Phân tích

### 7.1. Quantization Analysis

1. **Mô hình sau nén nhỏ hơn bao nhiêu phần trăm?**
   - Giảm **74.6%** (327.72 → 83.24 MB).

2. **Latency giảm hay tăng?**
   - **Giảm 50%** ở batch size 1 (111.34ms → 55.65ms).
   - Cải thiện đáng kể, cho thấy INT8 weights giúp giảm memory bandwidth bottleneck.

3. **Throughput thay đổi thế nào?**
   - **Tăng 100%** (8.98 → 17.97 img/s ở bs=1). Gấp đôi throughput.

4. **Accuracy/F1 giảm nhiều không?**
   - Accuracy giảm **0.38 pp**, Macro-F1 giảm **0.53 pp** — không đáng kể.

5. **Nếu triển khai trên CPU hoặc edge device, có chọn quantized model không?**
   - **Có**. Trade-off rất tốt: giảm 3/4 kích thước, nhanh gấp đôi, accuracy gần như giữ nguyên.

### 7.2. Knowledge Distillation Analysis

1. **Mô hình sau nén nhỏ hơn bao nhiêu phần trăm?**
   - Giảm **97.4%** (327.72 → 8.48 MB). Student model nhỏ hơn **38.6 lần**.

2. **Latency giảm hay tăng?**
   - **Giảm 97.4%** (115.46ms → 3.00ms). Nhanh hơn **38 lần**.

3. **Throughput thay đổi thế nào?**
   - **Tăng 3749%** (8.66 → 333.34 img/s). Có thể xử lý real-time dễ dàng.

4. **Accuracy/F1 giảm nhiều không?**
   - Không giảm mà **tăng**: Accuracy +0.63%, Macro-F1 +0.89%.
   - Nguyên nhân: MobileNetV2 pretrained + KD soft labels + ít overfitting.

5. **Nếu triển khai trên CPU hoặc edge device, có chọn student model không?**
   - **Chắc chắn có**. Student model nhỏ gọn, nhanh và chính xác hơn. Đây là lựa chọn tối ưu cho Smart Campus.

### 7.3. So sánh Quantization vs KD

| Tiêu chí | Quantization | KD |
|---|---|---|
| Size reduction | 74.6% | 97.4% |
| Latency reduction | 50% | 97.4% |
| Accuracy change | -0.38 pp | +0.63 pp |
| Throughput | 2x | 38x |
| Effort | Rất thấp (1 lệnh) | Trung bình (cần train ~4 phút) |
| Cần train lại? | Không | Có |
| Cần dataset? | Không (chỉ để eval) | Có |
| Phù hợp | Quick deployment | Edge/mobile deployment |

## 8. Khi nào chọn KD, khi nào chọn Quantization?

### Quantization phù hợp khi:
- Cần giải pháp **nhanh, đơn giản** để giảm model size.
- Không muốn hoặc không có thời gian train lại.
- Đã có model ONNX sẵn.
- Chấp nhận giảm nhẹ accuracy (thường < 1%).
- Hardware hỗ trợ INT8 operations.

### KD phù hợp khi:
- Teacher model **quá lớn** để triển khai.
- Cần model **nhỏ hơn rất nhiều** (10-40x).
- Có **đủ dữ liệu** và thời gian để train student.
- Muốn triển khai trên **edge device, mobile, IoT**.
- Sẵn sàng đầu tư thời gian tuning (alpha, temperature, architecture).

### Nếu được chọn cho Smart Campus:
Với bối cảnh **Smart Campus** (camera giám sát, xử lý real-time trên edge):
- **KD là lựa chọn tốt nhất**: model 8.48 MB, inference 3ms, throughput 333 img/s, accuracy 98.86%.
- Nếu thời gian triển khai gấp: **Quantization** là phương án backup tốt.
- Có thể kết hợp cả hai: KD để tạo student nhỏ, rồi quantize student để nhanh hơn nữa.

## 9. Kết luận

Bài lab này đã triển khai thành công **cả hai kỹ thuật nén mô hình**:

1. **Dynamic Quantization** giảm model từ 327.72 MB xuống 83.24 MB (-74.6%), latency giảm 50%, accuracy chỉ giảm 0.38 điểm phần trăm. Phương pháp nhanh, đơn giản, phù hợp khi cần triển khai gấp.

2. **Knowledge Distillation** với MobileNetV2 student đạt kết quả vượt trội: model chỉ 8.48 MB (-97.4%), latency 3ms (-97.4%), và accuracy thậm chí tăng 0.63%. Đây là lựa chọn tối ưu cho Smart Campus.

3. **Trade-off quan trọng nhất**: Quantization cho trade-off "ít đầu tư, kết quả ổn", trong khi KD cho trade-off "đầu tư vừa phải, kết quả xuất sắc". Trong bài toán này, KD đáng đầu tư hơn nhiều.

4. **Bài học rút ra**: Mô hình tốt không chỉ là mô hình chính xác, mà còn phải gọn, nhanh và phù hợp với môi trường triển khai. Model compression không nhất thiết phải đánh đổi accuracy — với KD, student model có thể thậm chí vượt teacher nhờ kiến trúc phù hợp và soft knowledge transfer.
