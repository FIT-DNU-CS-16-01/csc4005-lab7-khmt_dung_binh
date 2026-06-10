# RUBRIC_LAB7 – Compression: KD + Quantization Trade-offs

Tổng điểm: **10 điểm**

| Thành phần | Điểm |
|---|---:|
| A. Chuẩn bị baseline và dữ liệu | 1.5 |
| B. Triển khai kỹ thuật nén | 2.5 |
| C. Đánh giá accuracy/macro-F1 | 1.5 |
| D. Benchmark latency/throughput/model size | 1.5 |
| E. Phân tích trade-off | 2.0 |
| F. Tổ chức repo và báo cáo | 1.0 |

## A. Chuẩn bị baseline và dữ liệu – 1.5 điểm

- 1.3–1.5: Có baseline đúng từ Lab 6, dữ liệu đúng subset 5 lớp, lệnh chạy rõ ràng.
- 0.9–1.2: Có baseline và dữ liệu nhưng mô tả còn thiếu.
- 0.4–0.8: Baseline/dữ liệu thiếu hoặc khó kiểm chứng.
- 0–0.3: Không chuẩn bị được baseline.

## B. Triển khai kỹ thuật nén – 2.5 điểm

- 2.1–2.5: Triển khai đúng ít nhất một kỹ thuật nén; code rõ; chạy được; có artefact model sau nén.
- 1.5–2.0: Có nén được nhưng còn thiếu kiểm tra hoặc cấu hình chưa rõ.
- 0.8–1.4: Code chạy một phần; kết quả chưa đáng tin.
- 0–0.7: Không triển khai được kỹ thuật nén.

## C. Đánh giá accuracy/macro-F1 – 1.5 điểm

- 1.3–1.5: Có đánh giá trước/sau nén bằng cùng protocol.
- 0.9–1.2: Có metric nhưng thiếu giải thích.
- 0.4–0.8: Kết quả chưa nhất quán.
- 0–0.3: Không có đánh giá.

## D. Benchmark – 1.5 điểm

- 1.3–1.5: Có benchmark latency, p95, throughput, size; dùng warmup và repeat.
- 0.9–1.2: Có benchmark nhưng thiếu một vài metric.
- 0.4–0.8: Benchmark sơ sài, khó tin cậy.
- 0–0.3: Không benchmark.

## E. Phân tích trade-off – 2.0 điểm

- 1.7–2.0: Phân tích rõ accuracy–latency–size; biết khi nào chọn KD/quantization.
- 1.2–1.6: Có nhận xét dựa trên số liệu nhưng chưa sâu.
- 0.6–1.1: Nhận xét chung chung.
- 0–0.5: Không phân tích.

## F. Repo và báo cáo – 1.0 điểm

- 0.9–1.0: Repo sạch, báo cáo rõ, có lệnh tái lập.
- 0.6–0.8: Repo tương đối đủ.
- 0.3–0.5: Repo lộn xộn, thiếu hướng dẫn.
- 0.0–0.2: Không thể kiểm tra.
