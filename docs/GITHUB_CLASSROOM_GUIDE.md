# GitHub Classroom Guide

## 1. Nhận repo

1. Click vào link GitHub Classroom từ giảng viên.
2. Accept assignment.
3. Clone repo về máy:

```bash
git clone <student-repo-url>
cd csc4005_lab7_compression_kd_quantization_scaffold
```

## 2. Quy trình làm bài

1. Tạo environment và cài thư viện (xem README.md).
2. Chuẩn bị dataset và baseline model.
3. Chọn kỹ thuật nén: Quantization hoặc KD hoặc cả hai.
4. Chạy thí nghiệm và lưu kết quả vào `outputs/`.
5. Viết báo cáo theo mẫu `REPORT_TEMPLATE.md`.
6. Commit code và báo cáo lên GitHub.

## 3. Lưu ý khi commit

- **KHÔNG** commit file lớn: `.onnx`, `.pt`, `.pth`, dataset.
- Sử dụng `.gitignore` đã có sẵn.
- Lưu model lớn lên Google Drive / OneDrive và ghi link trong báo cáo.

## 4. Kiểm tra CI

Khi push lên GitHub, CI sẽ tự động kiểm tra cấu trúc repo:

```bash
python ci/check_structure.py
```

Đảm bảo tất cả file cần thiết đều có mặt.
