# Trade-off Analysis: Baseline vs Compressed

Batch size for latency comparison: **1**

## Comparison Table

| model      |   accuracy |   macro_f1 |   mean_latency_ms |   throughput_img_per_sec |   model_size_mb |
|:-----------|-----------:|-----------:|------------------:|-------------------------:|----------------:|
| baseline   |     0.9823 |     0.9761 |            111.34 |                     8.98 |          327.72 |
| compressed |     0.9785 |     0.9708 |             55.65 |                    17.97 |           83.24 |

## Change Summary

| Metric | Change |
|---|---:|
| Accuracy | -0.0038 (-0.39%) |
| Macro-F1 | -0.0053 (-0.54%) |
| Mean Latency | -50.02% |
| Throughput | +100.11% |
| Model Size | -74.60% (+244.48 MB) |

## Interpretation

- **Model size** giảm đáng kể (74.6%), từ 327.7 MB xuống 83.2 MB.
- **Accuracy** gần như không đổi (chênh 0.0038).
- **Latency** cải thiện rõ rệt (-50.0%).
