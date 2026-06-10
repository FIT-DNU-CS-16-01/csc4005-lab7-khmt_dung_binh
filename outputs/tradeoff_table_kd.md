# Trade-off Analysis: Baseline vs Compressed

Batch size for latency comparison: **1**

## Comparison Table

| model      |   accuracy |   macro_f1 |   mean_latency_ms |   throughput_img_per_sec |   model_size_mb |
|:-----------|-----------:|-----------:|------------------:|-------------------------:|----------------:|
| baseline   |     0.9823 |     0.9761 |            119.4  |                     8.38 |          327.72 |
| compressed |     0.9823 |     0.977  |              2.67 |                   374.1  |            8.48 |

## Change Summary

| Metric | Change |
|---|---:|
| Accuracy | +0.0000 (+0.00%) |
| Macro-F1 | +0.0009 (+0.09%) |
| Mean Latency | -97.76% |
| Throughput | +4364.20% |
| Model Size | -97.41% (+319.24 MB) |

## Interpretation

- **Model size** giảm đáng kể (97.4%), từ 327.7 MB xuống 8.5 MB.
- **Accuracy** gần như không đổi (chênh 0.0000).
- **Latency** cải thiện rõ rệt (-97.8%).
