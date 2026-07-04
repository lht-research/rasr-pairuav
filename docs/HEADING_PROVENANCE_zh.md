# Heading 来源

[English](HEADING_PROVENANCE.md)

精确冻结 artifact 路径将 heading 作为一个对齐 CSV column 使用，列名为 `heading_pred`。在论文记号中，该列是 PairUAV benchmark-specific output head 使用的冻结 predictor `phi_i`。

公开 from-scratch 路径会训练一个简单的逐行 heading baseline，用于透明端到端执行。该 baseline 不是归档线上分数使用的冻结 heading column。

## Per-Pair 约定

对每个正式行，上游 heading models 为该行的有序图像对预测 heading values。冻结 heading source 是对这些逐行预测做固定 circular weighted average 得到的：

```text
heading_pred[row] = circular_weighted_average(head_a[row], head_b[row], ...)
```

该 combiner 不检查邻居行，不构建测试图，不对隐藏 batch 排序，不做全局行匹配，不聚类样本，也不使用官方测试标签。

该操作的公开实现是 `models_src/heading_ensemble.py`。精确路径只存储冻结 output column，因为归档线上分数锚定到该导出列。

## PairUAV 输出变换

精确复现时，`heading_pred` 会通过 `models/heading_transform.json` 逐行变换：

```text
wrap_180(20 * round((1.014 * phi + 1.2) / 20))
```

20 degree snap 是 PairUAV benchmark-specific output head 的一部分。它用于复现归档分数，不声明为可迁移的控制器 resolution。

## Manifest

`models/heading_weights.json` 记录冻结 heading column contract：

- 期望文件名：`heading_predictions.csv`
- 期望列：`heading_pred`
- 官方测试集期望行数：`2773116`
- 归档 source CSV 文件的 SHA-256
- canonical heading column stream 的 SHA-256，其中每个值格式化为 `%.6f`，后接 LF，且无 header
- 关于 test-graph、assignment、clustering 和 cross-sample use 的 machine-readable provenance flags

## 验证

校验 manifest，并在可用时校验 heading CSV：

```bash
python scripts/verify_heading_provenance.py \
  --manifest models/heading_weights.json \
  --heading-csv /path/to/heading_predictions.csv
```

该检查会验证声明的 per-pair provenance flags，以及 CSV 的行数和列约定。它不是分数检查，也不使用官方标签。
