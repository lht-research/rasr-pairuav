# 归档 PairUAV 提交实现

[English](METHOD_RELEASE.md)

本目录在配合 `frozen_artifacts_manifest.json` 中列出的冻结中间 artifacts 使用时，可以复现归档 PairUAV 提交。

实现遵循论文中的 RASR 分解：

- 冻结成对几何上的 scale-recovery core；
- 四个冻结 distance candidates（`distance_head_a` ... `distance_head_d`）；
- 使用 `gate=head0` 的 range-aware convex combination；
- PairUAV benchmark-specific distance output head；
- 冻结 heading column 与固定 heading transform；
- 逐行 self-pair zeroing。

它不使用测试集图优化、batch 排序、隐藏邻居修正、隐藏测试集检索、全局分配、聚类或跨样本 bookkeeping。

## 距离配置

配置文件：`models/lastmeter_config.json`

选择的 candidate name：

```text
gate_boundary_head0_bounds_0129
```

Gate：

```text
head0
```

尺度代理：

```text
|h_1(z_i)|
```

Bucket boundaries：

```text
[5.596117113284452, 14.189340747346009, 33.73144867309181, 53.75,
 77.38132781381124, 114.79]
```

对每一行，配置的 bucket 会选择四个 distance candidates 上的凸权重。PairUAV output head 随后从 `models/lastmeter_config.json` 选择 bucket/sign/std-segment affine parameters，并应用 2.5 m snap。

## Heading Transform

Heading source 是冻结的 `heading_pred` CSV column。release transform 是逐行的：

```json
{
  "scale": 1.0139999999999996,
  "bias": 1.2000000000000002,
  "snap_step": 20.0,
  "snap_offset": 0.0,
  "wrap": true
}
```

这是下式的实现形式：

```text
wrap_180(20 * round((1.014 * phi + 1.2) / 20))
```

## 预期提交

预期 `submit.zip` SHA-256：

```text
2f742f2eff83e535b96a8dbd46db370fa3ac0538a9f3e53b684d65c253b34b77
```

该 archive 对应的线上指标：

```text
final_score          0.003189
distance_rel_error   0.003029
angle_rel_error      0.003350
```

## 公开从头复现路径

源码树也包含 `scripts/reproduce_from_scratch.sh`，它会构建 train/test pair tables、提取 features、训练公开 distance heads、训练简单公开逐行 heading baseline，并打包提交。该路径用于从官方数据进行透明端到端复现。它不预期匹配归档 SHA-256，因为归档线上结果冻结了 production prediction CSVs，尤其是更强的冻结 heading column。
