# 方法

[English](METHOD.md)

本文档沿用论文中的术语：

- **RASR:** Range-Aware Scale Recovery。
- **PairUAV:** last-meter relative-pose benchmark。
- **Scale-recovery core:** 方法中可迁移、连续、snap 前的部分。
- **PairUAV benchmark-specific output head:** 用于复现归档榜单提交的距离分桶、snapping 和数据集调参常数。

## 问题约定

每个 query 是一个有顺序的图像对。估计器为该图像对返回一个 heading 和 distance command：

```text
y_i = f(image_a_i, image_b_i; fixed_parameters)
```

所有参数都在正式推理前固定。第 `i` 行的预测不依赖任何其他隐藏测试行。这个约定排除了测试集图优化、batch 排序、隐藏邻居修正、隐藏测试集检索、全局分配、聚类和跨样本 bookkeeping。

## 冻结成对几何

尺度恢复核心从冻结的 dense pair geometry 开始。在 production system 中，两张图像都被 resize 到 512 像素，并使用冻结的 MASt3R-style ViT-L/16 metric checkpoint 在两个方向上对称运行。冻结 backbone 在可用时提供 point maps、cross-view point maps、confidence maps 和 descriptor maps。

这些张量被归约为每个图像对一个 422 维无元数据描述子。公开脚本暴露相同的 feature-table contract，并包含一个用于 smoke runs 的确定性轻量 feature builder。

## 距离候选与距离感知尺度恢复

四个冻结 distance heads 从同一个描述子产生候选米制距离：

- `distance_head_a`
- `distance_head_b`
- `distance_head_c`
- `distance_head_d`

这些 heads 在不同距离段具有互补行为。尺度代理 `|h_1(z_i)|` 在 release config 中实现为 `gate=head0`，用于通过 calibration-side cut points 将每行分配到七个 range buckets 之一：

```text
5.596117113284452, 14.189340747346009, 33.73144867309181, 53.75,
77.38132781381124, 114.79
```

在每个 bucket 内，四个候选结果由固定凸权重组合。这些权重在 distance relative-error 目标下用 SLSQP 拟合。四个候选的标准差作为 disagreement statistic 保留，供榜单专用 output head 使用。

这个尺度恢复核心是连续且逐行的。论文将这一部分视为可迁移对象。

## PairUAV 榜单专用输出头

归档线上分数额外使用一个根据 benchmark labels 和 scoring 拟合的 PairUAV-specific output head。distance branch 根据以下信息选择固定 affine correction：

- range bucket；
- 尺度恢复距离的符号；
- candidate disagreement statistic 的 segment。

修正后的距离随后 snap 到 2.5 m 网格。这些参数存储在 `models/lastmeter_config.json` 中。

Heading 使用冻结 predictor column `phi`，在冻结 artifact 包中存为 `heading_pred`，然后接论文中的固定变换：

```text
wrap_180(20 * round((1.014 * phi + 1.2) / 20))
```

2.5 m 和 20 degree snap 是面向榜单输出的 resolution，不是控制器 resolution。对于部署或分布偏移场景，应丢弃或重新调整 output head，同时保留尺度恢复 framing。

## Self-Pair 处理

Self-pair 行由同一行内的标识符检测。这些行会被设为：

```text
0.000000 0.000000
```

该修正是逐行的，不使用全局测试集结构。

## 只读分析

论文诊断展示了两个层级：

- 对冻结成对几何做一次全局校准可以消除大部分距离误差；
- 在 relative-error criterion 下，剩余误差具有距离结构，因此需要 range-aware correction。

这些诊断解释了为什么距离感知尺度恢复有用。它们不是在隐藏测试预测或官方反馈上调参的许可。
