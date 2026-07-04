# 合规说明

[English](COMPLIANCE.md)

本仓库的组织方式使 reviewers 可以验证推理是一个 per-pair function，与论文中声明的 contract 一致。

## Per-Pair 推理

每个正式行是一个有序图像对。Distance heads 只消耗该行的 feature vector。Heading stage 只消耗该行的一个对齐 prediction column。所有 learned weights 和 adapter parameters 都在正式推理前固定。

## 距离感知距离组合

尺度恢复阶段只使用：

- 当前行的四个 distance predictions；
- 正式推理前学习好的固定 bucket boundaries；
- 在 training/calibration data 上学习好的固定 convex bucket weights。

对每一行，release config 使用尺度代理 `|h_1(z_i)|` 选择一个 bucket；它在 `models/lastmeter_config.json` 中实现为 `gate=head0`，并返回相应的加权和。PairUAV submission adapter 随后应用固定 bucket/sign/std-segment parameters 和 distance submission quantization。在做出这个决定时，它不会读取任何其他隐藏测试行。

## Self-Pair 处理

Self-pair 行通过比较同一行中的标识符检测。一旦检测到 self-pair，heading 和 distance 只会对该行设为零。预计算 row-index list 仅作为显式 legacy audit fallback 保留，用于没有携带 image identifiers 的 CSV；默认不使用。

## Heading 来源

精确复现使用一个冻结 heading column。`models/heading_weights.json` 中的 manifest 记录了上游 heading source 是逐行 heading predictions 的固定 circular weighted average，并且不使用官方标签、测试图、跨测试关系、全局分配或聚类。`scripts/verify_heading_provenance.py` 会检查 manifest 和可选 heading CSV contract。

## 明确未使用

发布的推理路径不使用：

- test-set graph optimization；
- batch sorting；
- hidden-neighbor revision；
- retrieval over the hidden test set；
- global assignment；
- closure constraints across test samples；
- clustering over test samples；
- fitting or tuning on official test predictions；
- official test labels。

## 数据分发

本仓库不分发数据集图像或私有标签。用户必须从官方来源获取公开数据集，并提供本地路径。

## Review Checklist

- `inference/run_inference.py` 使用固定的 `lastmeter_config.json` 和 `heading_transform.json` 串联精确冻结 artifact 推理。
- `models_src/per_bucket_stacker.py` 实现通用逐行凸组合 stacker，供公开训练和审计工具使用。
- `inference/selfpair.py` 实现逐行 self-pair detection。
- `inference/package.py` 执行 row-aligned heading/distance packaging。
- `scripts/verify_heading_provenance.py` 检查冻结 heading manifest 和可选 heading CSV contract。
- `scripts/verify_result.py` 检查 row count、self-pair rows、zip root 和 archive hash。
