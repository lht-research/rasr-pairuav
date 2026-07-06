<div align="center">

# RASR：面向 PairUAV 的距离感知尺度恢复

**UAVM 2026 PairUAV 提交的可复现代码和冻结 artifacts。**

[English README](../README.md) · [复现说明](REPRODUCTION_zh.md) · [合规说明](../compliance/COMPLIANCE_zh.md) · [引用方式](#引用方式)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Task](https://img.shields.io/badge/Task-PairUAV-blue?style=flat-square)

<img src="assets/pipeline.png" alt="RASR pipeline overview" width="92%">

</div>

本仓库是 **RASR** 的公开发布版本。RASR 是下列论文中的推理时冻结、逐图像对系统：

> Hongtao Liang, Xinyu Shao, Chenxu Wang, Yiyao Wan, Jiahuan Ji, Fuhui Zhou,
> and Qihui Wu. **Range Aware Scale Recovery for Monocular Metric Grounding**.
> arXiv preprint, 2026. arXiv 编号将在上传后补充。

方法细节不在仓库文档中重复展开。算法说明请以论文为准；本仓库只保留运行、复现、验证和引用所需信息。

## 结果

精确冻结 artifact 路径可以复现归档 PairUAV 提交：

| Metric | Value |
| --- | ---: |
| `final_score` | `0.003189` |
| `distance_rel_error` | `0.003029` |
| `angle_rel_error` | `0.003350` |

预期 `submit.zip` SHA-256：

```text
2f742f2eff83e535b96a8dbd46db370fa3ac0538a9f3e53b684d65c253b34b77
```

## 仓库包含内容

- 数据目录、特征提取、训练、推理、打包和验证的公开源码。
- `models/` 中的发布 checkpoints 和固定协议特定校准配置。
- 从官方数据开始运行的透明 from-scratch 路径。
- 复现归档提交的精确 frozen-artifact 路径。

本仓库不包含数据集图像、私有标签或大型冻结 CSV artifacts。请从官方来源获取数据集，并从下方外部 bundle 下载冻结 artifacts。

## 快速开始

```bash
conda env create -f environment.yml
conda activate pairuav-lastmeter
bash scripts/smoke_test.sh
```

## 复现

完整命令见 [docs/REPRODUCTION_zh.md](REPRODUCTION_zh.md)。

Path A 从官方数据运行公开流程：

```bash
bash scripts/reproduce_from_scratch.sh \
  --data-root /path/to/pairuav-data \
  --output-dir outputs/from_scratch \
  --feature-mode smoke \
  --limit 4096 \
  --max-steps 50
```

Path B 精确复现归档提交。先下载完整 bundle：

- 百度网盘：https://pan.baidu.com/s/1_yCx9-8w6vvUesT1ES_onA
- 提取码：`yk7d`
- Bundle 文件：`pairuav_lastmeter_complete_release_bundle.zip`
- Bundle SHA-256：

```text
37f2111f5b19060281ce8b0942c655f3b18b2cc028c25f029029fd174f657019
```

然后从 bundle 中解压 `pairuav_lastmeter_frozen_artifacts.zip`，并运行：

```bash
python scripts/verify_frozen_artifacts.py \
  --artifact-root /path/to/frozen_artifacts \
  --manifest frozen_artifacts_manifest.json

bash scripts/reproduce_from_models.sh \
  --data-root /path/to/pairuav-data \
  --feature-root /path/to/frozen_artifacts \
  --output-dir outputs/exact
```

## 数据目录

请在仓库外准备数据目录：

```text
/path/to/pairuav-data/
├── University-1652/
└── PairUAV/
    ├── train/
    ├── test/
    └── metadata/
```

## 合规说明

推理以有序图像对为单位。发布路径使用固定参数，不使用测试集图优化、隐藏邻居修正、全局分配、聚类、官方测试标签或跨行 bookkeeping。见 [compliance/COMPLIANCE_zh.md](../compliance/COMPLIANCE_zh.md)。

## 引用方式

如果使用本代码或发布 artifacts，请引用论文：

```bibtex
@misc{liang2026rasr,
  title        = {Range Aware Scale Recovery for Monocular Metric Grounding},
  author       = {Liang, Hongtao and Shao, Xinyu and Wang, Chenxu and Wan, Yiyao and Ji, Jiahuan and Zhou, Fuhui and Wu, Qihui},
  year         = {2026},
  note         = {arXiv preprint, forthcoming},
  url          = {https://github.com/liang-hongtao/rasr-pairuav}
}
```

仓库引用元数据也放在 `CITATION.cff` 中。
