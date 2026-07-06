# 复现说明

[English](REPRODUCTION.md)

本页只保留运行发布版本所需的命令。方法细节请阅读论文
**RASR: Range Aware Scale Recovery for Metric UAV Navigation**。

## 数据

本仓库不分发数据集图像或标签。请从官方来源下载 University-1652 和 PairUAV 数据，并放在仓库外：

```text
/path/to/pairuav-data/
├── University-1652/
└── PairUAV/
    ├── train/
    ├── test/
    └── metadata/
```

## 环境

```bash
conda env create -f environment.yml
conda activate pairuav-lastmeter
bash scripts/smoke_test.sh
```

## Path A：公开从头复现路径

这一路径从官方数据运行公开代码，用于透明执行和独立实验；不用于逐 bit 复现归档线上分数。

```bash
bash scripts/reproduce_from_scratch.sh \
  --data-root /path/to/pairuav-data \
  --output-dir outputs/from_scratch \
  --feature-mode smoke \
  --limit 4096 \
  --max-steps 50
```

如需 production visual features：

```bash
bash scripts/reproduce_from_scratch.sh \
  --data-root /path/to/pairuav-data \
  --output-dir outputs/from_scratch_production \
  --feature-mode production \
  --runtime-root /path/to/mast3r_probe_runtime \
  --model-path /path/to/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
  --device cuda:0
```

## Path B：精确冻结 Artifact 复现

这一路径用于精确复现归档 PairUAV 提交。

预期线上结果：

```text
final_score          0.003189
distance_rel_error   0.003029
angle_rel_error      0.003350
```

下载云盘 archive：

- 百度网盘：https://pan.baidu.com/s/1S0-re-dET6ELinNfDkSPyA?pwd=6fgk（提取码：`6fgk`）
- Google Drive：https://drive.google.com/file/d/1T31XQwR4hr6naZ6zefX4EVLlMienXPBI/view?usp=sharing
- Archive 文件：`rasr_pairuav_frozen_artifacts_v1.0.zip`
- Archive SHA-256：

```text
37f2111f5b19060281ce8b0942c655f3b18b2cc028c25f029029fd174f657019
```

从 archive 中解压 frozen artifact zip。artifact 目录应包含：

```text
distance_head_a.csv
distance_head_b.csv
distance_head_c.csv
distance_head_d.csv
heading_predictions.csv
```

校验并运行：

```bash
python scripts/verify_frozen_artifacts.py \
  --artifact-root /path/to/frozen_artifacts \
  --manifest frozen_artifacts_manifest.json

bash scripts/reproduce_from_models.sh \
  --data-root /path/to/pairuav-data \
  --feature-root /path/to/frozen_artifacts \
  --output-dir outputs/exact
```

预期输出：

```text
outputs/exact/submit.zip
```

预期 `submit.zip` SHA-256：

```text
2f742f2eff83e535b96a8dbd46db370fa3ac0538a9f3e53b684d65c253b34b77
```
