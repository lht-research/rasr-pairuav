# 复现说明

[English](REPRODUCTION.md)

本发布版本支持两个层级的复现。这个区分与论文保持一致：RASR 的尺度恢复核心可以从公开代码和官方数据运行；而精确的归档线上分数需要冻结的 PairUAV 提交适配器预测 artifact。

## Path A：公开从头复现路径

当你希望只使用官方数据运行公开代码、且不下载冻结 prediction CSV 时，使用这一路径。

```bash
bash scripts/reproduce_from_scratch.sh \
  --data-root /path/to/pairuav-data \
  --output-dir outputs/from_scratch \
  --feature-mode smoke \
  --limit 4096 \
  --max-steps 50
```

如需 production features，使用：

```bash
bash scripts/reproduce_from_scratch.sh \
  --data-root /path/to/pairuav-data \
  --output-dir outputs/from_scratch_production \
  --feature-mode production \
  --runtime-root /path/to/mast3r_probe_runtime \
  --model-path /path/to/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
  --device cuda:0
```

这一路径会：

- 从本地 PairUAV 目录构建 train 和 test pair tables；
- 提取 pair features；
- 训练四个公开 distance heads；
- 训练一个公开的逐行 heading baseline；
- 预测测试集 distance 和 heading；
- 打包一个合法的 `submit.zip`。

这一路径用于方法透明和独立实验。它不预期匹配归档 SHA-256，因为论文中的正式线上结果使用冻结的 production prediction artifacts，其中包括更强的冻结 heading column。

## Path B：精确冻结 Artifact 复现

当你需要精确复现归档提交压缩包和论文报告的 PairUAV 线上结果时，使用这一路径：

```text
final_score          0.003189
distance_rel_error   0.003029
angle_rel_error      0.003350
```

这一路径需要冻结中间 artifact 包：

```text
pairuav_lastmeter_frozen_artifacts.zip
```

最简单的获取方式是下载完整 release bundle：

- 百度网盘：https://pan.baidu.com/s/1K1gDMw8mLJwAFC6jO-c9Fg
- 提取码：`t2c6`
- Bundle 文件：`pairuav_lastmeter_complete_release_bundle.zip`
- Bundle SHA-256：

```text
4680537d47c93f6b953d20fde6e55b260e4603f02743b06f6ad6900ec0ef729f
```

下载 bundle 后，将其中的 `pairuav_lastmeter_frozen_artifacts.zip` 解压到任意目录，例如：

```text
/path/to/frozen_artifacts/
├── distance_head_a.csv
├── distance_head_b.csv
├── distance_head_c.csv
├── distance_head_d.csv
└── heading_predictions.csv
```

先校验下载文件：

```bash
python scripts/verify_frozen_artifacts.py \
  --artifact-root /path/to/frozen_artifacts \
  --manifest frozen_artifacts_manifest.json
```

然后复现精确 archive：

```bash
bash scripts/reproduce_from_models.sh \
  --data-root /path/to/pairuav-data \
  --feature-root /path/to/frozen_artifacts \
  --output-dir outputs/exact
```

预期输出为：

```text
outputs/exact/submit.zip
```

预期 SHA-256：

```text
2f742f2eff83e535b96a8dbd46db370fa3ac0538a9f3e53b684d65c253b34b77
```

archive 内 `result.txt` 的预期 SHA-256 为：

```text
44b3be45f88893d58a17a8585147f31534d81992b7aa1f8a00006affa57946e8
```

这一路径是 bit-exact 的，因为它冻结了归档 PairUAV 提交使用的五个逐行对齐 prediction CSV。

## Artifact Manifest

精确路径由 `frozen_artifacts_manifest.json` 管理。最小 artifact 包包含：

```text
distance_head_a.csv      221939325 bytes
distance_head_b.csv      221939214 bytes
distance_head_c.csv      221930066 bytes
distance_head_d.csv      221973679 bytes
heading_predictions.csv   32855171 bytes
```

这五个文件未压缩约 878 MiB，使用 ZIP deflate 压缩后约 359 MiB。
