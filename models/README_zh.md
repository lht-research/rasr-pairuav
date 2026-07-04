# 模型

[English](README.md)

本目录包含归档 RASR 提交使用的冻结 checkpoints 和固定 PairUAV submission-adapter configuration。

- `distance_head_a.pt` ... `distance_head_d.pt`：基于 422 维 pair descriptor 的四个冻结 distance candidates。
- `lastmeter_config.json`：选择的 `gate=head0` range buckets、convex weights、bucket/sign/std-segment affine corrections 和 2.5 m distance snap。
- `heading_transform.json`：固定 heading scale、bias、20 degree snap 和 wrap。
- `heading_transform_identity.json`：公开 from-scratch heading baseline 使用的 identity transform。

预期精确 `submit.zip` SHA-256：

```text
2f742f2eff83e535b96a8dbd46db370fa3ac0538a9f3e53b684d65c253b34b77
```

精确复现还需要 `../frozen_artifacts_manifest.json` 中描述的冻结 CSV artifact package。
