# Models

[中文版本](README_zh.md)

This directory contains the frozen checkpoints and fixed PairUAV
submission-adapter configuration used by the archived RASR submission.

- `distance_head_a.pt` ... `distance_head_d.pt`: four frozen distance candidates
  over the 422-D pair descriptor.
- `lastmeter_config.json`: selected `gate=head0` range buckets, convex weights,
  bucket/sign/std-segment affine corrections, and 2.5 m distance snap.
- `heading_transform.json`: fixed heading scale, bias, 20 degree snap, and wrap.
- `heading_transform_identity.json`: identity transform used by the public
  from-scratch heading baseline.

Expected exact `submit.zip` SHA-256:

```text
2f742f2eff83e535b96a8dbd46db370fa3ac0538a9f3e53b684d65c253b34b77
```

Exact reproduction additionally requires the frozen CSV artifact package
described in `../frozen_artifacts_manifest.json`.
