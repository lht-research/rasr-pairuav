# Method

[中文版本](METHOD_zh.md)

This document follows the terminology used in the paper:

- **RASR:** Range-Aware Scale Recovery.
- **PairUAV:** the last-meter relative-pose benchmark.
- **Scale-recovery core:** the reusable, continuous, pre-snap part of the
  method.
- **Benchmark submission adapter:** the range-bucket residual correction,
  submission quantization, and dataset-tuned constants used to reproduce the
  archived leaderboard submission.

## Problem Contract

Each query is one ordered image pair. The estimator returns one heading and
distance command for that pair:

```text
y_i = f(image_a_i, image_b_i; fixed_parameters)
```

All parameters are fixed before official inference. A prediction for row `i`
does not depend on any other hidden test row. This contract rules out test-set
graph optimization, batch sorting, hidden-neighbor revision, retrieval over the
hidden test set, global assignment, clustering, and cross-sample bookkeeping.

## Frozen Pair Geometry

The scale-recovery core starts from frozen dense pair geometry. In the
production system, both images are resized to 512 pixels and a frozen
MASt3R-style ViT-L/16 metric checkpoint is run symmetrically in both directions.
The frozen backbone provides point maps, cross-view point maps, confidence maps,
and descriptor maps where those outputs are exposed.

These tensors are reduced to one metadata-free 422-D descriptor per pair:
144 point-map dimensions, 28 confidence dimensions, and 250 descriptor-statistic
dimensions. The public scripts expose this same feature-table contract and
include a deterministic lightweight feature builder for smoke runs.

## Distance Candidates and Range-Aware Scale Recovery

Four calibration-fitted, inference-frozen distance heads produce candidate
metric distances from the same descriptor:

- `distance_head_a`
- `distance_head_b`
- `distance_head_c`
- `distance_head_d`

The heads have complementary range behavior. A scale proxy
`|h_1(z_i)|`, implemented by the release config as `gate=head0`, assigns each row
to one of seven range buckets with calibration-side cut points:

```text
5.596117113284452, 14.189340747346009, 33.73144867309181, 53.75,
77.38132781381124, 114.79
```

Within each bucket, the four candidates are combined by fixed convex weights
fit with SLSQP under the distance relative-error objective. The standard
deviation of the four candidates is retained as a disagreement statistic for
the submission adapter.

The bucket-routed mixture defines the candidate pool and disagreement signal.
The transferable claim is that frozen pair geometry contains recoverable scale
and that the residual remains range-structured. This scale-recovery core is
continuous and row-local.

## Benchmark Submission Adapter

The archived online score additionally uses a PairUAV-fit submission adapter
that encodes benchmark labels and scoring. The distance branch applies a fixed
affine correction selected by:

- the range bucket;
- the sign of the scale-recovered distance;
- a segment of the candidate disagreement statistic.

The corrected distance is then quantized to a 2.5 m submission grid. These
parameters are stored in `models/lastmeter_config.json`.

Heading uses a frozen predictor column `phi`, stored as `heading_pred` in the
frozen artifact package, followed by the paper's fixed transform:

```text
wrap_180(20 * round((1.014 * phi + 1.2) / 20))
```

The 2.5 m and 20 degree snaps are submission quantization steps, not controller
resolutions. For deployment or distribution shifts, the adapter should be
discarded or retuned while preserving the scale-recovery framing.

## Self-Pair Handling

Self-pair rows are detected from identifiers in the same row. These rows are set
to:

```text
0.000000 0.000000
```

This correction is row-local and does not use global test-set structure.

## Read-Only Analysis

The paper's diagnostics show two levels:

- a single global calibration of frozen pair geometry removes most of the
  distance error;
- the remaining error is range-structured under the relative-error criterion,
  which motivates range-aware residual correction in the adapter.

Those diagnostics explain why range-aware scale recovery is useful. They are
not a license to tune on hidden test predictions or official feedback.
