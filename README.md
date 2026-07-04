<div align="center">

# RASR: Range-Aware Scale Recovery for PairUAV

**Frozen pair geometry, reusable scale recovery, and a separated PairUAV submission adapter.**

[中文介绍](docs/README_zh.md) · [Reproduction](docs/REPRODUCTION.md) · [Method](docs/METHOD.md) · [Compliance](compliance/COMPLIANCE.md)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Task](https://img.shields.io/badge/Task-PairUAV-blue?style=flat-square)
![Reproducibility](https://img.shields.io/badge/Reproduction-Exact%20%2B%20From--Scratch-orange?style=flat-square)

<img src="docs/assets/pipeline.png" alt="RASR pipeline overview" width="92%">

</div>

This repository contains the public implementation for **RASR
(Range-Aware Scale Recovery)**, the inference-frozen per-pair system described in
`Range-Aware Scale Recovery for Monocular Metric Grounding`.

The paper frames PairUAV as monocular metric grounding for last-meter
relative-pose estimation: from one ordered image pair, predict the heading and
distance command that moves the agent toward the goal view. Dense frozen pair
geometry already carries relative structure; the main bottleneck is recovering
metric scale under a relative-error objective.

RASR keeps two components separate:

- **Scale-recovery core:** frozen pair geometry, a metadata-free 422-D
  descriptor, global calibration, four calibration-fitted distance candidates,
  and a row-local convex mixture with a disagreement statistic.
- **PairUAV benchmark submission adapter:** range-bucket residual correction,
  submission quantization, and dataset-tuned constants used for the archived
  leaderboard submission. This adapter is included for reproduction, but it is
  not claimed to transfer and should be discarded or retuned for deployment.

Every official prediction is a function of one ordered image pair with all
parameters fixed before inference. The released path does not use test-set graph
optimization, batch sorting, hidden-neighbor revision, retrieval over the hidden
test set, global assignment, clustering, or cross-sample bookkeeping.

## Reproduction Paths

This repository supports two paths:

- **Path A, public from-scratch pipeline:** build pair tables from official
  data, extract pair features, train public distance heads and a simple public
  heading baseline, run inference, and package a valid PairUAV submission. This
  path is for transparent end-to-end execution and independent experimentation;
  it is not expected to reproduce the archived online score bit for bit.
- **Path B, exact frozen-artifact path:** download the frozen intermediate CSV
  package, verify it, and reproduce the archived final submission bit for bit.
  Use this path when you need the exact online result reported in the paper.

The exact frozen-artifact path reproduces this `submit.zip` SHA-256:

```text
2f742f2eff83e535b96a8dbd46db370fa3ac0538a9f3e53b684d65c253b34b77
```

The corresponding online PairUAV result is:

| Metric | Value |
| --- | ---: |
| `final_score` | `0.003189` |
| `distance_rel_error` | `0.003029` |
| `angle_rel_error` | `0.003350` |

See `docs/REPRODUCTION.md` for the full two-path protocol.

## Documentation

| Topic | English | Chinese |
| --- | --- | --- |
| Project overview | [README.md](README.md) | [docs/README_zh.md](docs/README_zh.md) |
| Reproduction protocol | [docs/REPRODUCTION.md](docs/REPRODUCTION.md) | [docs/REPRODUCTION_zh.md](docs/REPRODUCTION_zh.md) |
| Method | [docs/METHOD.md](docs/METHOD.md) | [docs/METHOD_zh.md](docs/METHOD_zh.md) |
| Archived implementation | [docs/METHOD_RELEASE.md](docs/METHOD_RELEASE.md) | [docs/METHOD_RELEASE_zh.md](docs/METHOD_RELEASE_zh.md) |
| Heading provenance | [docs/HEADING_PROVENANCE.md](docs/HEADING_PROVENANCE.md) | [docs/HEADING_PROVENANCE_zh.md](docs/HEADING_PROVENANCE_zh.md) |
| Compliance | [compliance/COMPLIANCE.md](compliance/COMPLIANCE.md) | [compliance/COMPLIANCE_zh.md](compliance/COMPLIANCE_zh.md) |
| Data request and layout | [data_processing/01_request_and_layout.md](data_processing/01_request_and_layout.md) | [data_processing/01_request_and_layout_zh.md](data_processing/01_request_and_layout_zh.md) |
| Models | [models/README.md](models/README.md) | [models/README_zh.md](models/README_zh.md) |
| License | [LICENSE](LICENSE) | [LICENSE_zh.md](LICENSE_zh.md) |

## Method Summary

The scale-recovery core starts from dense pairwise geometry extracted by a
frozen MASt3R-style ViT-L/16 metric checkpoint. Both images are resized to 512
pixels and symmetric pair inference is run in both directions. Point maps,
cross-view point maps, confidence maps, and descriptor maps are reduced to one
metadata-free 422-D descriptor per pair.

Four calibration-fitted, inference-frozen distance heads produce candidates
with complementary range behavior:

- `distance_head_a`
- `distance_head_b`
- `distance_head_c`
- `distance_head_d`

A scale proxy `|h_1(z_i)|`, implemented as `gate=head0`, assigns each pair to
one of seven range buckets using calibration-side cut points:

```text
5.596117113284452, 14.189340747346009, 33.73144867309181, 53.75,
77.38132781381124, 114.79
```

Within each bucket, the four candidates are combined by fixed convex weights fit
with SLSQP under the distance relative-error objective. The standard deviation
of the four candidates is retained as a disagreement statistic. The routed
mixture defines a candidate pool and disagreement signal; the benchmark-specific
gain is realized by the separate submission adapter.

The PairUAV benchmark submission adapter then applies fixed
bucket/sign/std-segment affine corrections and 2.5 m submission quantization
from `models/lastmeter_config.json`. Heading uses a frozen heading source
followed by the paper's fixed transform:

```text
wrap_180(20 * round((1.014 * phi + 1.2) / 20))
```

Self-pair rows are detected from identifiers in the same row and set to zero
heading and zero distance for that row only.

<p align="center">
  <img src="docs/assets/range_calibration.png" alt="Range-dependent calibration diagnostic" width="58%">
</p>

## Data

This repository does not distribute dataset images or labels. Obtain:

- University-1652 from the official University-1652 project page.
- PairUAV train/test data from the official benchmark release.

See `data_processing/01_request_and_layout.md` for the expected local layout.

## Environment

```bash
conda env create -f environment.yml
conda activate pairuav-lastmeter
```

A lightweight smoke test verifies the stacker, circular heading utilities, and
self-pair handling:

```bash
bash scripts/smoke_test.sh
```

## Path A: Public From-Scratch Pipeline

The public from-scratch script wires the full transparent pipeline:

```bash
bash scripts/reproduce_from_scratch.sh \
  --data-root /path/to/pairuav-data \
  --output-dir outputs/from_scratch \
  --feature-mode smoke \
  --limit 4096 \
  --max-steps 50
```

Full production-quality feature extraction can take multiple days depending on
GPU count and storage bandwidth. For production features, pass
`--feature-mode production`, `--runtime-root`, and `--model-path`:

```bash
bash scripts/reproduce_from_scratch.sh \
  --data-root /path/to/pairuav-data \
  --output-dir outputs/from_scratch_production \
  --feature-mode production \
  --runtime-root /path/to/mast3r_probe_runtime \
  --model-path /path/to/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
  --device cuda:0
```

`data_processing/03_extract_features.py` has two modes:

- `--mode smoke`: deterministic image statistics for fast local pipeline checks.
- `--mode production`: frozen visual pair features matching the released
  checkpoint contract. It requires the frozen visual runtime and model path, and
  supports sharding, limits, and resume.

The production extractor defaults to strict per-pair forward passes. Use
`--batched-pair-forward` only for speed-oriented experiments where tiny numeric
drift is acceptable.

The public from-scratch path trains a simple row-local sin/cos heading baseline
from pair features. It is meant to expose the full workflow from official data;
use Path B for the exact archived score.

## Path B: Exact Frozen-Artifact Reproduction

Path B is the only path that requires the external artifact bundle. The bundle
contains the source package, frozen row-aligned prediction CSVs for exact
reproduction, checksum file, and archived submit package.

- Baidu Netdisk: https://pan.baidu.com/s/1K1gDMw8mLJwAFC6jO-c9Fg
- Extraction code: `t2c6`
- Bundle file: `pairuav_lastmeter_complete_release_bundle.zip`
- Bundle SHA-256:

```text
4680537d47c93f6b953d20fde6e55b260e4603f02743b06f6ad6900ec0ef729f
```

Unpack `pairuav_lastmeter_frozen_artifacts.zip` from the complete bundle. The
unpacked directory must contain four distance-head prediction CSVs and one
heading CSV:

```text
/path/to/frozen_artifacts/
├── distance_head_a.csv
├── distance_head_b.csv
├── distance_head_c.csv
├── distance_head_d.csv
└── heading_predictions.csv
```

Each distance CSV must include `range_pred`; the heading CSV must include
`heading_pred`. If metadata columns such as `manifest_index`, `scene_id`,
`pair_id`, `image_a`, and `image_b` are present, they are checked for alignment
across heads and used for row-local self-pair handling.

Verify the downloaded artifacts:

```bash
python scripts/verify_frozen_artifacts.py \
  --artifact-root /path/to/frozen_artifacts \
  --manifest frozen_artifacts_manifest.json
```

Run exact reproduction:

```bash
bash scripts/reproduce_from_models.sh \
  --data-root /path/to/pairuav-data \
  --feature-root /path/to/frozen_artifacts \
  --output-dir outputs/exact
```

Full verification checks row count, self-pair zeroing, zip root contents, and
the expected archive hash. The exact hash is anchored by the five frozen,
row-aligned prediction CSVs.

## Recomputing Distance Heads From Frozen Pair Features

If you have a 422-dimensional pair feature cache compatible with the released
checkpoints, the released-model script can recompute the four distance-head CSVs
before applying the same PairUAV submission adapter:

```bash
bash scripts/reproduce_from_models.sh \
  --data-root /path/to/pairuav-data \
  --feature-root /path/to/features \
  --pair-feature-csv /path/to/pair_features.csv \
  --heading-csv /path/to/heading_predictions.csv \
  --output-dir outputs/released-models
```

Use `--pair-feature-glob 'features/shard_*.csv'` or `--pair-feature-dir
features/` for sharded caches. By default, checkpoint inference uses the
per-head batch sizes used for the archived exports; pass
`--no-legacy-batch-policy` only for experiments where tiny runtime-dependent
numeric drift is acceptable.

For a small check:

```bash
bash scripts/reproduce_from_models.sh \
  --data-root /path/to/pairuav-data \
  --feature-root /path/to/frozen_artifacts \
  --output-dir outputs/smoke \
  --limit 4096
```

## Verification Commands

Package from aligned heading and distance CSVs:

```bash
python inference/package.py from-csv \
  --heading-csv /path/to/heading_predictions.csv \
  --distance-csv /path/to/distance_col.csv \
  --output-dir outputs/package
```

Verify a result archive:

```bash
python scripts/verify_result.py \
  --result-zip outputs/package/result.zip
```

Verify that a feature CSV matches the released checkpoint schema, optionally
against a frozen reference cache:

```bash
python scripts/verify_features.py \
  --feature-csv outputs/features/pair_features.csv \
  --reference-csv /path/to/frozen_pair_features.csv \
  --checkpoint models/distance_head_a.pt \
  --limit 4096 \
  --atol 1e-5
```

## Repository Layout

```text
pairuav_lastmeter/
├── compliance/
├── configs/
├── data_processing/
├── docs/
├── inference/
├── models/
├── models_src/
└── scripts/
```

## Compliance

Inference is per pair. The distance-head and adapter stages process one
ordered image pair at a time using only that row's predictions and fixed
train/calibration-derived parameters. See `compliance/COMPLIANCE.md` and
`docs/HEADING_PROVENANCE.md`.
