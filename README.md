<div align="center">

# RASR: Range-Aware Scale Recovery for PairUAV

**Reproducible code and frozen artifacts for the UAVM 2026 PairUAV submission.**

[中文介绍](docs/README_zh.md) · [Reproduction](docs/REPRODUCTION.md) · [Compliance](compliance/COMPLIANCE.md) · [Citation](#citation)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Task](https://img.shields.io/badge/Task-PairUAV-blue?style=flat-square)

<img src="docs/assets/pipeline.png" alt="RASR pipeline overview" width="92%">

</div>

This repository contains the public release for **RASR**, the inference-frozen
per-pair system from:

> Hongtao Liang, Xinyu Shao, Chenxu Wang, Yiyao Wan, Jiahuan Ji, Fuhui Zhou,
> and Qihui Wu. **Range-Aware Scale Recovery for Monocular Metric Grounding**.
> arXiv preprint, 2026. The arXiv identifier will be added after posting.

Method details are intentionally not duplicated here. Please use the paper for
the algorithmic description and this repository for running and verifying the
release.

## Result

The exact frozen-artifact path reproduces the archived PairUAV submission:

| Metric | Value |
| --- | ---: |
| `final_score` | `0.003189` |
| `distance_rel_error` | `0.003029` |
| `angle_rel_error` | `0.003350` |

Expected `submit.zip` SHA-256:

```text
2f742f2eff83e535b96a8dbd46db370fa3ac0538a9f3e53b684d65c253b34b77
```

## What Is Included

- Public source code for data layout, feature extraction, training, inference,
  packaging, and verification.
- Released model checkpoints and fixed submission configuration under `models/`.
- A transparent from-scratch pipeline from official data.
- An exact frozen-artifact pipeline for reproducing the archived submission.

This repository does **not** include dataset images, private labels, or the
large frozen CSV artifacts. Download the official datasets from their original
sources and the frozen artifacts from the external bundle listed below.

## Quick Start

```bash
conda env create -f environment.yml
conda activate pairuav-lastmeter
bash scripts/smoke_test.sh
```

## Reproduction

See [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for the full commands.

Path A runs the public pipeline from official data:

```bash
bash scripts/reproduce_from_scratch.sh \
  --data-root /path/to/pairuav-data \
  --output-dir outputs/from_scratch \
  --feature-mode smoke \
  --limit 4096 \
  --max-steps 50
```

Path B reproduces the archived submission exactly. Download the complete bundle:

- Baidu Netdisk: https://pan.baidu.com/s/1K1gDMw8mLJwAFC6jO-c9Fg
- Extraction code: `t2c6`
- Bundle file: `pairuav_lastmeter_complete_release_bundle.zip`
- Bundle SHA-256:

```text
4680537d47c93f6b953d20fde6e55b260e4603f02743b06f6ad6900ec0ef729f
```

Then unpack `pairuav_lastmeter_frozen_artifacts.zip` from that bundle and run:

```bash
python scripts/verify_frozen_artifacts.py \
  --artifact-root /path/to/frozen_artifacts \
  --manifest frozen_artifacts_manifest.json

bash scripts/reproduce_from_models.sh \
  --data-root /path/to/pairuav-data \
  --feature-root /path/to/frozen_artifacts \
  --output-dir outputs/exact
```

## Expected Data Layout

Use a data directory outside this repository:

```text
/path/to/pairuav-data/
├── University-1652/
└── PairUAV/
    ├── train/
    ├── test/
    └── metadata/
```

## Compliance

Inference is per ordered image pair. The released path uses fixed parameters
and does not use test-set graph optimization, hidden-neighbor revision, global
assignment, clustering, official test labels, or cross-row bookkeeping. See
[compliance/COMPLIANCE.md](compliance/COMPLIANCE.md).

## Citation

If you use this code or the released artifacts, please cite the paper:

```bibtex
@misc{liang2026rasr,
  title        = {Range-Aware Scale Recovery for Monocular Metric Grounding},
  author       = {Liang, Hongtao and Shao, Xinyu and Wang, Chenxu and Wan, Yiyao and Ji, Jiahuan and Zhou, Fuhui and Wu, Qihui},
  year         = {2026},
  note         = {arXiv preprint, forthcoming},
  url          = {https://github.com/liang-hongtao/rasr-pairuav}
}
```

The repository citation metadata is also available in `CITATION.cff`.
