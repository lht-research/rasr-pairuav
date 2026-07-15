<div align="center">

# RASR: Range-Aware Scale Recovery for Metric UAV Navigation

**Reproducible code and frozen artifacts for the RASR result on the UAVs in Multimedia 2026 PairUAV official online evaluation.**

[中文介绍](docs/README_zh.md) · [Reproduction](docs/REPRODUCTION.md) · [Compliance](compliance/COMPLIANCE.md) · [Citation](#citation)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Task](https://img.shields.io/badge/Task-PairUAV-blue?style=flat-square)
[![arXiv](https://img.shields.io/badge/arXiv-2607.09815-b31b1b?style=flat-square)](https://arxiv.org/abs/2607.09815)

<img src="docs/assets/pipeline.png" alt="RASR pipeline overview" width="92%">

</div>

This repository contains the public release for **RASR**, a per-pair system for
metric distance and heading estimation in image-goal UAV navigation:

> Hongtao Liang, Xinyu Shao, Chenxu Wang, Yiyao Wan, Jiahuan Ji, Fangwei Ye,
> Fuhui Zhou, and Qihui Wu. **RASR: Range-Aware Scale Recovery for Metric UAV Navigation**.
> arXiv:2607.09815, 2026. https://arxiv.org/abs/2607.09815

RASR complements global scale calibration with range-aware residual correction.
Method details are intentionally not duplicated here; please use the paper for
the algorithmic description and this repository for running and verifying the
release.

## Result

The exact frozen-artifact path reproduces the UAVs in Multimedia 2026 PairUAV
official online evaluation result:

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
- Released model checkpoints and fixed task-specific command calibration under `models/`.
- A transparent from-scratch pipeline from official data.
- An exact frozen-artifact pipeline for reproducing the official online evaluation result.

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

Path B reproduces the official online evaluation result exactly. Download the cloud archive:

- Baidu Netdisk: https://pan.baidu.com/s/1S0-re-dET6ELinNfDkSPyA?pwd=6fgk (extraction code: `6fgk`)
- Google Drive: https://drive.google.com/file/d/1T31XQwR4hr6naZ6zefX4EVLlMienXPBI/view?usp=sharing
- Archive file: `rasr_pairuav_frozen_artifacts_v1.0.zip`
- Archive SHA-256:

```text
37f2111f5b19060281ce8b0942c655f3b18b2cc028c25f029029fd174f657019
```

Then unpack the frozen artifact zip from that archive and run:

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
  title        = {RASR: Range-Aware Scale Recovery for Metric UAV Navigation},
  author       = {Liang, Hongtao and Shao, Xinyu and Wang, Chenxu and Wan, Yiyao and Ji, Jiahuan and Ye, Fangwei and Zhou, Fuhui and Wu, Qihui},
  year         = {2026},
  eprint       = {2607.09815},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi          = {10.48550/arXiv.2607.09815},
  url          = {https://arxiv.org/abs/2607.09815}
}
```

The repository citation metadata is also available in `CITATION.cff`.
