# Reproduction

[中文版本](REPRODUCTION_zh.md)

This page keeps only the commands needed to run the release. For method details,
please read the paper: **RASR: Range Aware Scale Recovery for Metric UAV Navigation**.

## Data

This repository does not distribute dataset images or labels. Download
University-1652 and the official PairUAV data from their original sources, then
place them outside the repository:

```text
/path/to/pairuav-data/
├── University-1652/
└── PairUAV/
    ├── train/
    ├── test/
    └── metadata/
```

## Environment

```bash
conda env create -f environment.yml
conda activate pairuav-lastmeter
bash scripts/smoke_test.sh
```

## Path A: Public From-Scratch Pipeline

This path runs the public code from official data. It is intended for
transparent execution and experimentation, not bit-exact reproduction of the
archived online score.

```bash
bash scripts/reproduce_from_scratch.sh \
  --data-root /path/to/pairuav-data \
  --output-dir outputs/from_scratch \
  --feature-mode smoke \
  --limit 4096 \
  --max-steps 50
```

For production visual features:

```bash
bash scripts/reproduce_from_scratch.sh \
  --data-root /path/to/pairuav-data \
  --output-dir outputs/from_scratch_production \
  --feature-mode production \
  --runtime-root /path/to/mast3r_probe_runtime \
  --model-path /path/to/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
  --device cuda:0
```

## Path B: Exact Frozen-Artifact Reproduction

Use this path to reproduce the archived PairUAV submission exactly.

Expected online result:

```text
final_score          0.003189
distance_rel_error   0.003029
angle_rel_error      0.003350
```

Download the cloud archive:

- Baidu Netdisk: https://pan.baidu.com/s/1S0-re-dET6ELinNfDkSPyA?pwd=6fgk (extraction code: `6fgk`)
- Google Drive: https://drive.google.com/file/d/1T31XQwR4hr6naZ6zefX4EVLlMienXPBI/view?usp=sharing
- Archive file: `rasr_pairuav_frozen_artifacts_v1.0.zip` (Baidu may display `rasr_pairuav_frozen_artifacts_v1.0.zip.zip`)
- Archive SHA-256:

```text
37f2111f5b19060281ce8b0942c655f3b18b2cc028c25f029029fd174f657019
```

Unpack the frozen artifact zip from the archive. The artifact directory should
contain:

```text
distance_head_a.csv
distance_head_b.csv
distance_head_c.csv
distance_head_d.csv
heading_predictions.csv
```

Verify and run:

```bash
python scripts/verify_frozen_artifacts.py \
  --artifact-root /path/to/frozen_artifacts \
  --manifest frozen_artifacts_manifest.json

bash scripts/reproduce_from_models.sh \
  --data-root /path/to/pairuav-data \
  --feature-root /path/to/frozen_artifacts \
  --output-dir outputs/exact
```

Expected output:

```text
outputs/exact/submit.zip
```

Expected `submit.zip` SHA-256:

```text
2f742f2eff83e535b96a8dbd46db370fa3ac0538a9f3e53b684d65c253b34b77
```
