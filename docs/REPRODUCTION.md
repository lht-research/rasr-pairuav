# Reproduction

[中文版本](REPRODUCTION_zh.md)

This release supports two reproducibility levels. The distinction matches the
paper: RASR's scale-recovery core can be run from public code and official data,
while the exact archived online score requires the frozen PairUAV
benchmark-specific prediction artifacts.

## Path A: Public From-Scratch Pipeline

Use this path when you want to run the public code from official data without
downloading the frozen prediction CSVs.

```bash
bash scripts/reproduce_from_scratch.sh \
  --data-root /path/to/pairuav-data \
  --output-dir outputs/from_scratch \
  --feature-mode smoke \
  --limit 4096 \
  --max-steps 50
```

For production features, use:

```bash
bash scripts/reproduce_from_scratch.sh \
  --data-root /path/to/pairuav-data \
  --output-dir outputs/from_scratch_production \
  --feature-mode production \
  --runtime-root /path/to/mast3r_probe_runtime \
  --model-path /path/to/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
  --device cuda:0
```

This path:

- builds train and test pair tables from the local PairUAV layout;
- extracts pair features;
- trains four public distance heads;
- trains a public row-local heading baseline;
- predicts test distance and heading;
- packages a valid `submit.zip`.

This path is intended for method transparency and independent experimentation.
It is not expected to match the archived SHA-256 because the official online
result in the paper uses frozen production prediction artifacts, including a
stronger frozen heading column.

## Path B: Exact Frozen-Artifact Reproduction

Use this path when you need the exact archived submission archive and the
reported online PairUAV result:

```text
final_score          0.003189
distance_rel_error   0.003029
angle_rel_error      0.003350
```

This path requires the frozen intermediate artifact package:

```text
pairuav_lastmeter_frozen_artifacts.zip
```

The easiest way to obtain it is to download the complete release bundle:

- Baidu Netdisk: https://pan.baidu.com/s/1K1gDMw8mLJwAFC6jO-c9Fg
- Extraction code: `t2c6`
- Bundle file: `pairuav_lastmeter_complete_release_bundle.zip`
- Bundle SHA-256:

```text
4680537d47c93f6b953d20fde6e55b260e4603f02743b06f6ad6900ec0ef729f
```

After downloading the bundle, unpack `pairuav_lastmeter_frozen_artifacts.zip`
into any directory, for example:

```text
/path/to/frozen_artifacts/
├── distance_head_a.csv
├── distance_head_b.csv
├── distance_head_c.csv
├── distance_head_d.csv
└── heading_predictions.csv
```

Verify the downloaded files:

```bash
python scripts/verify_frozen_artifacts.py \
  --artifact-root /path/to/frozen_artifacts \
  --manifest frozen_artifacts_manifest.json
```

Then reproduce the exact archive:

```bash
bash scripts/reproduce_from_models.sh \
  --data-root /path/to/pairuav-data \
  --feature-root /path/to/frozen_artifacts \
  --output-dir outputs/exact
```

The expected output is:

```text
outputs/exact/submit.zip
```

Expected SHA-256:

```text
2f742f2eff83e535b96a8dbd46db370fa3ac0538a9f3e53b684d65c253b34b77
```

The expected `result.txt` SHA-256 inside the archive is:

```text
44b3be45f88893d58a17a8585147f31534d81992b7aa1f8a00006affa57946e8
```

This path is bit-exact because it freezes the five row-aligned prediction CSVs
used by the archived PairUAV submission.

## Artifact Manifest

The exact path is governed by `frozen_artifacts_manifest.json`. The minimum
artifact package contains:

```text
distance_head_a.csv      221939325 bytes
distance_head_b.csv      221939214 bytes
distance_head_c.csv      221930066 bytes
distance_head_d.csv      221973679 bytes
heading_predictions.csv   32855171 bytes
```

The five files are about 878 MiB uncompressed and about 359 MiB as ZIP deflate.
