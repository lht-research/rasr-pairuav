# Compliance Statement

[中文版本](COMPLIANCE_zh.md)

This repository is organized so reviewers can verify that inference is a
per-pair function, matching the contract stated in the paper.

## Per-Pair Inference

Each official row is one ordered image pair. Distance heads consume only the
feature vector for that row. The heading stage consumes one aligned prediction
column for that row. All learned weights and output-head parameters are fixed
before official inference.

## Range-Aware Distance Combination

The scale-recovery stage uses only:

- the four distance predictions for the current row;
- fixed bucket boundaries learned before official inference;
- fixed convex bucket weights learned on training/calibration data.

For each row, the release config selects a bucket using the scale proxy
`|h_1(z_i)|`, implemented as `gate=head0` in `models/lastmeter_config.json`, and
returns the corresponding weighted sum. The PairUAV output head then applies
fixed bucket/sign/std-segment parameters and the distance snap. It does not read
any other hidden test row while making that decision.

## Self-Pair Handling

Self-pair rows are detected by comparing identifiers from the same row. When a
self-pair is detected, heading and distance are set to zero for that row only. A
precomputed row-index list is retained only as an explicit legacy audit fallback
for CSVs that do not carry image identifiers; it is not used by default.

## Heading Provenance

Exact reproduction consumes one frozen heading column. The manifest in
`models/heading_weights.json` records that the upstream heading source is a
fixed circular weighted average over per-row heading predictions and does not use
official labels, test graphs, cross-test relationships, global assignment, or
clustering. `scripts/verify_heading_provenance.py` checks the manifest and the
optional heading CSV contract.

## Explicitly Not Used

The released inference path does not use:

- test-set graph optimization;
- batch sorting;
- hidden-neighbor revision;
- retrieval over the hidden test set;
- global assignment;
- closure constraints across test samples;
- clustering over test samples;
- fitting or tuning on official test predictions;
- official test labels.

## Data Distribution

No dataset images or private labels are distributed in this repository. Users
must obtain public datasets from their official sources and provide local paths.

## Review Checklist

- `inference/run_inference.py` wires exact frozen-artifact inference with fixed
  `lastmeter_config.json` and `heading_transform.json`.
- `models_src/per_bucket_stacker.py` implements a generic row-local convex
  stacker used by public training and audit utilities.
- `inference/selfpair.py` implements row-local self-pair detection.
- `inference/package.py` performs row-aligned heading/distance packaging.
- `scripts/verify_heading_provenance.py` checks the frozen heading manifest and
  optional heading CSV contract.
- `scripts/verify_result.py` checks row count, self-pair rows, zip root, and
  archive hash.
