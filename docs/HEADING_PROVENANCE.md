# Heading Provenance

[中文版本](HEADING_PROVENANCE_zh.md)

The exact frozen-artifact path consumes heading as one aligned CSV column named
`heading_pred`. In the paper's notation this column is the frozen predictor
`phi_i` used by the PairUAV benchmark-specific output head.

The public from-scratch path trains a simple row-local heading baseline for
transparent end-to-end execution. That baseline is not the frozen heading column
used for the archived online score.

## Per-Pair Contract

For each official row, upstream heading models predict heading values for that
row's ordered image pair. The frozen heading source was produced by a fixed
circular weighted average over those per-row predictions:

```text
heading_pred[row] = circular_weighted_average(head_a[row], head_b[row], ...)
```

The combiner does not inspect neighboring rows, construct a test graph, sort the
hidden batch, match rows globally, cluster samples, or use official test labels.

The public implementation of this operation is `models_src/heading_ensemble.py`.
The exact path stores only the frozen output column because the archived online
score is anchored to that exported column.

## PairUAV Output Transform

During exact reproduction, `heading_pred` is transformed row by row with
`models/heading_transform.json`:

```text
wrap_180(20 * round((1.014 * phi + 1.2) / 20))
```

The 20 degree snap is part of the PairUAV benchmark-specific output head. It is
included to reproduce the archived score, not claimed as a transferable
controller resolution.

## Manifest

`models/heading_weights.json` records the frozen heading column contract:

- expected file name: `heading_predictions.csv`
- expected column: `heading_pred`
- expected rows for the official test set: `2773116`
- SHA-256 of the archived source CSV file
- SHA-256 of the canonical heading column stream, where each value is formatted
  as `%.6f` followed by LF and no header
- machine-readable provenance flags for test-graph, assignment, clustering, and
  cross-sample use

## Verification

Validate the manifest and, when available, a heading CSV:

```bash
python scripts/verify_heading_provenance.py \
  --manifest models/heading_weights.json \
  --heading-csv /path/to/heading_predictions.csv
```

This check verifies the declared per-pair provenance flags and the CSV row and
column contract. It is not a score check and does not use official labels.
