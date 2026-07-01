# Archived PairUAV Submission Implementation

This directory reproduces the archived PairUAV submission when used with the
frozen intermediate artifacts listed in `frozen_artifacts_manifest.json`.

The implementation follows the paper's RASR decomposition:

- scale-recovery core over frozen pair geometry;
- four frozen distance candidates (`distance_head_a` ... `distance_head_d`);
- range-aware convex combination using `gate=head0`;
- PairUAV benchmark-specific distance output head;
- frozen heading column with the fixed heading transform;
- row-local self-pair zeroing.

It does not use test-set graph optimization, batch sorting,
hidden-neighbor revision, retrieval over the hidden test set, global assignment,
clustering, or cross-sample bookkeeping.

## Distance Configuration

Config file: `models/lastmeter_config.json`

Selected candidate name:

```text
gate_boundary_head0_bounds_0129
```

Gate:

```text
head0
```

Scale proxy:

```text
|h_1(z_i)|
```

Bucket boundaries:

```text
[5.596117113284452, 14.189340747346009, 33.73144867309181, 53.75,
 77.38132781381124, 114.79]
```

For each row, the configured bucket selects convex weights over the four
distance candidates. The PairUAV output head then selects bucket/sign/std-segment
affine parameters from `models/lastmeter_config.json` and applies the 2.5 m snap.

## Heading Transform

The heading source is the frozen `heading_pred` CSV column. The release transform
is row-local:

```json
{
  "scale": 1.0139999999999996,
  "bias": 1.2000000000000002,
  "snap_step": 20.0,
  "snap_offset": 0.0,
  "wrap": true
}
```

This is the implementation form of:

```text
wrap_180(20 * round((1.014 * phi + 1.2) / 20))
```

## Expected Submission

Expected `submit.zip` SHA-256:

```text
2f742f2eff83e535b96a8dbd46db370fa3ac0538a9f3e53b684d65c253b34b77
```

Reported online metrics for this archive:

```text
final_score          0.003189
distance_rel_error   0.003029
angle_rel_error      0.003350
```

## Public From-Scratch Path

The source tree also includes `scripts/reproduce_from_scratch.sh`, which builds
train/test pair tables, extracts features, trains public distance heads, trains
a simple public row-local heading baseline, and packages a submission. That path
is meant for transparent end-to-end reproduction from official data. It is not
expected to match the archived SHA-256 because the archived online result freezes
production prediction CSVs, especially the stronger frozen heading column.
