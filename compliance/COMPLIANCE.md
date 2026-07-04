# Compliance Statement

[中文版本](COMPLIANCE_zh.md)

This release is organized so reviewers can check that inference is a fixed
per-pair function.

## Inference Contract

Each official row is one ordered image pair. The released inference path uses
only that row's features, that row's frozen heading value, and parameters fixed
before official inference.

## Not Used

The released path does not use:

- official test labels;
- fitting or tuning on official test predictions;
- test-set graph optimization;
- hidden-neighbor revision;
- retrieval over the hidden test set;
- global assignment;
- clustering;
- batch sorting or cross-row bookkeeping.

## Data

No dataset images or private labels are distributed in this repository. Users
must obtain datasets from official sources and provide local paths.

For algorithmic details and motivation, please cite and read the paper rather
than treating this repository documentation as the method description.
