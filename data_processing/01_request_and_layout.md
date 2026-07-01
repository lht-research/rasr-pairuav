# Data Request and Layout

This repository does not redistribute images or labels.

## Required Sources

1. University-1652: request the dataset from the official University-1652
   project page and follow its terms.
2. PairUAV official data: download the official training and test data from the
   benchmark release page.

## Expected Layout

Use a local data root outside this repository:

```text
/path/to/pairuav-data/
├── University-1652/
└── PairUAV/
    ├── train/
    ├── test/
    └── metadata/
```

Scripts accept `--data-root` so the repository stays free of dataset files.

## Smoke Mode

Most scripts accept `--limit N` to process a small prefix for code validation.
Smoke mode is intended to verify wiring, shapes, serialization, and row-order
stability. It is not used to report benchmark performance.
