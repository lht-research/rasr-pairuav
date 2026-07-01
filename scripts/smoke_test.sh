#!/usr/bin/env bash
set -euo pipefail

"${PYTHON:-python3}" - <<'PY'
import csv
import tempfile
from pathlib import Path

import numpy as np

from inference.package import package_from_csvs
from inference.selfpair import apply_self_pair_zero
from models_src.heading_ensemble import HeadingEnsemble
from models_src.per_bucket_stacker import PerBucketStacker, write_distance_csv_streaming

stacker = PerBucketStacker(boundaries_meters=(5.0,), weights=np.array([[0.5, 0.5], [0.25, 0.75]]))
pred = stacker.predict(np.array([[2.0, 4.0], [-8.0, -10.0]]))
assert np.allclose(pred, [3.0, -9.5])

heading = HeadingEnsemble(weights=np.array([0.5, 0.5]))
combined = heading.combine_degrees(np.array([[350.0, 10.0]]))
assert np.allclose(combined, [360.0], atol=1e-8) or np.allclose(combined, [0.0], atol=1e-8)

fixed = apply_self_pair_zero(["a.jpg", "b.jpg"], ["a.jpg", "c.jpg"], [3.0, 4.0])
assert fixed == [0.0, 4.0]

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    head_a = root / "distance_head_a.csv"
    head_b = root / "distance_head_b.csv"
    heading = root / "heading_predictions.csv"
    distance = root / "distance_col.csv"
    out = root / "package"
    fieldnames = ["manifest_index", "scene_id", "pair_id", "image_a", "image_b", "range_pred"]
    rows = [
        {
            "manifest_index": "0",
            "scene_id": "s",
            "pair_id": "p0",
            "image_a": "a.jpg",
            "image_b": "a.jpg",
        },
        {
            "manifest_index": "1",
            "scene_id": "s",
            "pair_id": "p1",
            "image_a": "b.jpg",
            "image_b": "c.jpg",
        },
    ]
    for path, values in [(head_a, [2.0, -8.0]), (head_b, [4.0, -10.0])]:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row, value in zip(rows, values):
                writer.writerow({**row, "range_pred": f"{value:.6f}"})
    rows_written = write_distance_csv_streaming(
        distance,
        [head_a, head_b],
        stacker=stacker,
    )
    assert rows_written == 2
    with heading.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["manifest_index", "scene_id", "pair_id", "image_a", "image_b", "heading_pred"],
        )
        writer.writeheader()
        for row, value in zip(rows, [12.0, 34.0]):
            writer.writerow({**row, "heading_pred": f"{value:.6f}"})
    manifest = package_from_csvs(
        heading_csv=heading,
        distance_csv=distance,
        output_dir=out,
        expected_rows=2,
        expected_selfpair_rows=1,
    )
    assert manifest["selfpair_mode"] == "row_metadata"
    assert (out / "result.txt").read_text(encoding="utf-8").splitlines() == [
        "0.000000 0.000000",
        "34.000000 -9.500000",
    ]

print("skeleton smoke ok")
PY

"${PYTHON:-python3}" scripts/verify_heading_provenance.py \
  --manifest models/heading_weights.json
