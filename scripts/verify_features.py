"""Verify released pair-feature CSV files against checkpoint schema or a reference CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

META_COLUMNS = ["manifest_index", "scene_id", "pair_id", "json_rel_path", "image_a", "image_b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/distance_head_a.pt"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def safe_float(value: str | None) -> float:
    try:
        parsed = float(value) if value is not None else math.nan
    except ValueError:
        parsed = math.nan
    return parsed


def compare(args: argparse.Namespace, feature_names: list[str]) -> dict[str, object]:
    rows = 0
    metadata_mismatch = 0
    max_abs_diff = 0.0
    max_abs_diff_column = ""
    missing_columns: list[str] = []
    with args.feature_csv.open("r", encoding="utf-8", newline="") as feature_handle:
        feature_reader = csv.DictReader(feature_handle)
        feature_fields = feature_reader.fieldnames or []
        missing_columns = [name for name in feature_names if name not in feature_fields]
        if missing_columns:
            return {
                "rows": 0,
                "feature_count": len(feature_names),
                "missing_columns": missing_columns,
                "passed": False,
            }
        if args.reference_csv is None:
            for rows, _row in enumerate(feature_reader, start=1):
                if args.limit is not None and rows >= args.limit:
                    break
            return {
                "rows": rows,
                "feature_count": len(feature_names),
                "missing_columns": [],
                "reference_csv": None,
                "max_abs_diff": None,
                "passed": True,
            }
        with args.reference_csv.open("r", encoding="utf-8", newline="") as reference_handle:
            reference_reader = csv.DictReader(reference_handle)
            reference_fields = reference_reader.fieldnames or []
            reference_missing = [name for name in feature_names if name not in reference_fields]
            if reference_missing:
                return {
                    "rows": 0,
                    "feature_count": len(feature_names),
                    "missing_columns": [],
                    "reference_missing_columns": reference_missing,
                    "passed": False,
                }
            for rows, (row, ref_row) in enumerate(
                zip(feature_reader, reference_reader), start=1  # noqa: B905
            ):
                if args.limit is not None and rows > args.limit:
                    rows -= 1
                    break
                for key in META_COLUMNS:
                    if key in row and key in ref_row and row.get(key, "") != ref_row.get(key, ""):
                        metadata_mismatch += 1
                        break
                values = np.asarray(
                    [safe_float(row.get(name)) for name in feature_names], dtype=np.float64
                )
                ref_values = np.asarray(
                    [safe_float(ref_row.get(name)) for name in feature_names], dtype=np.float64
                )
                diffs = np.abs(values - ref_values)
                diffs = np.where(np.isfinite(diffs), diffs, np.inf)
                idx = int(np.argmax(diffs))
                if float(diffs[idx]) > max_abs_diff:
                    max_abs_diff = float(diffs[idx])
                    max_abs_diff_column = feature_names[idx]
    return {
        "rows": rows,
        "feature_count": len(feature_names),
        "missing_columns": missing_columns,
        "reference_csv": str(args.reference_csv) if args.reference_csv else None,
        "metadata_mismatch": metadata_mismatch,
        "max_abs_diff": max_abs_diff,
        "max_abs_diff_column": max_abs_diff_column,
        "atol": args.atol,
        "passed": metadata_mismatch == 0 and max_abs_diff <= args.atol,
    }


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    feature_names = [str(name) for name in checkpoint["feature_names"]]
    report = compare(args, feature_names)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report.get("passed", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
