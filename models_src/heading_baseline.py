"""Train and apply the public from-scratch heading baseline from pair features.

This baseline is intentionally simple and fully reproducible. It predicts
heading by fitting two ridge-regression heads for sin(theta) and cos(theta) from
the same pair-feature table used by the distance heads. It is provided for the
from-scratch public path; the archived online submission uses the frozen
``heading_predictions.csv`` artifact listed in ``frozen_artifacts_manifest.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

META_COLUMNS = ["manifest_index", "scene_id", "pair_id", "json_rel_path", "image_a", "image_b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit")
    fit.add_argument("--features", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--target-column", default="heading_num")
    fit.add_argument("--metadata-column", action="append", default=META_COLUMNS)
    fit.add_argument("--ridge-alpha", type=float, default=1e-2)
    fit.add_argument("--limit", type=int, default=None)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--features", type=Path, required=True)
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def safe_float(value: str | None) -> float:
    try:
        parsed = float(value) if value is not None else math.nan
    except ValueError:
        parsed = math.nan
    return parsed


def read_feature_table(
    path: Path,
    *,
    target_column: str | None,
    metadata_columns: list[str],
    feature_names: list[str] | None = None,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[str], list[dict[str, str]]]:
    metadata_set = set(metadata_columns)
    if target_column:
        metadata_set.add(target_column)

    rows: list[list[float]] = []
    targets: list[float] = []
    metadata: list[dict[str, str]] = []
    names = feature_names

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if names is None:
            names = [
                name
                for name in (reader.fieldnames or [])
                if name not in metadata_set
                and name not in {"range_num", "range_true", "heading_true"}
            ]
        if target_column and target_column not in (reader.fieldnames or []):
            raise RuntimeError(f"missing target column {target_column!r} in {path}")
        for row_idx, row in enumerate(reader):
            if limit is not None and row_idx >= limit:
                break
            values = [safe_float(row.get(name)) for name in names]
            if not all(math.isfinite(value) for value in values):
                continue
            if target_column:
                target = safe_float(row.get(target_column))
                if not math.isfinite(target):
                    continue
                targets.append(target)
            rows.append(values)
            metadata.append({key: row.get(key, "") for key in META_COLUMNS if key in row})

    if not rows:
        raise RuntimeError(f"no usable feature rows loaded from {path}")
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64) if target_column else None
    return x, y, names or [], metadata


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std, mean, std


def fit_ridge(x: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    gram = x_aug.T @ x_aug
    reg = np.eye(gram.shape[0], dtype=x.dtype) * alpha
    reg[-1, -1] = 0.0
    return np.linalg.solve(gram + reg, x_aug.T @ target)


def fit(args: argparse.Namespace) -> None:
    metadata_columns = [str(value) for value in args.metadata_column]
    x_raw, heading_deg, feature_names, _ = read_feature_table(
        args.features,
        target_column=args.target_column,
        metadata_columns=metadata_columns,
        limit=args.limit,
    )
    if heading_deg is None:
        raise RuntimeError("heading target was not loaded")
    x, mean, std = standardize(x_raw)
    radians = np.deg2rad(heading_deg)
    sin_weights = fit_ridge(x, np.sin(radians), args.ridge_alpha)
    cos_weights = fit_ridge(x, np.cos(radians), args.ridge_alpha)
    payload = {
        "model_type": "public_heading_ridge_sincos",
        "target_column": args.target_column,
        "feature_names": feature_names,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "sin_weights": sin_weights.tolist(),
        "cos_weights": cos_weights.tolist(),
        "ridge_alpha": float(args.ridge_alpha),
        "rows": int(x.shape[0]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": str(args.output), "rows": int(x.shape[0])}, indent=2))


def predict(args: argparse.Namespace) -> None:
    payload = json.loads(args.model.read_text(encoding="utf-8"))
    feature_names = [str(name) for name in payload["feature_names"]]
    x_raw, _, _, metadata = read_feature_table(
        args.features,
        target_column=None,
        metadata_columns=META_COLUMNS,
        feature_names=feature_names,
        limit=args.limit,
    )
    mean = np.asarray(payload["mean"], dtype=np.float64)
    std = np.asarray(payload["std"], dtype=np.float64)
    x = (x_raw - mean) / np.where(std < 1e-8, 1.0, std)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    sin_pred = x_aug @ np.asarray(payload["sin_weights"], dtype=np.float64)
    cos_pred = x_aug @ np.asarray(payload["cos_weights"], dtype=np.float64)
    heading = (np.rad2deg(np.arctan2(sin_pred, cos_pred)) + 360.0) % 360.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [key for key in META_COLUMNS if metadata and key in metadata[0]]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, "heading_pred"])
        writer.writeheader()
        for meta, value in zip(metadata, heading):  # noqa: B905
            row = {key: meta.get(key, "") for key in fieldnames}
            row["heading_pred"] = f"{float(value):.6f}"
            writer.writerow(row)
    print(json.dumps({"heading_csv": str(args.output), "rows": int(len(heading))}, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "fit":
        fit(args)
        return
    if args.command == "predict":
        predict(args)
        return
    raise RuntimeError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
