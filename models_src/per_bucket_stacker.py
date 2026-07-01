"""Generic row-local per-bucket convex distance stacker.

The exact archived RASR path uses ``inference/run_inference.py`` with
``models/lastmeter_config.json``, where the range bucket is selected by
``gate=head0``. This module is retained for public training and audit utilities.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

DEFAULT_BOUNDARIES_METERS = (5.0, 15.0, 30.0, 60.0, 100.0, 132.0)
RANGE_CLIP = 0.5


@dataclass(frozen=True)
class PerBucketStacker:
    """Apply fixed bucket weights to per-row distance head predictions."""

    boundaries_meters: tuple[float, ...]
    weights: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        if weights.ndim != 2:
            raise ValueError("weights must have shape [num_buckets, num_heads]")
        if len(self.boundaries_meters) + 1 != weights.shape[0]:
            raise ValueError("number of buckets must equal len(boundaries) + 1")
        if np.any(weights < -1e-12):
            raise ValueError("convex weights must be non-negative")
        row_sums = weights.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-8):
            raise ValueError("each bucket weight row must sum to one")
        object.__setattr__(self, "weights", weights)

    def bucket_indices(self, predictions: np.ndarray) -> np.ndarray:
        predictions = _as_2d(predictions)
        medians = np.abs(np.median(predictions, axis=1))
        return np.searchsorted(np.asarray(self.boundaries_meters), medians, side="right")

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        predictions = _as_2d(predictions)
        buckets = self.bucket_indices(predictions)
        row_weights = self.weights[buckets]
        return np.sum(predictions * row_weights, axis=1)

    def predict_row(self, predictions: Sequence[float] | np.ndarray) -> float:
        """Apply fixed stacker weights to one row of head predictions."""

        row = np.asarray(predictions, dtype=np.float64)
        if row.ndim != 1:
            raise ValueError("row predictions must have shape [num_heads]")
        if row.shape[0] != self.weights.shape[1]:
            raise ValueError(
                f"expected {self.weights.shape[1]} head predictions, got {row.shape[0]}"
            )
        median = abs(float(np.median(row)))
        bucket = int(np.searchsorted(np.asarray(self.boundaries_meters), median, side="right"))
        return float(np.sum(row * self.weights[bucket]))

    @classmethod
    def from_json(cls, path: Path) -> PerBucketStacker:
        payload = json.loads(path.read_text(encoding="utf-8"))
        boundaries = payload.get("boundaries_meters") or payload.get("bucket_boundaries")
        if boundaries is None:
            boundaries = DEFAULT_BOUNDARIES_METERS
        raw_weights = payload.get("weights") or payload.get("bucket_weights")
        if raw_weights is None:
            raise ValueError(f"missing bucket weights in {path}")
        if isinstance(raw_weights, dict):
            weights = [raw_weights[str(bucket)] for bucket in range(len(boundaries) + 1)]
        else:
            weights = raw_weights
        return cls(tuple(float(v) for v in boundaries), np.asarray(weights, dtype=np.float64))

    def to_json_payload(self, head_names: Sequence[str] | None = None) -> dict[str, object]:
        return {
            "model_type": "convex_per_bucket_linear",
            "head_names": list(head_names or []),
            "boundaries_meters": [float(v) for v in self.boundaries_meters],
            "weights": [[float(v) for v in row] for row in self.weights],
        }


def load_head_prediction_csvs(
    paths: Sequence[Path],
    *,
    value_column: str = "range_pred",
    limit: int | None = None,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Load aligned distance-head CSV files and return prediction matrix plus metadata."""

    if not paths:
        raise ValueError("at least one head prediction CSV is required")

    readers: list[csv.DictReader] = []
    handles = []
    try:
        for path in paths:
            handle = path.open("r", encoding="utf-8", newline="")
            handles.append(handle)
            reader = csv.DictReader(handle)
            if value_column not in (reader.fieldnames or []):
                raise ValueError(f"missing column {value_column!r} in {path}")
            readers.append(reader)

        rows: list[list[float]] = []
        metadata: list[dict[str, str]] = []
        for row_idx, row_tuple in enumerate(zip_longest(*readers, fillvalue=None)):
            if limit is not None and row_idx >= limit:
                break
            if any(row is None for row in row_tuple):
                raise ValueError(f"head CSV row count mismatch near row {row_idx}")
            metas = [_metadata_from_row(row) for row in row_tuple if row is not None]
            identities = [_metadata_identity(item) for item in metas]
            identities = [item for item in identities if item is not None]
            if identities and any(item != identities[0] for item in identities):
                raise ValueError(f"head CSV metadata mismatch at row {row_idx}: {identities}")
            rows.append([float(row[value_column]) for row in row_tuple if row is not None])
            metadata.append(_merge_metadata(metas))
        return np.asarray(rows, dtype=np.float64), metadata
    finally:
        for handle in handles:
            handle.close()


def iter_stacked_distance_rows(
    paths: Sequence[Path],
    *,
    stacker: PerBucketStacker,
    value_column: str = "range_pred",
    limit: int | None = None,
) -> Iterator[tuple[dict[str, str], float]]:
    """Yield stacked predictions row by row without materializing the test set."""

    if not paths:
        raise ValueError("at least one head prediction CSV is required")
    if len(paths) != stacker.weights.shape[1]:
        raise ValueError(
            f"stacker expects {stacker.weights.shape[1]} heads, got {len(paths)} CSV files"
        )

    readers: list[csv.DictReader] = []
    handles = []
    try:
        for path in paths:
            handle = path.open("r", encoding="utf-8", newline="")
            handles.append(handle)
            reader = csv.DictReader(handle)
            if value_column not in (reader.fieldnames or []):
                raise ValueError(f"missing column {value_column!r} in {path}")
            readers.append(reader)

        for row_idx, row_tuple in enumerate(zip_longest(*readers, fillvalue=None)):
            if limit is not None and row_idx >= limit:
                break
            if any(row is None for row in row_tuple):
                raise ValueError(f"head CSV row count mismatch near row {row_idx}")
            rows = [row for row in row_tuple if row is not None]
            metas = [_metadata_from_row(row) for row in rows]
            identities = [_metadata_identity(item) for item in metas]
            identities = [item for item in identities if item is not None]
            if identities and any(item != identities[0] for item in identities):
                raise ValueError(f"head CSV metadata mismatch at row {row_idx}: {identities}")
            head_values = [float(row[value_column]) for row in rows]
            yield _merge_metadata(metas), stacker.predict_row(head_values)
    finally:
        for handle in handles:
            handle.close()


def write_distance_csv(
    path: Path, predictions: np.ndarray, metadata: Sequence[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_keys = _ordered_metadata_keys(metadata)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*metadata_keys, "range_pred"])
        writer.writeheader()
        for meta, value in zip(metadata, predictions):  # noqa: B905
            row = {key: meta.get(key, "") for key in metadata_keys}
            row["range_pred"] = f"{float(value):.9f}"
            writer.writerow(row)


def write_distance_csv_streaming(
    path: Path,
    paths: Sequence[Path],
    *,
    stacker: PerBucketStacker,
    value_column: str = "range_pred",
    limit: int | None = None,
) -> int:
    """Write stacked distance predictions while reading only aligned current rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_keys: list[str] | None = None
    rows_written = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer: csv.DictWriter[str] | None = None
        for meta, value in iter_stacked_distance_rows(
            paths,
            stacker=stacker,
            value_column=value_column,
            limit=limit,
        ):
            if writer is None:
                metadata_keys = _ordered_metadata_keys([meta])
                writer = csv.DictWriter(handle, fieldnames=[*metadata_keys, "range_pred"])
                writer.writeheader()
            row = {key: meta.get(key, "") for key in metadata_keys or []}
            row["range_pred"] = f"{float(value):.9f}"
            writer.writerow(row)
            rows_written += 1
        if writer is None:
            writer = csv.DictWriter(handle, fieldnames=["range_pred"])
            writer.writeheader()
    return rows_written


def fit_per_bucket_stacker(
    head_predictions: np.ndarray,
    targets: np.ndarray,
    *,
    boundaries_meters: Sequence[float] = DEFAULT_BOUNDARIES_METERS,
    min_bucket_rows: int = 100,
) -> PerBucketStacker:
    """Fit convex per-bucket weights on training or calibration rows."""

    head_predictions = _as_2d(head_predictions)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 1 or targets.shape[0] != head_predictions.shape[0]:
        raise ValueError("targets must have shape [num_rows]")
    boundaries = tuple(float(v) for v in boundaries_meters)
    global_weights = _optimize_convex_weights(head_predictions, targets)
    buckets = np.searchsorted(
        np.asarray(boundaries), np.abs(np.median(head_predictions, axis=1)), side="right"
    )
    bucket_weights: list[np.ndarray] = []
    for bucket in range(len(boundaries) + 1):
        mask = buckets == bucket
        if int(np.sum(mask)) < min_bucket_rows:
            bucket_weights.append(global_weights)
        else:
            bucket_weights.append(_optimize_convex_weights(head_predictions[mask], targets[mask]))
    return PerBucketStacker(boundaries, np.vstack(bucket_weights))


def fit_from_csv(
    train_csv: Path,
    *,
    head_columns: Sequence[str],
    target_column: str,
    output: Path,
    boundaries_meters: Sequence[float] = DEFAULT_BOUNDARIES_METERS,
    min_bucket_rows: int = 100,
) -> dict[str, object]:
    with train_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [
            col for col in [*head_columns, target_column] if col not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(f"missing columns in {train_csv}: {missing}")
        head_rows: list[list[float]] = []
        targets: list[float] = []
        for row in reader:
            head_rows.append([float(row[col]) for col in head_columns])
            targets.append(float(row[target_column]))
    stacker = fit_per_bucket_stacker(
        np.asarray(head_rows, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        boundaries_meters=boundaries_meters,
        min_bucket_rows=min_bucket_rows,
    )
    payload = stacker.to_json_payload(head_columns)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _optimize_convex_weights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_heads = x.shape[1]
    if len(y) == 0:
        return np.ones(n_heads, dtype=np.float64) / float(n_heads)
    denom = np.maximum(np.abs(y), RANGE_CLIP)

    def objective(weights: np.ndarray) -> float:
        pred = x @ weights
        return float(np.mean(np.abs(pred - y) / denom))

    constraints = [{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}]
    bounds = [(0.0, 1.0) for _ in range(n_heads)]
    starts = [np.ones(n_heads, dtype=np.float64) / float(n_heads)]
    best_weights = starts[0]
    best_value = objective(best_weights)
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300},
        )
        if result.success:
            value = objective(np.asarray(result.x, dtype=np.float64))
            if value < best_value:
                best_value = value
                best_weights = np.asarray(result.x, dtype=np.float64)
    total = float(np.sum(best_weights))
    if not np.isfinite(total) or total <= 0:
        return np.ones(n_heads, dtype=np.float64) / float(n_heads)
    return best_weights / total


def _metadata_from_row(row: dict[str, str]) -> dict[str, str]:
    keys = [
        "manifest_index",
        "scene_id",
        "pair_id",
        "json_rel_path",
        "image_a",
        "image_b",
        "left_image_id",
        "right_image_id",
    ]
    return {key: row[key] for key in keys if key in row and row[key] != ""}


def _metadata_identity(meta: dict[str, str]) -> tuple[str, ...] | None:
    keys = [key for key in ("manifest_index", "scene_id", "pair_id") if key in meta]
    if not keys:
        return None
    return tuple(meta[key] for key in keys)


def _merge_metadata(metas: Sequence[dict[str, str]]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for meta in metas:
        for key, value in meta.items():
            if key in merged and merged[key] != value:
                raise ValueError(
                    f"metadata value conflict for {key!r}: {merged[key]!r} vs {value!r}"
                )
            merged[key] = value
    return merged


def _ordered_metadata_keys(metadata: Sequence[dict[str, str]]) -> list[str]:
    preferred = [
        "manifest_index",
        "scene_id",
        "pair_id",
        "json_rel_path",
        "image_a",
        "image_b",
        "left_image_id",
        "right_image_id",
    ]
    present = set().union(*(item.keys() for item in metadata)) if metadata else set()
    return [key for key in preferred if key in present]


def _as_2d(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("predictions must have shape [num_rows, num_heads]")
    return array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="Apply released stacker weights to head CSVs.")
    predict.add_argument("--stacker-config", type=Path, required=True)
    predict.add_argument("--head-csv", type=Path, action="append", required=True)
    predict.add_argument("--value-column", default="range_pred")
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--limit", type=int, default=None)
    predict.add_argument(
        "--materialize",
        action="store_true",
        help="Load all rows before stacking. Default is streaming row-local inference.",
    )

    fit = subparsers.add_parser("fit", help="Fit stacker weights from a calibration CSV.")
    fit.add_argument("--train-csv", type=Path, required=True)
    fit.add_argument("--head-column", action="append", required=True)
    fit.add_argument("--target-column", default="range_true")
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--min-bucket-rows", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "predict":
        stacker = PerBucketStacker.from_json(args.stacker_config)
        if args.materialize:
            head_predictions, metadata = load_head_prediction_csvs(
                args.head_csv,
                value_column=args.value_column,
                limit=args.limit,
            )
            distances = stacker.predict(head_predictions)
            write_distance_csv(args.output, distances, metadata)
            rows_written = len(distances)
        else:
            rows_written = write_distance_csv_streaming(
                args.output,
                args.head_csv,
                stacker=stacker,
                value_column=args.value_column,
                limit=args.limit,
            )
        print(f"wrote {rows_written} rows to {args.output}")
        return
    if args.command == "fit":
        payload = fit_from_csv(
            args.train_csv,
            head_columns=args.head_column,
            target_column=args.target_column,
            output=args.output,
            min_bucket_rows=args.min_bucket_rows,
        )
        print(json.dumps(payload, indent=2))
        return
    raise RuntimeError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
