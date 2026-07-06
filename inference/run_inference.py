"""Run the archived RASR PairUAV release inference.

This entry point preserves the paper contract: each output is a per-pair
function with fixed parameters. It consumes four distance candidates, applies
the range-aware scale-recovery configuration and the protocol-specific
calibration module, transforms the frozen heading column, and performs row-local
self-pair zeroing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from itertools import zip_longest

from inference.package import package_from_csvs

import numpy as np

DEFAULT_HEAD_FILES = [
    "distance_head_a.csv",
    "distance_head_b.csv",
    "distance_head_c.csv",
    "distance_head_d.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--head-csv", type=Path, action="append", default=None)
    parser.add_argument("--pair-feature-csv", type=Path, default=None)
    parser.add_argument("--pair-feature-glob", default=None)
    parser.add_argument("--pair-feature-dir", type=Path, default=None)
    parser.add_argument("--heading-csv", type=Path, default=None)
    parser.add_argument("--selfpair-indices", type=Path, default=None)
    parser.add_argument("--stacker-config", type=Path, default=None)
    parser.add_argument("--lastmeter-config", type=Path, default=None)
    parser.add_argument("--heading-transform", type=Path, default=None)
    parser.add_argument("--head-value-column", default="range_pred")
    parser.add_argument("--heading-column", default="heading_pred")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--legacy-batch-policy", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-rows", type=int, default=2_773_116)
    parser.add_argument("--expected-selfpair-rows", type=int, default=51_354)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)[:180]


def gate_value(values: np.ndarray, gate: str) -> float:
    if gate == "median":
        return float(np.median(values))
    if gate == "mean":
        return float(np.mean(values))
    if gate.startswith("head"):
        return float(values[int(gate[4:])])
    raise ValueError(f"unsupported gate: {gate}")


def snap_with_params(raw: float, params: dict[str, object]) -> float:
    step = float(params["step"])
    offset = float(params.get("offset", params.get("snap_offset", 0.0)))
    value = raw * float(params["scale"]) + float(params["bias"])
    return float(round((value - offset) / step) * step + offset)


def predict_distance(values: np.ndarray, config: dict[str, object]) -> float:
    stacker = config["stacker_candidate"]
    gate = str(stacker["gate"])
    boundaries = np.asarray(stacker["boundaries"], dtype=np.float64)
    weights = np.asarray(stacker["weights"], dtype=np.float64)
    bucket = int(np.searchsorted(boundaries, abs(gate_value(values, gate)), side="right"))
    raw = float(np.sum(values * weights[bucket]))
    sign = "pos" if raw >= 0.0 else "neg"
    params = config["params"][str(bucket)][sign]
    segment_params = config.get("segment_params", {})
    segment = segment_params.get(f"{bucket}_{sign}")
    if segment is not None:
        std_value = float(np.std(values))
        cuts = np.asarray([float(v) for v in segment["std_cuts"]], dtype=np.float64)
        seg_idx = int(np.searchsorted(cuts, std_value, side="right"))
        params = segment["params"][seg_idx]
    return snap_with_params(raw, params)


def build_distance_csv(
    output: Path,
    head_csvs: list[Path],
    *,
    config: dict[str, object],
    value_column: str,
    limit: int | None,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    handles = [path.open("r", encoding="utf-8", newline="") for path in head_csvs]
    try:
        readers = [csv.DictReader(handle) for handle in handles]
        rows = 0
        metadata_fields: list[str] = []
        writer: csv.DictWriter[str] | None = None
        with output.open("w", encoding="utf-8", newline="") as out_handle:
            for row_tuple in zip_longest(*readers, fillvalue=None):
                if limit is not None and rows >= limit:
                    break
                if any(row is None for row in row_tuple):
                    raise RuntimeError(f"head row count mismatch near row {rows}")
                head_rows = list(row_tuple)
                keys = [
                    (row.get("manifest_index", ""), row.get("scene_id", ""), row.get("pair_id", ""))
                    for row in head_rows
                ]
                if any(key != keys[0] for key in keys):
                    raise RuntimeError(f"metadata mismatch at row {rows}: {keys}")
                base = head_rows[0]
                if writer is None:
                    metadata_fields = [
                        key
                        for key in [
                            "manifest_index",
                            "scene_id",
                            "pair_id",
                            "json_rel_path",
                            "image_a",
                            "image_b",
                        ]
                        if key in base
                    ]
                    writer = csv.DictWriter(
                        out_handle, fieldnames=[*metadata_fields, "range_pred"]
                    )
                    writer.writeheader()
                values = np.asarray([float(row[value_column]) for row in head_rows], dtype=np.float64)
                distance = predict_distance(values, config)
                out_row = {key: base.get(key, "") for key in metadata_fields}
                out_row["range_pred"] = f"{distance:.9f}"
                writer.writerow(out_row)
                rows += 1
        return rows
    finally:
        for handle in handles:
            handle.close()


def wrap_deg(value: float) -> float:
    return math.fmod(value + 180.0, 360.0) - 180.0 if value + 180.0 >= 0 else ((value + 180.0) % 360.0) - 180.0


def transform_heading(value: float, variant: dict[str, object]) -> float:
    transformed = value * float(variant.get("scale", 1.0)) + float(variant.get("bias", 0.0))
    if "snap_step" in variant:
        step = float(variant["snap_step"])
        offset = float(variant.get("snap_offset", 0.0))
        transformed = round((transformed - offset) / step) * step + offset
    if bool(variant.get("wrap", True)):
        transformed = wrap_deg(transformed)
    return transformed


def build_heading_csv(
    output: Path,
    heading_csv: Path,
    *,
    transform: dict[str, object],
    heading_column: str,
    limit: int | None,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with (
        heading_csv.open("r", encoding="utf-8", newline="") as in_handle,
        output.open("w", encoding="utf-8", newline="") as out_handle,
    ):
        reader = csv.DictReader(in_handle)
        if heading_column not in (reader.fieldnames or []):
            raise RuntimeError(f"missing heading column {heading_column!r} in {heading_csv}")
        metadata_fields = [
            key
            for key in [
                "manifest_index",
                "scene_id",
                "pair_id",
                "json_rel_path",
                "image_a",
                "image_b",
            ]
            if key in (reader.fieldnames or [])
        ]
        writer = csv.DictWriter(out_handle, fieldnames=[*metadata_fields, heading_column])
        writer.writeheader()
        for row in reader:
            if limit is not None and rows >= limit:
                break
            heading = transform_heading(float(row[heading_column]), transform)
            out_row = {key: row.get(key, "") for key in metadata_fields}
            out_row[heading_column] = f"{heading:.6f}"
            writer.writerow(out_row)
            rows += 1
    return rows


def main() -> None:
    args = parse_args()
    if args.head_csv:
        head_csvs = args.head_csv
    elif any(
        value is not None
        for value in [args.pair_feature_csv, args.pair_feature_glob, args.pair_feature_dir]
    ):
        from models_src.released_head_inference import predict_all_heads

        generated_dir = args.output_dir / "head_predictions"
        head_csvs = predict_all_heads(
            feature_csv=args.pair_feature_csv,
            feature_glob=args.pair_feature_glob,
            feature_dir=args.pair_feature_dir,
            model_dir=args.model_dir,
            output_dir=generated_dir,
            batch_size=args.batch_size,
            device=args.device,
            limit=args.limit,
            legacy_batch_policy=args.legacy_batch_policy,
        )
    else:
        head_csvs = [args.feature_root / name for name in DEFAULT_HEAD_FILES]
    heading_csv = args.heading_csv or args.feature_root / "heading_predictions.csv"
    lastmeter_config = args.lastmeter_config or args.model_dir / "lastmeter_config.json"
    heading_transform = args.heading_transform or args.model_dir / "heading_transform.json"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    distance_csv = args.output_dir / "distance_col.csv"
    transformed_heading_csv = args.output_dir / "heading_col.csv"
    config = json.loads(lastmeter_config.read_text(encoding="utf-8"))
    heading_variant = json.loads(heading_transform.read_text(encoding="utf-8"))
    distance_rows = build_distance_csv(
        distance_csv,
        head_csvs,
        config=config,
        value_column=args.head_value_column,
        limit=args.limit,
    )
    heading_rows = build_heading_csv(
        transformed_heading_csv,
        heading_csv,
        transform=heading_variant,
        heading_column=args.heading_column,
        limit=args.limit,
    )
    if heading_rows != distance_rows:
        raise RuntimeError(f"heading/distance row mismatch: {heading_rows} vs {distance_rows}")

    manifest = package_from_csvs(
        heading_csv=transformed_heading_csv,
        distance_csv=distance_csv,
        output_dir=args.output_dir,
        heading_column=args.heading_column,
        distance_column="range_pred",
        selfpair_indices_path=args.selfpair_indices,
        expected_rows=args.expected_rows,
        expected_selfpair_rows=args.expected_selfpair_rows,
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "distance_csv": str(distance_csv),
                "heading_csv": str(transformed_heading_csv),
                "distance_rows": distance_rows,
                "heading_rows": heading_rows,
                "lastmeter_config": str(lastmeter_config),
                "heading_transform": str(heading_transform),
                "stacker_mode": "streaming_row_local_bucket_sign_stdseg_snap",
                "selfpair_mode": "row_metadata"
                if args.selfpair_indices is None
                else "explicit_index_list",
                "package": manifest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
