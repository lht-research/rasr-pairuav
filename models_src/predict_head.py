"""Run a trained distance head on a CSV or NPZ feature table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from models_src.distance_head import DistanceHead
from models_src.train_head import load_config, load_training_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_metadata(path: Path, limit: int | None) -> list[dict[str, str]]:
    if path.suffix.lower() != ".csv":
        return []
    keys = ["manifest_index", "scene_id", "pair_id", "json_rel_path", "image_a", "image_b"]
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            rows.append({key: row.get(key, "") for key in keys if key in row})
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    metadata_columns = [str(v) for v in config.get("metadata_columns", [])]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    target_column = args.target_column or str(checkpoint.get("target_column", "range_num"))
    x_raw, _, _ = load_training_arrays(
        args.features,
        target_column=target_column,
        metadata_columns=metadata_columns,
        limit=args.limit,
    )
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    x = (x_raw.astype(np.float32) - mean) / std
    model = DistanceHead(
        feature_dim=int(checkpoint["feature_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x)).cpu().numpy() * float(checkpoint["target_scale"])

    metadata = read_metadata(args.features, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(metadata[0]) if metadata else []
    fieldnames.append("range_pred")
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, value in enumerate(pred):
            row = dict(metadata[idx]) if metadata else {}
            row["range_pred"] = f"{float(value):.9f}"
            writer.writerow(row)
    print(f"wrote {len(pred)} rows to {args.output}")


if __name__ == "__main__":
    main()
