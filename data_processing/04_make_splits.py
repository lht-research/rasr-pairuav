"""Create deterministic scene-disjoint training and calibration splits."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.pairs.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for idx, row in enumerate(reader):
            if args.limit is not None and idx >= args.limit:
                break
            rows.append(row)
        fieldnames = reader.fieldnames or []

    scenes = sorted({row.get("scene_id", "") for row in rows})
    rng = random.Random(args.seed)
    rng.shuffle(scenes)
    calib_count = max(1, int(round(len(scenes) * args.calib_fraction))) if scenes else 0
    calib_scenes = set(scenes[:calib_count])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train_heads.csv": [row for row in rows if row.get("scene_id", "") not in calib_scenes],
        "train_calib.csv": [row for row in rows if row.get("scene_id", "") in calib_scenes],
    }
    for name, split_rows in outputs.items():
        path = args.output_dir / name
        with path.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"wrote {len(split_rows)} rows to {path}")


if __name__ == "__main__":
    main()
