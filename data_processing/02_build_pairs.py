"""Build pair-level tables from a local PairUAV JSON layout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def find_split_root(data_root: Path, split: str) -> Path:
    candidates = [
        data_root / "PairUAV" / split,
        data_root / "pairUAV" / split,
        data_root / split,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"could not find {split!r} JSON directory under {data_root}")


def sort_key(path: Path) -> tuple[str, int, str, int, str]:
    def first_int(value: str) -> int:
        digits = "".join(ch for ch in value if ch.isdigit())
        return int(digits) if digits else 10**12

    return (
        path.parent.name,
        first_int(path.parent.name),
        path.stem,
        first_int(path.stem),
        path.name,
    )


def iter_pair_rows(split_root: Path, *, limit: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for idx, json_path in enumerate(sorted(split_root.rglob("*.json"), key=sort_key)):
        if limit is not None and idx >= limit:
            break
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        rel_path = json_path.relative_to(split_root).as_posix()
        scene_id = json_path.parent.name
        pair_id = json_path.stem
        rows.append(
            {
                "manifest_index": str(idx),
                "scene_id": scene_id,
                "pair_id": pair_id,
                "json_rel_path": rel_path,
                "image_a": str(payload.get("image_a", "")),
                "image_b": str(payload.get("image_b", "")),
                "heading_num": str(payload.get("heading_num", "")),
                "range_num": str(payload.get("range_num", "")),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    split_root = find_split_root(args.data_root, args.split)
    rows = iter_pair_rows(split_root, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "manifest_index",
        "scene_id",
        "pair_id",
        "json_rel_path",
        "image_a",
        "image_b",
        "heading_num",
        "range_num",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
