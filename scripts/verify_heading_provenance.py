"""Validate the released heading manifest and optional heading CSV contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("models/heading_weights.json"))
    parser.add_argument("--heading-csv", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def sha256_column(path: Path, column: str, limit: int | None = None) -> tuple[int, str]:
    digest = hashlib.sha256()
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise RuntimeError(f"missing heading column {column!r} in {path}")
        for row in reader:
            if limit is not None and rows >= limit:
                break
            digest.update(f"{float(row[column]):.6f}\n".encode())
            rows += 1
    return rows, digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    provenance = payload.get("provenance") or {}
    forbidden_true = [
        "uses_test_labels",
        "uses_test_graph",
        "uses_cross_test_sample_relationships",
        "uses_global_assignment_or_clustering",
    ]
    bad_flags = [key for key in forbidden_true if provenance.get(key) is not False]
    if bad_flags:
        raise SystemExit(f"heading provenance has non-compliant flags: {bad_flags}")

    result: dict[str, object] = {
        "manifest": str(args.manifest),
        "model_type": payload.get("model_type"),
        "column": payload.get("column"),
        "rows_declared": payload.get("rows"),
        "row_local_provenance": True,
    }
    if args.heading_csv is not None:
        file_sha = sha256_file(args.heading_csv)
        rows, column_sha = sha256_column(
            args.heading_csv,
            str(payload.get("column", "heading_pred")),
            limit=args.limit,
        )
        if args.limit is None and payload.get("rows") is not None and rows != int(payload["rows"]):
            raise SystemExit(f"heading row count mismatch: expected {payload['rows']}, got {rows}")
        if args.limit is None and payload.get("source_column_sha256") and (
            column_sha != payload["source_column_sha256"]
        ):
            raise SystemExit(
                "heading column sha256 mismatch: "
                f"expected {payload['source_column_sha256']}, got {column_sha}"
            )
        if args.limit is None and payload.get("source_file_sha256") and (
            file_sha != payload["source_file_sha256"]
        ):
            raise SystemExit(
                "heading file sha256 mismatch: "
                f"expected {payload['source_file_sha256']}, got {file_sha}"
            )
        result.update(
            {
                "heading_csv": str(args.heading_csv),
                "rows_checked": rows,
                "source_file_sha256": file_sha,
                "source_column_sha256_decimal6_lf": column_sha,
                "source_column_sha256_format": payload.get("source_column_sha256_format"),
            }
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
