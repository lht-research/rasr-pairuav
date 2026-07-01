"""Canonical PairUAV submission packager."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from itertools import zip_longest
from pathlib import Path

from inference.selfpair import is_self_pair

EXPECTED_ROWS = 2_773_116
EXPECTED_SELFPAIR_ROWS = 51_354
EXPECTED_ARCHIVE_SHA256 = "ac18c886d363f4c0a787bb4bc110fc7e91c4f75c76a053bff7d0218e2099265e"
RELEASE_ZIP_TIMESTAMP = (2026, 6, 13, 3, 9, 54)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_tuple(row: dict[str, str]) -> tuple[str, ...] | None:
    keys = [key for key in ("manifest_index", "scene_id", "pair_id") if key in row]
    if not keys:
        return None
    return tuple(row[key] for key in keys)


def load_selfpair_indices(path: Path | None) -> set[int] | None:
    if path is None:
        return None
    values = json.loads(path.read_text(encoding="utf-8"))
    return {int(value) for value in values}


def row_is_selfpair(row: dict[str, str]) -> bool:
    candidates = [
        ("image_a", "image_b"),
        ("left_image_id", "right_image_id"),
        ("query_image", "reference_image"),
    ]
    for left_key, right_key in candidates:
        if left_key in row and right_key in row:
            return is_self_pair(row[left_key], row[right_key])
    return False


def package_from_csvs(
    *,
    heading_csv: Path,
    distance_csv: Path,
    output_dir: Path,
    heading_column: str = "heading_pred",
    distance_column: str = "range_pred",
    selfpair_indices_path: Path | None = None,
    expected_rows: int | None = EXPECTED_ROWS,
    expected_selfpair_rows: int | None = EXPECTED_SELFPAIR_ROWS,
    limit: int | None = None,
    candidate_name: str | None = None,
) -> dict[str, object]:
    """Build result.txt and result.zip from aligned heading and distance CSV files.

    The default self-pair policy is row-local: compare image identifiers that are
    present in the current heading or distance row. ``selfpair_indices_path`` is
    retained only for legacy archive verification when row metadata is absent.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    source_result_txt = output_dir / "source_result.txt"
    result_txt = output_dir / "result.txt"
    result_zip = output_dir / "result.zip"
    manifest_path = output_dir / "candidate_manifest.json"

    index_selfpairs = load_selfpair_indices(selfpair_indices_path)
    selfpair_mode = "row_metadata" if index_selfpairs is None else "explicit_index_list"
    if index_selfpairs is not None and limit is None and expected_selfpair_rows is not None:
        if len(index_selfpairs) != expected_selfpair_rows:
            raise RuntimeError(
                f"selfpair row count mismatch: {len(index_selfpairs)} vs {expected_selfpair_rows}"
            )

    rows = 0
    zeroed_rows = 0
    metadata_checks = 0

    with (
        heading_csv.open("r", encoding="utf-8", newline="") as heading_handle,
        distance_csv.open("r", encoding="utf-8", newline="") as distance_handle,
        source_result_txt.open("w", encoding="utf-8") as source_handle,
        result_txt.open("w", encoding="utf-8") as result_handle,
    ):
        heading_reader = csv.DictReader(heading_handle)
        distance_reader = csv.DictReader(distance_handle)
        if heading_column not in (heading_reader.fieldnames or []):
            raise RuntimeError(f"missing heading column {heading_column!r} in {heading_csv}")
        if distance_column not in (distance_reader.fieldnames or []):
            raise RuntimeError(f"missing distance column {distance_column!r} in {distance_csv}")

        for idx, pair in enumerate(zip_longest(heading_reader, distance_reader, fillvalue=None)):
            if limit is not None and idx >= limit:
                break
            heading_row, distance_row = pair
            if heading_row is None or distance_row is None:
                raise RuntimeError(f"CSV row mismatch near row {idx}")

            heading_meta = metadata_tuple(heading_row)
            distance_meta = metadata_tuple(distance_row)
            if heading_meta is not None and distance_meta is not None:
                metadata_checks += 1
                if heading_meta != distance_meta:
                    raise RuntimeError(
                        f"metadata mismatch at row {idx}: "
                        f"heading={heading_meta} distance={distance_meta}"
                    )

            heading = float(heading_row[heading_column])
            distance = float(distance_row[distance_column])
            source_line = f"{heading:.6f} {distance:.6f}"
            source_handle.write(source_line + "\n")

            zero_this_row = False
            if index_selfpairs is not None:
                zero_this_row = idx in index_selfpairs
            else:
                zero_this_row = row_is_selfpair(distance_row) or row_is_selfpair(heading_row)

            if zero_this_row:
                result_handle.write("0.000000 0.000000\n")
                zeroed_rows += 1
            else:
                result_handle.write(source_line + "\n")
            rows += 1

    if limit is None and expected_rows is not None and rows != expected_rows:
        raise RuntimeError(f"row count mismatch: {rows} vs {expected_rows}")
    if (
        limit is None
        and expected_selfpair_rows is not None
        and zeroed_rows != expected_selfpair_rows
    ):
        raise RuntimeError(
            f"selfpair zeroed row count mismatch: {zeroed_rows} vs {expected_selfpair_rows}"
        )

    archive_sha = package_result(result_txt, result_zip)
    manifest: dict[str, object] = {
        "candidate_name": candidate_name or output_dir.name,
        "heading_csv": str(heading_csv),
        "distance_csv": str(distance_csv),
        "heading_column": heading_column,
        "distance_column": distance_column,
        "source_result_txt": str(source_result_txt),
        "result_txt": str(result_txt),
        "result_zip": str(result_zip),
        "selfpair_indices": str(selfpair_indices_path) if selfpair_indices_path else None,
        "selfpair_mode": selfpair_mode,
        "rows": rows,
        "selfpair_zeroed_rows": zeroed_rows,
        "metadata_checks": metadata_checks,
        "result_txt_sha256": sha256_file(result_txt),
        "result_zip_sha256": archive_sha,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def package_result(result_txt: Path, output_zip: Path) -> str:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo("result.txt", date_time=RELEASE_ZIP_TIMESTAMP)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, result_txt.read_bytes())
    return sha256_file(output_zip)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    from_txt = subparsers.add_parser("from-result-txt")
    from_txt.add_argument("--result-txt", type=Path, required=True)
    from_txt.add_argument("--output-zip", type=Path, required=True)

    from_csv = subparsers.add_parser("from-csv")
    from_csv.add_argument("--heading-csv", type=Path, required=True)
    from_csv.add_argument("--distance-csv", type=Path, required=True)
    from_csv.add_argument("--output-dir", type=Path, required=True)
    from_csv.add_argument("--candidate-name", default=None)
    from_csv.add_argument("--heading-column", default="heading_pred")
    from_csv.add_argument("--distance-column", default="range_pred")
    from_csv.add_argument(
        "--selfpair-indices",
        type=Path,
        default=None,
        help=(
            "Legacy audit fallback for CSVs without image identifiers. "
            "Default packaging detects self-pairs from identifiers in the same row."
        ),
    )
    from_csv.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    from_csv.add_argument("--expected-selfpair-rows", type=int, default=EXPECTED_SELFPAIR_ROWS)
    from_csv.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "from-result-txt":
        archive_sha = package_result(args.result_txt, args.output_zip)
        print(json.dumps({"result_zip": str(args.output_zip), "result_zip_sha256": archive_sha}))
        return
    if args.command == "from-csv":
        manifest = package_from_csvs(
            heading_csv=args.heading_csv,
            distance_csv=args.distance_csv,
            output_dir=args.output_dir,
            candidate_name=args.candidate_name,
            heading_column=args.heading_column,
            distance_column=args.distance_column,
            selfpair_indices_path=args.selfpair_indices,
            expected_rows=args.expected_rows,
            expected_selfpair_rows=args.expected_selfpair_rows,
            limit=args.limit,
        )
        print(json.dumps(manifest, indent=2))
        return
    raise RuntimeError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
