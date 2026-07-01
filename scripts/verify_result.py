"""Verify a packaged PairUAV result archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

EXPECTED_ARCHIVE_SHA256 = "2f742f2eff83e535b96a8dbd46db370fa3ac0538a9f3e53b684d65c253b34b77"
EXPECTED_ROWS = 2_773_116
EXPECTED_SELFPAIR_ROWS = 51_354


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-zip", type=Path, required=True)
    parser.add_argument("--expected-sha256", default=None)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    parser.add_argument("--selfpair-indices", type=Path, default=None)
    parser.add_argument("--expected-selfpair-rows", type=int, default=EXPECTED_SELFPAIR_ROWS)
    parser.add_argument("--skip-sha256", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actual = sha256_file(args.result_zip)
    expected_sha = args.expected_sha256 or EXPECTED_ARCHIVE_SHA256
    if not args.skip_sha256 and actual != expected_sha:
        raise SystemExit(f"sha256 mismatch: expected {args.expected_sha256}, got {actual}")
    with zipfile.ZipFile(args.result_zip) as archive:
        names = archive.namelist()
        if names != ["result.txt"]:
            raise SystemExit(f"archive root mismatch: expected ['result.txt'], got {names}")
        with archive.open("result.txt") as handle:
            lines = [line.decode("utf-8").rstrip("\n") for line in handle]
    if args.expected_rows is not None and len(lines) != args.expected_rows:
        raise SystemExit(f"row count mismatch: expected {args.expected_rows}, got {len(lines)}")
    selfpair_zeroed = None
    if args.selfpair_indices:
        indices = {
            int(value) for value in json.loads(args.selfpair_indices.read_text(encoding="utf-8"))
        }
        selfpair_zeroed = sum(
            1 for idx in indices if idx < len(lines) and lines[idx] == "0.000000 0.000000"
        )
        if selfpair_zeroed != args.expected_selfpair_rows:
            raise SystemExit(
                "selfpair count mismatch: "
                f"expected {args.expected_selfpair_rows}, got {selfpair_zeroed}"
            )
    print(
        json.dumps(
            {
                "result_zip": str(args.result_zip),
                "result_zip_sha256": actual,
                "zip_names": names,
                "rows": len(lines),
                "selfpair_zeroed_rows": selfpair_zeroed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
