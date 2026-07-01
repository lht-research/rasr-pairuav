"""Export a heading prediction CSV from a two-column result text file.

The released packager consumes heading as a CSV column. This helper converts an
existing prediction text file whose first whitespace-separated column is heading
in degrees into that CSV contract.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result-txt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with (
        args.source_result_txt.open("r", encoding="utf-8") as input_handle,
        args.output.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as output_handle,
    ):
        writer = csv.DictWriter(output_handle, fieldnames=["heading_pred"])
        writer.writeheader()
        for line in input_handle:
            if args.limit is not None and rows >= args.limit:
                break
            parts = line.strip().split()
            if not parts:
                continue
            writer.writerow({"heading_pred": f"{float(parts[0]):.6f}"})
            rows += 1
    print(f"wrote {rows} heading rows to {args.output}")


if __name__ == "__main__":
    main()
