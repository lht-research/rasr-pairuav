"""Verify the frozen intermediate CSV artifacts used for exact reproduction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("frozen_artifacts_manifest.json"),
        help="Manifest shipped with this release repository.",
    )
    parser.add_argument(
        "--check-columns",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check CSV headers in addition to file size and SHA-256.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration as exc:
            raise RuntimeError(f"empty CSV: {path}") from exc


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    artifact_root = args.artifact_root
    results: list[dict[str, object]] = []
    errors: list[str] = []

    for entry in manifest["files"]:
        rel_path = Path(entry["path"])
        path = artifact_root / rel_path
        result: dict[str, object] = {"path": str(path), "ok": False}
        file_errors: list[str] = []
        if not path.is_file():
            message = f"missing file: {path}"
            errors.append(message)
            file_errors.append(message)
            result["error"] = "missing"
            results.append(result)
            continue

        size = path.stat().st_size
        file_sha = sha256_file(path)
        result.update({"bytes": size, "sha256": file_sha})
        if size != int(entry["bytes"]):
            message = f"size mismatch for {path}: expected {entry['bytes']}, got {size}"
            errors.append(message)
            file_errors.append(message)
        if file_sha != entry["sha256"]:
            message = f"sha256 mismatch for {path}: expected {entry['sha256']}, got {file_sha}"
            errors.append(message)
            file_errors.append(message)

        if args.check_columns:
            columns = read_columns(path)
            required = [str(value) for value in entry.get("required_columns", [])]
            missing = [column for column in required if column not in columns]
            result["columns_checked"] = required
            if missing:
                message = f"missing columns for {path}: {missing}"
                errors.append(message)
                file_errors.append(message)

        result["ok"] = not file_errors
        if file_errors:
            result["errors"] = file_errors
        results.append(result)

    report = {
        "artifact_root": str(artifact_root),
        "manifest": str(args.manifest),
        "expected_submit_zip_sha256": manifest.get("expected_submit_zip_sha256"),
        "files": results,
        "ok": not errors,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
