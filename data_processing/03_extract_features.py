"""Extract pair features for training and released-checkpoint inference.

The default smoke mode writes deterministic image-statistics features for fast
local pipeline checks. Production mode runs a frozen visual pair backbone and
writes the 422-column feature contract expected by the released checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

BASE_COLUMNS = [
    "manifest_index",
    "scene_id",
    "pair_id",
    "json_rel_path",
    "image_a",
    "image_b",
    "heading_num",
    "range_num",
    "is_valid",
    "failure_reason",
    "pair_wall_seconds",
]
POINT_STATS = ["mean", "std", "p05", "p25", "p50", "p75", "p95", "min", "max"]
CONF_STATS = ["mean", "std", "p10", "p25", "p50", "p75", "p90", "min", "max"]
DESC_STATS = ["mean", "std", "p10", "p50", "p90"]
CONF_BUCKETS = (1.0, 2.0, 3.0, 5.0, 10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "production"], default="smoke")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--json-dir", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--pair-batch-size", type=int, default=2)
    parser.add_argument(
        "--batched-pair-forward",
        action="store_true",
        help=(
            "Batch multiple pairs into one visual-model forward pass. "
            "Strict per-pair forward is the reproducible default."
        ),
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_image(path_value: str, image_root: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    for candidate in (image_root / path, image_root / path.name):
        if candidate.is_file():
            return candidate
    return None


def image_stats(path: Path | None) -> np.ndarray:
    if path is None:
        return np.zeros(14, dtype=np.float32)
    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    flat = arr.reshape(-1, 3)
    return np.asarray(
        [
            arr.shape[1],
            arr.shape[0],
            *flat.mean(axis=0),
            *flat.std(axis=0),
            *np.percentile(flat, 10, axis=0),
            *np.percentile(flat, 90, axis=0),
        ],
        dtype=np.float32,
    )


def default_image_root(data_root: Path) -> Path:
    candidates = [
        data_root / "PairUAV" / "train_tour",
        data_root / "pairUAV" / "train_tour",
        data_root / "train_tour",
        data_root / "PairUAV",
        data_root,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return data_root


def run_smoke_mode(args: argparse.Namespace) -> None:
    if args.pairs is None:
        raise ValueError("--pairs is required in smoke mode")
    image_root = args.image_root or default_image_root(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with args.pairs.open("r", encoding="utf-8", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        for idx, row in enumerate(reader):
            if args.limit is not None and idx >= args.limit:
                break
            a_stats = image_stats(resolve_image(row.get("image_a", ""), image_root))
            b_stats = image_stats(resolve_image(row.get("image_b", ""), image_root))
            values = np.concatenate([a_stats, b_stats, np.abs(a_stats - b_stats)]).astype(
                np.float32
            )
            out_row = {
                "manifest_index": row.get("manifest_index", str(idx)),
                "scene_id": row.get("scene_id", ""),
                "pair_id": row.get("pair_id", ""),
                "image_a": row.get("image_a", ""),
                "image_b": row.get("image_b", ""),
                "range_num": row.get("range_num", row.get("range_true", "")),
            }
            for feature_idx, value in enumerate(values):
                out_row[f"feature_{feature_idx:03d}"] = f"{float(value):.9f}"
            rows.append(out_row)

    fieldnames = ["manifest_index", "scene_id", "pair_id", "image_a", "image_b", "range_num"]
    if rows:
        fieldnames.extend(key for key in rows[0] if key.startswith("feature_"))
    with args.output.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} feature rows to {args.output}")


def parse_manifest_line(line: str, logical_index: int) -> tuple[int, int, str] | None:
    text = line.strip()
    if not text:
        return None
    if "\t" in text:
        maybe_index, rel_path = text.split("\t", 1)
        if maybe_index.strip().isdigit():
            return logical_index, int(maybe_index.strip()), rel_path.strip()
    return logical_index, logical_index, text


def count_manifest_rows(path: Path) -> int:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    return rows


def manifest_bounds(path: Path, start_offset: int, limit: int | None) -> tuple[int, int, int]:
    total = count_manifest_rows(path)
    if start_offset < 0 or start_offset > total:
        raise ValueError(f"invalid start_offset={start_offset} for manifest rows={total}")
    end = min(total, start_offset + limit) if limit else total
    if end <= start_offset:
        raise RuntimeError(f"empty manifest window: start_offset={start_offset} limit={limit}")
    return start_offset, end, total


def shard_bounds(total_rows: int, shard_id: int, num_shards: int) -> tuple[int, int]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards})")
    return total_rows * shard_id // num_shards, total_rows * (shard_id + 1) // num_shards


def iter_manifest_slice(path: Path, start: int, end: int) -> Iterable[tuple[int, int, str]]:
    logical_index = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parsed = parse_manifest_line(raw_line, logical_index)
            if parsed is None:
                continue
            current, source_index, rel_path = parsed
            if current >= end:
                break
            if current >= start:
                yield current, source_index, rel_path
            logical_index += 1


def resolve_pair_image(image_dir: Path, image_rel_path: str) -> Path:
    rel = Path(image_rel_path)
    candidates = [image_dir / rel, image_dir / rel.name]
    for ext in (".jpeg", ".jpg", ".png", ".webp"):
        candidates.append(image_dir / rel.parent / f"{rel.stem}{ext}")
        candidates.append(image_dir / f"{rel.stem}{ext}")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"could not resolve image {image_rel_path} under {image_dir}")


def first_line(exc: Exception) -> str:
    return str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__


def load_runtime(runtime_root: Path) -> dict[str, Any]:
    mast3r_repo = runtime_root / "mast3r"
    dust3r_repo = mast3r_repo / "dust3r"
    for path in (runtime_root, mast3r_repo, dust3r_repo):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    if os.environ.get("PAIRUAV_TORCHVISION_NMS_STUB", "1") == "1":
        try:
            import torch
            from torch.library import Library

            try:
                torch._C._dispatch_find_schema_or_throw("torchvision::nms", "")
            except Exception:
                Library("torchvision", "DEF").define(
                    "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
                )
        except Exception:
            pass

    import run_mast3r_probe as probe_ref
    import torch
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.device import collate_with_cat, to_cpu
    from mast3r.model import AsymmetricMASt3R

    return {
        "torch": torch,
        "probe_ref": probe_ref,
        "make_pairs": make_pairs,
        "collate_with_cat": collate_with_cat,
        "to_cpu": to_cpu,
        "AsymmetricMASt3R": AsymmetricMASt3R,
    }


def amp_dtype(name: str, torch_mod: Any) -> Any:
    if name == "fp32":
        return None
    if name == "fp16":
        return torch_mod.float16
    if name == "bf16":
        return torch_mod.bfloat16
    raise ValueError(f"unsupported dtype {name}")


def sync_device(device: Any, torch_mod: Any) -> None:
    if device.type == "cuda":
        torch_mod.cuda.synchronize(device)


def load_record(
    json_dir: Path, image_dir: Path, manifest_index: int, rel_path: str
) -> dict[str, Any]:
    json_path = (json_dir / rel_path).resolve()
    if not json_path.is_file():
        raise FileNotFoundError(json_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    image_a = str(payload["image_a"])
    image_b = str(payload["image_b"])
    return {
        "manifest_index": manifest_index,
        "scene_id": json_path.parent.name,
        "pair_id": json_path.stem,
        "json_rel_path": rel_path,
        "image_a": image_a,
        "image_b": image_b,
        "heading_num": float(payload.get("heading_num", math.nan)),
        "range_num": float(payload.get("range_num", math.nan)),
        "image_a_path": resolve_pair_image(image_dir, image_a),
        "image_b_path": resolve_pair_image(image_dir, image_b),
    }


def prepare_directed_pairs(
    runtime: dict[str, Any], record: dict[str, Any], image_size: int, patch_size: int
) -> list[Any]:
    probe_ref = runtime["probe_ref"]
    imgs = [
        probe_ref.prepare_image(
            record["image_a_path"], idx=0, size=image_size, patch_size=patch_size
        ),
        probe_ref.prepare_image(
            record["image_b_path"], idx=1, size=image_size, patch_size=patch_size
        ),
    ]
    return list(runtime["make_pairs"](imgs, scene_graph="complete", symmetrize=True))


def run_forward(
    runtime: dict[str, Any], directed_pairs: list[Any], model: Any, device: Any, dtype: Any
) -> dict[str, Any]:
    torch_mod = runtime["torch"]
    view1, view2 = runtime["collate_with_cat"](directed_pairs)
    ignore = {"depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"}
    for view in (view1, view2):
        for name, value in list(view.items()):
            if name not in ignore and isinstance(value, torch_mod.Tensor):
                view[name] = value.to(device, non_blocking=True)
    enabled = dtype is not None and device.type == "cuda"
    with torch_mod.inference_mode():
        with torch_mod.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            pred1, pred2 = model(view1, view2)
    return runtime["to_cpu"]({"view1": view1, "view2": view2, "pred1": pred1, "pred2": pred2})


def finite_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)


def add_array_stats(row: dict[str, Any], prefix: str, arr: np.ndarray, names: list[str]) -> None:
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        for name in names:
            row[f"{prefix}_{name}"] = math.nan
        return
    percentiles = {"p05": 5, "p10": 10, "p25": 25, "p50": 50, "p75": 75, "p90": 90, "p95": 95}
    for name in names:
        if name == "mean":
            value = np.mean(flat)
        elif name == "std":
            value = np.std(flat)
        elif name == "min":
            value = np.min(flat)
        elif name == "max":
            value = np.max(flat)
        else:
            value = np.percentile(flat, percentiles[name])
        row[f"{prefix}_{name}"] = float(value)


def add_channel_stats(row: dict[str, Any], prefix: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim < 2:
        add_array_stats(row, prefix, arr, POINT_STATS)
        return
    for channel in range(arr.shape[-1]):
        add_array_stats(row, f"{prefix}_c{channel}", arr[..., channel], POINT_STATS)
    add_array_stats(row, f"{prefix}_norm", np.linalg.norm(arr, axis=-1), POINT_STATS)


def add_conf_bucket_stats(row: dict[str, Any], prefix: str, conf: np.ndarray) -> None:
    finite = np.asarray(conf, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    total = max(int(finite.size), 1)
    for threshold in CONF_BUCKETS:
        row[f"{prefix}_ge_{str(threshold).replace('.', 'p')}_ratio"] = float(
            np.count_nonzero(finite >= threshold) / total
        )


def directed_positions(output: dict[str, Any], pos_a: int, pos_b: int) -> tuple[int, int]:
    mapping = {
        (int(output["view1"]["idx"][pos_a]), int(output["view2"]["idx"][pos_a])): pos_a,
        (int(output["view1"]["idx"][pos_b]), int(output["view2"]["idx"][pos_b])): pos_b,
    }
    if (0, 1) not in mapping or (1, 0) not in mapping:
        raise RuntimeError(f"expected both directed edges, got {sorted(mapping.keys())}")
    return mapping[(0, 1)], mapping[(1, 0)]


def build_feature_row(
    record: dict[str, Any], output: dict[str, Any], pos_a: int, pos_b: int, pair_seconds: float
) -> dict[str, Any]:
    idx_ab, idx_ba = directed_positions(output, pos_a, pos_b)
    pts3d_a = finite_np(output["pred1"]["pts3d"][idx_ab])
    pts3d_b = finite_np(output["pred1"]["pts3d"][idx_ba])
    pts3d_b_in_a = finite_np(output["pred2"]["pts3d_in_other_view"][idx_ab])
    conf_a = finite_np(output["pred1"]["conf"][idx_ab])
    conf_b = finite_np(output["pred2"]["conf"][idx_ab])
    delta = (
        pts3d_b_in_a - pts3d_a
        if pts3d_b_in_a.shape == pts3d_a.shape
        else np.asarray([], dtype=np.float32)
    )
    row: dict[str, Any] = {
        "manifest_index": int(record["manifest_index"]),
        "scene_id": str(record["scene_id"]),
        "pair_id": str(record["pair_id"]),
        "json_rel_path": str(record["json_rel_path"]),
        "image_a": str(record["image_a"]),
        "image_b": str(record["image_b"]),
        "heading_num": float(record["heading_num"]),
        "range_num": float(record["range_num"]),
        "is_valid": True,
        "failure_reason": "",
        "pair_wall_seconds": float(pair_seconds),
    }
    add_channel_stats(row, "pts3d_a", pts3d_a)
    add_channel_stats(row, "pts3d_b", pts3d_b)
    add_channel_stats(row, "pts3d_b_in_a", pts3d_b_in_a)
    add_channel_stats(row, "delta_pts3d_ba", delta)
    add_array_stats(row, "conf_a", conf_a, CONF_STATS)
    add_array_stats(row, "conf_b", conf_b, CONF_STATS)
    add_conf_bucket_stats(row, "conf_a", conf_a)
    add_conf_bucket_stats(row, "conf_b", conf_b)
    if "desc" in output["pred1"] and "desc" in output["pred2"]:
        desc_a = finite_np(output["pred1"]["desc"][idx_ab])
        desc_b = finite_np(output["pred2"]["desc"][idx_ab])
        for prefix, desc in (("desc_a", desc_a), ("desc_b", desc_b)):
            channels = min(desc.shape[-1], 24) if desc.ndim >= 3 else 0
            for channel in range(channels):
                add_array_stats(row, f"{prefix}_c{channel}", desc[..., channel], DESC_STATS)
            if channels:
                add_array_stats(
                    row, f"{prefix}_norm", np.linalg.norm(desc[..., :channels], axis=-1), DESC_STATS
                )
    return row


def failure_row(record: dict[str, Any], reason: str, pair_seconds: float = 0.0) -> dict[str, Any]:
    return {
        "manifest_index": int(record["manifest_index"]),
        "scene_id": str(record.get("scene_id", "")),
        "pair_id": str(record.get("pair_id", "")),
        "json_rel_path": str(record.get("json_rel_path", "")),
        "image_a": str(record.get("image_a", "")),
        "image_b": str(record.get("image_b", "")),
        "heading_num": float(record.get("heading_num", math.nan)),
        "range_num": float(record.get("range_num", math.nan)),
        "is_valid": False,
        "failure_reason": reason,
        "pair_wall_seconds": float(pair_seconds),
    }


def process_batch(
    records: list[dict[str, Any]],
    runtime: dict[str, Any],
    model: Any,
    device: Any,
    dtype: Any,
    image_size: int,
    patch_size: int,
) -> list[dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    directed_pairs: list[Any] = []
    prepared: list[tuple[dict[str, Any], int, int]] = []
    for record in records:
        try:
            pair_directed = prepare_directed_pairs(runtime, record, image_size, patch_size)
            if len(pair_directed) != 2:
                raise RuntimeError(f"expected 2 directed pairs, got {len(pair_directed)}")
            pos = len(directed_pairs)
            directed_pairs.extend(pair_directed)
            prepared.append((record, pos, pos + 1))
        except Exception as exc:
            out[int(record["manifest_index"])] = failure_row(
                record, f"prepare:{exc.__class__.__name__}:{first_line(exc)}"
            )
    if prepared:
        try:
            start = time.perf_counter()
            sync_device(device, runtime["torch"])
            output = run_forward(runtime, directed_pairs, model, device, dtype)
            sync_device(device, runtime["torch"])
            pair_seconds = (time.perf_counter() - start) / max(len(prepared), 1)
            for record, pos_a, pos_b in prepared:
                try:
                    out[int(record["manifest_index"])] = build_feature_row(
                        record, output, pos_a, pos_b, pair_seconds
                    )
                except Exception as exc:
                    out[int(record["manifest_index"])] = failure_row(
                        record, f"stats:{exc.__class__.__name__}:{first_line(exc)}", pair_seconds
                    )
        except Exception as exc:
            if len(prepared) == 1:
                record, _, _ = prepared[0]
                out[int(record["manifest_index"])] = failure_row(
                    record, f"forward:{exc.__class__.__name__}:{first_line(exc)}"
                )
            else:
                mid = len(prepared) // 2
                for row in process_batch(
                    [item[0] for item in prepared[:mid]],
                    runtime,
                    model,
                    device,
                    dtype,
                    image_size,
                    patch_size,
                ):
                    out[int(row["manifest_index"])] = row
                for row in process_batch(
                    [item[0] for item in prepared[mid:]],
                    runtime,
                    model,
                    device,
                    dtype,
                    image_size,
                    patch_size,
                ):
                    out[int(row["manifest_index"])] = row
    return [out[int(record["manifest_index"])] for record in records]


def process_records(
    records: list[dict[str, Any]],
    runtime: dict[str, Any],
    model: Any,
    device: Any,
    dtype: Any,
    image_size: int,
    patch_size: int,
    batched_pair_forward: bool,
) -> list[dict[str, Any]]:
    if batched_pair_forward:
        return process_batch(records, runtime, model, device, dtype, image_size, patch_size)
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.extend(
            process_batch([dict(record)], runtime, model, device, dtype, image_size, patch_size)
        )
    return rows


def default_fieldnames() -> list[str]:
    names = list(BASE_COLUMNS)
    for prefix in ["pts3d_a", "pts3d_b", "pts3d_b_in_a", "delta_pts3d_ba"]:
        for suffix in ["c0", "c1", "c2", "norm"]:
            names.extend(f"{prefix}_{suffix}_{stat}" for stat in POINT_STATS)
    for prefix in ["conf_a", "conf_b"]:
        names.extend(f"{prefix}_{stat}" for stat in CONF_STATS)
        names.extend(
            f"{prefix}_ge_{str(threshold).replace('.', 'p')}_ratio" for threshold in CONF_BUCKETS
        )
    for prefix in ["desc_a", "desc_b"]:
        for channel in range(24):
            names.extend(f"{prefix}_c{channel}_{stat}" for stat in DESC_STATS)
        names.extend(f"{prefix}_norm_{stat}" for stat in DESC_STATS)
    return names


def scan_existing(path: Path) -> set[int]:
    processed: set[int] = set()
    if not path.exists():
        return processed
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                processed.add(int(row.get("manifest_index", "")))
            except ValueError:
                pass
    return processed


def append_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if need_header:
            writer.writeheader()
        writer.writerows(rows)


def production_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    data_root = args.data_root.resolve()
    runtime_root = (args.runtime_root or data_root.parent / "mast3r_probe").resolve()
    model_path = (
        args.model_path
        or runtime_root / "models" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    ).resolve()
    manifest = (
        args.manifest or data_root / "baseline" / "split_manifests" / "train_manifest.txt"
    ).resolve()
    json_dir = (args.json_dir or data_root / "pairUAV" / "train").resolve()
    image_dir = (args.image_dir or data_root / "pairUAV" / "train_tour").resolve()
    return runtime_root, model_path, manifest, json_dir, image_dir


def run_production_mode(args: argparse.Namespace) -> None:
    runtime_root, model_path, manifest, json_dir, image_dir = production_paths(args)
    window_start, window_end, manifest_total = manifest_bounds(
        manifest, args.start_offset, args.limit
    )
    total_rows = window_end - window_start
    rel_start, rel_end = shard_bounds(total_rows, args.shard_id, args.num_shards)
    start_index = window_start + rel_start
    end_index = window_start + rel_end
    fieldnames = default_fieldnames()
    processed = scan_existing(args.output) if args.resume else set()

    runtime = load_runtime(runtime_root)
    torch_mod = runtime["torch"]
    device = torch_mod.device(args.device)
    model = runtime["AsymmetricMASt3R"].from_pretrained(str(model_path)).to(device)
    model.eval()
    model.requires_grad_(False)
    patch_size = (
        int(model.patch_size if isinstance(model.patch_size, int) else model.patch_size[0])
        if hasattr(model, "patch_size")
        else 16
    )
    dtype = amp_dtype(args.dtype, torch_mod)
    print(
        f"[features] rows=[{start_index},{end_index}) shard={args.shard_id}/{args.num_shards} "
        f"device={device} dtype={args.dtype} batch={args.pair_batch_size}",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    buffer: list[dict[str, Any]] = []
    success = 0
    failure = 0
    start_wall = time.perf_counter()

    def flush(force: bool = False) -> None:
        nonlocal buffer, success, failure
        if not buffer or (not force and len(buffer) < args.flush_every):
            return
        append_rows(args.output, buffer, fieldnames)
        success += sum(1 for row in buffer if bool(row.get("is_valid", False)))
        failure += sum(1 for row in buffer if not bool(row.get("is_valid", False)))
        print(f"[features] wrote batch success={success} failure={failure}", flush=True)
        buffer = []

    for _logical_index, source_index, rel_path in iter_manifest_slice(
        manifest, start_index, end_index
    ):
        if source_index in processed:
            continue
        try:
            records.append(load_record(json_dir, image_dir, source_index, rel_path))
        except Exception as exc:
            stub = {
                "manifest_index": source_index,
                "scene_id": Path(rel_path).parent.name,
                "pair_id": Path(rel_path).stem,
                "json_rel_path": rel_path,
            }
            buffer.append(failure_row(stub, f"record:{exc.__class__.__name__}:{first_line(exc)}"))
        if len(records) >= args.pair_batch_size:
            buffer.extend(
                process_records(
                    records,
                    runtime,
                    model,
                    device,
                    dtype,
                    args.image_size,
                    patch_size,
                    args.batched_pair_forward,
                )
            )
            records = []
            flush()
            if device.type == "cuda":
                torch_mod.cuda.empty_cache()
    if records:
        buffer.extend(
            process_records(
                records,
                runtime,
                model,
                device,
                dtype,
                args.image_size,
                patch_size,
                args.batched_pair_forward,
            )
        )
    flush(force=True)

    print(
        json.dumps(
            {
                "mode": "production",
                "manifest_total_rows": manifest_total,
                "start_index": start_index,
                "end_index": end_index,
                "success_count": success,
                "failure_count": failure,
                "total_wall_seconds": time.perf_counter() - start_wall,
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.mode == "production":
        run_production_mode(args)
    else:
        run_smoke_mode(args)


if __name__ == "__main__":
    main()
