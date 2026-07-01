"""Run released production checkpoints on pair feature CSV files or shards."""

from __future__ import annotations

import argparse
import csv
import glob
import math
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

META_COLUMNS = ["manifest_index", "scene_id", "pair_id", "json_rel_path", "image_a", "image_b"]
DISTANCE_EDGES = np.asarray(
    [-132.0, -110.0, -90.0, -65.0, -45.0, -30.0, -18.0, -10.0, 0.0, 10.0, 30.0, 65.0, 132.0],
    dtype=np.float32,
)
DISTANCE_CENTERS = 0.5 * (DISTANCE_EDGES[:-1] + DISTANCE_EDGES[1:])
HEADING_BUCKETS = 16
HEADING_WIDTH = 360.0 / HEADING_BUCKETS
RANGE_DENOM = 132.0


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class HierarchicalDistanceHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.distance_bucket = nn.Linear(hidden_dim, len(DISTANCE_CENTERS))
        self.distance_residual = nn.Linear(hidden_dim, len(DISTANCE_CENTERS))
        self.heading_bucket = nn.Linear(hidden_dim, HEADING_BUCKETS)
        self.heading_residual = nn.Linear(hidden_dim, HEADING_BUCKETS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(x)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm(hidden)
        dist_logits = self.distance_bucket(hidden)
        dist_residuals = self.distance_residual(hidden)
        centers = torch.from_numpy(DISTANCE_CENTERS).to(x.device)
        probs = torch.softmax(dist_logits, dim=-1)
        return (probs * (centers[None, :] + dist_residuals)).sum(dim=-1)


class BucketHybridHead(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, depth: int, dropout: float, centers: np.ndarray
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(max(depth, 1)):
            layers.extend([nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(dim, 3 + len(centers))
        self.register_buffer(
            "centers", torch.from_numpy(centers.astype(np.float32)), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.out(self.backbone(x))
        logits = pred[:, 3 : 3 + self.centers.numel()]
        bucket_pred = torch.argmax(logits, dim=1)
        return self.centers[bucket_pred] + pred[:, 2] * RANGE_DENOM


class DistanceSpecialist(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_blocks: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(x)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(self.norm(hidden)).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--feature-csv", type=Path)
    source.add_argument("--feature-glob")
    source.add_argument("--feature-dir", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--legacy-batch-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the per-head batch sizes used for the archived exports.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def safe_float(value: str | None) -> float:
    try:
        parsed = float(value) if value is not None else 0.0
    except ValueError:
        parsed = 0.0
    return parsed if math.isfinite(parsed) else 0.0


def load_checkpoint(path: Path) -> dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_feature_paths(
    *,
    feature_csv: Path | None = None,
    feature_glob: str | None = None,
    feature_dir: Path | None = None,
) -> list[Path]:
    if feature_csv is not None:
        return [feature_csv]
    if feature_glob:
        paths = [Path(path) for path in glob.glob(feature_glob, recursive=True)]
    elif feature_dir is not None:
        paths = sorted(feature_dir.glob("*.csv"))
    else:
        paths = []
    paths = sorted(path for path in paths if path.is_file())
    if not paths:
        raise FileNotFoundError("no feature CSV files found")
    return paths


def iter_rows(paths: Sequence[Path], limit: int | None) -> Iterator[dict[str, str]]:
    yielded = 0
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if limit is not None and yielded >= limit:
                    return
                yielded += 1
                yield row


def read_batches(
    paths: Sequence[Path], batch_size: int, limit: int | None
) -> Iterator[list[dict[str, str]]]:
    batch: list[dict[str, str]] = []
    for row in iter_rows(paths, limit):
        batch.append(row)
        if len(batch) >= max(batch_size, 1):
            yield batch
            batch = []
    if batch:
        yield batch


def feature_matrix(
    rows: list[dict[str, str]], feature_names: list[str], mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    if rows:
        missing = [name for name in feature_names if name not in rows[0]]
        if missing:
            preview = ", ".join(missing[:8])
            raise ValueError(
                f"feature CSV is missing {len(missing)} required columns; "
                f"first missing columns: {preview}"
            )
    std = np.where(std == 0, 1.0, std)
    x = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        if str(row.get("is_valid", "true")).strip().lower() not in {"1", "true", "yes"}:
            continue
        for feature_idx, name in enumerate(feature_names):
            x[row_idx, feature_idx] = safe_float(row.get(name))
    return ((x - mean) / std).astype(np.float32)


def build_model(checkpoint: dict[str, object]) -> nn.Module:
    args = checkpoint["args"]
    feature_count = len(checkpoint["feature_names"])
    state = checkpoint["model_state_dict"]
    if "distance_bucket.weight" in state:
        return HierarchicalDistanceHead(
            input_dim=feature_count,
            hidden_dim=int(args["hidden_dim"]),
            depth=int(args["depth"]),
            dropout=float(args["dropout"]),
        )
    if "out.weight" in state:
        return BucketHybridHead(
            input_dim=feature_count,
            hidden_dim=int(args["hidden_dim"]),
            depth=int(args["depth"]),
            dropout=float(args["dropout"]),
            centers=np.asarray(checkpoint["bucket_centers"], dtype=np.float32),
        )
    return DistanceSpecialist(
        input_dim=feature_count,
        hidden_dim=int(args["hidden_dim"]),
        n_blocks=int(args["n_blocks"]),
        dropout=float(args["dropout"]),
    )


def predict_checkpoint(
    *,
    checkpoint_path: Path,
    feature_paths: Sequence[Path],
    output_csv: Path,
    batch_size: int,
    device: torch.device,
    limit: int | None,
) -> None:
    checkpoint = load_checkpoint(checkpoint_path)
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    feature_names = list(checkpoint["feature_names"])
    model = build_model(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    writer: csv.DictWriter[str] | None = None
    with output_csv.open("w", encoding="utf-8", newline="") as handle, torch.no_grad():
        for rows in read_batches(feature_paths, batch_size=batch_size, limit=limit):
            if writer is None:
                metadata_keys = [key for key in META_COLUMNS if key in rows[0]]
                writer = csv.DictWriter(handle, fieldnames=[*metadata_keys, "range_pred"])
                writer.writeheader()
            features = feature_matrix(rows, feature_names, mean, std)
            batch = torch.from_numpy(features).to(device)
            pred = model(batch).detach().cpu().numpy().astype(np.float64)
            for row, value in zip(rows, pred.tolist()):  # noqa: B905
                out = {
                    key: row.get(key, "") for key in writer.fieldnames or [] if key != "range_pred"
                }
                out["range_pred"] = f"{float(value):.6f}"
                writer.writerow(out)
        if writer is None:
            writer = csv.DictWriter(handle, fieldnames=[*META_COLUMNS, "range_pred"])
            writer.writeheader()


def predict_all_heads(
    *,
    feature_csv: Path | None = None,
    feature_glob: str | None = None,
    feature_dir: Path | None = None,
    model_dir: Path,
    output_dir: Path,
    batch_size: int = 1024,
    device: str = "cpu",
    limit: int | None = None,
    legacy_batch_policy: bool = True,
) -> list[Path]:
    feature_paths = resolve_feature_paths(
        feature_csv=feature_csv,
        feature_glob=feature_glob,
        feature_dir=feature_dir,
    )
    device_obj = torch.device(device)
    outputs: list[Path] = []
    legacy_batch_sizes = {
        "distance_head_a": 1024,
        "distance_head_b": 1024,
        "distance_head_c": 8192,
        "distance_head_d": 8192,
    }
    for head_name in ["distance_head_a", "distance_head_b", "distance_head_c", "distance_head_d"]:
        output_csv = output_dir / f"{head_name}.csv"
        head_batch_size = legacy_batch_sizes[head_name] if legacy_batch_policy else batch_size
        predict_checkpoint(
            checkpoint_path=model_dir / f"{head_name}.pt",
            feature_paths=feature_paths,
            output_csv=output_csv,
            batch_size=head_batch_size,
            device=device_obj,
            limit=limit,
        )
        outputs.append(output_csv)
    return outputs


def main() -> None:
    args = parse_args()
    outputs = predict_all_heads(
        feature_csv=args.feature_csv,
        feature_glob=args.feature_glob,
        feature_dir=args.feature_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        legacy_batch_policy=args.legacy_batch_policy,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
