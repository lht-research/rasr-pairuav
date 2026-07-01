"""Train a distance head from frozen pair features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models_src.distance_head import DistanceHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--head-name", required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_training_arrays(
    path: Path,
    *,
    target_column: str,
    metadata_columns: list[str],
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if path.suffix.lower() == ".npz":
        payload = np.load(path, allow_pickle=False)
        feature_key = "features" if "features" in payload else "x"
        target_key = target_column if target_column in payload else "targets"
        if target_key not in payload:
            target_key = "y"
        if feature_key not in payload or target_key not in payload:
            raise ValueError(f"NPZ must contain features/x and {target_column}/targets/y: {path}")
        x = np.asarray(payload[feature_key], dtype=np.float32)
        y = np.asarray(payload[target_key], dtype=np.float32)
        feature_names = [f"feature_{idx}" for idx in range(x.shape[1])]
    elif path.suffix.lower() == ".csv":
        x, y, feature_names = load_csv_arrays(
            path,
            target_column=target_column,
            metadata_columns=metadata_columns,
            limit=limit,
        )
        return x, y, feature_names
    else:
        raise ValueError(f"unsupported training feature file: {path}")

    if limit is not None:
        x = x[:limit]
        y = y[:limit]
    return x.astype(np.float32), y.astype(np.float32).reshape(-1), feature_names


def load_csv_arrays(
    path: Path,
    *,
    target_column: str,
    metadata_columns: list[str],
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if target_column not in (reader.fieldnames or []):
            raise ValueError(f"missing target column {target_column!r} in {path}")
        ignored = set(metadata_columns)
        ignored.add(target_column)
        numeric_columns = [col for col in (reader.fieldnames or []) if col not in ignored]
        rows: list[list[float]] = []
        targets: list[float] = []
        for row_idx, row in enumerate(reader):
            if limit is not None and row_idx >= limit:
                break
            values: list[float] = []
            for col in numeric_columns:
                try:
                    values.append(float(row[col]))
                except (TypeError, ValueError):
                    break
            else:
                rows.append(values)
                targets.append(float(row[target_column]))
    if not rows:
        raise ValueError(f"no numeric training rows loaded from {path}")
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        numeric_columns,
    )


def standardize_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std, mean.astype(np.float32), std.astype(np.float32)


def train(args: argparse.Namespace) -> dict[str, object]:
    config = load_config(args.config)
    head_config = (config.get("heads") or {}).get(args.head_name, {})
    if not isinstance(head_config, dict):
        raise ValueError(f"head config must be a mapping: {args.head_name}")
    seed = int(head_config.get("seed", 2026))
    set_seed(seed)

    target_column = args.target_column or str(config.get("target_column", "range_num"))
    metadata_columns = [str(v) for v in config.get("metadata_columns", [])]
    x_raw, y_raw, feature_names = load_training_arrays(
        args.train_features,
        target_column=target_column,
        metadata_columns=metadata_columns,
        limit=args.limit,
    )
    x, mean, std = standardize_features(x_raw)
    y_scale = max(float(np.max(np.abs(y_raw))), 1.0)
    y = y_raw / y_scale

    device = torch.device(args.device)
    features = torch.from_numpy(x.astype(np.float32))
    targets = torch.from_numpy(y.astype(np.float32))
    dataset = TensorDataset(features, targets)
    batch_size = min(int(config.get("batch_size", 256)), len(dataset))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    model = DistanceHead(
        feature_dim=features.shape[1],
        hidden_dim=int(config.get("hidden_dim", 256)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    criterion: nn.Module = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    max_steps = int(args.max_steps or max(1, int(config.get("max_epochs", 1)) * len(loader)))
    losses: list[float] = []
    step = 0
    model.train()
    while step < max_steps:
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            step += 1
            if step >= max_steps:
                break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "head_name": args.head_name,
        "feature_dim": int(features.shape[1]),
        "hidden_dim": int(config.get("hidden_dim", 256)),
        "dropout": float(config.get("dropout", 0.1)),
        "feature_names": feature_names,
        "target_column": target_column,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "target_scale": y_scale,
        "losses": losses,
    }
    torch.save(checkpoint, args.output)

    reloaded = torch.load(args.output, map_location="cpu", weights_only=False)
    if "model_state_dict" not in reloaded:
        raise RuntimeError("checkpoint reload failed: missing model_state_dict")

    summary = {
        "head_name": args.head_name,
        "rows": int(len(dataset)),
        "feature_dim": int(features.shape[1]),
        "steps": len(losses),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_decreased": bool(losses[-1] < losses[0]),
        "checkpoint": str(args.output),
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
