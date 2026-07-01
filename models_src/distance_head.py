"""Distance head modules used by the PairUAV last-meter pipeline."""

from __future__ import annotations

import torch
from torch import nn


class DistanceHead(nn.Module):
    """Small MLP regressor for pairwise distance."""

    def __init__(self, feature_dim: int, hidden_dim: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)
