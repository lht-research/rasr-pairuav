"""Frozen heading ensemble utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeadingEnsemble:
    """Weighted circular average for heading predictions in degrees."""

    weights: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        if weights.ndim != 1:
            raise ValueError("weights must be one-dimensional")
        if np.any(weights < -1e-12):
            raise ValueError("weights must be non-negative")
        if not np.isclose(weights.sum(), 1.0, atol=1e-8):
            raise ValueError("weights must sum to one")
        object.__setattr__(self, "weights", weights)

    def combine_degrees(self, headings: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        headings_array = np.asarray(headings, dtype=np.float64)
        if headings_array.ndim != 2:
            raise ValueError("headings must have shape [num_rows, num_models]")
        radians = np.deg2rad(headings_array)
        sin_mean = np.sum(np.sin(radians) * self.weights, axis=1)
        cos_mean = np.sum(np.cos(radians) * self.weights, axis=1)
        return (np.rad2deg(np.arctan2(sin_mean, cos_mean)) + 360.0) % 360.0
