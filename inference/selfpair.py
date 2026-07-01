"""Row-local self-pair detection and correction."""

from __future__ import annotations

from collections.abc import Iterable


def is_self_pair(left_image_id: str, right_image_id: str) -> bool:
    """Return true when both row-local image identifiers refer to the same image."""

    return _normalize_image_id(left_image_id) == _normalize_image_id(right_image_id)


def apply_self_pair_zero(
    left_image_ids: Iterable[str],
    right_image_ids: Iterable[str],
    distances: Iterable[float],
) -> list[float]:
    """Set distance to zero for rows whose two image identifiers match."""

    left_values = list(left_image_ids)
    right_values = list(right_image_ids)
    distance_values = list(distances)
    if not (len(left_values) == len(right_values) == len(distance_values)):
        raise ValueError("left ids, right ids, and distances must have the same length")

    corrected: list[float] = []
    for left_id, right_id, distance in zip(  # noqa: B905
        left_values, right_values, distance_values
    ):
        corrected.append(0.0 if is_self_pair(left_id, right_id) else float(distance))
    return corrected


def _normalize_image_id(image_id: str) -> str:
    value = str(image_id).strip().replace("\\", "/")
    return value.rsplit("/", 1)[-1]
