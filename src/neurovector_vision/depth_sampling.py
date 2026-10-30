"""Depth map sampling without model or runtime state."""

from collections.abc import Iterable

import numpy as np

from .models import BoundingBox


def valid_values(region: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    """Retain finite in-range values in the original traversal order."""
    mask = np.isfinite(region) & (region >= min_depth) & (region <= max_depth)
    return region[mask]


def median_depth(
    depth_map: np.ndarray,
    box: BoundingBox,
    *,
    inset_ratio: float = 0.2,
    min_pixels: int = 12,
    min_depth: float,
    max_depth: float,
) -> float | None:
    """바운딩 박스 중앙 영역의 유효 깊이 중앙값을 반환한다."""
    if depth_map.ndim != 2 or depth_map.size == 0:
        return None
    height, width = depth_map.shape
    if not all(np.isfinite(value) for value in (box.x1, box.y1, box.x2, box.y2)):
        return None
    box_width = box.x2 - box.x1
    box_height = box.y2 - box.y1
    if box_width <= 0.0 or box_height <= 0.0:
        return None
    if not 0.0 <= inset_ratio < 0.5:
        raise ValueError("inset_ratio must be between 0 and 0.5")
    inset_x = box_width * inset_ratio
    inset_y = box_height * inset_ratio
    left, right = box.x1 + inset_x, box.x2 - inset_x
    top, bottom = box.y1 + inset_y, box.y2 - inset_y
    if left >= width or right <= 0.0 or top >= height or bottom <= 0.0:
        return None
    x1 = max(0, min(width, int(np.floor(left))))
    x2 = max(0, min(width, int(np.ceil(right))))
    y1 = max(0, min(height, int(np.floor(top))))
    y2 = max(0, min(height, int(np.ceil(bottom))))
    region = depth_map[y1:y2, x1:x2]
    values = valid_values(region, min_depth, max_depth)
    if values.size < min_pixels:
        return None
    return float(np.median(values))


def median_depth_around_points(
    depth_map: np.ndarray,
    points: Iterable[tuple[float, float]],
    *,
    radius_px: float = 6.0,
    min_pixels: int = 8,
    min_depth: float,
    max_depth: float,
) -> float | None:
    """키포인트 주변 작은 패치들의 유효 깊이 중앙값을 반환한다.

    팔/다리처럼 얇은 부위는 바운딩 박스 중앙값을 쓰면 배경이 섞이기
    쉽다. 키포인트를 따라 좁은 패치만 샘플링해 실제 신체 표면 깊이에
    가깝게 만든다.
    """
    if depth_map.ndim != 2 or depth_map.size == 0:
        return None
    height, width = depth_map.shape
    radius = max(1, int(round(radius_px)))
    patches: list[np.ndarray] = []
    for x_px, y_px in points:
        if not np.isfinite(x_px) or not np.isfinite(y_px):
            continue
        if not (0.0 <= x_px < width and 0.0 <= y_px < height):
            continue
        center_x = int(round(x_px))
        center_y = int(round(y_px))
        x1 = max(0, center_x - radius)
        x2 = min(width, center_x + radius + 1)
        y1 = max(0, center_y - radius)
        y2 = min(height, center_y + radius + 1)
        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            continue
        values = valid_values(region, min_depth, max_depth)
        if values.size:
            patches.append(values)
    if not patches:
        return None
    values = np.concatenate(patches)
    if values.size < min_pixels:
        return None
    return float(np.median(values))
