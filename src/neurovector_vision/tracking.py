"""Short-lived box tracks for one continuous camera stream.

IDs are local to a tracker. IoU association cannot preserve identity reliably
through crossings or occlusion and does not establish person/part ownership.
Distance filtering is deliberately conservative: closer readings take effect
immediately, while a short median suppresses isolated farther readings.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from statistics import median

from neurovector_vision.models import BoundingBox, Detection


@dataclass
class _Track:
    label: str
    box: BoundingBox
    seen_at: float
    distances: list[float] = field(default_factory=list)
    filtered_distance: float | None = None


def _iou(first: BoundingBox, second: BoundingBox) -> float:
    intersection = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1)) * max(
        0.0, min(first.y2, second.y2) - max(first.y1, second.y1)
    )
    first_area = max(0.0, first.x2 - first.x1) * max(0.0, first.y2 - first.y1)
    second_area = max(0.0, second.x2 - second.x1) * max(0.0, second.y2 - second.y1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


class TemporalTracker:
    """Greedy, unique IoU matching within each exact detection label.

    Call from one camera worker only, with monotonic timestamps in seconds.
    Missing or invalid current distances are never replaced by old values.
    """

    def __init__(
        self, *, iou_threshold: float = 0.3, max_age_s: float = 1.0, window_size: int = 5
    ) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        if not math.isfinite(max_age_s) or max_age_s <= 0.0:
            raise ValueError("max_age_s must be finite and positive")
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self._iou_threshold = iou_threshold
        self._max_age_s = max_age_s
        self._window_size = window_size
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        """Discard associations and distance history without recycling track IDs."""
        self._tracks.clear()

    def update(self, detections: list[Detection], *, now: float | None = None) -> list[Detection]:
        now = time.monotonic() if now is None else now
        self._tracks = {
            key: track
            for key, track in self._tracks.items()
            if 0.0 <= now - track.seen_at <= self._max_age_s
        }
        candidates = sorted(
            (
                (_iou(track.box, detection.box), index, track_id)
                for index, detection in enumerate(detections)
                for track_id, track in self._tracks.items()
                if track.label == detection.label
            ),
            reverse=True,
        )
        assignments: dict[int, int] = {}
        used_tracks: set[int] = set()
        for overlap, index, track_id in candidates:
            if overlap < self._iou_threshold:
                break
            if index not in assignments and track_id not in used_tracks:
                assignments[index] = track_id
                used_tracks.add(track_id)

        output: list[Detection] = []
        for index, detection in enumerate(detections):
            track_id = assignments.get(index)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _Track(detection.label, detection.box, now)
            track = self._tracks[track_id]
            track.box = detection.box.model_copy()
            track.seen_at = now
            distance = detection.distance_m
            if distance is None or not math.isfinite(distance) or distance <= 0.0:
                track.distances.clear()
                track.filtered_distance = None
                distance = None
            else:
                if track.filtered_distance is not None and distance < track.filtered_distance:
                    # Keep a real approach visible, even if it is a large step.
                    track.distances.clear()
                track.distances.append(distance)
                track.distances = track.distances[-self._window_size :]
                distance = float(median(track.distances))
                track.filtered_distance = distance
            output.append(
                detection.model_copy(update={"track_id": track_id, "distance_m": distance})
            )
        return output
