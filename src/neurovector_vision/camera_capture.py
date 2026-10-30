from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class LatestFrameCapture:
    """Own the camera on one worker; never queue stale frames behind inference."""

    def __init__(self, capture: cv2.VideoCapture) -> None:
        self._capture = capture
        self._condition = threading.Condition()
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._consumed = 0
        self._running = True
        self._thread = threading.Thread(target=self._read, name="camera-capture", daemon=True)
        self._thread.start()

    def _read(self) -> None:
        try:
            while self._running and self._capture.isOpened():
                ok, frame = self._capture.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                with self._condition:
                    self._frame = frame
                    self._sequence += 1
                    self._condition.notify_all()
        finally:
            self._capture.release()
            with self._condition:
                self._running = False
                self._condition.notify_all()

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence != self._consumed or not self._running,
                timeout=1.0,
            )
            if self._sequence == self._consumed:
                return False, None
            self._consumed = self._sequence
            return True, self._frame

    def get(self, prop: int) -> float:
        return self._capture.get(prop)

    def isOpened(self) -> bool:
        return self._capture.isOpened()

    def release(self) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()
        self._thread.join(timeout=2.0)
