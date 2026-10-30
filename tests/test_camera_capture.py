import threading
from queue import Queue
from unittest.mock import Mock

import numpy as np

from neurovector_vision.camera_capture import LatestFrameCapture


def test_capture_discards_old_frames_while_inference_is_busy() -> None:
    frames: Queue[tuple[bool, np.ndarray | None]] = Queue()
    first = np.zeros((8, 8, 3), dtype=np.uint8)
    newest = np.ones((8, 8, 3), dtype=np.uint8)
    frames.put((True, first))
    frames.put((True, newest))
    drained = threading.Event()
    reads = 0

    def read() -> tuple[bool, np.ndarray | None]:
        nonlocal reads
        reads += 1
        if reads == 3:
            drained.set()
        return frames.get(timeout=2.0)

    camera = Mock()
    camera.isOpened.return_value = True
    camera.read.side_effect = read
    capture = LatestFrameCapture(camera)
    try:
        assert drained.wait(timeout=2.0)
        ok, actual = capture.read()
        assert ok
        assert actual is newest
    finally:
        frames.put((False, None))
        capture.release()
    camera.release.assert_called_once()
