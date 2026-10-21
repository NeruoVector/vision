"""체커보드로 카메라 내부 파라미터를 보정한다.

사용 예:
    cd vision
    .venv/bin/python tools/calibrate_camera.py --pattern 9x6 --square-size 0.025

체커보드를 카메라 앞에서 여러 방향/거리로 보여주면 자동으로 샘플을 모으고,
충분히 모이면 ``camera_calibration.json`` 을 저장한다. 저장된 파일은
``tools/camera_ui.py`` 가 자동으로 읽어 렌즈 왜곡 보정과 초점 거리(fx)에
사용한다.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from neurovector_vision.camera_calibration import CameraCalibration


def parse_pattern(value: str) -> tuple[int, int]:
    try:
        cols, rows = (int(part) for part in value.lower().split("x", maxsplit=1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pattern must look like 9x6") from exc
    if cols < 3 or rows < 3:
        raise argparse.ArgumentTypeError("pattern must be at least 3x3")
    return cols, rows


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(index)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, 30)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def find_corners(gray: np.ndarray, pattern: tuple[int, int]):
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        flags |= getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0)
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
        if found:
            return True, corners.astype(np.float32)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
    if found:
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
    return found, corners


def calibrate(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> CameraCalibration:
    rms, matrix, distortion, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    return CameraCalibration(
        camera_matrix=matrix,
        distortion_coefficients=distortion,
        image_size=image_size,
        reprojection_error=float(rms),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroVector checkerboard camera calibration")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--pattern", type=parse_pattern, default=(9, 6))
    parser.add_argument("--square-size", type=float, default=0.025, help="chess square size (m)")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output", default="camera_calibration.json")
    parser.add_argument("--headless", action="store_true", help="no preview window; auto-capture")
    args = parser.parse_args()

    cols, rows = args.pattern
    pattern = (cols, rows)
    capture = open_camera(args.camera, args.width, args.height)
    if not capture.isOpened():
        raise SystemExit("카메라를 열 수 없습니다. --camera 인덱스를 확인하세요.")

    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image_size = (actual_width, actual_height)
    print(
        f"camera {actual_width}x{actual_height}, pattern {cols}x{rows}, square {args.square_size}m"
    )
    print("체커보드를 여러 각도/거리로 보여주세요. q=종료, c=지금 보정")

    grid = np.zeros((cols * rows, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * args.square_size
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    last_corners: np.ndarray | None = None
    last_capture = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = find_corners(gray, pattern)
            now = time.monotonic()
            if found and now - last_capture > 0.6:
                different = True
                if last_corners is not None:
                    different = float(np.mean(np.abs(corners - last_corners))) > 8.0
                if different:
                    object_points.append(grid.copy())
                    image_points.append(corners)
                    last_corners = corners
                    last_capture = now
                    print(f"sample {len(object_points)}/{args.samples}")

            preview = frame.copy()
            if found:
                cv2.drawChessboardCorners(preview, pattern, corners, found)
            cv2.putText(
                preview,
                f"samples {len(object_points)}/{args.samples}  q=quit c=calibrate",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if found else (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
            if not args.headless:
                cv2.imshow("NeuroVector Camera Calibration", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c") and len(object_points) >= 6:
                    break

            if len(object_points) >= args.samples:
                break

        if len(object_points) < 6:
            raise SystemExit(f"샘플이 부족합니다 ({len(object_points)}개, 최소 6개).")

        calibration = calibrate(object_points, image_points, image_size)
        calibration.save(args.output)
        print(f"saved {Path(args.output).resolve()}")
        print(
            "reprojection error: "
            f"{calibration.reprojection_error:.4f} px, fx={calibration.focal_px:.2f} px"
        )
        print("이제 tools/camera_ui.py 를 다시 시작하면 자동으로 적용됩니다.")
    finally:
        capture.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
