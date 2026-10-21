# NeuroVector Vision

드론 영상에서 장애물을 감지하고 백엔드로 전달하는 Python 서비스입니다.

- **Python 3.13 + uv**
- FastAPI + YOLO(ultralytics)
- 단안 카메라 기반 거리 추정 (`distance_m`)
- 감지 결과를 `backend` 의 `/api/v1/detections` 로 전송

## 요구 사항

- [uv](https://docs.astral.sh/uv/)

## 시작하기

```bash
uv sync                 # 기본 의존성 (FastAPI)
uv sync --extra vision  # YOLO/OpenCV 포함 (torch 다운로드로 시간이 걸릴 수 있음)
cp .env.example .env
uv run neurovector-vision   # http://localhost:8082
```

## NVIDIA GPU와 OpenCV

이 환경의 OpenCV는 NVIDIA OpenCL 장치를 사용할 수 있습니다. OpenCV에서 OpenCL을 쓰려면
지원되는 영상 연산에 `cv2.UMat`을 사용하고 OpenCL을 켭니다. NumPy 배열만 받는 함수는
`UMat.get()`으로 CPU 배열로 변환하세요.

```python
import cv2

cv2.ocl.setUseOpenCL(True)
device = cv2.ocl.Device.getDefault()
print("OpenCL:", cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL())
print("Device:", device.name() if device.available() else "none")

gpu_frame = cv2.UMat(frame)  # OpenCL 지원 OpenCV 연산에 GPU 메모리 사용
gpu_blurred = cv2.GaussianBlur(gpu_frame, (5, 5), 0)
blurred = gpu_blurred.get()  # NumPy 배열이 필요한 코드에 전달
```

RTX 3050 GPU 사용량은 앱 목록의 **RTX 3050 GPU Monitor**에서 확인할 수 있습니다.

API 문서: http://localhost:8082/docs

## 카메라 테스트 UI

로컬 카메라와 YOLO 감지 결과를 브라우저에서 바로 확인하는 개발용 도구입니다.
서버가 카메라를 직접 열어 MJPEG 로 전송하므로 브라우저 카메라 권한은 필요 없습니다.

```bash
uv sync --extra vision
uv run --extra vision python tools/camera_ui.py
# http://127.0.0.1:8090
```

- 감지 박스, 클래스, confidence, 추정 거리, 추론 시간/FPS 를 표시합니다.
- `--camera` 로 V4L2 인덱스를, `--model` 로 가중치 경로를 지정할 수 있습니다.
- 기본 모델 `yolov8n.pt` 는 첫 실행 시 자동 다운로드됩니다.
- 화면에서 **Distance Calibration** 에 실제 거리를 입력하면 초점 거리를 바로
  보정할 수 있습니다. 보정값은 실행 중에만 적용되며, 유지하려면 표시되는
  `NEUROVECTOR_CAMERA_FOCAL_LENGTH_PX` 를 `.env` 에 넣고 재시작하세요.
- 운영 API 가 아니며, 로컬 카메라/모델/거리 추정 확인용입니다.

```bash
# 다른 카메라 + 사용자 학습 모델
uv run --extra vision python tools/camera_ui.py \
  --camera 0 --model runs/detect/obstacles/weights/best.pt --port 8090

# 보정된 초점 거리로 시작
uv run --extra vision python tools/camera_ui.py --focal 812.5
```

## API

| 메서드 | 경로 | 설명 |
| --- | --- | --- |
| `GET` | `/health` | 헬스 체크 |
| `POST` | `/detect` | 이미지 업로드 → 감지 → 백엔드 전송 |

```bash
curl -F "file=@sample.jpg" http://localhost:8082/detect
```

응답의 각 감지 항목에는 선택적으로 `distance_m` 이 포함됩니다.

```json
{
  "drone_id": "drone-001",
  "detections": [
    {
      "label": "person",
      "confidence": 0.91,
      "box": {"x1": 100, "y1": 80, "x2": 180, "y2": 260},
      "distance_m": 12.4,
      "distance_note": null
    }
  ],
  "forwarded": 1,
  "elapsed_ms": 84.2
}
```

`ultralytics` 가 설치되지 않은 환경에서는 `/detect` 가 `503` 을 반환합니다.
크기 정보가 없는 클래스, 화면 경계에 잘린 객체, 너비/높이 형상이 크게
어긋나는 객체는 `distance_m` 이 `null` 이 되고 `distance_note` 에 사유가
들어갑니다. 잘못된 거리를 표시하는 것보다 `null` 을 반환하는 편이 안전합니다.

## 거리 추정

핀홀 카메라 근사식을 사용합니다.

```text
distance_m = real_dimension_m * focal_px / bbox_dimension_px
focal_px   = image_width_px / (2 * tan(horizontal_fov_deg / 2))
```

- 초점 거리는 `NEUROVECTOR_CAMERA_HORIZONTAL_FOV_DEG` 로 계산합니다.
- 실제 픽셀 초점 거리를 알고 있다면 `NEUROVECTOR_CAMERA_FOCAL_LENGTH_PX` 로
  덮어쓸 수 있습니다. 이 값이 있으면 FOV 는 사용하지 않습니다.
- 클래스별 실제 너비/높이는 기본 표(`distance.py`)에 있고,
  `NEUROVECTOR_OBJECT_SIZES` JSON 으로 덮어쓰거나 새 클래스를 추가할 수 있습니다.
- 너비와 높이가 모두 신뢰 가능하면 두 거리의 **기하평균(면적 기반)** 을 사용합니다.
  두 값이 `NEUROVECTOR_DISTANCE_MIN_AXIS_AGREEMENT` 보다 크게 어긋나면 객체가
  부분적으로 가려졌거나 형상이 다른 것으로 보고 거리를 버립니다.
- 바운딩 박스가 화면 경계에서 `NEUROVECTOR_DISTANCE_EDGE_MARGIN_RATIO` 안쪽에
  닿으면 잘린 것으로 보고 해당 축을 사용하지 않습니다. 양쪽 축이 모두 잘리면
  `distance_m` 은 `null` 입니다.
- 단안 방식은 객체의 전체 형상이 보이고 실제 크기를 알 때만 정확합니다. 사람의
  머리만 보이거나 차량 일부만 보이는 경우 거리를 표시하지 않습니다.

`NEUROVECTOR_OBJECT_SIZES` 예시:

```bash
export NEUROVECTOR_OBJECT_SIZES='{
  "fallen_tree": {"width_m": 3.0, "height_m": 1.0},
  "power_line": {"width_m": 2.0, "height_m": 0.05}
}'
```

키는 YOLO 클래스 이름과 같아야 하며 대소문자는 무시됩니다. 기본 표에 있는
클래스도 같은 방식으로 크기를 덮어쓸 수 있습니다.

### 카메라 캘리브레이션

카메라 UI의 **Distance Calibration** 을 사용하는 방법이 가장 빠릅니다.

1. 실제 크기를 아는 객체(예: 1.7 m 사람)의 **전체 모습**을 화면에 담습니다.
2. 감지된 객체의 실제 거리를 줄자로 잽니다.
3. UI의 Distance Calibration 에 실제 거리를 입력하고 **보정 적용**을 누릅니다.
4. 표시되는 `NEUROVECTOR_CAMERA_FOCAL_LENGTH_PX` 를 `.env` 에 넣고 재시작합니다.
5. 다른 거리에서 다시 측정해 오차를 확인합니다.

수동 계산은 다음과 같습니다.

```text
focal_px = actual_distance_m * bbox_height_px / real_height_m
```

- 해상도가 바뀌면 초점 거리도 비례해서 바뀝니다.
- 기준 객체는 화면 경계에 잘리지 않아야 하며, 가능한 한 가까이 두고 크게
  보이는 상태에서 보정하는 것이 픽셀 오차에 유리합니다.
- 보정은 기본 크기 표의 실제 크기를 기준으로 합니다. 예를 들어 기준 사람의 키가
  1.8 m 라면 먼저 `NEUROVECTOR_OBJECT_SIZES='{"person":{"width_m":0.5,"height_m":1.8}}'`
  처럼 실제 크기를 맞춘 뒤 보정하세요.
- 보정 후에도 객체가 부분적으로 가려지거나 기울어지면 거리 오차가 커집니다.

## 새 장애물 클래스 추가 (YOLO 파인튜닝)

감지기 코드는 수정하지 않습니다. Ultralytics 로 클래스를 학습한 뒤 모델과
클래스 크기만 교체하면 됩니다.

1. YOLO 형식 데이터셋을 준비합니다.

```yaml
# datasets/obstacles/obstacles.yaml
path: datasets/obstacles
train: images/train
val: images/val
names:
  0: fallen_tree
  1: power_line
  2: construction_barrel
```

2. 모델을 파인튜닝합니다.

```bash
uv run --extra vision yolo detect train \
  model=yolov8n.pt \
  data=datasets/obstacles/obstacles.yaml \
  epochs=100 \
  imgsz=640 \
  project=runs/detect \
  name=obstacles
```

3. 학습된 가중치와 클래스 크기를 설정합니다.

```bash
export NEUROVECTOR_MODEL_PATH=runs/detect/obstacles/weights/best.pt
export NEUROVECTOR_OBJECT_SIZES='{
  "fallen_tree": {"width_m": 3.0, "height_m": 1.0},
  "power_line": {"width_m": 2.0, "height_m": 0.05}
}'
```

4. `/detect` 를 호출해 새 클래스의 `label`, `distance_m` 을 확인합니다.

학습 데이터가 적다면 `yolov8s.pt`/`yolo11n.pt` 같은 더 큰 모델을 쓰거나,
`epochs`, augmentation, `imgsz` 를 조정하세요. 파인튜닝 후에는 기존 COCO
클래스가 사라질 수 있으므로, 필요한 클래스는 데이터셋에 함께 포함하는 것이 좋습니다.

## 구조

```
src/neurovector_vision/
├── api.py        # FastAPI 엔드포인트
├── backend.py    # 백엔드 API 클라이언트
├── config.py     # 환경변수 설정
├── detector.py   # YOLO 감지기
├── distance.py   # 단안 거리 추정 / 클래스별 크기
├── models.py     # Pydantic 모델
└── __main__.py   # uvicorn 진입점
tests/
tools/
├── camera_ui.py   # 로컬 카메라 테스트 UI 서버
└── camera_ui.html # 테스트 UI 페이지
```

## 개발

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

## 환경 변수

`NEUROVECTOR_` 접두사를 사용합니다. `.env.example` 참고.

| 변수 | 기본값 |
| --- | --- |
| `NEUROVECTOR_PORT` | `8082` |
| `NEUROVECTOR_BACKEND_BASE_URL` | `http://localhost:8081` |
| `NEUROVECTOR_MODEL_PATH` | `yolov8n.pt` |
| `NEUROVECTOR_CONFIDENCE_THRESHOLD` | `0.35` |
| `NEUROVECTOR_IOU_THRESHOLD` | `0.45` |
| `NEUROVECTOR_CAMERA_HORIZONTAL_FOV_DEG` | `70.0` |
| `NEUROVECTOR_CAMERA_FOCAL_LENGTH_PX` | (없음, FOV 사용) |
| `NEUROVECTOR_DISTANCE_MIN_AXIS_AGREEMENT` | `0.6` |
| `NEUROVECTOR_DISTANCE_EDGE_MARGIN_RATIO` | `0.02` |
| `NEUROVECTOR_OBJECT_SIZES` | `{}` (기본 크기 표 사용) |
| `NEUROVECTOR_DRONE_ID` | `drone-001` |
