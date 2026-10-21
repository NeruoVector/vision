# NeuroVector Vision

드론 영상에서 장애물을 감지하고 백엔드로 전달하는 Python 서비스입니다.

- **Python 3.13 + uv**
- FastAPI + **YOLO26** (ultralytics)
- NVIDIA GPU(CUDA) 추론, 실패 시 CPU 자동 폴백
- 단안 깊이(YOLO26-depth) + 얼굴 키포인트 기반 거리 추정 (`distance_m`)
- 체커보드 카메라 캘리브레이션(렌즈 왜곡/초점 거리)
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

## NVIDIA GPU (CUDA)

`uv sync --extra vision` 은 PyPI 기본 torch(CPU)를 설치할 수 있습니다. NVIDIA GPU를
쓰려면 CUDA 12.8 wheel 로 교체하세요. (RTX 3050 4GB에서 검증)

```bash
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.9.1+cu128' 'torchvision==0.24.1+cu128'

.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`NEUROVECTOR_DEVICE=auto` 이면 CUDA 사용 가능 시 `0`(GPU), 아니면 `cpu`를 선택합니다.
GPU 메모리가 부족하면 `--no-depth` 또는 `NEUROVECTOR_DEPTH_ENABLED=false` 로 깊이
모델을 끄고 얼굴 키포인트 거리만 사용할 수 있습니다.

API 문서: http://localhost:8082/docs

## 카메라 테스트 UI (얼굴 거리)

로컬 카메라에서 **얼굴만** 감지하고, YOLO26-pose 키포인트와 YOLO26-depth 깊이 맵을
융합해 거리를 표시하는 개발용 도구입니다. 서버가 카메라를 직접 열어 MJPEG 로
전송하므로 브라우저 카메라 권한은 필요 없습니다.

```bash
uv sync --extra vision
uv run --extra vision python tools/camera_ui.py
# http://127.0.0.1:8090
```

- 얼굴 박스, confidence, **깊이 기반 거리**, 눈 간격 기반 보조 거리, 추론 시간/FPS.
- `--model yolo26n-pose.pt` (얼굴 키포인트), `--depth-model yolo26n-depth.pt`.
- `--device auto|0|cpu`, `--no-depth`, `--focal`, `--face-scale` 지원.
- YOLO26 가중치는 첫 실행 시 자동 다운로드됩니다. (`yolo26n-pose.pt`, `yolo26n-depth.pt`)
- GPU 추론은 약 50~70ms/프레임 (RTX 3050, pose+depth, warmup 제외).

### 얼굴 거리 보정

1. 카메라 앞에 얼굴을 보여주고 UI에 얼굴이 잡히게 합니다.
2. 얼굴까지 실제 거리를 줄자로 잽니다.
3. **Face Distance Calibration** 에 실제 거리를 입력하고 **보정 적용**을 누릅니다.
4. UI에 표시되는 두 값을 `.env` 에 넣고 재시작합니다.

```env
NEUROVECTOR_FACE_IPD_SCALE_PX_M=...
NEUROVECTOR_DEPTH_SCALE=...
```

- `FACE_IPD_SCALE` 은 `실제 거리 × 눈 간격(px)` 로 계산됩니다.
- `DEPTH_SCALE` 은 `실제 거리 / 깊이 모델 출력(m)` 로 계산됩니다.
- 깊이 모델이 없거나 실패하면 눈 간격 기반 거리로 자동 폴백합니다.

### 체커보드 카메라 캘리브레이션

렌즈 왜곡과 실제 초점 거리를 구하려면 체커보드를 사용합니다.

```bash
# 9x6 내부 코너, 정사각형 25mm 체커보드
.venv/bin/python tools/calibrate_camera.py --pattern 9x6 --square-size 0.025

# 샘플 30장, 다른 카메라
.venv/bin/python tools/calibrate_camera.py --camera 0 --samples 30
```

체커보드를 여러 각도/거리로 보여주면 자동으로 샘플을 모으고
`camera_calibration.json` 을 저장합니다. `tools/camera_ui.py` 는 이 파일이 있으면
자동으로 렌즈 왜곡을 제거하고 `fx` 를 거리 계산에 사용합니다.

```bash
# 파일 경로 직접 지정
.venv/bin/python tools/camera_ui.py --calibration camera_calibration.json
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
- 높이가 0.5 m 이상인 객체는 **높이만** 주 축으로 사용합니다. 사람이 걷거나 팔을
  벌려 너비가 변해도 거리가 흔들리지 않게 하기 위해서입니다. 낮은 객체는 너비를 씁니다.
- 주 축이 화면 경계에서 잘리면 보조 축으로 근사하고
  `distance_note=approximate_bbox_truncated` 를 붙입니다.
- 양쪽 축이 모두 온전할 때 종횡비가 클래스 기대값에서 5배 이상 벗어나면
  `inconsistent_axes` 로 거리를 표시하지 않습니다. (예: 사람 얼굴만 보이는 경우)
- 바운딩 박스가 화면 경계에서 `NEUROVECTOR_DISTANCE_EDGE_MARGIN_RATIO` 안쪽에
  닿으면 잘린 것으로 봅니다.
- 얼굴 근거리 거리는 전체 사람 크기 대신 눈/귀 키포인트와 YOLO26-depth를 사용합니다.
  `distance_note=depth_keypoint_mismatch` 는 두 추정이 3배 이상 다르다는 뜻이며,
  이때는 깊이 모델 값을 우선합니다.

`NEUROVECTOR_OBJECT_SIZES` 예시:

```bash
export NEUROVECTOR_OBJECT_SIZES='{
  "fallen_tree": {"width_m": 3.0, "height_m": 1.0},
  "power_line": {"width_m": 2.0, "height_m": 0.05}
}'
```

키는 YOLO 클래스 이름과 같아야 하며 대소문자는 무시됩니다. 기본 표에 있는
클래스도 같은 방식으로 크기를 덮어쓸 수 있습니다.

### 거리 보정 요약

| 대상 | 방법 | 저장 값 |
| --- | --- | --- |
| 카메라 렌즈/초점 | 체커보드 `tools/calibrate_camera.py` | `camera_calibration.json` |
| 얼굴 근거리 | UI에서 실제 얼굴 거리 입력 | `NEUROVECTOR_FACE_IPD_SCALE_PX_M` |
| 깊이 모델 | UI에서 실제 얼굴 거리 입력 | `NEUROVECTOR_DEPTH_SCALE` |
| 일반 객체 | `NEUROVECTOR_CAMERA_FOCAL_LENGTH_PX` 또는 FOV | `.env` |

- 해상도가 바뀌면 초점 거리와 얼굴 스케일도 비례해서 바뀝니다. 같은 해상도에서
  보정하세요.
- 기준 얼굴은 화면 경계에 잘리지 않고, 정면을 바라보는 상태가 좋습니다.
- 체커보드 캘리브레이션은 렌즈 왜곡을 제거하므로 화면 가장자리 얼굴의 키포인트
  거리 오차도 줄여줍니다.

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
  model=yolo26n.pt \
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

학습 데이터가 적다면 `yolo26s.pt` 같은 더 큰 모델을 쓰거나,
`epochs`, augmentation, `imgsz` 를 조정하세요. 파인튜닝 후에는 기존 COCO
클래스가 사라질 수 있으므로, 필요한 클래스는 데이터셋에 함께 포함하는 것이 좋습니다.

## 구조

```
src/neurovector_vision/
├── api.py        # FastAPI 엔드포인트
├── backend.py    # 백엔드 API 클라이언트
├── config.py     # 환경변수 설정
├── detector.py     # YOLO26 객체 감지기
├── face_detector.py# YOLO26-pose 얼굴 키포인트 + depth 융합
├── depth.py        # YOLO26-depth 단안 깊이 추정
├── camera_calibration.py # 체커보드 캘리브레이션 로드/왜곡 보정
├── distance.py     # 단안 거리 추정 / 얼굴 거리 / 클래스별 크기
├── models.py       # Pydantic 모델
└── __main__.py     # uvicorn 진입점
tests/
tools/
├── camera_ui.py         # 얼굴 거리 테스트 UI 서버
├── camera_ui.html       # 테스트 UI 페이지
└── calibrate_camera.py  # 체커보드 카메라 캘리브레이션
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
| `NEUROVECTOR_MODEL_PATH` | `yolo26n.pt` |
| `NEUROVECTOR_FACE_MODEL_PATH` | `yolo26n-pose.pt` |
| `NEUROVECTOR_DEPTH_MODEL_PATH` | `yolo26n-depth.pt` |
| `NEUROVECTOR_DEVICE` | `auto` (CUDA 가능 시 GPU) |
| `NEUROVECTOR_CONFIDENCE_THRESHOLD` | `0.35` |
| `NEUROVECTOR_IOU_THRESHOLD` | `0.45` |
| `NEUROVECTOR_DEPTH_ENABLED` | `true` |
| `NEUROVECTOR_DEPTH_IMGSZ` | `768` |
| `NEUROVECTOR_DEPTH_SCALE` | `1.0` |
| `NEUROVECTOR_FACE_IPD_M` | `0.063` |
| `NEUROVECTOR_FACE_KEYPOINT_CONFIDENCE` | `0.5` |
| `NEUROVECTOR_FACE_IPD_SCALE_PX_M` | (없음, focal × IPD) |
| `NEUROVECTOR_CAMERA_CALIBRATION_PATH` | `camera_calibration.json` |
| `NEUROVECTOR_CAMERA_HORIZONTAL_FOV_DEG` | `70.0` |
| `NEUROVECTOR_CAMERA_FOCAL_LENGTH_PX` | (없음, FOV/캘리브레이션 사용) |
| `NEUROVECTOR_DISTANCE_MIN_AXIS_AGREEMENT` | `0.6` |
| `NEUROVECTOR_DISTANCE_EDGE_MARGIN_RATIO` | `0.02` |
| `NEUROVECTOR_OBJECT_SIZES` | `{}` (기본 크기 표 사용) |
| `NEUROVECTOR_DRONE_ID` | `drone-001` |
