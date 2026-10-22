# NeuroVector Vision

드론 영상에서 장애물을 감지하고 백엔드로 전달하는 Python 서비스입니다.

- **Python 3.13 + uv**
- FastAPI + **YOLO26** (ultralytics)
- NVIDIA GPU(CUDA) 추론, 실패 시 CPU 자동 폴백
- 단안 깊이(YOLO26-depth) + 얼굴/신체 키포인트 기반 거리 추정 (`distance_m`)
- 얼굴·몸통·양팔·양손·양다리·양발 감지 (YOLO26-pose 17 키포인트)
- 화면 가장자리 사선거리 보정 (`R = Z × √(1 + x² + y²)`)
- 감지 우선순위 1~3단계 (`priority_level`)
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
깊이 추론 중 CUDA 오류가 발생하면 CPU로 전환하고 30초 후 GPU를 다시 시도합니다.
객체/pose 추론 실패는 API에서 503으로 보고됩니다. GPU 메모리가 부족하면 `--no-depth` 또는 `NEUROVECTOR_DEPTH_ENABLED=false` 로 깊이
모델을 끄고 얼굴 키포인트 거리만 사용할 수 있습니다.

API 문서: http://localhost:8082/docs

## 카메라 테스트 UI (얼굴/신체 부위 거리)

로컬 카메라에서 **일반 객체·얼굴·신체 부위**를 감지하고, YOLO26 객체 검출·pose 키포인트와
YOLO26-depth 깊이 맵을 융합해 부위별 거리를 표시하는 개발용 도구입니다.
서버가 카메라를 직접 열어 MJPEG 로 전송하므로 브라우저 카메라 권한은 필요 없습니다.

```bash
uv sync --extra vision
uv run --extra vision python tools/camera_ui.py
# http://127.0.0.1:8090

# 얼굴만 표시하고 싶을 때
uv run --extra vision python tools/camera_ui.py --face-only
```

- 일반 객체와 얼굴, 몸통, 왼/오른팔, 왼/오른손, 왼/오른다리, 왼/오른발 박스와 거리.
- 얼굴은 눈 간격 기반 보조 거리도 함께 표시하고, 나머지 부위는 깊이 모델 거리를 사용합니다.
- `--object-model yolo26n.pt` (일반 객체), `--model yolo26n-pose.pt` (사람 키포인트),
  `--depth-model yolo26n-depth.pt` (거리). 세 모델을 함께 사용합니다.
- `--device auto|0|cpu`, `--no-depth`, `--face-only`, `--focal`, `--face-scale` 지원.
- YOLO26 가중치는 첫 실행 시 자동 다운로드됩니다. (`yolo26n.pt`, `yolo26n-pose.pt`, `yolo26n-depth.pt`)
- 실제 RTX 3050에서 세 모델의 예열 후 샘플 통합 추론은 평균 약 39~45ms였습니다.
  [측정 조건과 결과](docs/runtime-validation-yolo26.md)를 참고하세요.
- 손/팔은 YOLO26-pose가 사람을 감지했을 때 키포인트로 계산합니다. 손만 화면을
  가득 채우는 초근접 장면은 전용 손 키포인트 모델이 필요합니다.

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
  `distance_note=depth_keypoint_mismatch` 는 두 추정이 3배 넘게 다르다는 뜻이며,
  이때는 거리를 미확정(`distance_m=null`)으로 반환합니다.

`NEUROVECTOR_OBJECT_SIZES` 예시:

```bash
export NEUROVECTOR_OBJECT_SIZES='{
  "fallen_tree": {"width_m": 3.0, "height_m": 1.0},
  "power_line": {"width_m": 2.0, "height_m": 0.05}
}'
```

키는 YOLO 클래스 이름과 같아야 하며 대소문자는 무시됩니다. 기본 표에 있는
클래스도 같은 방식으로 크기를 덮어쓸 수 있습니다.

### 감지 우선순위

| 단계 | 대상 | 예시 |
| --- | --- | --- |
| 1단계 · 위험 | 얼굴, 신체 부위, 동물 | `face`, `body`, `left_hand`, `right_arm`, `person`, `dog`, `고양이` |
| 2단계 · 주의 | 차량, 일반 사물 | `car`, `truck`, `bicycle`, 알 수 없는 클래스 |
| 3단계 · 낮음 | 손상되어도 큰 지장이 없는 자연물 | `leaf`, `fallen leaf`, `나뭇잎` |

나뭇잎 데이터셋은 아직 없지만 라벨을 미리 등록해 두었습니다. 데이터셋/모델이
추가되면 `leaf` 클래스가 자동으로 3단계로 분류됩니다. 클래스 이름이 다르면
`NEUROVECTOR_PRIORITY_LEVELS` JSON으로 덮어쓸 수 있습니다.

```bash
export NEUROVECTOR_PRIORITY_LEVELS='{"leaf_damage": 3, "my-obstacle": 1}'
```

### 거리 보정 요약

| 대상 | 방법 | 저장 값 |
| --- | --- | --- |
| 카메라 렌즈/초점 | 체커보드 `tools/calibrate_camera.py` | `camera_calibration.json` |
| 얼굴 근거리 | UI에서 실제 얼굴 거리 입력 | `NEUROVECTOR_FACE_IPD_SCALE_PX_M` |
| 깊이 모델 | UI에서 실제 얼굴 거리 입력 | `NEUROVECTOR_DEPTH_SCALE` |
| 일반 객체 | `NEUROVECTOR_CAMERA_FOCAL_LENGTH_PX` 또는 FOV | `.env` |
| 사선거리 | 체커보드 캘리브레이션(주점/초점) | `camera_calibration.json` |

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
├── face_detector.py# YOLO26-pose 얼굴/신체 부위 키포인트 + depth 융합
├── body_parts.py   # 키포인트 → 몸통/팔/손/다리/발 박스
├── priority.py     # 1~3단계 위험 우선순위 매핑
├── depth.py        # YOLO26-depth 단안 깊이 추정
├── camera_calibration.py # 체커보드 캘리브레이션 로드/왜곡 보정
├── distance.py     # 단안 거리 추정 / 얼굴 거리 / 클래스별 크기
├── models.py       # Pydantic 모델
└── __main__.py     # uvicorn 진입점
tests/
tools/
├── camera_ui.py         # 얼굴/신체 부위 거리 테스트 UI 서버
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
| `NEUROVECTOR_BODY_PARTS_ENABLED` | `true` |
| `NEUROVECTOR_FACE_IPD_SCALE_PX_M` | (없음, focal × IPD) |
| `NEUROVECTOR_CAMERA_CALIBRATION_PATH` | `camera_calibration.json` |
| `NEUROVECTOR_CAMERA_HORIZONTAL_FOV_DEG` | `70.0` |
| `NEUROVECTOR_CAMERA_FOCAL_LENGTH_PX` | (없음, FOV/캘리브레이션 사용) |
| `NEUROVECTOR_DISTANCE_MIN_AXIS_AGREEMENT` | `0.6` |
| `NEUROVECTOR_DISTANCE_EDGE_MARGIN_RATIO` | `0.02` |
| `NEUROVECTOR_OBJECT_SIZES` | `{}` (기본 크기 표 사용) |
| `NEUROVECTOR_PRIORITY_LEVELS` | `{}` (기본 1~3단계 매핑 사용) |
| `NEUROVECTOR_DRONE_ID` | `drone-001` |

## 추론 파이프라인과 상태

`/detect`는 객체 감지와 얼굴·신체 부위 추정을 함께 실행합니다.
`NEUROVECTOR_FACE_ENABLED=false`이면 객체 감지만 실행합니다.
캘리브레이션 파일이 있으면 입력 해상도에 맞게 내부 파라미터를 조정하고
렌즈 왜곡을 제거한 영상에서 `yolo26n-depth.pt` 깊이 맵을 **프레임당 한 번** 계산해
일반 객체와 얼굴·신체 감지기가 공유합니다. 사람이나 얼굴이 없는 장면도 깊이를
계산하며, `NEUROVECTOR_FACE_ENABLED=false`여도 객체의 깊이 거리는 유지됩니다.
일반 객체는 박스 중앙 영역의 깊이 중앙값에 보정 스케일과 사선 보정을 적용합니다.
클래스별 크기가 없는 객체나 화면에 잘린 객체도 유효 깊이가 있으면 거리를 반환합니다.
깊이가 없거나 유효 픽셀이 부족하면 기존 박스 크기 기반 거리로 폴백하고 사유를 표시합니다.
얼굴은 깊이/키포인트를 비교하며 신체는 부위별 키포인트 주변 깊이를 사용합니다. person과 그 신체 부위는
각각 반환되므로 결과 개수가 실제 사람 수와 같지는 않습니다.

모델은 서버 프로세스마다 유지하며 기본적으로 시작 시 예열합니다.
선택 의존성이 없거나 예열이 실패해도 서버는 시작하고 `/health`에
`status=degraded`, `inference_error`를 반환합니다. 깊이 장애는 `depth_error`로,
사용 중인 객체·pose·깊이 가중치는 `models`로 확인할 수 있습니다. 다음 감지 요청에서 다시 시도합니다.
실제 추론은 전용 작업자 1개에서 직렬 실행합니다. 실행 중 요청을 포함해 기본
2개까지 접수하며, 슬롯을 2초 내 확보하지 못하면 503을 반환합니다.
웹 작업자 프로세스를 늘리면 각 프로세스가 별도의 모델과 GPU 메모리를 사용합니다.

깊이 추정 실패는 기본 5초 후 재시도합니다. 얼굴의 두 거리 추정이 3배 넘게
충돌하면 검출은 유지하고 거리는 비웁니다. 응답에 다음 필드가 추가됩니다.

- `distance_source`: `bbox`, `depth`, `keypoint` 또는 `null`
- `distance_confidence`: 미보정 상태에서는 `null`, 거리 충돌이 확인되면 `0.0`
- `region_is_approximate`: 얼굴·신체 키포인트로 근사한 영역이면 `true`
- `track_id`: 연속 카메라 UI에서만 사용하는 추적 ID. 개별 API 업로드는 `null`

카메라 UI는 같은 라벨의 박스를 IoU로 연결하고 최근 거리 중앙값을 사용합니다.
가까워진 거리는 즉시 반영하며 미확정 거리를 과거 값으로 대체하지 않습니다.
보정·해상도 변경 시 이력을 초기화합니다. `--no-tracking`으로 끌 수 있습니다.
교차·가림 상황에서는 ID가 바뀔 수 있으며 신체 부위와 사람의 소속 관계,
속도·충돌 예상 시간은 아직 추정하지 않습니다.

## 실제 영상 평가

검출 정밀도·재현율, 일치한 객체의 거리 MAE·거리 반환율, 추론 p95 지연은
라벨링한 영상으로 측정할 수 있습니다. 데이터셋은 별도로 준비해야 합니다.

```bash
uv run --extra vision python tools/evaluate.py dataset/manifest.json --warmup 2
```

manifest 예시(이미지 경로는 manifest 파일 기준):

```json
{"images": [
  {"image": "frames/001.jpg", "annotations": [
    {"label": "person", "box": [10, 20, 100, 180], "distance_m": 2.4}
  ]},
  {"image": "frames/empty.jpg", "annotations": []}
]}
```

빈 장면도 포함하고 출력 대상인 얼굴·신체 부위도 해당 라벨로 빠짐없이 표시하세요.
지연은 디코딩과 추론을 포함하며 파일 읽기·백엔드 전송은 제외합니다.


## 기존 구현에서 보완한 내용

기존 `3e94c3a`의 얼굴 IPD·깊이 보정, `72cbeb5`의 신체 부위·키포인트 패치 깊이와
직선거리 보정, `0a4e0c1`의 손·발 영역 확장을 확인하고 그 동작을 유지했습니다.
깊이 모델의 실행·오류 복구 상태를 공유해 중복 추론을 없애고 일반 객체에도 깊이를
적용했습니다. 이미지 밖 박스·키포인트의 잘못된 깊이 샘플링을 차단하고,
입력 영상과 깊이 맵 크기 일치를 확인합니다. 카메라 미러 영상은 주점도 반전합니다.

단안 모델이 출력하는 미터 값은 추정치입니다. 실제 카메라의 거리 보정과
실측 거리 라벨 평가가 필요하며, 현재 검증은 추론 실행과 데이터 흐름을 확인한 것입니다.
