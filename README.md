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

API 문서: http://localhost:8082/docs

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
      "distance_m": 12.4
    }
  ],
  "forwarded": 1,
  "elapsed_ms": 84.2
}
```

`ultralytics` 가 설치되지 않은 환경에서는 `/detect` 가 `503` 을 반환합니다.
크기 정보가 없는 클래스는 `distance_m` 이 `null` 이 됩니다.

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
- 너비/높이 추정값 중 **더 가까운 값**을 사용합니다. 장애물 회피에서는 거리
  과대평가보다 과소평가가 안전하기 때문입니다.
- 바운딩 박스가 잘리거나 객체가 기울어져 있으면 오차가 커집니다. 안전 임계값은
  여유 있게 잡으세요.

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

1. 실제 크기를 아는 객체(예: 1.7 m 사람)를 알려진 거리(예: 20 m)에 세웁니다.
2. 해당 객체의 바운딩 박스 높이를 픽셀로 측정합니다.
3. `focal_px = distance_m * bbox_height_px / real_height_m` 로 초점 거리를 계산합니다.
4. `NEUROVECTOR_CAMERA_FOCAL_LENGTH_PX` 에 넣고 다른 거리에서 검증합니다.
5. 해상도가 바뀌면 초점 거리도 비례해서 바뀌므로, FOV 방식이 더 편할 수 있습니다.

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
| `NEUROVECTOR_OBJECT_SIZES` | `{}` (기본 크기 표 사용) |
| `NEUROVECTOR_DRONE_ID` | `drone-001` |
