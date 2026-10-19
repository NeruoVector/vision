# NeuroVector Vision

드론 영상에서 장애물을 감지하고 백엔드로 전달하는 Python 서비스입니다.

- **Python 3.13 + uv**
- FastAPI + YOLO(ultralytics)
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

`ultralytics` 가 설치되지 않은 환경에서는 `/detect` 가 `503` 을 반환합니다.

## 구조

```
src/neurovector_vision/
├── api.py        # FastAPI 엔드포인트
├── backend.py    # 백엔드 API 클라이언트
├── config.py     # 환경변수 설정
├── detector.py   # YOLO 감지기
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
| `NEUROVECTOR_DRONE_ID` | `drone-001` |
