FROM python:3.13-slim-bookworm

# uv 바이너리만 가져와서 사용
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# YOLO/OpenCV 포함 여부 (이미지가 커지므로 기본은 미포함)
ARG WITH_VISION=0

# 의존성 레이어 캐시: 프로젝트 설치 전에 의존성만 먼저 설치
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$WITH_VISION" = "1" ]; then \
        uv sync --frozen --no-install-project --no-dev --extra vision; \
    else \
        uv sync --frozen --no-install-project --no-dev; \
    fi

# 실제 소스
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$WITH_VISION" = "1" ]; then \
        uv sync --frozen --no-dev --extra vision; \
    else \
        uv sync --frozen --no-dev; \
    fi

EXPOSE 8082

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8082/health').status == 200 else 1)"

CMD ["uvicorn", "neurovector_vision.api:app", "--host", "0.0.0.0", "--port", "8082"]
