# H-JEPA Studio web platform (API + UI). CPU only: GPU work runs on RunPod workers.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 HJS_DATA_DIR=/data HJS_WORKER_DIR=/app/worker
WORKDIR /app
COPY pyproject.toml README.md ./
COPY hjepa_studio ./hjepa_studio
COPY worker ./worker
RUN pip install --no-cache-dir ".[postgres]"

VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "hjepa_studio.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
