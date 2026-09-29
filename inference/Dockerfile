# CPU inference gateway image: no CUDA libraries, no torch.
# GPU support lives in Dockerfile.gpu. Build with the repo root as context:
#   docker build -f inference/Dockerfile .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/models

WORKDIR /srv

COPY pyproject.toml README.md ./
COPY common/ ./common/
COPY inference/app/ ./inference/app/

RUN pip install --no-cache-dir ".[gateway]"

VOLUME ["/models"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

CMD ["uvicorn", "inference.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
