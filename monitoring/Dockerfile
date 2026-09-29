# Freqtrade Prometheus exporter (Task 9). Polls the bot REST API and
# exposes metrics on :9108/metrics. Build with the repo root as context:
#   docker build -f monitoring/Dockerfile .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FT_API_URL=http://freqtrade:8080 \
    EXPORTER_PORT=9108 \
    POLL_INTERVAL_S=15

WORKDIR /srv

COPY pyproject.toml README.md ./
COPY common/ ./common/
COPY mcp_server/ ./mcp_server/
COPY monitoring/ ./monitoring/

RUN pip install --no-cache-dir ".[gateway]"

EXPOSE 9108

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9108/metrics')"

CMD ["python", "-m", "monitoring.freqtrade_exporter"]
