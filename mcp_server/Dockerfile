# Bot monitoring/control MCP server (Task 7). stdio transport: no ports.
# Build with the repo root as context:
#   docker build -f mcp_server/Dockerfile .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FT_API_URL=http://freqtrade:8080

WORKDIR /srv

COPY pyproject.toml README.md ./
COPY common/ ./common/
COPY mcp_server/ ./mcp_server/

RUN pip install --no-cache-dir ".[mcp]"

# MCP_ALLOW_WRITE=yes to enable pause_trading / force_exit. Default: read-only.
CMD ["python", "-m", "mcp_server.server"]
