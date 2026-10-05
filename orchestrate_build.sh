#!/usr/bin/env bash
set -euo pipefail

echo "===================================================="
echo " Starting Headless Multi-Agent Build for Freqtrade App"
echo "===================================================="

# 1. Create necessary directory structure
mkdir -p .opencode/agent common config ftctl freqtrade inference mcp_server deploy/compose deploy/helm tests

# 2. Automatically generate OpenCode Subagent Definitions
echo "--- Writing OpenCode Agent Configurations ---"

cat << 'EOF' > .opencode/agent/infra.md
---
description: Docker, Compose, Helm, Kubernetes, and Monorepo Infrastructure.
mode: subagent
permission:
  edit: deny
  bash:
    "kubectl apply*": ask
    "helm install*": ask
    "*": allow
  read:
    ".": allow
  write:
    ".": allow
---
You are the infrastructure subagent. Handle repo scaffolding, Dockerfiles, Docker Compose stacks, Helm charts, and deployment configurations.
EOF

cat << 'EOF' > .opencode/agent/backend.md
---
description: Python logic, ftctl CLI renderer, FastAPI Inference Gateway, and PyTorch models.
mode: subagent
permission:
  edit: deny
  bash:
    "*": allow
  read:
    ".": allow
  write:
    ".": allow
---
You are the backend subagent. Handle Python code, Pydantic contracts, CLI tooling (`ftctl`), and the inference gateway service.
EOF

cat << 'EOF' > .opencode/agent/integration.md
---
description: Freqtrade strategies, FreqAI models, MCP clients, servers, and CI test pipelines.
mode: subagent
permission:
  edit: deny
  bash:
    "*": allow
  read:
    ".": allow
  write:
    ".": allow
---
You are the integration subagent. Handle Freqtrade custom strategies, MCP protocol integrations, and testing suites.
EOF

# 3. Automatically generate Base Config Examples
echo "--- Writing Base Configuration Files ---"

cat << 'EOF' > config/app.yaml.example
mode: dry_run
exchange:
  name: binance
  pairs: ["BTC/USDT", "ETH/USDT"]
  stake_currency: USDT
  stake_amount: 50
  max_stake: 100
strategy:
  name: AiSignalStrategy
  timeframe: 5m
  entry_signal_min: 0.4
  entry_confidence_min: 0.6
  exit_signal_max: -0.2
freqai:
  enabled: false
  identifier: ft-freqai-v1
  model: LightGBMRegressor
  train_period_days: 30
  backtest_period_days: 7
  live_retrain_hours: 1
  device: auto
  feature_parameters:
    include_timeframes: ["5m", "1h"]
    include_corr_pairlist: ["ETH/USDT"]
    label_period_candles: 24
  data_split_parameters: {test_size: 0.33, shuffle: false}
  model_training_parameters: {}
inference:
  url: http://inference:8000
  default_model: local-gru
  models:
    local-gru: {backend: local, device: auto}
    remote-llm: {backend: mcp, server: llm-tools, tool: predict}
mcp:
  servers:
    llm-tools:
      transport: stdio
      command: ["python", "-m", "some_mcp_server"]
secrets:
  binance_key: ${BINANCE_API_KEY}
  binance_secret: ${BINANCE_API_SECRET}
  ft_api_username: ${FT_API_USERNAME}
  ft_api_password: ${FT_API_PASSWORD}
EOF

cat << 'EOF' > config/.env.example
BINANCE_API_KEY=test_key_placeholder
BINANCE_API_SECRET=test_secret_placeholder
FT_API_USERNAME=freqtrader
FT_API_PASSWORD=changeme
FT_ALLOW_LIVE=no
EOF

# 4. Check if opencode is installed
if ! command -v opencode &> /dev/null; then
    echo "Error: opencode CLI is not installed or not in PATH."
    exit 1
fi

echo "--- WAVE 0 & 1: Foundation, CLI, Strategy & Gateway ---"

# Task 1: Repo scaffold and shared contracts
echo "[Task 1/11] Running Infra Agent: Scaffold & Contracts..."
opencode run --agent infra "@infra Implement Task 1 from @SPEC.md: Create pyproject.toml, Makefile, .pre-commit-config.yaml, .gitignore, and common/contracts.py using Pydantic v2."

# Task 2: Centralized configuration (ftctl)
echo "[Task 2/11] Running Backend Agent: ftctl CLI..."
opencode run --agent backend "@backend Implement Task 2 from @SPEC.md: Build the ftctl CLI loader, variable interpolator, environment guards, and target renderers."

# Task 3: Freqtrade image and AI-signal strategy
echo "[Task 3/11] Running Integration Agent: Freqtrade strategy..."
opencode run --agent integration "@integration Implement Task 3 from @SPEC.md: Set up freqtrade/Dockerfile, AiSignalStrategy.py, and the HTTP inference client with fallback logic."

# Task 5: Inference gateway with local GPU backend
echo "[Task 5/11] Running Backend Agent: FastAPI Inference Gateway..."
opencode run --agent backend "@backend Implement Task 5 from @SPEC.md: Build the FastAPI inference gateway under inference/app/ with local PyTorch/GRU backend support and dummy training script."

# Task 11: FreqAI strategies and configuration
echo "[Task 11/11 (Wave 1)] Running Integration Agent: FreqAI & Hybrid Strategies..."
opencode run --agent integration "@integration Implement Task 11 from @SPEC.md: Add FreqAiStrategy.py, HybridStrategy.py, and config plumbing."

echo "--- WAVE 2: Infrastructure & Connectors ---"

# Task 4: Docker Compose stack (CPU and GPU)
echo "[Task 4/11] Running Infra Agent: Docker Compose Stack..."
opencode run --agent infra "@infra Implement Task 4 from @SPEC.md: Create docker-compose.yml and docker-compose.gpu.yml override."

# Task 6: MCP client backend for external models
echo "[Task 6/11] Running Integration Agent: MCP Client Backend..."
opencode run --agent integration "@integration Implement Task 6 from @SPEC.md: Implement MCP client backend, connection pool, and circuit breaker under inference/app/backends/mcp_backend.py."

# Task 7: MCP server for bot monitoring and control
echo "[Task 7/11] Running Integration Agent: Bot Control MCP Server..."
opencode run --agent integration "@integration Implement Task 7 from @SPEC.md: Build the bot monitoring and control MCP server under mcp_server/."

# Task 8: Kubernetes Helm chart
echo "[Task 8/11] Running Infra Agent: Helm Chart..."
opencode run --agent infra "@infra Implement Task 8 from @SPEC.md: Generate Helm templates under deploy/helm/ with GPU toggles and Recreate strategy for Freqtrade."

# Task 9: Observability
echo "[Task 9/11] Running Infra Agent: Observability..."
opencode run --agent infra "@infra Implement Task 9 from @SPEC.md: Set up Prometheus configs and Grafana dashboards for trading and inference metrics."

echo "--- WAVE 3: Testing, Safety Checks & CI ---"

# Task 10: Testing, safety checks and CI
echo "[Task 10/11] Running Integration Agent: Tests & CI Pipeline..."
opencode run --agent integration "@integration Implement Task 10 from @SPEC.md: Create GitHub Actions CI workflow, safety tests, and end-to-end dry-run verification script."

echo "===================================================="
echo " Multi-Agent Headless Build Wave Execution Complete!"
echo "===================================================="
