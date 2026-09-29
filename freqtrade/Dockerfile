# Build arg selecting the Freqtrade image variant (Task 4, consumed from the
# ftctl-rendered Compose env as FREQTRADE_TAG):
#   stable (default)            gateway / hybrid-signal strategies
#   stable_freqai               FreqAI with classic ML models (LightGBM, XGBoost)
#   stable_freqaitorch          FreqAI with PyTorch models
# NOTE: the stock FreqAI images may not ship CUDA-enabled PyTorch; for GPU
# training/inference with PyTorch FreqAI models see Task 11
# (freqtrade/Dockerfile.freqai, built on a CUDA base) and
# deploy/compose/docker-compose.freqai-gpu.yml.
ARG FREQTRADE_TAG=stable
FROM freqtradeorg/freqtrade:${FREQTRADE_TAG}

# Extra runtime dependency for the AI-signal strategy's inference gateway
# client. Freqtrade itself is left unmodified; all custom logic lives in
# user_data/strategies (see AiSignalStrategy.py and _inference_client.py).
USER root
RUN pip install --no-cache-dir "httpx>=0.27"
USER ftuser
