# Agent worker image for LiveKit Cloud (`lk agent deploy`).
#
# Only the agent runs here. The web console is a separate deployment — it is an ordinary HTTP
# service, while this is a long-lived worker holding a WebSocket to LiveKit.

FROM python:3.11-slim

# libgomp is required by onnxruntime, which backs Silero VAD and the turn detector.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# pyproject.toml declares agent/, analysis/, observability/ and services/ as packages, so they
# must be present for `pip install` to build at all.
COPY pyproject.toml main.py ./
COPY agent/ ./agent/
COPY analysis/ ./analysis/
COPY observability/ ./observability/
COPY services/ ./services/
COPY data/ ./data/
RUN pip install --upgrade pip && pip install ".[agent]"

# Model weights are fetched at build time. Downloading them on first call would block the event
# loop for several seconds and delay the greeting on a live call.
RUN python main.py download-files

# `start` rather than `dev`: production logging, graceful SIGTERM handling, no reload.
CMD ["python", "main.py", "start"]
