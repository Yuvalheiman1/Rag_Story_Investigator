# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (kept minimal)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel \
    # Force CPU-only PyTorch wheels in Docker to avoid pulling multi-GB CUDA runtime deps.
    # (PyTorch GPU wheels bring in /site-packages/nvidia and can explode image size.)
    && PIP_INDEX_URL=https://download.pytorch.org/whl/cpu \
       PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
       pip install -r /app/requirements.txt

# Copy app code
COPY config.yaml /app/config.yaml
COPY data /app/data
COPY src /app/src


# Default command runs the interactive console
CMD ["python", "-m", "src.main"]
