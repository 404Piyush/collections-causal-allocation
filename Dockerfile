# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies (CBC solver comes bundled with PuLP)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.lock ./
RUN pip install -r requirements.lock

# Copy source
COPY pyproject.toml ./
COPY src/ ./src/
COPY data/ ./data/
COPY tests/ ./tests/
COPY run_pipeline.py ./
COPY notebooks/ ./notebooks/

# Create non-root user for runtime
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Default: run the full pipeline
ENTRYPOINT ["python", "run_pipeline.py"]
CMD ["--help"]