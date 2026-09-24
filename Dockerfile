FROM ghcr.io/astral-sh/uv:debian-slim

WORKDIR /app
COPY . /app

# Only the packages the eval sweep actually needs at runtime -- skips
# download-pre-embedded's datasets/HF Hub stack, which is only used offline
# to fetch data (already done, and now served from the shared NFS volume).
RUN uv sync --package eval-harness --package qdrant-load --frozen

ENV PATH="/app/.venv/bin:$PATH"

CMD [ "true" ]
