# Voice engine image. Built on the server itself (Oracle Always Free is ARM64).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app/engine

# Dependencies first, so code changes don't reinstall them.
COPY engine/pyproject.toml engine/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Keep the repo layout: the engine finds devpage/ and client/dist-browser/ relative to its source.
COPY engine/ ./
COPY devpage/ /app/devpage/
COPY client/dist-browser/client.js /app/client/dist-browser/client.js
RUN uv sync --frozen --no-dev

CMD ["uv", "run", "--frozen", "--no-dev", "voice-engine"]
