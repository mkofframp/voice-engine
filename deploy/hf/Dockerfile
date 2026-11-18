# Hugging Face Space image: the engine on 127.0.0.1:7861 behind Caddy on :7860 (password gate).
# HF runs the container as user 1000 with one public port and terminates HTTPS itself.
FROM caddy:2 AS caddy
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
    HOME=/tmp XDG_DATA_HOME=/tmp/caddy-data XDG_CONFIG_HOME=/tmp/caddy-config \
    HOST=127.0.0.1 PORT=7861
COPY --from=caddy /usr/bin/caddy /usr/bin/caddy
WORKDIR /app/engine

COPY engine/pyproject.toml engine/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Keep the repo layout: the engine finds devpage/ and client/dist-browser/ relative to its source.
COPY engine/ ./
COPY devpage/ /app/devpage/
COPY client/dist-browser/client.js /app/client/dist-browser/client.js
RUN uv sync --frozen --no-dev
COPY deploy/hf/start.sh deploy/hf/Caddyfile.template /app/deploy/hf/

EXPOSE 7860
CMD ["bash", "/app/deploy/hf/start.sh"]
