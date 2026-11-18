#!/usr/bin/env bash
# Container entry point: password gate (Caddy) in front of the engine. Exits if either stops.
set -euo pipefail
: "${DEMO_PASSWORD:?DEMO_PASSWORD secret is not set}"
HASH="$(caddy hash-password --plaintext "$DEMO_PASSWORD")"
sed -e "s|__USER__|${DEMO_USER:-demo}|" -e "s|__HASH__|$HASH|" \
  /app/deploy/hf/Caddyfile.template > /tmp/Caddyfile

/app/engine/.venv/bin/voice-engine &
caddy run --config /tmp/Caddyfile --adapter caddyfile &
wait -n
exit 1
