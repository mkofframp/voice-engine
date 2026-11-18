#!/usr/bin/env bash
# Deploy the voice engine demo to a fresh Ubuntu server (Oracle Cloud Always Free).
# Run from your Mac:   deploy/deploy.sh <server-public-ip> [path/to/ssh-private-key]
# Re-run any time to push new code; the passcode stays the same.
set -euo pipefail

IP="${1:?usage: deploy/deploy.sh <server-public-ip> [ssh-key]}"
SSH_KEY="${2:-}"
SSH_USER="${SSH_USER:-ubuntu}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
[[ -n "$SSH_KEY" ]] && SSH_OPTS+=(-i "$SSH_KEY")
TARGET="$SSH_USER@$IP"

# --- API keys from engine/.env (never committed) ---
envval() { grep -E "^$1=" engine/.env | head -1 | cut -d= -f2-; }
for k in DEEPGRAM_API_KEY ANTHROPIC_API_KEY CARTESIA_API_KEY ENGINE_API_KEY; do
  [[ -n "$(envval "$k")" ]] || { echo "engine/.env is missing $k" >&2; exit 1; }
done

# --- browser bundle for the test page ---
if [[ ! -f client/dist-browser/client.js ]]; then
  echo "Building the browser client (needs Node 22)..."
  (cd client && npm ci && npm run build:browser)
fi

# --- passcode, generated once and kept in deploy/.demo-passcode ---
PASS_FILE=deploy/.demo-passcode
[[ -f "$PASS_FILE" ]] || { openssl rand -base64 12 | tr -d '/+=' | cut -c1-12 > "$PASS_FILE"; chmod 600 "$PASS_FILE"; }
PASS="$(cat "$PASS_FILE")"
DEMO_USER="${DEMO_USER:-demo}"

SITE="$(echo "$IP" | tr . -).sslip.io"

# --- server environment ---
umask 077
cat > deploy/server.env <<EOF
DEEPGRAM_API_KEY=$(envval DEEPGRAM_API_KEY)
ANTHROPIC_API_KEY=$(envval ANTHROPIC_API_KEY)
CARTESIA_API_KEY=$(envval CARTESIA_API_KEY)
ENGINE_API_KEY=$(envval ENGINE_API_KEY)
HOST=127.0.0.1
PORT=7860
LOG_LEVEL=INFO
ICE_SERVERS=["stun:stun.l.google.com:19302","stun:stun.cloudflare.com:3478"]
MAX_CONNECTED_SESSIONS=${MAX_CONNECTED_SESSIONS:-3}
EOF
umask 022

echo "Copying the project to $TARGET ..."
ssh "${SSH_OPTS[@]}" "$TARGET" 'command -v rsync >/dev/null || (sudo apt-get update -qq && sudo apt-get install -y -qq rsync)'
rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
  --exclude .git --exclude .superpowers --exclude '**/.venv' --exclude '**/node_modules' \
  --exclude '**/__pycache__' --exclude engine/.env --exclude client/dist \
  --exclude 'engine/scripts/fixtures/*.wav' --exclude deploy/Caddyfile \
  ./ "$TARGET:kid-voice/"

echo "Setting up the server (first run takes ~10 minutes) ..."
ssh "${SSH_OPTS[@]}" "$TARGET" bash -s -- "$SITE" "$DEMO_USER" "$PASS" <<'REMOTE'
set -euo pipefail
SITE="$1"; DEMO_USER="$2"; PASS="$3"

if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
fi

# Oracle's Ubuntu images block everything but SSH in iptables. Open web + WebRTC audio.
allow() { sudo iptables -C INPUT "$@" -j ACCEPT 2>/dev/null || sudo iptables -I INPUT 1 "$@" -j ACCEPT; }
allow -p tcp --dport 80
allow -p tcp --dport 443
allow -p udp --dport 32768:60999
if command -v netfilter-persistent >/dev/null; then sudo netfilter-persistent save >/dev/null; fi

cd ~/kid-voice/deploy
HASH="$(sudo docker run --rm caddy:2 caddy hash-password --plaintext "$PASS")"
sed -e "s|__SITE__|$SITE|" -e "s|__USER__|$DEMO_USER|" -e "s|__HASH__|$HASH|" Caddyfile.template > Caddyfile
chmod 600 server.env

sudo docker compose up -d --build
sudo docker compose restart caddy >/dev/null  # pick up a changed Caddyfile

echo "Waiting for the engine and the HTTPS certificate ..."
for _ in $(seq 1 60); do
  code="$(curl -s -o /dev/null -w '%{http_code}' -u "$DEMO_USER:$PASS" "https://$SITE/dev" || true)"
  [[ "$code" == 200 ]] && break
  sleep 5
done
if [[ "$code" != 200 ]]; then
  echo "Not up yet (last status: $code). Logs:" >&2
  sudo docker compose logs --tail 40 >&2
  exit 1
fi
REMOTE

cat <<EOF

Done. Share with teammates:
  Link:      https://$SITE/
  Username:  $DEMO_USER
  Password:  $PASS
(Up to ${MAX_CONNECTED_SESSIONS:-3} live conversations at once.)
EOF
