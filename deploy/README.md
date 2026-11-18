# Hosted demo (Hugging Face Spaces, free)

The engine runs in a free Hugging Face Space behind HTTPS and a shared username/password.
Your API keys are stored as Space secrets; teammates only open a link.

Hugging Face gives the Space no direct UDP, so live conversation goes through Cloudflare's
TURN relay (free up to 1,000 GB a month — a conversation uses a few MB a minute).
Text-to-voice works without it.

## 1. Accounts (once, ~10 min)

1. **Hugging Face**: sign up at <https://huggingface.co/join>, then create a token with
   *Write* access at <https://huggingface.co/settings/tokens>, and on this Mac run
   `uvx --from huggingface_hub hf auth login` and paste it.
2. **Cloudflare TURN**: sign up at <https://dash.cloudflare.com/sign-up>, open
   *Realtime → TURN Server → Create*, and copy the **Turn Token ID** and **API Token**.
   Add them to `engine/.env`:
   ```
   TURN_KEY_ID=...
   TURN_API_TOKEN=...
   ```

## 2. Deploy (from this Mac)

```bash
(cd client && npm run build:browser)              # Node 22
uv run --no-project --with huggingface_hub python deploy/hf/deploy.py   # optional: space name
```

It creates the public Space `<you>/kid-voice-demo`, stores the keys as secrets, uploads only the
engine, test page and browser bundle (never `engine/.env`), waits for the build and prints the
link, username and password. The password is kept in `deploy/.demo-passcode`; delete it and
redeploy to change it. Re-run the same command after code changes.

Teammates open `https://<you>-kid-voice-demo.hf.space/dev` directly. On the huggingface.co
page the app is inside a frame, where the password prompt and microphone don't work.

## Limits and costs

- The Space's code is public (its *Files* tab); secrets are not.
- A free Space sleeps after 48 hours without visitors; the next visit wakes it in about a minute.
- `MAX_CONNECTED_SESSIONS` (default 3) live conversations at once.
- Each conversation spends Deepgram, Anthropic and Cartesia credit from your keys.

## Your own server instead

`deploy/deploy.sh <ip> [ssh-key]` deploys to any Ubuntu 24.04 server you can SSH into
(Docker + Caddy with an automatic HTTPS certificate). Open TCP 80/443 and UDP 32768–60999.
