"""Deploy the voice engine demo to a Hugging Face Space (free CPU tier).

    uv run --with huggingface_hub python deploy/hf/deploy.py [space-name]

Needs: `hf auth login` (or HF_TOKEN) with a write token, and in engine/.env the three
service keys plus TURN_KEY_ID and TURN_API_TOKEN (Cloudflare TURN, for live conversation).
Re-run after code changes; the password stays in deploy/.demo-passcode."""

import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = ("DEEPGRAM_API_KEY", "ANTHROPIC_API_KEY", "CARTESIA_API_KEY")
TURN = ("TURN_KEY_ID", "TURN_API_TOKEN")
PASS_FILE = ROOT / "deploy" / ".demo-passcode"
# Only these go into the Space. Never engine/.env, .venv, node_modules or fixtures.
SHIP = [
    ("engine/pyproject.toml", None), ("engine/uv.lock", None), ("engine/README.md", None),
    ("engine/src", None), ("devpage", None), ("client/dist-browser/client.js", None),
    ("deploy/hf/start.sh", None), ("deploy/hf/Caddyfile.template", None),
    ("deploy/hf/Dockerfile", "Dockerfile"),
]


def read_env(path: Path) -> dict[str, str]:
    out = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip("'\"")
    return out


def stage(dst: Path, space_title: str) -> None:
    for src, name in SHIP:
        s, d = ROOT / src, dst / (name or src)
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            shutil.copytree(s, d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".env"))
        else:
            shutil.copy2(s, d)
    (dst / "README.md").write_text(
        f"---\ntitle: {space_title}\nemoji: 🎙️\ncolorFrom: indigo\ncolorTo: pink\n"
        "sdk: docker\napp_port: 7860\npinned: false\n---\n"
    )


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "kid-voice-demo"
    env = {**read_env(ROOT / "engine" / ".env"), **{k: v for k, v in os.environ.items() if k in REQUIRED + TURN}}
    missing = [k for k in REQUIRED if not env.get(k)]
    if missing:
        sys.exit(f"Missing in engine/.env: {', '.join(missing)}")
    if not all(env.get(k) for k in TURN):
        print("WARNING: TURN_KEY_ID / TURN_API_TOKEN not set. Text-to-voice will work, "
              "live conversation will not (the Space has no direct UDP).")

    bundle = ROOT / "client" / "dist-browser" / "client.js"
    if not bundle.is_file():
        sys.exit("Missing client/dist-browser/client.js: run `npm run build:browser` in client/ (Node 22).")

    if not PASS_FILE.is_file():
        PASS_FILE.write_text(secrets.token_urlsafe(9)[:12])
        PASS_FILE.chmod(0o600)
    password = PASS_FILE.read_text().strip()
    user = os.environ.get("DEMO_USER", "demo")

    api = HfApi()
    owner = api.whoami()["name"]
    repo = f"{owner}/{name}"
    api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
    host = f"{owner}-{name}".lower().replace("_", "-").replace(".", "-") + ".hf.space"

    secrets_ = {k: env[k] for k in REQUIRED}
    secrets_["ENGINE_API_KEY"] = env.get("ENGINE_API_KEY") or secrets.token_urlsafe(32)
    secrets_["DEMO_PASSWORD"] = password
    for k in TURN:
        if env.get(k):
            secrets_[k] = env[k]
    for k, v in secrets_.items():
        api.add_space_secret(repo, k, v)
    variables = {
        "DEMO_USER": user,
        "MAX_CONNECTED_SESSIONS": os.environ.get("MAX_CONNECTED_SESSIONS", "3"),
        "ICE_SERVERS": '["stun:stun.cloudflare.com:3478"]',
        "ALLOWED_ORIGINS": f'["https://{host}"]',
        "LOG_LEVEL": "INFO",
    }
    for k, v in variables.items():
        api.add_space_variable(repo, k, v)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp), name)
        api.upload_folder(
            repo_id=repo, repo_type="space", folder_path=tmp,
            commit_message=f"deploy {commit or 'local'}",
            delete_patterns=["engine/**", "devpage/**", "client/**", "deploy/**"],
        )

    print(f"Uploaded. Building on Hugging Face (first build ~5-10 min): https://huggingface.co/spaces/{repo}")
    stage_ = ""
    for _ in range(120):
        stage_ = api.get_space_runtime(repo).stage
        if stage_ in ("RUNNING", "BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"):
            break
        time.sleep(10)
    print(f"Space status: {stage_}")
    if stage_ != "RUNNING":
        sys.exit(f"See the logs: https://huggingface.co/spaces/{repo}?logs=container")
    print(f"""
Share with your team:
  Link:      https://{host}/dev
  Username:  {user}
  Password:  {password}
Open the link directly (not the huggingface.co page) so the browser can ask for the mic.""")


if __name__ == "__main__":
    main()
