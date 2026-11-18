# Hosted demo (Oracle Cloud Always Free)

One small always-on server runs the engine behind HTTPS and a shared username/password.
Your API keys stay on the server; teammates only open a link.

## 1. Create the server (once, ~20 min, in the Oracle console)

1. Sign up at <https://signup.cloud.oracle.com> (a card is asked for verification; Always Free resources are not charged). Pick a home region close to your team. It can't be changed later.
2. **Compute → Instances → Create instance**
   - Image: **Canonical Ubuntu 24.04**
   - Shape: **Ampere → VM.Standard.A1.Flex**, 2 OCPUs, 12 GB memory (inside the free allowance)
   - Networking: keep "Assign a public IPv4 address" on
   - SSH keys: upload your public key (`~/.ssh/id_ed25519.pub`), or download the generated private key
   - If it says "Out of capacity", try another availability domain, or retry later.
3. **Open the ports** — on the instance page: Subnet → Security list → *Add ingress rules*, source `0.0.0.0/0`:
   - TCP, destination port `80`
   - TCP, destination port `443`
   - UDP, destination port range `32768-60999` (live-conversation audio)
4. Copy the instance's **public IP address**.

## 2. Deploy (from this Mac)

```bash
deploy/deploy.sh <public-ip>                      # uses your default SSH key
deploy/deploy.sh <public-ip> ~/Downloads/key.pem  # or the key Oracle generated
```

It copies the project, installs Docker, opens the server firewall, builds the engine,
gets an HTTPS certificate for `<ip-with-dashes>.sslip.io`, and prints the link, username and password.
The password is kept in `deploy/.demo-passcode`; delete that file and redeploy to change it.
Re-run the same command after code changes.

## Limits and costs

- `MAX_CONNECTED_SESSIONS` (default 3) live conversations at once; set it before `deploy.sh` to change.
- Each conversation still spends Deepgram, Anthropic and Cartesia credit from the keys in `engine/.env`.
- Live audio needs UDP. A teammate on a network that blocks UDP (some corporate VPNs) can use text-to-voice but not live conversation.

## Operating

```bash
ssh ubuntu@<ip> 'cd kid-voice/deploy && sudo docker compose logs -f engine'   # logs
ssh ubuntu@<ip> 'cd kid-voice/deploy && sudo docker compose down'            # stop
```
