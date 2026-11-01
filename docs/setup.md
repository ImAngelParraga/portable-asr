# Deploy on a new machine

Use Ubuntu/Debian with systemd for the provided service installer. Other systems can run the foreground [quickstart](../README.md#quickstart). Start with CPU transcription; add acceleration and cleanup after one real recording succeeds.

## Persistent service

From a fresh checkout, install the prerequisites and select a profile:

```bash
sudo apt update
sudo apt install git python3 python3-venv
sudo env ASR_ENV_SOURCE=config/profiles/cpu.env.example ./scripts/install-service.sh
curl --fail http://127.0.0.1:9000/health
```

The installer installs Python dependencies into `/opt/portable-asr/venv`, creates a private `/opt/portable-asr/asr.env`, replaces `change-me` with a random bearer token, and enables `portable-asr.service`. By default it runs as the account invoking sudo. Read the token locally with `sudoedit /opt/portable-asr/asr.env` and enter it into each client. Do not post the environment file in support requests.

Custom paths and accounts are optional. Create the service account and group first, then pass options **after sudo**, so its environment filtering does not drop them:

```bash
sudo env ASR_DIR=/opt/portable-asr SERVICE_NAME=portable-asr \
  SERVICE_USER=asr SERVICE_GROUP=asr \
  ASR_ENV_SOURCE=config/profiles/cpu.env.example ./scripts/install-service.sh
```

Re-running the installer updates code and dependencies while preserving an existing env file. To change profiles on an existing installation, edit its env file explicitly and restart; the profile source does not overwrite existing settings.

```bash
sudo systemctl restart portable-asr
sudo systemctl status portable-asr
sudo journalctl -u portable-asr -n 50 --no-pager
```

First inference downloads weights. Test a real recording before relying on the service, and allow a longer client timeout for that first request.

## Reach it from another device

The default `ASR_LISTEN_HOST=127.0.0.1` accepts only local connections. For a phone or another PC, put a TLS reverse proxy on the server with a certificate trusted by the client, forwarding to `http://127.0.0.1:9000`. Enter `https://asr.example.com` as the client's base URL and keep the bearer token enabled. Replace the example domain with one you control and configure its DNS and certificate; this repository does not provision them.

Alternatively, for a private network deployment, set `ASR_LISTEN_HOST` to the desired interface address, restrict port 9000 in the firewall to your clients, and provide HTTPS before using an Android release APK. `0.0.0.0` binds every IPv4 interface; it is not a client URL. A VPN does not by itself turn an HTTP URL into HTTPS.

## Add CUDA or cleanup

For CUDA, install a supported NVIDIA driver and confirm `nvidia-smi` works. Use the single-GPU profile on a fresh install or edit `ASR_DEVICE=cuda`, `ASR_COMPUTE_TYPE=float16`, and `ASR_MODEL_SIZE=large-v3` in the existing env. The installer sets library paths for the pip-installed CUDA runtime; manual launches must follow the [faster-whisper library guidance](https://github.com/SYSTRAN/faster-whisper#gpu).

Follow [models and cleanup](models-and-cleanup.md) to add a local or external cleanup model. Keep the first setup small: neither cleanup nor a second ASR engine is required.

## Recreate or recover

Keep the selected code revision, a private copy of the environment file, and any custom cleanup instructions. Cache directories can be downloaded again. On the new machine, install the same revision and dependencies, restore the environment with mode `0600` for the service account, adjust device selection and paths, then test health and a real transcription. Tokens saved in client credential stores must be entered again on a new device. Public examples do not reproduce a particular host's private GPU or network setup.
