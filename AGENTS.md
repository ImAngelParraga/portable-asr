# Repository Guidelines

## Project Structure

This repository contains an OpenAI-compatible ASR service. The main application lives in `asr_server.py`. Configuration examples live in `config/`. Deployment helpers live in `scripts/`, systemd examples live in `systemd/`, and generic public documentation lives in `docs/`. Tests live in `tests/` and use `unittest`.

Keep host-specific operations, live environment files, private network details, hardware inventory, and latency baselines in a private companion repository. See `docs/private-companion.md`.

## Development

Create a local environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Run tests:

```bash
.venv/bin/python -m unittest discover -s tests
```

Run locally on CPU:

```bash
ASR_BEARER_TOKEN=dev-token ASR_DEVICE=cpu ASR_POSTPROCESS_ENABLED=0 \
  uvicorn asr_server:app --host 127.0.0.1 --port 9000
```

## Configuration

Runtime behavior is controlled with `ASR_*` environment variables. Public examples must be safe placeholders and must not include live tokens, private hosts, private IP addresses, GPU UUIDs from real machines, or operator-specific paths.

Default public examples should favor:

- `ASR_LISTEN_HOST=127.0.0.1`
- `ASR_DEVICE=cpu`
- `ASR_POSTPROCESS_ENABLED=0`
- generated or placeholder bearer tokens

GPU, multi-GPU, and local LLM settings are optional profiles, not required defaults.

## Installation

Production install scripts must be idempotent and configurable through environment variables or documented flags. Do not hardcode one host's service name, paths, user, model id, GPU layout, network exposure, or sudoers rules.

Generic install:

```bash
sudo ./scripts/install-service.sh
```

Optional local LLM install:

```bash
sudo ./scripts/install-llm-runtime.sh
```

## Testing

Tests must not require:

- GPU hardware
- network access
- root privileges
- systemd
- live Whisper or llama.cpp models
- private companion repository

When changing transcript cleanup, include regression cases for meaning preservation, punctuation, paragraph breaks, expression commands, and failure behavior.

## Security

Do not commit:

- bearer tokens
- live `.env` files
- private overlays
- private hostnames or tailnet domains
- real GPU UUIDs
- logs or pid files
- model caches
- downloaded llama.cpp binaries
- virtualenvs

Before a public release, run:

```bash
scripts/secret-scan.sh
.venv/bin/python -m unittest discover -s tests
git status --short
```

## Style

Use Python 3 with 4-space indentation. Keep helper functions private with leading underscores when they are not API handlers. Prefer explicit, conservative text-processing logic over broad rewrites. Shell scripts should remain safe to re-run and should fail clearly.

## Commits

Use short imperative commit subjects. Keep commits focused on one behavior, docs change, or operational risk. Mention tests or scans in pull request notes.
