# Public Boundary

This repository is the public core for an OpenAI-compatible ASR service. It should contain portable code, generic examples, tests, and installer templates.

## Public Core

Keep these in this repository:

- `asr_server.py`: API server, transcription orchestration, optional post-processing, and backend selection.
- `config/*.env.example`: sanitized configuration templates and hardware profiles.
- `scripts/install-service.sh`: configurable service installer.
- `scripts/install-llm-runtime.sh`: configurable optional local LLM installer.
- `systemd/*.example`: generic unit examples.
- `tests/`: unit tests that do not require GPUs, network, root, systemd, or live models.
- `docs/`: generic setup, security, release, and companion-overlay guidance.

## Generic Examples

Public examples may mention:

- CPU-only local development.
- Single Nvidia GPU.
- Multi-GPU setup with placeholder ordinal or UUID values.
- External OpenAI-compatible post-processing endpoint.
- Local llama.cpp post-processing with a replaceable model id.

Public examples must use placeholders, not real host details.

## Private Companion

Move these to a private companion repository:

- Real hostnames, private DNS names, Tailscale names, LAN addresses, and public exposure details.
- Live env files, bearer tokens, generated secrets, and private service account assumptions.
- Real GPU UUIDs, hardware inventory, failure history, and latency baselines.
- Host-specific deployment commands, sudoers assumptions, warmup commands, and rollback notes.
- Personal workflow names or examples that only make sense for one operator.

## Obsolete Local Migration Material

Migration scripts for a previous private service name are not part of the public product unless rewritten as generic migration examples. Keep private migrations in the companion repository.

## Release Gate

Before making this repository public, run:

```bash
scripts/secret-scan.sh
.venv/bin/python -m unittest discover -s tests
git status --short
```
