# Portable ASR

OpenAI-compatible speech-to-text service built with FastAPI and `faster-whisper`. It can run as a small CPU-only local service, use Nvidia CUDA for faster transcription, and optionally clean transcripts with a local or external OpenAI-compatible LLM.

## Related Project

[OpenVoiceIME](https://github.com/ImAngelParraga/OpenVoiceIME) is an Android voice keyboard that can send recordings to any OpenAI-compatible `/v1/audio/transcriptions` endpoint. This service can be used as a self-hosted transcription backend for OpenVoiceIME or other compatible clients.

## Features

- `POST /v1/audio/transcriptions` for OpenAI-compatible audio transcription.
- `POST /v1/text/postprocess` for transcript cleanup.
- `POST /v1/chat/completions` for OpenAI-compatible transcript cleanup requests.
- `POST /v1/llm/chat/completions` for non-streaming local or proxied LLM chat.
- Bearer-token auth for `/v1/*` endpoints.
- CPU, single-GPU, multi-GPU, local llama.cpp, and external postprocess profiles.

## Quickstart

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp config/profiles/cpu.env.example .env
ASR_BEARER_TOKEN=dev-token ASR_DEVICE=cpu ASR_POSTPROCESS_ENABLED=0 \
  uvicorn asr_server:app --host 127.0.0.1 --port 9000
```

Health check:

```bash
curl http://127.0.0.1:9000/health
```

Transcribe audio:

```bash
curl -sS \
  -H "Authorization: Bearer dev-token" \
  -F model=whisper-1 \
  -F language=en \
  -F 'file=@/path/to/audio.wav;type=audio/wav' \
  http://127.0.0.1:9000/v1/audio/transcriptions
```

Optional prompt context:

```bash
curl -sS \
  -H "Authorization: Bearer dev-token" \
  -F model=whisper-1 \
  -F language=en \
  -F prompt='Project names: ExampleApp, ExampleDB' \
  -F 'file=@/path/to/audio.wav;type=audio/wav' \
  http://127.0.0.1:9000/v1/audio/transcriptions
```

Keep prompts short and in the same language as the audio. Long or unrelated prompts can bias Whisper toward hallucinated words.

## Spoken Formatted Lists

When post-processing is enabled, bilingual voice controls can create Markdown lists:

- `lista de ítems` / `item list`: start a bulleted list.
- `lista numerada` / `numbered list`: start a numbered list.
- `nuevo ítem` / `new item`: start the next item in either list type.
- `fin de lista` / `end of list`: optionally end the list before continuing with normal prose.

The clause before the list control becomes the introduction with an implicit colon. For example, `hoy tengo que comprar lista de ítems plátanos nuevo ítem tomates` becomes:

```markdown
Hoy tengo que comprar:

- Plátanos
- Tomates
```

Clear comma-separated short items also work. Use `nuevo ítem` / `new item` when an item contains commas or item boundaries could be ambiguous.

## Configuration Profiles

Start from one profile and copy it to your runtime env file:

```bash
cp config/profiles/cpu.env.example .env
```

Profiles:

- `config/profiles/cpu.env.example`: safest default; no GPU or postprocess LLM required.
- `config/profiles/cuda-single-gpu.env.example`: Whisper on one Nvidia GPU.
- `config/profiles/cuda-multi-gpu.env.example`: separate devices for Whisper and local LLM.
- `config/profiles/local-llama.env.example`: optional local llama.cpp post-processing.
- `config/profiles/external-postprocess.env.example`: optional external OpenAI-compatible LLM endpoint.

Base template:

- `config/asr.env.example`: full list of supported settings with safe placeholder values.

## Production Install

Install the ASR service:

```bash
sudo ./scripts/install-service.sh
```

Common overrides:

```bash
ASR_DIR=/opt/portable-asr \
SERVICE_NAME=portable-asr \
SERVICE_USER=asr \
ASR_ENV_SOURCE=config/profiles/cpu.env.example \
sudo ./scripts/install-service.sh
```

Optional local llama.cpp runtime:

```bash
LLM_MODEL=unsloth/Qwen3.5-9B-GGUF:Q4_K_M \
LLM_CONTEXT=32768 \
sudo ./scripts/install-llm-runtime.sh
```

The local LLM runtime is optional. For transcription-only use, set:

```env
ASR_POSTPROCESS_ENABLED=0
```

## CPU Mode

CPU mode is the most portable setup:

```env
ASR_DEVICE=cpu
ASR_COMPUTE_TYPE=int8
ASR_MODEL_SIZE=base
ASR_POSTPROCESS_ENABLED=0
```

## CUDA Mode

CUDA mode requires Nvidia drivers and compatible Python packages:

```env
ASR_DEVICE=cuda
ASR_COMPUTE_TYPE=float16
ASR_WHISPER_CUDA_VISIBLE_DEVICES=0
```

For multi-GPU hosts, use ordinals or GPU UUIDs from `nvidia-smi -L`:

```env
ASR_WHISPER_CUDA_VISIBLE_DEVICES=0
ASR_LLM_CUDA_VISIBLE_DEVICES=1
```

UUIDs are useful on systems where device order may change. Use placeholder examples in public docs and keep real UUIDs private.

## Optional Qwen3-ASR

Qwen3-ASR can run beside resident Whisper in a separate Python worker. Keep the base ASR environment unchanged. Create a separate environment with `qwen-asr==0.0.6` and a PyTorch CUDA wheel compatible with your GPU. Older Pascal/Volta GPUs require the CUDA 12.6 PyTorch wheel; see [PyTorch installation guidance](https://pytorch.org/get-started/locally/).

```bash
python3 -m venv /opt/asr-qwen/venv
/opt/asr-qwen/venv/bin/pip install --index-url https://download.pytorch.org/whl/cu126 'torch==2.12.1+cu126'
/opt/asr-qwen/venv/bin/pip install 'qwen-asr==0.0.6'
```

Set these values in the service environment:

```env
ASR_QWEN_ENABLED=1
ASR_QWEN_PYTHON=/opt/asr-qwen/venv/bin/python
ASR_QWEN_CHECKPOINT=Qwen/Qwen3-ASR-1.7B
ASR_QWEN_CUDA_VISIBLE_DEVICES=0
```

Send `model=qwen3-asr-1.7b` to `/v1/audio/transcriptions` to use Qwen. Existing model values continue to use Whisper. The Qwen worker starts on first Qwen request and remains loaded; allow extra time for first model download/load, or pre-download the checkpoint. `language=en` and `language=es` map to Qwen's forced-language names, while omitted language uses automatic detection. Multipart `prompt` becomes Qwen request context. Same authentication and optional transcript cleanup apply to both engines. Use OpenVoiceDesktop's ASR **Model** field or OpenVoiceIME's custom-provider **Model** field to select Qwen for that app.

## Optional Post-Processing

Post-processing cleans punctuation, casing, paragraph breaks, and clear ASR artifacts while preserving meaning and language. It is disabled in the CPU profile.

Local llama.cpp:

```env
ASR_POSTPROCESS_ENABLED=1
ASR_POSTPROCESS_PROVIDER=local-llama
ASR_POSTPROCESS_BASE_URL=http://127.0.0.1:9100/v1
```

External OpenAI-compatible endpoint:

```env
ASR_POSTPROCESS_ENABLED=1
ASR_POSTPROCESS_PROVIDER=openai-compatible
ASR_POSTPROCESS_BASE_URL=http://llm.example.local:8000/v1
ASR_POSTPROCESS_MODEL=example-cleanup-model
```

## Files

- `asr_server.py`: FastAPI service.
- `requirements.txt`: Python dependencies.
- `config/asr.env.example`: complete sanitized configuration template.
- `config/profiles/`: environment-specific sanitized profiles.
- `scripts/install-service.sh`: configurable systemd installer.
- `scripts/install-llm-runtime.sh`: optional configurable llama.cpp installer.
- `scripts/secret-scan.sh`: public-release leak scanner.
- `systemd/portable-asr.service.example`: generic systemd unit example.
- `docs/`: public setup and release docs.
- `tests/`: unit tests.

## Test

```bash
.venv/bin/python -m unittest discover -s tests
```

Tests mock external model behavior and should not require GPU, network, root, systemd, or live model downloads.

## Security

Do not commit:

- bearer tokens
- live env files
- private overlay files
- private hostnames or network domains
- real GPU UUIDs
- logs or pid files
- model caches
- virtualenvs
- downloaded llama.cpp binaries

Before making a public release:

```bash
scripts/secret-scan.sh
.venv/bin/python -m unittest discover -s tests
git status --short
```

Host-specific operations belong in a private companion repository. See `docs/private-companion.md`.
