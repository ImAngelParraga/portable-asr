# Models and transcript cleanup

These are starting points for this service, not universal accuracy or latency rankings. Compare recordings in your language on your own hardware. Model downloads and their licenses come from the linked upstream projects. Guidance checked on 2026-11-01.

## Transcription

| Setup | Server configuration | Client model | Tradeoff |
| --- | --- | --- | --- |
| First install, CPU | `ASR_MODEL_SIZE=base`, `ASR_DEVICE=cpu`, `ASR_COMPUTE_TYPE=int8` | `whisper-1` | Small download; establish a working baseline |
| More CPU capacity | `ASR_MODEL_SIZE=small`, CPU/int8 | `whisper-1` | Try when base misses words; measure latency |
| NVIDIA CUDA | `ASR_MODEL_SIZE=large-v3`, CUDA/float16 | `whisper-1` | Larger model and more VRAM; try `turbo` if latency matters more |
| Optional Qwen worker | `ASR_QWEN_ENABLED=1`, separate environment | `qwen3-asr-1.7b` | Multilingual alternative; additional PyTorch setup |
| Optional Parakeet worker | `ASR_PARAKEET_ENABLED=1`, Photon or cpp backend | `parakeet-ultra` | Additional runtime/hardware constraints |

The Whisper request model is a compatibility selector: changing it to a different Whisper size does not reload a new checkpoint. Set `ASR_MODEL_SIZE` and restart the server instead. Optional engines appear in authenticated `GET /v1/audio/models` only when enabled.

Upstream: [faster-whisper models, CPU/CUDA examples and requirements](https://github.com/SYSTRAN/faster-whisper), [Qwen3-ASR model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), [Parakeet Ultra model card](https://huggingface.co/moondream/parakeet-ultra). See [advanced engine setup](advanced.md) before enabling an optional worker.

## Cleanup choices

| Choice | Recommendation |
| --- | --- |
| No LLM | Recommended first install and simplest CPU setup: `ASR_POSTPROCESS_ENABLED=0` |
| Local LLM | `unsloth/Qwen3.5-9B-GGUF:Q4_K_M`, the model configured by the local profile and runtime installer |
| External LLM | Any compatible chat-completions endpoint that follows conservative editing instructions; use its actual served model ID |

The [Qwen quantization card](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) lists Q4_K_M weights at roughly 5.8 GB. Runtime memory also includes context, buffers, and any ASR model sharing the device. The installer requires an 8 GiB GPU by default; that check is a starting threshold, not a promise that ASR and cleanup fit together. CPU cleanup is possible but may add substantial latency. Avoid adding another model until you have measured the baseline.

### Local llama.cpp

On Ubuntu/Debian, run from the service checkout:

```bash
sudo env LLM_MODEL=unsloth/Qwen3.5-9B-GGUF:Q4_K_M \
  LLM_CONTEXT=8192 ./scripts/install-llm-runtime.sh
```

This installs build dependencies, builds the latest llama.cpp release for CUDA, downloads the model, and runs a smoke test. It needs a working NVIDIA driver and a compatible CUDA compiler. For CPU builds, add `LLM_BACKEND=cpu` to the same `sudo env` invocation. The runtime installer does not edit your ASR environment.

Set these values in `/opt/portable-asr/asr.env` (or your foreground environment):

```env
ASR_POSTPROCESS_ENABLED=1
ASR_POSTPROCESS_PROVIDER=local-llama
ASR_POSTPROCESS_MODEL=unsloth/Qwen3.5-9B-GGUF:Q4_K_M
ASR_POSTPROCESS_CONTEXT=8192
ASR_POSTPROCESS_BASE_URL=http://127.0.0.1:9100/v1
ASR_POSTPROCESS_CLI=/opt/asr-llm/bin/llama-cli
ASR_LLM_SERVER_BIN=/opt/asr-llm/bin/llama-server
ASR_POSTPROCESS_CACHE_DIR=/opt/asr-llm/hf-cache
ASR_POSTPROCESS_LIBRARY_PATH=/opt/asr-llm/bin
```

Restart `portable-asr` and test a real recording. The ASR process starts a resident `llama-server` on demand and stops it after its idle timeout. The installer uses `llama-cli` only for cache priming and the smoke test. Keep port 9100 on localhost. See [llama.cpp build requirements](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) if the compiler or GPU architecture is unsupported.

### External endpoint

Copy the settings from `config/profiles/external-postprocess.env.example`, substitute the real base URL ending in `/v1` and served model ID, and restart. The current proxy sends JSON without an upstream Authorization header; use a trusted local/private proxy if your provider requires an API key. The public ASR bearer token authenticates clients to this service; it is not forwarded as a cleanup-provider credential.

External cleanup sends transcript text to that endpoint. Optional unavailable cleanup returns raw audio transcription; `ASR_AUDIO_POSTPROCESS_REQUIRED=1` instead makes audio requests fail if cleanup is unavailable. `POST /v1/text/postprocess` fails with HTTP 503 when cleanup is unavailable.

## Ready-to-use generic prompt

[config/cleanup-prompt.txt](../config/cleanup-prompt.txt) contains portable formatting instructions with no personal names, vocabulary, or spoken-command conventions. Use it as the system prompt of a standalone cleanup LLM. Supply the raw transcript as a separate user message.

To use it with the existing text cleanup API, run from the checkout with the same bearer token exported in your shell:

```bash
.venv/bin/python - <<'PYTHON' > /tmp/cleanup-request.json
import json
from pathlib import Path
print(json.dumps({
    "text": "hello this is a transcription example",
    "instruction": Path("config/cleanup-prompt.txt").read_text(),
}))
PYTHON
curl --fail -H "Authorization: Bearer $ASR_BEARER_TOKEN" \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/cleanup-request.json \
  http://127.0.0.1:9000/v1/text/postprocess
rm /tmp/cleanup-request.json
```

To use this generic prompt for all audio dictation and default text cleanup, set:

```env
ASR_POSTPROCESS_PROMPT_FILE=/opt/portable-asr/config/cleanup-prompt.txt
```

The service installer copies the bundled template to that location. For the foreground quickstart, export `ASR_POSTPROCESS_PROMPT_FILE="$PWD/config/cleanup-prompt.txt"` before starting Python. Restart after changing the file; missing, unreadable, empty, or invalid UTF-8 files fail startup rather than silently selecting another prompt. Store personal customizations in a separate file so reinstalling the bundled template does not overwrite them.

Leave the setting unset to use the built-in audio cleanup prompt, which needs no prompt file. It adds the service's optional [dictation controls](dictation.md). A per-request `instruction` replaces the LLM instructions only: deterministic formatting and safety checks still apply, including recognized voice controls. The audio request's `prompt` field remains vocabulary context. Per-request text cleanup instructions take precedence over the configured file.
