# Advanced engines and diagnostics

Get the [CPU quickstart](../README.md#quickstart) working before enabling an optional engine. These workers add separate environments, model downloads, and GPU compatibility requirements. Install `ffmpeg` for Qwen and Parakeet audio decoding.

## Recent Audio Diagnostics

Audio history is disabled by default. To diagnose recognition and cleanup separately, enable bounded local retention in the service environment:

```env
ASR_AUDIO_HISTORY_ENABLED=1
ASR_AUDIO_HISTORY_DIR=/opt/portable-asr/audio-history
ASR_AUDIO_HISTORY_MAX_FILES=20
ASR_AUDIO_HISTORY_MAX_AGE_SECONDS=86400
ASR_AUDIO_HISTORY_MAX_BYTES=268435456
```

Choose a directory writable by the service user and allowed by any systemd filesystem restrictions. Restart the service after changing settings. Every authenticated, accepted audio request creates a directory named with its server-generated request ID. It contains the original upload as `audio.<extension>` and `metadata.json` with UTC creation time, engine/model, language, temperature, prompt length, processing status, raw ASR text, final text, and timings. Failed inference or cleanup retains the available audio/text with an error type and HTTP status; exception messages, original filenames, bearer tokens, and prompt contents are excluded.

Successful archived responses include `request_id`; failed archived responses include the `X-Request-ID` header. Logs contain the ID and operational diagnostics, never audio or transcript bodies. Inspect the matching directory locally to compare the recording with `raw_text` and `final_text`. The service provides no archive download endpoint. Records left incomplete after a process crash remain available until normal eviction.

Directories use mode `0700`; audio and metadata files use `0600`. Oldest records are removed when any count, age, or total file-byte limit is reached. Limits must be positive. Cleanup runs on writes and every 60 seconds, including startup and idle periods, while history is enabled and the service is running. Storage failures preserve normal transcription behavior and emit a diagnostic without sensitive content. Record IDs can outlive their files after eviction. Disabling capture or stopping the service stops periodic cleanup; delete existing records locally if they must be removed immediately. Keep diagnostic data outside source control.

## Optional Parakeet Ultra

Select `parakeet-ultra` to use `moondream/parakeet-ultra` through the local Photon runtime. It is disabled by default and does not change Whisper or Qwen selection. Both OpenVoiceIME and OpenVoiceDesktop can discover it through the server model list; refresh that list and select it explicitly in the app you want to test.

Use a separate Python environment with `moondream==2.6.1` and a CUDA-compatible PyTorch build. Photon requires an NVIDIA Ampere or newer GPU; Pascal cards cannot run this engine. Install `ffmpeg` on the host for WAV/AAC/M4A decoding. Keep the runtime and writable model cache inside paths accessible to your service's systemd sandbox.

```bash
sudo install -d -o "$(id -un)" -g "$(id -gn)" /opt/portable-asr/parakeet-venv
python3 -m venv /opt/portable-asr/parakeet-venv
/opt/portable-asr/parakeet-venv/bin/pip install 'moondream==2.6.1'
```

```env
ASR_PARAKEET_ENABLED=1
ASR_PARAKEET_MODEL_ID=parakeet-ultra
ASR_PARAKEET_CHECKPOINT=moondream/parakeet-ultra
ASR_PARAKEET_PYTHON=/opt/portable-asr/parakeet-venv/bin/python
ASR_PARAKEET_CUDA_VISIBLE_DEVICES=0
ASR_PARAKEET_CACHE_DIR=/opt/portable-asr/parakeet-cache
ASR_PARAKEET_KEEP_WARM=0
ASR_PARAKEET_STOP_LLM_FOR_TRANSCRIPTION=1
```

Prefer a GPU UUID for `ASR_PARAKEET_CUDA_VISIBLE_DEVICES` when multiple GPUs have different capabilities. The worker removes Whisper's `LD_LIBRARY_PATH` so PyTorch resolves its own libraries. It starts only when selected. First use may download weights; pre-download them into the configured cache before serving requests.

The example releases the worker and CUDA memory after every request, before normal transcript cleanup. With `ASR_PARAKEET_STOP_LLM_FOR_TRANSCRIPTION=1`, local cleanup is stopped before Parakeet inference and starts again when cleanup is needed. This allows both models to share limited VRAM, but adds model loading time to each dictation. Set `ASR_PARAKEET_KEEP_WARM=1` only when Parakeet and cleanup have enough independent GPU memory; leave the stop setting at `0` if they do not need to share memory.

Parakeet detects language automatically. The service accepts existing client `language`, `prompt`, and `temperature` fields for compatibility, but Parakeet does not use them for forced language, vocabulary biasing, or sampling. Authentication, optional cleanup, request timing, and diagnostic history behave like the other engines. Worker logs and failures exclude transcript contents. For an explicit `language=es` request whose Parakeet result contains Cyrillic, the service retries once using the same loaded model with 0.5 seconds of silence appended to decoded PCM. Speech samples remain unchanged; no Whisper fallback or model switch occurs. This is a short-clip boundary recovery, not forced Spanish decoding: Ultra has no supported language prompt. Failed, empty, or still-Cyrillic recovery returns sanitized HTTP 503 without evicting a warm worker. Other language hints and automatic detection keep their existing behavior. Fallback timings identify the retry; enabled audio history retains `initial_raw_text`, recovered `raw_text`, `recognition_engine=parakeet`, and `fallback_reason`. The retry is not a guarantee of correct language or recognition for every clip.

### Ultra on a Pascal GPU

The `cpp` backend loads a locally built [parakeet.cpp](https://github.com/mudler/parakeet.cpp) shared library and a converted Ultra GGUF. This avoids Photon's Ampere requirement. Use a CUDA toolkit and compiler compatible with your card; for a GTX 1070, build for CUDA architecture `61`. The F16 GGUF preserves the Ultra checkpoint without low-bit quantization; it is not NVIDIA's original checkpoint.

Build with `PARAKEET_GGML_CUDA=ON`, `PARAKEET_SHARED=ON`, and `CMAKE_CUDA_ARCHITECTURES=61`. Copy `libparakeet.so` and its `libggml*.so*` dependencies into one service-accessible directory, and put `ultra-f16.gguf` in a readable model directory. The worker uses that library directory instead of Whisper's CUDA library overrides. No Photon or PyTorch package is needed in this worker's Python environment.

```env
ASR_PARAKEET_BACKEND=cpp
ASR_PARAKEET_LIBRARY=/opt/portable-asr/parakeet-cpp/lib/libparakeet.so
ASR_PARAKEET_GGUF=/opt/portable-asr/parakeet-cpp/models/ultra-f16.gguf
ASR_PARAKEET_PYTHON=/opt/portable-asr/venv/bin/python
ASR_PARAKEET_CUDA_VISIBLE_DEVICES=0
ASR_PARAKEET_KEEP_WARM=1
ASR_PARAKEET_STOP_LLM_FOR_TRANSCRIPTION=0
ASR_WHISPER_KEEP_WARM=1
ASR_SINGLE_RESIDENT_MODEL=1
```

With single-resident mode, selecting Whisper, Qwen, or Parakeet unloads the other ASR workers before inference. Repeated requests keep the selected model loaded through cleanup. Keep all ASR engines on the same GPU and cleanup on its own GPU. Switching models incurs loading time; ordinary requests avoid that cost. The default model selection and model list order do not change.

## Optional Qwen3-ASR

Qwen3-ASR can run beside resident Whisper in a separate Python worker. Keep the base ASR environment unchanged. Create a separate environment with `qwen-asr==0.0.6` and a PyTorch CUDA wheel compatible with your GPU. For older GPUs, verify that the selected PyTorch wheel still includes your CUDA architecture; CUDA version alone does not establish support. See [PyTorch installation guidance](https://pytorch.org/get-started/locally/).

Install the `ffmpeg` command on the service host. The Qwen worker decodes uploads, including OpenVoiceIME's AAC/M4A recordings, to temporary mono 16 kHz PCM WAV before inference. Decoded files are removed after each request; the Whisper path is unchanged.

```bash
sudo install -d -o "$(id -un)" -g "$(id -gn)" /opt/asr-qwen
python3 -m venv /opt/asr-qwen/venv
/opt/asr-qwen/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126
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

`GET /v1/audio/models` returns the currently enabled transcription model IDs in OpenAI list format. It requires the same bearer token as transcription requests and does not include the cleanup LLM. Clients can refresh this list when settings open or the endpoint changes; `GET /v1/models` remains the broader compatibility catalog.

## GPU sharing


With `ASR_GPU_WATCH_ENABLED=1`, cleanup yields only to external NVIDIA compute processes whose executable name matches `ASR_GPU_WATCH_PROCESS_PATTERNS` (default `ffmpeg`). CPU-only FFmpeg jobs do not unload or block cleanup. Set `ASR_LLM_CUDA_VISIBLE_DEVICES` to physical GPU UUIDs to limit arbitration to the cleanup GPU(s); without it, all NVIDIA GPUs are considered. Media-server NVENC on a separate GPU can coexist with warm cleanup. Matching transcoding on the cleanup GPU pauses cleanup until its GPU context exits; there is no additional fixed cooldown. Other GPU processes do not force eviction or veto startup. NVIDIA telemetry failure leaves cleanup running.
