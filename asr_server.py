#!/usr/bin/env python3
"""
OpenAI-compatible ASR microservice using faster-whisper.
Defaults favor portable localhost CPU startup; override with env vars.
"""

import os
import io
import json
import time
import uuid
import gc
import re
import select
import tempfile
import threading
import asyncio
import subprocess
import sys
import unicodedata
import urllib.error
import urllib.request
import urllib.parse
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ---- Configuration (from environment) ----
ASR_MODEL_SIZE = os.environ.get("ASR_MODEL_SIZE", "base")
ASR_DEVICE = os.environ.get("ASR_DEVICE", "cuda")
ASR_COMPUTE_TYPE = os.environ.get("ASR_COMPUTE_TYPE", "float16")
ASR_WHISPER_CUDA_VISIBLE_DEVICES = os.environ.get("ASR_WHISPER_CUDA_VISIBLE_DEVICES", "").strip()
ASR_WHISPER_KEEP_WARM = os.environ.get("ASR_WHISPER_KEEP_WARM", "0") == "1"
ASR_QWEN_ENABLED = os.environ.get("ASR_QWEN_ENABLED", "0") == "1"
ASR_QWEN_MODEL_ID = os.environ.get("ASR_QWEN_MODEL_ID", "qwen3-asr-1.7b")
ASR_QWEN_CHECKPOINT = os.environ.get("ASR_QWEN_CHECKPOINT", "Qwen/Qwen3-ASR-1.7B")
ASR_QWEN_PYTHON = os.environ.get("ASR_QWEN_PYTHON", sys.executable)
ASR_QWEN_CUDA_VISIBLE_DEVICES = os.environ.get("ASR_QWEN_CUDA_VISIBLE_DEVICES", "").strip()
ASR_BEAM_SIZE = int(os.environ.get("ASR_BEAM_SIZE", "5"))
ASR_VAD_FILTER = os.environ.get("ASR_VAD_FILTER", "0") == "1"
ASR_VAD_MIN_SILENCE_MS = int(os.environ.get("ASR_VAD_MIN_SILENCE_MS", "500"))
ASR_BEARER_TOKEN = os.environ.get("ASR_BEARER_TOKEN", "")
ASR_CPU_THREADS = int(os.environ.get("ASR_CPU_THREADS", "4"))
ASR_NUM_WORKERS = int(os.environ.get("ASR_NUM_WORKERS", "1"))
ASR_MAX_FILE_SIZE = int(os.environ.get("ASR_MAX_FILE_SIZE", str(25 * 1024 * 1024)))  # 25 MB
ASR_REQUEST_TIMEOUT = int(os.environ.get("ASR_REQUEST_TIMEOUT", "120"))
ASR_MAX_CONCURRENCY = int(os.environ.get("ASR_MAX_CONCURRENCY", "1"))
ASR_BUSY_WAIT_TIMEOUT = int(os.environ.get("ASR_BUSY_WAIT_TIMEOUT", "600"))
ASR_MODEL_CACHE_DIR = os.environ.get("ASR_MODEL_CACHE_DIR", "/opt/portable-asr/models")
ASR_LISTEN_HOST = os.environ.get("ASR_LISTEN_HOST", "127.0.0.1")
ASR_LISTEN_PORT = int(os.environ.get("ASR_LISTEN_PORT", "9000"))
ASR_POSTPROCESS_ENABLED = os.environ.get("ASR_POSTPROCESS_ENABLED", "0") == "1"
ASR_POSTPROCESS_PROVIDER = os.environ.get("ASR_POSTPROCESS_PROVIDER", "local-llama").strip()
ASR_POSTPROCESS_BASE_URL = os.environ.get("ASR_POSTPROCESS_BASE_URL", "http://127.0.0.1:9100/v1").rstrip("/")
ASR_POSTPROCESS_MODEL = os.environ.get("ASR_POSTPROCESS_MODEL", "example-cleanup-model")
ASR_POSTPROCESS_TIMEOUT = int(os.environ.get("ASR_POSTPROCESS_TIMEOUT", "180"))
ASR_POSTPROCESS_CLI = os.environ.get("ASR_POSTPROCESS_CLI", "/opt/asr-llm/bin/llama-cli")
ASR_POSTPROCESS_CONTEXT = os.environ.get("ASR_POSTPROCESS_CONTEXT", "8192")
ASR_POSTPROCESS_CACHE_DIR = os.environ.get("ASR_POSTPROCESS_CACHE_DIR", "/opt/asr-llm/hf-cache")
ASR_POSTPROCESS_LIBRARY_PATH = os.environ.get("ASR_POSTPROCESS_LIBRARY_PATH", "/opt/asr-llm/bin")
ASR_AUDIO_POSTPROCESS_REQUIRED = os.environ.get("ASR_AUDIO_POSTPROCESS_REQUIRED", "0") == "1"
ASR_POST_WHISPER_GPU_SETTLE_SECONDS = float(os.environ.get("ASR_POST_WHISPER_GPU_SETTLE_SECONDS", "0"))
ASR_STOP_LLM_FOR_TRANSCRIPTION = os.environ.get("ASR_STOP_LLM_FOR_TRANSCRIPTION", "0") == "1"
ASR_LLM_START_RETRIES = int(os.environ.get("ASR_LLM_START_RETRIES", "2"))
ASR_LLM_CUDA_VISIBLE_DEVICES = os.environ.get("ASR_LLM_CUDA_VISIBLE_DEVICES", "").strip()
ASR_LLM_DEVICE = os.environ.get("ASR_LLM_DEVICE", "").strip()
ASR_LLM_SERVER_HOST = os.environ.get("ASR_LLM_SERVER_HOST", "127.0.0.1")
ASR_LLM_SERVER_PORT = int(os.environ.get("ASR_LLM_SERVER_PORT", "9100"))
ASR_LLM_IDLE_TIMEOUT = int(os.environ.get("ASR_LLM_IDLE_TIMEOUT", "3600"))
ASR_LLM_WATCH_INTERVAL = int(os.environ.get("ASR_LLM_WATCH_INTERVAL", "3"))
ASR_LLM_START_TIMEOUT = int(os.environ.get("ASR_LLM_START_TIMEOUT", "180"))
ASR_LLM_BUSY_WAIT_TIMEOUT = int(os.environ.get("ASR_LLM_BUSY_WAIT_TIMEOUT", "600"))
ASR_LLM_LOG_FILE = os.environ.get("ASR_LLM_LOG_FILE", "/opt/portable-asr/llama-server.log")
ASR_LLM_PID_FILE = os.environ.get("ASR_LLM_PID_FILE", "/opt/portable-asr/llama-server.pid")
ASR_LLM_CACHE_REUSE = int(os.environ.get("ASR_LLM_CACHE_REUSE", "256"))
ASR_LLM_SERVER_BIN = os.environ.get("ASR_LLM_SERVER_BIN", "/opt/asr-llm/bin/llama-server")
ASR_GPU_WATCH_ENABLED = os.environ.get("ASR_GPU_WATCH_ENABLED", "0") == "1"
ASR_GPU_WATCH_PROCESS_PATTERNS = [
    item.strip().lower()
    for item in os.environ.get("ASR_GPU_WATCH_PROCESS_PATTERNS", "ffmpeg").split(",")
    if item.strip()
]
ASR_POSTPROCESS_LOCAL_PROVIDERS = {"local-llama", "llama.cpp", "llamacpp"}
ASR_POSTPROCESS_REMOTE_PROVIDERS = {"openai-compatible", "external-openai", "remote-openai"}

MAX_TRANSCRIPTION_PROMPT_CHARS = 1000

# ---- Global state ----
_model_lock = threading.Lock()
_model_instance = None
_model_size_loaded = None
_concurrency_semaphore = threading.Semaphore(ASR_MAX_CONCURRENCY)
_llm_lock = threading.RLock()
_llm_process = None
_llm_last_used = 0.0
_llm_block_until = 0.0
_whisper_worker_lock = threading.RLock()
_whisper_worker_process = None
_qwen_worker_lock = threading.RLock()
_qwen_worker_process = None

security = HTTPBearer(auto_error=False)

class PostprocessRequest(BaseModel):
    text: str
    instruction: str | None = None

class ChatMessage(BaseModel):
    role: str
    content: object

class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = 0.0
    stream: bool | None = False
    max_tokens: int | None = None
    thinking: bool | None = True

class PostprocessUnavailable(RuntimeError):
    """Raised when transcript post-processing cannot complete."""

def _postprocess_uses_local_server() -> bool:
    return ASR_POSTPROCESS_PROVIDER.lower() in ASR_POSTPROCESS_LOCAL_PROVIDERS

def _postprocess_uses_remote_server() -> bool:
    return ASR_POSTPROCESS_PROVIDER.lower() in ASR_POSTPROCESS_REMOTE_PROVIDERS

def _gpu_watch_enabled() -> bool:
    return ASR_GPU_WATCH_ENABLED

def _validate_runtime_config() -> list[str]:
    warnings = []
    provider = ASR_POSTPROCESS_PROVIDER.lower()
    if ASR_POSTPROCESS_ENABLED and provider not in ASR_POSTPROCESS_LOCAL_PROVIDERS | ASR_POSTPROCESS_REMOTE_PROVIDERS:
        warnings.append(f"unknown ASR_POSTPROCESS_PROVIDER={ASR_POSTPROCESS_PROVIDER!r}; using configured base URL without local server management")
    if ASR_DEVICE == "cpu" and ASR_COMPUTE_TYPE == "float16":
        warnings.append("ASR_COMPUTE_TYPE=float16 is usually invalid for CPU; use int8 or int8_float32")
    return warnings

DEFAULT_POSTPROCESS_PROMPT = """Clean this transcript.

You are a conservative transcript formatter. Your job is to add punctuation, capitalization, accents, paragraph breaks, and only the safest transcript-artifact fixes while preserving the speaker's exact words.

Most important constraint:
- Preserve every meaningful word, sentence, detail, tense, mood, person, number, negation, uncertainty, and time reference from the raw transcript.
- Do not summarize, shorten, combine, simplify, reinterpret, normalize, or improve the speaker's phrasing.
- Do not change Spanish verb tenses or modes. For example, do not change present to past, conditional to indicative, subjunctive to indicative, or first person to third person.
- Do not normalize colloquial Spanish, diminutives, augmentatives, idioms, or regional wording. Preserve words such as "cortitos", "poquito", "rapidito", "grandote", "chiquitito", and similar forms exactly when they appear in the raw transcript.
- Do not normalize Spanish discourse markers or short acknowledgements. Preserve words such as "ya", "vale", "bueno", "pues", "oye", "venga", and "anda" when they appear in the raw transcript; for example, do not change "Ya, tienes razón" to "Sí, tienes razón".
- If a phrase sounds awkward but is understandable, keep it awkward and only add punctuation/capitalization.
- You may insert a short missing function word only when the omission is an obvious local ASR drop and the correction does not change word order, meaning, speaker, tense, person, or style. This includes Spanish auxiliary verbs such as "ha" in "me ha dicho".
- Never move names, subjects, objects, pronouns, or attribution phrases to make the sentence sound smoother.

Core rules:
1. Fix spelling, capitalization, punctuation, and accent errors when the intended word is clear.
2. Fix clearly bad transcription words only when the correction is obvious from the local phrase and does not change meaning, tense, or style.
3. Do not add missing words to make the text more grammatical unless the word is a short, obvious local omission such as a Spanish auxiliary, or the word was clearly spoken as a control marker or punctuation command.
4. Remove filler sounds only when they are clearly non-meaningful speech noises:
   - um, uh, eh, mmm
   Do not remove repeated words, hesitations, self-corrections, or partial thoughts if they carry meaning.
5. Convert clear spoken numbers to digits:
   - twenty-five -> 25
   - ten percent -> 10%
   - five dollars -> $5
   - doscientos euros -> 200 €
6. Keep every part in its original language. Do not translate.
7. Preserve meaning and original word order as much as possible. Do not paraphrase or rewrite style.

Paragraphing rules:
8. Paragraphing is mandatory for medium or long transcripts.
9. First identify distinct topical units, then place blank lines between them.
10. A paragraph break is formatting only; it does not count as paraphrasing, reordering, or changing style.
11. Prefer 2-5 short paragraphs over one large paragraph when the transcript includes personal commentary, questions, sports/news, work, plans, or unrelated side notes.
12. Start a new paragraph when the speaker moves to an unrelated aside, even without an explicit transition phrase.
13. Start a new paragraph at clear transition phrases such as "dicho esto", "ahora cambio de tema", "cambiando de tema", "por otro lado", "por cierto", or "en otro orden de cosas".

List formatting control rules:
- Treat `lista de ítems` / `item list` / `list of items` as a bulleted-list control only when followed by at least two clear items.
- Treat `lista numerada` / `numbered list` as a numbered-list control only when followed by at least two clear items.
- In either list type, `nuevo ítem` / `new item` starts the next item.
- `fin de lista` / `end of list` ends the list; preserve following speech as normal prose.
- Remove recognized list control phrases from the output.
- Preserve the clause immediately before the list control as the introduction and add an implicit colon. Do not invent a title when there is no introduction.
- Preserve every item's language, content, quantity, and order. Never combine, split, reorder, omit, or invent items.
- Clear comma-separated short items may become separate items. If commas belong inside an item or the enumeration is uncertain, preserve them inside the item.
- Use Markdown `- ` markers for bulleted lists and sequential `1. `, `2. ` markers for numbered lists.
- Do not treat ordinary-language mentions such as `una lista de ítems defectuosos` or `we discussed an item list` as controls.

Spoken punctuation and technical literals:
14. Replace spoken punctuation commands with symbols when they are clearly commands:
   - period / full stop / punto -> .
   - comma / coma -> ,
   - question mark / signo de interrogación -> ?
   - slash / barra -> /
   - colon / dos puntos -> :
   - semicolon / punto y coma -> ;
15. Reconstruct URLs, emails, file paths, command flags, and identifiers only when the transcript clearly contains a technical literal.
   Technical-context cues work in both English and Spanish:
   - English: variable, field, identifier, file, filename, name, token, key, column, table, class, function, method, endpoint, folder, directory, path, document.
   - Spanish: variable, campo, identificador, archivo, fichero, nombre, token, clave, columna, tabla, clase, función, método, endpoint, carpeta, directorio, ruta, documento.
16. Inside technical literals, convert spoken symbols to symbols and remove unnecessary spaces:
   - dot / punto -> .
   - slash / barra -> /
   - backslash -> \\
   - hyphen / dash / guion / guión -> -
   - underscore / guion bajo / guión bajo / barra baja -> _
   - colon / dos puntos -> :
   - at / arroba -> @
17. Do not convert these words when they are ordinary words in a sentence.
18. If uncertain whether something is a technical literal, leave it unchanged.

Question control rules:
19. If a sentence, fragment, or transcript starts with "pregunta", remove that marker and turn the remaining content into a direct question.
    - In Spanish, use opening and closing question marks: ¿...?
    - In English, use only the closing question mark: ...?
    - In any other language, use the standard question punctuation for that language.
20. Only treat "pregunta" as a control marker when it appears at the very beginning of a sentence, fragment, or transcript.
21. Do not treat "pregunta" as a control marker when it appears inside a sentence.
22. If the transcript contains "abre interrogación" and/or "cierra interrogación", replace them with ¿ and ? where appropriate.
23. If a question is already obvious from the words, add the correct question marks even without a control marker.

Expression control rules:
24. If the transcript contains a standalone laughter command, convert it to natural written laughter for the surrounding language:
   - English: "laugh" -> "hahaha"
   - Spanish: "risa" / "risas" -> "jajaja"
25. Only treat laughter commands as commands when they appear standalone, not when they are part of a normal sentence.

Safety rules:
26. Do not add new information.
27. Do not remove meaningful words, clauses, sentences, caveats, examples, repetitions, or side notes.
28. Do not change verb tense, grammatical person, modality, certainty, or chronology.
29. Do not replace diminutives or colloquial wording with more formal wording.
30. Do not guess uncommon names, brands, URLs, commands, or identifiers.
31. Keep the meaningful words in their original order.
32. If you are not confident a correction is right, leave it unchanged.
33. Prefer under-correction over over-correction.
34. Do not explain the changes.
35. Return only the cleaned transcript.

Examples:
Raw: pregunta como estas
Clean: ¿cómo estás?

Raw: Si yo tuviera tiempo lo haría mañana pero ahora no puedo
Clean: Si yo tuviera tiempo, lo haría mañana, pero ahora no puedo.

Raw: Los mensajes son cortitos y quiero dejarlos cortitos
Clean: Los mensajes son cortitos y quiero dejarlos cortitos.

Raw: Ramón me dicho que no le gusta tu actitud
Clean: Ramón me ha dicho que no le gusta tu actitud.

Raw: Ya tienes razón
Clean: Ya, tienes razón.

Raw: example dot com slash docs
Clean: example.com/docs

Raw: user arroba example punto com
Clean: user@example.com

Raw: la clase Debian guion Developer guion Install
Clean: La clase debian-developer-install.

Raw: the class Debian underscore Developer underscore Install
Clean: The class debian_developer_install.

Raw: hoy tengo que comprar lista de items plátanos, tomates
Clean: Hoy tengo que comprar:

- Plátanos
- Tomates

Raw: deployment steps numbered list run tests new item copy the file
Clean: Deployment steps:

1. Run tests
2. Copy the file

Raw: I drew a dot on the paper
Clean: I drew a dot on the paper.

Raw: risa
Clean: jajaja

Raw: laugh
Clean: hahaha

Raw: Estoy probando esto. Risas. Ahora cambio de tema y hablo del trabajo.
Clean: Estoy probando esto. Jajaja.

Ahora cambio de tema y hablo del trabajo.

Raw: Estoy revisando la GPU y el servidor. Ahora cambio completamente de tema y hablo del partido de Brasil contra Japón.
Clean: Estoy revisando la GPU y el servidor.

Ahora cambio completamente de tema y hablo del partido de Brasil contra Japón.

Raw: Estoy probando el sistema y mi mujer se ríe de mí. Pregunta tú crees que soy tonto. Dicho esto estoy viendo Brasil contra Japón. Ah por cierto llamé a mi mejor amigo.
Clean: Estoy probando el sistema y mi mujer se ríe de mí. ¿Tú crees que soy tonto?

Dicho esto, estoy viendo Brasil contra Japón.

Ah, por cierto, llamé a mi mejor amigo."""

# ---- Helpers ----

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not ASR_BEARER_TOKEN:
        raise HTTPException(status_code=500, detail="ASR_BEARER_TOKEN not configured")
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    token = credentials.credentials
    if token != ASR_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
    return token

def _message_content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == "text" and isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("content"), str):
            return content["content"]
    return str(content)

def _llm_base_url() -> str:
    return f"http://{ASR_LLM_SERVER_HOST}:{ASR_LLM_SERVER_PORT}/v1"

def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

def _pid_file_process() -> int | None:
    try:
        value = Path(ASR_LLM_PID_FILE).read_text().strip()
        pid = int(value)
    except (OSError, ValueError):
        return None
    return pid if _process_exists(pid) else None

def _llm_server_processes() -> set[int]:
    current = set()
    pid = _pid_file_process()
    if pid is not None:
        current.add(pid)
    try:
        output = subprocess.check_output(
            ["pgrep", "-af", Path(ASR_LLM_SERVER_BIN).name],
            text=True,
        )
    except subprocess.CalledProcessError:
        return current
    except OSError:
        return current
    for line in output.splitlines():
        parts = line.strip().split(" ", 1)
        if len(parts) != 2:
            continue
        try:
            candidate_pid = int(parts[0])
        except ValueError:
            continue
        cmdline = parts[1]
        first = cmdline.split(" ", 1)[0]
        if Path(first).name == Path(ASR_LLM_SERVER_BIN).name:
            current.add(candidate_pid)
    return current

def _gpu_compute_apps() -> list[tuple[int, str, str]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    apps = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            apps.append((int(parts[0]), parts[1], parts[2]))
        except ValueError:
            continue
    return apps

def _external_gpu_users() -> list[tuple[int, str, str]]:
    llama_pids = _llm_server_processes()
    own_pid = os.getpid()
    whisper_pid = None
    with _whisper_worker_lock:
        if _whisper_worker_process is not None and _whisper_worker_process.poll() is None:
            whisper_pid = _whisper_worker_process.pid
    external = []
    for pid, name, memory in _gpu_compute_apps():
        if pid in llama_pids or pid == own_pid or pid == whisper_pid:
            continue
        external.append((pid, name, memory))
    return external

def _watched_gpu_processes() -> list[tuple[int, str]]:
    if not ASR_GPU_WATCH_PROCESS_PATTERNS:
        return []
    try:
        output = subprocess.check_output(["pgrep", "-af", "."], text=True)
    except subprocess.CalledProcessError:
        return []
    except OSError:
        return []
    current_pid = os.getpid()
    matches = []
    for line in output.splitlines():
        parts = line.strip().split(" ", 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmdline = parts[1]
        if pid == current_pid or "pgrep -af" in cmdline:
            continue
        executable = cmdline.split(" ", 1)[0].lower()
        executable_name = Path(executable).name.lower()
        if any(
            pattern in executable_name or pattern in executable
            for pattern in ASR_GPU_WATCH_PROCESS_PATTERNS
        ):
            matches.append((pid, cmdline))
    return matches

def _llm_health() -> bool:
    try:
        with urllib.request.urlopen(f"{_llm_base_url()}/models", timeout=2) as response:
            return 200 <= response.status < 300
    except Exception:
        return False

def _stop_llm_server(reason: str = "requested"):
    global _llm_process
    with _llm_lock:
        pids = _llm_server_processes()
        if _llm_process is not None and _llm_process.poll() is None:
            pids.add(_llm_process.pid)
        for pid in pids:
            try:
                os.kill(pid, 15)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                print(f"[llm] Could not stop llama-server pid {pid}: {exc}")
        deadline = time.time() + 20
        while time.time() < deadline and _llm_server_processes():
            time.sleep(0.25)
        for pid in _llm_server_processes():
            try:
                os.kill(pid, 9)
            except Exception:
                pass
        _llm_process = None
        try:
            Path(ASR_LLM_PID_FILE).unlink(missing_ok=True)
        except Exception:
            pass
        print(f"[llm] llama-server stopped ({reason})")

def _start_llm_server():
    global _llm_process
    with _llm_lock:
        if _llm_health():
            return
        if _gpu_watch_enabled():
            external = _external_gpu_users()
            watched = _watched_gpu_processes()
            if watched:
                names = ", ".join(f"{pid}:{cmd[:80]}" for pid, cmd in watched)
                raise RuntimeError(f"GPU reserved for watched process: {names}")
            if external:
                names = ", ".join(f"{pid}:{name}:{memory}" for pid, name, memory in external)
                raise RuntimeError(f"GPU busy with external process: {names}")
        env = os.environ.copy()
        env["HF_HOME"] = ASR_POSTPROCESS_CACHE_DIR
        env["HUGGINGFACE_HUB_CACHE"] = f"{ASR_POSTPROCESS_CACHE_DIR}/hub"
        env["LLAMA_CACHE"] = ASR_POSTPROCESS_CACHE_DIR
        env["LD_LIBRARY_PATH"] = ASR_POSTPROCESS_LIBRARY_PATH
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        if ASR_LLM_CUDA_VISIBLE_DEVICES:
            env["CUDA_VISIBLE_DEVICES"] = ASR_LLM_CUDA_VISIBLE_DEVICES
        cmd = [
            ASR_LLM_SERVER_BIN,
            "-hf", ASR_POSTPROCESS_MODEL,
            "--offline",
            "--host", ASR_LLM_SERVER_HOST,
            "--port", str(ASR_LLM_SERVER_PORT),
            "-c", ASR_POSTPROCESS_CONTEXT,
            "-ctk", "q8_0",
            "-ctv", "q8_0",
            "-ngl", "all",
            "-np", "1",
            "--split-mode", "none",
            "-fa", "on",
            "--jinja",
            "--reasoning", "off",
            "--reasoning-format", "deepseek",
            "--no-webui",
            "--no-slots",
            "--log-disable",
        ]
        if ASR_LLM_DEVICE:
            cmd.extend(["--device", ASR_LLM_DEVICE])
        log_file = open(ASR_LLM_LOG_FILE, "ab", buffering=0)
        _llm_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        Path(ASR_LLM_PID_FILE).write_text(f"{_llm_process.pid}\n")
        deadline = time.time() + ASR_LLM_START_TIMEOUT
        while time.time() < deadline:
            if _llm_process.poll() is not None:
                raise RuntimeError(f"llama-server exited with {_llm_process.returncode}")
            if _llm_health():
                print(f"[llm] llama-server ready on {_llm_base_url()}")
                return
            time.sleep(1)
        _stop_llm_server("startup timeout")
        raise RuntimeError("llama-server startup timed out")

def _ensure_llm_server():
    global _llm_last_used
    _llm_last_used = time.monotonic()
    deadline = time.monotonic() + ASR_LLM_BUSY_WAIT_TIMEOUT
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if time.monotonic() < _llm_block_until:
            time.sleep(1)
            continue
        for attempt in range(ASR_LLM_START_RETRIES + 1):
            try:
                _start_llm_server()
                return
            except RuntimeError as exc:
                last_error = exc
                message = str(exc)
                if "GPU busy" in message or "watched process" in message:
                    break
                _stop_llm_server(f"retry after startup failure: {message}")
                if attempt >= ASR_LLM_START_RETRIES:
                    raise
                time.sleep(max(1, ASR_POST_WHISPER_GPU_SETTLE_SECONDS))
        time.sleep(1)
    if last_error is not None:
        raise TimeoutError(f"timed out waiting for GPU to start llama-server after: {last_error}")
    raise TimeoutError("timed out waiting for GPU to start llama-server")

def _llm_chat(
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.0,
    thinking: bool = False,
    postprocess_mode: bool = False,
) -> dict:
    global _llm_last_used
    _llm_last_used = time.monotonic()
    if _postprocess_uses_local_server():
        _ensure_llm_server()
    payload = {
        "model": ASR_POSTPROCESS_MODEL,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "repeat_penalty": 1.18,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if postprocess_mode:
        payload["stop"] = ["<|im_end|>", "<|endoftext|>"]
        payload["cache_prompt"] = True
        payload["cache_reuse"] = ASR_LLM_CACHE_REUSE
        payload["timings_per_token"] = True
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{_llm_base_url()}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=ASR_POSTPROCESS_TIMEOUT) as response:
        result = json.loads(response.read().decode("utf-8"))
    _llm_last_used = time.monotonic()
    return result

def _log_llm_postprocess_timing(result: dict, raw_text: str, corrected_text: str) -> None:
    timings = result.get("timings") or {}
    usage = result.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    fields = {
        "raw_chars": len(raw_text),
        "corrected_chars": len(corrected_text),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens", timings.get("cache_n")),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_ms": round(timings["prompt_ms"]) if "prompt_ms" in timings else None,
        "predicted_ms": round(timings["predicted_ms"]) if "predicted_ms" in timings else None,
        "prompt_tps": round(timings["prompt_per_second"], 1) if "prompt_per_second" in timings else None,
        "predicted_tps": round(timings["predicted_per_second"], 1) if "predicted_per_second" in timings else None,
    }
    print(
        "[llm] postprocess timing "
        + " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    )

def _llm_watcher_loop():
    global _llm_block_until
    while True:
        try:
            if not _gpu_watch_enabled():
                time.sleep(max(1, ASR_LLM_WATCH_INTERVAL))
                continue
            external = _external_gpu_users()
            watched = _watched_gpu_processes()
            watched_external = [
                item for item in external
                if any(
                    pattern in Path(item[1].lower()).name or pattern in item[1].lower()
                    for pattern in ASR_GPU_WATCH_PROCESS_PATTERNS
                )
            ]
            if watched or watched_external:
                _llm_block_until = time.monotonic() + max(ASR_LLM_WATCH_INTERVAL * 2, 10)
                if _llm_server_processes():
                    _stop_llm_server("watched GPU process activity")
            if (
                ASR_LLM_IDLE_TIMEOUT > 0
                and _llm_last_used
                and _llm_server_processes()
                and time.monotonic() - _llm_last_used > ASR_LLM_IDLE_TIMEOUT
            ):
                _stop_llm_server(f"idle for {ASR_LLM_IDLE_TIMEOUT}s")
        except Exception as exc:
            print(f"[llm] watcher error: {exc}")
        time.sleep(max(1, ASR_LLM_WATCH_INTERVAL))

def load_model():
    """Load the Whisper model (thread-safe, lazy)."""
    global _model_instance, _model_size_loaded
    with _model_lock:
        if _model_instance is not None and _model_size_loaded == ASR_MODEL_SIZE:
            return _model_instance
        # Release previous model if size changed
        if _model_instance is not None:
            del _model_instance
            _model_instance = None
            gc.collect()
            _cleanup_gpu()
        # Import and load
        from faster_whisper import WhisperModel
        print(f"[asr] Loading model '{ASR_MODEL_SIZE}' on {ASR_DEVICE} ({ASR_COMPUTE_TYPE})...")
        t0 = time.time()
        _model_instance = WhisperModel(
            ASR_MODEL_SIZE,
            device=ASR_DEVICE,
            compute_type=ASR_COMPUTE_TYPE,
            cpu_threads=ASR_CPU_THREADS,
            num_workers=ASR_NUM_WORKERS,
            download_root=ASR_MODEL_CACHE_DIR,
        )
        _model_size_loaded = ASR_MODEL_SIZE
        print(f"[asr] Model loaded in {time.time() - t0:.2f}s")
        return _model_instance

def _cleanup_gpu():
    """Attempt to release GPU memory."""
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
    try:
        import ctypes
        libcudart = ctypes.CDLL("libcudart.so")
        libcudart.cudaDeviceSynchronize()
    except Exception:
        pass
    gc.collect()

def _settle_after_whisper_unload():
    """Give accelerator memory a short window to settle before starting the postprocess LLM."""
    _cleanup_gpu()
    if ASR_POST_WHISPER_GPU_SETTLE_SECONDS > 0:
        time.sleep(ASR_POST_WHISPER_GPU_SETTLE_SECONDS)

def unload_model():
    """Force-unload the model and release GPU memory."""
    global _model_instance, _model_size_loaded
    with _model_lock:
        if _model_instance is not None:
            print("[asr] Unloading model and releasing GPU memory...")
            del _model_instance
            _model_instance = None
            _model_size_loaded = None
            _cleanup_gpu()
            print("[asr] GPU memory released")

def _normalize_transcription_prompt(prompt: str | None) -> str | None:
    if prompt is None:
        return None
    normalized = prompt.strip()
    if not normalized:
        return None
    return normalized[:MAX_TRANSCRIPTION_PROMPT_CHARS]

def _transcribe_file_once(audio_path: str, language: str | None, temperature: float, prompt: str | None = None) -> list[str]:
    """Transcribe one file in the current process. Used by the subprocess entrypoint."""
    from faster_whisper import WhisperModel

    model_obj = WhisperModel(
        ASR_MODEL_SIZE,
        device=ASR_DEVICE,
        compute_type=ASR_COMPUTE_TYPE,
        cpu_threads=ASR_CPU_THREADS,
        num_workers=ASR_NUM_WORKERS,
        download_root=ASR_MODEL_CACHE_DIR,
    )
    segments, _ = model_obj.transcribe(
        audio_path,
        language=language or None,
        temperature=temperature,
        beam_size=ASR_BEAM_SIZE,
        word_timestamps=False,
        condition_on_previous_text=False,
        initial_prompt=prompt,
        vad_filter=ASR_VAD_FILTER,
        vad_parameters={"min_silence_duration_ms": ASR_VAD_MIN_SILENCE_MS},
    )
    return [seg.text for seg in segments]

def _run_transcription_subprocess(audio_path: str, language: str | None, temperature: float, prompt: str | None = None) -> tuple[list[str], dict]:
    """Run Whisper in a short-lived child process so CUDA state exits cleanly."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--transcribe-once",
        audio_path,
        language or "",
        str(temperature),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if prompt is not None:
        env["ASR_TRANSCRIBE_ONCE_PROMPT"] = prompt
    if ASR_WHISPER_CUDA_VISIBLE_DEVICES:
        env["CUDA_VISIBLE_DEVICES"] = ASR_WHISPER_CUDA_VISIBLE_DEVICES
    completed = subprocess.run(
        cmd,
        env=env,
        text=True,
        capture_output=True,
        timeout=ASR_REQUEST_TIMEOUT,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "unknown transcription subprocess error").strip()
        raise RuntimeError(detail[-1000:])
    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        segments = payload.get("segments")
        if isinstance(segments, list):
            timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}
            return [str(segment) for segment in segments], timings
    raise RuntimeError("transcription subprocess returned no JSON segments")

def _timing_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)

def _whisper_worker_env() -> dict:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if ASR_WHISPER_CUDA_VISIBLE_DEVICES:
        env["CUDA_VISIBLE_DEVICES"] = ASR_WHISPER_CUDA_VISIBLE_DEVICES
    return env

def _stop_whisper_worker(reason: str = "requested"):
    global _whisper_worker_process
    with _whisper_worker_lock:
        process = _whisper_worker_process
        _whisper_worker_process = None
        if process is None:
            return
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        print(f"[asr] Whisper worker stopped ({reason})")

def _start_whisper_worker_locked():
    global _whisper_worker_process
    if _whisper_worker_process is not None and _whisper_worker_process.poll() is None:
        return
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--transcribe-worker",
    ]
    _whisper_worker_process = subprocess.Popen(
        cmd,
        env=_whisper_worker_env(),
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    print(f"[asr] Whisper worker started pid={_whisper_worker_process.pid}")

def _read_whisper_worker_response(process: subprocess.Popen, deadline: float, request_id: str, worker_name: str = "Whisper") -> dict:
    assert process.stdout is not None
    fd = process.stdout.fileno()
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], min(1.0, max(0.0, deadline - time.monotonic())))
        if not ready:
            continue
        line = process.stdout.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            print(f"[asr:{worker_name.lower()}-worker] {line}")
            continue
        if isinstance(payload, dict):
            if payload.get("ready") is True:
                continue
            if payload.get("request_id") != request_id:
                print(f"[asr:{worker_name.lower()}-worker] ignoring stale response for request {payload.get('request_id')}")
                continue
            return payload
    raise TimeoutError(f"timed out waiting for {worker_name} worker response")

def _run_transcription_persistent(audio_path: str, language: str | None, temperature: float, prompt: str | None = None) -> tuple[list[str], dict]:
    """Transcribe through a resident worker so Whisper stays loaded on its GPU."""
    with _whisper_worker_lock:
        request_started = time.monotonic()
        _start_whisper_worker_locked()
        assert _whisper_worker_process is not None
        process = _whisper_worker_process
        if process.stdin is None:
            _stop_whisper_worker("missing stdin")
            raise RuntimeError("Whisper worker stdin unavailable")
        request_id = uuid.uuid4().hex
        request = {
            "request_id": request_id,
            "audio_path": audio_path,
            "language": language or "",
            "temperature": temperature,
            "prompt": prompt or "",
        }
        try:
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            payload = _read_whisper_worker_response(process, time.monotonic() + ASR_REQUEST_TIMEOUT, request_id)
        except Exception:
            _stop_whisper_worker("request failure")
            raise
        if process.poll() is not None:
            _stop_whisper_worker("worker exited")
            raise RuntimeError("Whisper worker exited")
        if payload.get("ok") is not True:
            raise RuntimeError(str(payload.get("error") or "Whisper worker failed"))
        segments = payload.get("segments")
        if not isinstance(segments, list):
            raise RuntimeError("Whisper worker returned no JSON segments")
        timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}
        timings["whisper_worker_roundtrip_ms"] = round((time.monotonic() - request_started) * 1000)
        return [str(segment) for segment in segments], timings

def _run_transcription(audio_path: str, language: str | None, temperature: float, prompt: str | None = None) -> tuple[list[str], dict]:
    if ASR_WHISPER_KEEP_WARM:
        return _run_transcription_persistent(audio_path, language, temperature, prompt)
    return _run_transcription_subprocess(audio_path, language, temperature, prompt)

def _transcription_engine(model: str | None) -> str:
    if model == ASR_QWEN_MODEL_ID:
        if not ASR_QWEN_ENABLED:
            raise HTTPException(status_code=400, detail="Qwen3-ASR is not enabled")
        return "qwen"
    # Preserve legacy clients: model names other than the Qwen selector used Whisper.
    return "whisper"

def _stop_qwen_worker(reason: str = "requested"):
    global _qwen_worker_process
    with _qwen_worker_lock:
        process = _qwen_worker_process
        _qwen_worker_process = None
        if process is None:
            return
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:
                process.kill()
        print(f"[asr] Qwen worker stopped ({reason})")

def _start_qwen_worker_locked():
    global _qwen_worker_process
    if _qwen_worker_process is not None and _qwen_worker_process.poll() is None:
        return
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    device = ASR_QWEN_CUDA_VISIBLE_DEVICES or ASR_WHISPER_CUDA_VISIBLE_DEVICES
    if device:
        env["CUDA_VISIBLE_DEVICES"] = device
    env["ASR_QWEN_CHECKPOINT"] = ASR_QWEN_CHECKPOINT
    _qwen_worker_process = subprocess.Popen(
        [ASR_QWEN_PYTHON, str(Path(__file__).with_name("qwen_asr_worker.py"))],
        env=env,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    print(f"[asr] Qwen worker started pid={_qwen_worker_process.pid}")

def _run_qwen_transcription(audio_path: str, language: str | None, temperature: float, prompt: str | None = None) -> tuple[list[str], dict]:
    with _qwen_worker_lock:
        request_started = time.monotonic()
        _start_qwen_worker_locked()
        assert _qwen_worker_process is not None
        process = _qwen_worker_process
        if process.stdin is None:
            _stop_qwen_worker("missing stdin")
            raise RuntimeError("Qwen worker stdin unavailable")
        request_id = uuid.uuid4().hex
        request = {
            "request_id": request_id,
            "audio_path": audio_path,
            "language": language or "",
            "prompt": prompt or "",
        }
        try:
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            payload = _read_whisper_worker_response(
                process, time.monotonic() + ASR_REQUEST_TIMEOUT, request_id, "Qwen"
            )
        except Exception:
            _stop_qwen_worker("request failure")
            raise
        if process.poll() is not None:
            _stop_qwen_worker("worker exited")
            raise RuntimeError("Qwen worker exited")
        if payload.get("ok") is not True:
            raise RuntimeError(str(payload.get("error") or "Qwen worker failed"))
        segments = payload.get("segments")
        if not isinstance(segments, list):
            raise RuntimeError("Qwen worker returned no JSON segments")
        timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}
        timings["qwen_worker_roundtrip_ms"] = _timing_ms(request_started)
        return [str(segment) for segment in segments], timings

def _transcribe_worker_loop():
    model_obj = load_model()
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            request_id = request.get("request_id")
            transcribe_started = time.monotonic()
            segments, _ = model_obj.transcribe(
                request["audio_path"],
                language=request.get("language") or None,
                temperature=float(request.get("temperature") or 0.0),
                beam_size=ASR_BEAM_SIZE,
                word_timestamps=False,
                condition_on_previous_text=False,
                initial_prompt=request.get("prompt") or None,
                vad_filter=ASR_VAD_FILTER,
                vad_parameters={"min_silence_duration_ms": ASR_VAD_MIN_SILENCE_MS},
            )
            segment_texts = [seg.text for seg in segments]
            print(
                json.dumps(
                    {
                        "request_id": request_id,
                        "ok": True,
                        "segments": segment_texts,
                        "timings": {
                            "whisper_transcribe_ms": round((time.monotonic() - transcribe_started) * 1000),
                        },
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            request_id = None
            try:
                request_id = request.get("request_id")
            except Exception:
                pass
            print(json.dumps({"request_id": request_id, "ok": False, "error": str(exc)[-1000:]}, ensure_ascii=False), flush=True)

def _strip_reasoning_blocks(text: str) -> str:
    """Remove common reasoning wrappers if a thinking model ignores disable flags."""
    while "<think>" in text and "</think>" in text:
        before, rest = text.split("<think>", 1)
        _, after = rest.split("</think>", 1)
        text = before + after
    for marker in ("<|channel>thought", "<channel|>"):
        text = text.replace(marker, "")
    return text.strip().strip('"').strip()

def _remove_question_directives(text: str) -> str:
    """Remove sentence-leading pregunta markers if the LLM leaves them."""
    directive = re.compile(r'(^|(?<=[.!?])\s+)(pregunta)\s*[,;:\-]*\s*', re.IGNORECASE)
    return directive.sub(lambda match: match.group(1), text).strip()

def _isolated_expression_command_value(text: str) -> str | None:
    """Return expression text only when the whole text is an isolated command."""
    normalized = text.strip()
    if re.fullmatch(r'(?i)(risa|risas)[.!?¡¿]*', normalized):
        return "jajaja"
    if re.fullmatch(r'(?i)laugh[.!?¡¿]*', normalized):
        return "hahaha"
    return None

def _apply_isolated_expression_command(text: str) -> str:
    """Apply expression commands only when the whole text is an isolated command."""
    replacement = _isolated_expression_command_value(text)
    if replacement is not None:
        return replacement
    return text

def _apply_isolated_expression_sentences(text: str) -> str:
    """Apply expression commands when they are isolated as their own sentence or line."""
    def replace_match(match: re.Match) -> str:
        command = match.group("command")
        replacement = _isolated_expression_command_value(command)
        if replacement is None:
            return match.group(0)
        return f"{match.group('prefix')}{replacement}{match.group('suffix')}"

    return re.sub(
        r'(?P<prefix>^|(?<=[.!?¡¿])\s+|\n+)(?P<command>risas|risa|laugh)(?P<suffix>[.!?¡¿]*)(?=\s*(?:$|[A-ZÁÉÍÓÚÜÑ¿¡]|\n))',
        replace_match,
        text,
        flags=re.IGNORECASE,
    )

def _remove_quoted_and_empty_wrapper_lines(text: str) -> str:
    """Drop wrapper/quote lines while preserving intentional paragraph breaks."""
    lines = [line.rstrip() for line in text.splitlines()]
    kept = [line for line in lines if not line.lstrip().startswith(">")]
    text = "\n".join(kept).strip()
    return re.sub(r'\n{3,}', '\n\n', text)

def _apply_explicit_paragraph_breaks(text: str) -> str:
    """Insert paragraph breaks before explicit topic-transition phrases."""
    transition = (
        r'dicho esto|ahora cambio(?: completamente)? de tema|cambiando de tema|'
        r'por otro lado|(?:ah,?\s+)?por cierto|en otro orden de cosas'
    )
    text = re.sub(
        rf'(?i)(?<=[.!?¡¿])\s+(?=({transition})\b)',
        '\n\n',
        text,
    )
    return re.sub(r'\n{3,}', '\n\n', text).strip()

_TECH_TOKEN_PATTERN = r"[^\W_]+"
_TECH_SPOKEN_SYMBOL_PATTERN = (
    r"underscore|(?:guion|guión|barra)\s+baj[ao]|"
    r"hyphen|dash|guion|guión|"
    r"dot|punto|slash|barra|backslash|colon|dos\s+puntos|at|arroba"
)
_TECHNICAL_CUE_PATTERN = (
    r"variable|field|campo|identifier|identificador|file|archivo|fichero|"
    r"filename|name|nombre|token|key|clave|column|columna|table|tabla|"
    r"class|clase|function|función|funcion|method|método|metodo|endpoint|"
    r"folder|carpeta|directory|directorio|path|ruta|document|documento"
)
_TECH_LITERAL_SEQUENCE_RE = re.compile(
    rf"(?<!\w){_TECH_TOKEN_PATTERN}(?:\s+(?:{_TECH_SPOKEN_SYMBOL_PATTERN})\s+{_TECH_TOKEN_PATTERN})+(?!\w)",
    re.IGNORECASE,
)

def _spoken_symbol_to_literal(symbol: str) -> str:
    normalized = _normalized_plain_text(symbol)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized in {"underscore", "guion bajo", "barra baja", "guion baja", "barra bajo"}:
        return "_"
    if normalized in {"hyphen", "dash", "guion"}:
        return "-"
    if normalized in {"dot", "punto"}:
        return "."
    if normalized in {"slash", "barra"}:
        return "/"
    if normalized == "backslash":
        return "\\"
    if normalized in {"colon", "dos puntos"}:
        return ":"
    if normalized in {"at", "arroba"}:
        return "@"
    return symbol

def _technical_literal_parts(sequence: str) -> tuple[list[str], list[str]]:
    parts = re.split(
        rf"\s+({_TECH_SPOKEN_SYMBOL_PATTERN})\s+",
        sequence,
        flags=re.IGNORECASE,
    )
    return parts[::2], parts[1::2]

def _technical_literal_value(words: list[str], spoken_symbols: list[str]) -> str:
    rebuilt = [words[0].lower()]
    for symbol, word in zip(spoken_symbols, words[1:]):
        rebuilt.extend((_spoken_symbol_to_literal(symbol), word.lower()))
    return "".join(rebuilt)

def _technical_literal_pattern(
    words: list[str],
    separators: list[str],
) -> re.Pattern:
    pattern = re.escape(words[0])
    for separator, word in zip(separators, words[1:]):
        pattern += separator + re.escape(word)
    return re.compile(rf"(?<!\w){pattern}(?!\w)", re.IGNORECASE)

def _joined_spoken_symbol_separator(symbol: str) -> str:
    joiner = r"\s*[-_]\s*"
    symbol_words = re.split(r"\s+", symbol.strip())
    return joiner + joiner.join(re.escape(word) for word in symbol_words) + joiner

def _spoken_literal_is_high_confidence(
    raw_text: str,
    match: re.Match,
    words: list[str],
    spoken_symbols: list[str],
) -> bool:
    if len(spoken_symbols) > 1 or any(any(char.isdigit() for char in word) for word in words):
        return True
    if re.fullmatch(
        rf"\s*{re.escape(match.group(0))}\s*[.!?¡¿]*\s*",
        raw_text,
        re.IGNORECASE,
    ):
        return True
    return bool(
        re.search(
            rf"(?i)\b(?:{_TECHNICAL_CUE_PATTERN})\b"
            r"(?:\s+(?:is|called|named|es|llamad[oa]|denominad[oa]))?\s*$",
            raw_text[:match.start()],
        )
    )

def _restore_low_confidence_spoken_literal(
    raw_match: re.Match,
    words: list[str],
    spoken_symbols: list[str],
    corrected: str,
) -> str:
    """Restore ordinary spoken-symbol words when cleanup treats them as commands."""
    separator_patterns = []
    for spoken in spoken_symbols:
        literal = _spoken_symbol_to_literal(spoken)
        variants = [
            rf"\s*{re.escape(literal)}\s*",
            r"\s+",
        ]
        spoken_words = re.split(r"\s+", spoken.strip())
        spoken_words_with_spaces = r"\s+".join(
            re.escape(word)
            for word in spoken_words
        )
        variants.extend(
            (
                rf"\s+{spoken_words_with_spaces}\s*{re.escape(literal)}\s+",
                rf"\s*{re.escape(literal)}\s*{spoken_words_with_spaces}\s+",
            )
        )
        if len(spoken_words) > 1:
            joined_words = r"\s*[-_]\s*".join(
                re.escape(word)
                for word in spoken_words
            )
            variants.extend(
                (
                    rf"\s+{joined_words}\s+",
                    _joined_spoken_symbol_separator(spoken),
                )
            )
        separator_patterns.append(rf"(?:{'|'.join(variants)})")

    raw_sequence = raw_match.group(0)

    def restore_match(corrected_match: re.Match) -> str:
        if corrected_match.group(0)[:1].isupper() and raw_sequence[:1].islower():
            return raw_sequence[:1].upper() + raw_sequence[1:]
        return raw_sequence

    return _technical_literal_pattern(words, separator_patterns).sub(
        restore_match,
        corrected,
        count=1,
    )

def _repair_spoken_technical_literals(raw_text: str, corrected: str) -> str:
    """Align spoken separators with matching words in the corrected transcript."""
    for match in _TECH_LITERAL_SEQUENCE_RE.finditer(raw_text):
        words, spoken_symbols = _technical_literal_parts(match.group(0))
        value = _technical_literal_value(words, spoken_symbols)
        literal_symbols = [_spoken_symbol_to_literal(symbol) for symbol in spoken_symbols]
        high_confidence = _spoken_literal_is_high_confidence(
            raw_text,
            match,
            words,
            spoken_symbols,
        )
        if not high_confidence:
            corrected = _restore_low_confidence_spoken_literal(
                match,
                words,
                spoken_symbols,
                corrected,
            )
            continue
        written_separators = [
            rf"(?:\s*[-_]\s*|{_joined_spoken_symbol_separator(spoken)}|\s+)"
            if literal in {"-", "_"}
            else rf"\s*{re.escape(literal)}\s*"
            for literal, spoken in zip(literal_symbols, spoken_symbols)
        ]
        corrected = _technical_literal_pattern(words, written_separators).sub(value, corrected)

        spoken_separators = [
            rf"\s+(?:{_TECH_SPOKEN_SYMBOL_PATTERN})\s+"
            for _ in spoken_symbols
        ]
        corrected = _technical_literal_pattern(words, spoken_separators).sub(value, corrected)
    return corrected

def _lowercase_joined_technical_literals(text: str) -> str:
    """Keep identifier/path-like words lowercase across hyphen/underscore joins."""
    word = r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+"

    def replace_match(match: re.Match) -> str:
        return match.group(0).lower()

    return re.sub(rf"\b{word}(?:[_-]{word})+\b", replace_match, text)

def _repair_technical_literal_format(raw_text: str, corrected: str) -> str:
    """Apply deterministic formatting for spoken technical literals."""
    corrected = _repair_spoken_technical_literals(raw_text, corrected)
    return _lowercase_joined_technical_literals(corrected)

def _protect_non_isolated_expression_rewrite(raw_text: str, corrected: str) -> str:
    """Reject laughter-only rewrites when the raw text was not an isolated command."""
    if _isolated_expression_command_value(raw_text) is not None:
        return corrected
    if (
        re.fullmatch(r'(?i)(jajaja|hahaha)[.!?¡¿]*', corrected.strip())
        and re.search(r'(?i)\b(risa|risas|laugh)\b', raw_text)
    ):
        return raw_text
    return corrected

def _spanish_diminutive_tokens(text: str) -> set[str]:
    """Return likely Spanish diminutive/colloquial tokens that should be preserved exactly."""
    tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", text)
    diminutive = re.compile(
        r"(?i).{3,}(?:"
        r"it[oa]s?|"
        r"cit[oa]s?|"
        r"ecit[oa]s?|"
        r"ill[oa]s?|"
        r"cill[oa]s?|"
        r"ic[oa]s?"
        r")$"
    )
    return {token.lower() for token in tokens if diminutive.fullmatch(token)}

def _protect_spanish_diminutive_rewrite(raw_text: str, corrected: str) -> str:
    """Reject rewrites that drop or normalize Spanish diminutive-style words."""
    raw_diminutives = _spanish_diminutive_tokens(raw_text)
    if not raw_diminutives:
        return corrected
    corrected_tokens = {token.lower() for token in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", corrected)}
    if raw_diminutives - corrected_tokens:
        print(f"[asr] post-processing removed diminutive(s): {', '.join(sorted(raw_diminutives - corrected_tokens))}; returning raw transcript")
        return raw_text
    return corrected

def _normalized_plain_text(text: str) -> str:
    """Return lowercased text with accents removed for conservative comparisons."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char)).lower()

def _normalized_word_tokens(text: str) -> list[str]:
    """Return normalized word tokens."""
    return re.findall(r"[a-z0-9]+", _normalized_plain_text(text))

_LIST_TRIGGER_RE = re.compile(
    r"(?P<numbered>\b(?:lista\s+numerada|numbered\s+list)\b)|"
    r"(?P<bulleted>\b(?:lista\s+de\s+[íi]tems?|item\s+list|list\s+of\s+items?)\b)",
    re.IGNORECASE,
)
_LIST_ITEM_SEPARATOR_RE = re.compile(
    r"\b(?:nuevo\s+[íi]tem|new\s+item)\b",
    re.IGNORECASE,
)
_LIST_END_RE = re.compile(
    r"\b(?:fin\s+de\s+(?:la\s+)?lista|end\s+of\s+(?:the\s+)?list)\b",
    re.IGNORECASE,
)
_FORMATTED_LIST_LINE_RE = re.compile(
    r"^\s*(?P<marker>[-*+]|\d+[.)])\s+(?P<content>\S.*)$",
    re.MULTILINE,
)

def _clear_comma_list_items(body: str) -> list[str] | None:
    """Return conservative comma-separated items, or None for prose-like text."""
    parts = [part.strip() for part in re.split(r"\s*,\s*", body)]
    if len(parts) < 2 or any(not part for part in parts):
        return None
    clause_starters = {
        "although", "because", "but", "if", "that", "when", "which", "while",
        "aunque", "pero", "porque", "que", "cuando", "mientras", "si",
    }
    for part in parts:
        words = _normalized_word_tokens(part)
        if not words or len(words) > 8 or words[0] in clause_starters:
            return None
        if re.search(r"[.!?¡¿]", part[:-1]):
            return None
    return parts

def _list_control_parts(text: str) -> tuple[str, str, list[str], str] | None:
    """Parse one valid bilingual list-control region from text."""
    trigger = _LIST_TRIGGER_RE.search(text)
    if trigger is None:
        return None
    mode = "numbered" if trigger.group("numbered") else "bulleted"
    prefix = text[:trigger.start()].strip()
    remainder = text[trigger.end():]
    end = _LIST_END_RE.search(remainder)
    if end is None:
        body = remainder
        tail = ""
    else:
        body = remainder[:end.start()]
        tail = remainder[end.end():].strip(" \t\r\n,;:.-–—")
    body = body.strip(" \t\r\n,;:-–—")
    if not body:
        return None

    if _LIST_ITEM_SEPARATOR_RE.search(body):
        items = [
            part.strip(" \t\r\n,;")
            for part in _LIST_ITEM_SEPARATOR_RE.split(body)
        ]
        if len(items) < 2 or any(not item for item in items):
            return None
    else:
        items = _clear_comma_list_items(body)
        if items is None:
            return None
    return mode, prefix, items, tail

def _list_introduction(prefix: str) -> str:
    """Add an implicit colon to a spoken list introduction."""
    prefix = prefix.strip()
    if not prefix:
        return ""
    if prefix.endswith(("?", "!", "¿", "¡")):
        return prefix
    prefix = re.sub(r"[\s,;:.\-–—]+$", "", prefix)
    return f"{prefix}:" if prefix else ""

def _prepare_formatted_list_transcript(text: str) -> str:
    """Convert valid bilingual list controls into deterministic Markdown."""
    parsed = _list_control_parts(text)
    if parsed is None:
        return text
    mode, prefix, items, tail = parsed
    if mode == "bulleted":
        list_text = "\n".join(f"- {item}" for item in items)
    else:
        list_text = "\n".join(
            f"{index}. {item}" for index, item in enumerate(items, start=1)
        )
    introduction = _list_introduction(prefix)
    sections = [section for section in (introduction, list_text, tail) if section]
    return "\n\n".join(sections)

def _meaningful_order_tokens(text: str) -> list[str]:
    text = _normalized_plain_text(text)
    text = re.sub(
        r"\b(full stop|question mark|signo de interrogacion|(?:guion|barra) baj[ao]|"
        r"dos puntos|abre interrogacion|cierra interrogacion)\b",
        " ",
        text,
    )
    ignored = {
        "um", "uh", "eh", "mmm",
        "period", "full", "stop", "comma", "question", "mark",
        "slash", "backslash", "hyphen", "dash", "underscore",
        "colon", "semicolon", "dot", "at",
        "punto", "coma", "interrogacion", "barra",
        "guion", "arroba", "pregunta",
        "abre", "cierra",
    }
    return [token for token in re.findall(r"[a-z0-9]+", text) if token not in ignored]

def _lcs_length(left: list[str], right: list[str]) -> int:
    """Return the longest common subsequence length using two rolling rows."""
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]

def _formatted_list_items(text: str, mode: str) -> list[str] | None:
    """Extract items only when Markdown markers match requested list type."""
    matches = list(_FORMATTED_LIST_LINE_RE.finditer(text))
    if len(matches) < 2:
        return None
    markers = [match.group("marker") for match in matches]
    if mode == "bulleted":
        if any(marker[0].isdigit() for marker in markers):
            return None
    else:
        if any(not marker[0].isdigit() for marker in markers):
            return None
        numbers = [int(re.match(r"\d+", marker).group(0)) for marker in markers]
        if numbers != list(range(1, len(numbers) + 1)):
            return None
    return [match.group("content").strip() for match in matches]

def _protect_formatted_list_rewrite(raw_text: str, prepared_text: str, corrected: str) -> str:
    """Reject list output that loses, invents, reorders, or restyles items."""
    if prepared_text == raw_text:
        return corrected
    parsed = _list_control_parts(raw_text)
    if parsed is None:
        return corrected
    mode = parsed[0]
    expected_items = _formatted_list_items(prepared_text, mode)
    corrected_items = _formatted_list_items(corrected, mode)
    if expected_items is None or corrected_items is None or len(expected_items) != len(corrected_items):
        print("[asr] post-processing changed formatted list structure; returning prepared list")
        return prepared_text

    expected_item_tokens = _normalized_word_tokens(" ".join(expected_items))
    corrected_item_tokens = _normalized_word_tokens(" ".join(corrected_items))
    item_lcs = _lcs_length(expected_item_tokens, corrected_item_tokens)
    expected_ratio = item_lcs / max(1, len(expected_item_tokens))
    corrected_ratio = item_lcs / max(1, len(corrected_item_tokens))
    if expected_ratio < 0.8 or corrected_ratio < 0.8:
        print("[asr] post-processing changed formatted list content/order; returning prepared list")
        return prepared_text

    prepared_tokens = _meaningful_order_tokens(prepared_text)
    corrected_tokens = _meaningful_order_tokens(corrected)
    overall_lcs = _lcs_length(prepared_tokens, corrected_tokens)
    if (
        overall_lcs / max(1, len(prepared_tokens)) < 0.8
        or overall_lcs / max(1, len(corrected_tokens)) < 0.8
    ):
        print("[asr] post-processing changed formatted list context; returning prepared list")
        return prepared_text
    return corrected

def _protect_meaningful_order_rewrite(raw_text: str, corrected: str) -> str:
    """Reject broad rewrites that reorder or replace too many meaningful words."""
    raw_tokens = _meaningful_order_tokens(raw_text)
    if not raw_tokens:
        return corrected
    corrected_tokens = _meaningful_order_tokens(corrected)
    if not corrected_tokens:
        return raw_text
    preserved_ratio = _lcs_length(raw_tokens, corrected_tokens) / len(raw_tokens)
    if preserved_ratio < 0.8:
        print(
            "[asr] post-processing changed meaningful word order/content "
            f"too much ({preserved_ratio:.2f}); returning raw transcript"
        )
        return raw_text
    return corrected

def _spanish_pronoun_tokens(text: str) -> list[str]:
    pronouns = {
        "me", "te", "se", "nos", "os",
        "lo", "la", "los", "las",
        "le", "les",
        "mi", "mis", "tu", "tus", "su", "sus",
        "mio", "mia", "mios", "mias",
        "tuyo", "tuya", "tuyos", "tuyas",
        "suyo", "suya", "suyos", "suyas",
        "yo", "tu", "el", "ella", "ellos", "ellas",
        "nosotros", "nosotras", "vosotros", "vosotras",
        "usted", "ustedes",
    }
    return [token for token in _normalized_word_tokens(text) if token in pronouns]

def _restore_spanish_pronoun_sequence(raw_text: str, corrected: str) -> str | None:
    raw_pronouns = _spanish_pronoun_tokens(raw_text)
    corrected_pronouns = _spanish_pronoun_tokens(corrected)
    if len(raw_pronouns) != len(corrected_pronouns):
        return None
    raw_index = 0

    def replace_match(match: re.Match) -> str:
        nonlocal raw_index
        token = match.group(0)
        normalized = _normalized_plain_text(token)
        if raw_index < len(raw_pronouns) and normalized == corrected_pronouns[raw_index]:
            replacement = raw_pronouns[raw_index]
            raw_index += 1
            if token[:1].isupper():
                return replacement.capitalize()
            return replacement
        return token

    restored = re.sub(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", replace_match, corrected)
    if raw_index != len(raw_pronouns):
        return None
    return restored

def _protect_spanish_pronoun_rewrite(raw_text: str, corrected: str) -> str:
    """Reject rewrites that change existing Spanish pronouns or possessives."""
    raw_pronouns = _spanish_pronoun_tokens(raw_text)
    if not raw_pronouns:
        return corrected
    corrected_pronouns = _spanish_pronoun_tokens(corrected)
    if _lcs_length(raw_pronouns, corrected_pronouns) < len(raw_pronouns):
        restored = _restore_spanish_pronoun_sequence(raw_text, corrected)
        if restored is not None:
            print("[asr] post-processing changed Spanish pronoun(s); restored raw pronoun sequence")
            return restored
        print("[asr] post-processing changed Spanish pronoun(s); returning raw transcript")
        return raw_text
    return corrected

def _spanish_discourse_marker_tokens(text: str) -> list[str]:
    markers = {
        "ya", "vale", "bueno", "pues", "oye", "venga", "anda",
    }
    return [token for token in _normalized_word_tokens(text) if token in markers]

def _restore_spanish_discourse_markers(raw_text: str, corrected: str) -> str | None:
    raw_markers = _spanish_discourse_marker_tokens(raw_text)
    if not raw_markers:
        return corrected
    replacement_candidates = {
        "ya", "vale", "bueno", "pues", "oye", "venga", "anda",
        "si", "ok", "okay", "claro",
    }
    corrected_candidates = [
        token for token in _normalized_word_tokens(corrected)
        if token in replacement_candidates
    ]
    if len(raw_markers) != len(corrected_candidates):
        return None
    raw_index = 0

    def replace_match(match: re.Match) -> str:
        nonlocal raw_index
        token = match.group(0)
        normalized = _normalized_plain_text(token)
        if raw_index < len(raw_markers) and normalized == corrected_candidates[raw_index]:
            replacement = raw_markers[raw_index]
            raw_index += 1
            if token[:1].isupper():
                return replacement.capitalize()
            return replacement
        return token

    restored = re.sub(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", replace_match, corrected)
    if raw_index != len(raw_markers):
        return None
    return restored

def _protect_spanish_discourse_marker_rewrite(raw_text: str, corrected: str) -> str:
    """Preserve Spanish discourse markers instead of accepting normalized synonyms."""
    raw_markers = _spanish_discourse_marker_tokens(raw_text)
    if not raw_markers:
        return corrected
    corrected_markers = _spanish_discourse_marker_tokens(corrected)
    if _lcs_length(raw_markers, corrected_markers) < len(raw_markers):
        restored = _restore_spanish_discourse_markers(raw_text, corrected)
        if restored is not None:
            print("[asr] post-processing changed Spanish discourse marker(s); restored raw marker sequence")
            return restored
        print("[asr] post-processing changed Spanish discourse marker(s); returning raw transcript")
        return raw_text
    return corrected

async def _acquire_concurrency_slot(kind: str):
    deadline = time.monotonic() + ASR_BUSY_WAIT_TIMEOUT
    while time.monotonic() < deadline:
        if _concurrency_semaphore.acquire(blocking=False):
            return
        await asyncio.sleep(0.25)
    raise HTTPException(
        status_code=503,
        detail=f"Server busy, timed out waiting to start {kind}",
    )

def _postprocess_transcript(
    raw_text: str,
    instruction: str | None = None,
    direct_prompt: bool = False,
    fail_closed: bool = False,
) -> str:
    """Use the local LLM to clean transcripts while preserving meaning."""
    if not ASR_POSTPROCESS_ENABLED or not raw_text.strip():
        return raw_text
    expression_result = _apply_isolated_expression_command(raw_text)
    if expression_result != raw_text:
        return expression_result

    prepared_text = _prepare_formatted_list_transcript(raw_text)
    system_prompt = instruction.strip() if instruction and instruction.strip() else DEFAULT_POSTPROCESS_PROMPT
    user_prompt = prepared_text if direct_prompt else f"Raw transcript:\n{prepared_text}\n\nCorrected transcript:"
    max_tokens = max(64, min(4096, len(raw_text) // 2 + 128))
    try:
        result = _llm_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            thinking=False,
            postprocess_mode=True,
        )
    except Exception as exc:
        message = f"post-processing unavailable: {exc}"
        if fail_closed:
            print(f"[asr] {message}")
            raise PostprocessUnavailable(message) from exc
        print(f"[asr] {message}; returning raw transcript")
        return raw_text

    corrected = (
        result.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    if "Corrected transcript:" in corrected:
        corrected = corrected.rsplit("Corrected transcript:", 1)[1]
    if "Exiting..." in corrected:
        corrected = corrected.split("Exiting...", 1)[0]
    corrected = _remove_quoted_and_empty_wrapper_lines(corrected)

    corrected = _apply_isolated_expression_command(_remove_question_directives(_strip_reasoning_blocks(corrected)))
    corrected = _apply_isolated_expression_sentences(corrected)
    corrected = _apply_explicit_paragraph_breaks(corrected)
    corrected = _protect_non_isolated_expression_rewrite(raw_text, corrected)
    corrected = re.sub(
        r'^\s*(cleaned\s+text|text\s+cleaned|transcript\s+cleaned|cleaned\s+transcript|texto\s+limpio|texto\s+corregido|transcripción\s+limpia|transcripcion\s+limpia)\s*[:：\-–—]\s*',
        '',
        corrected,
        flags=re.IGNORECASE,
    )
    corrected = _repair_technical_literal_format(raw_text, corrected)
    corrected = re.sub(r'([?.!,;:])\1{3,}', r'\1', corrected)
    corrected = re.sub(r'\s+([?.!,;:])', r'\1', corrected)
    corrected = _protect_formatted_list_rewrite(raw_text, prepared_text, corrected)
    corrected = _protect_spanish_diminutive_rewrite(raw_text, corrected)
    corrected = _protect_spanish_pronoun_rewrite(raw_text, corrected)
    corrected = _protect_spanish_discourse_marker_rewrite(raw_text, corrected)
    corrected = _protect_meaningful_order_rewrite(prepared_text, corrected)
    _log_llm_postprocess_timing(result, raw_text, corrected)
    if not corrected:
        return raw_text
    if len(corrected) > max(len(raw_text) * 2, len(raw_text) + 80):
        message = "post-processing output was too long"
        if fail_closed:
            print(f"[asr] {message}")
            raise PostprocessUnavailable(message)
        print(f"[asr] {message}; returning raw transcript")
        return raw_text
    return corrected

async def _run_postprocess_locked(
    raw_text: str,
    instruction: str | None = None,
    direct_prompt: bool = False,
    fail_closed: bool = False,
) -> str:
    await _acquire_concurrency_slot("post-processing")
    try:
        return await asyncio.get_event_loop().run_in_executor(
            None,
            _postprocess_transcript,
            raw_text,
            instruction,
            direct_prompt,
            fail_closed,
        )
    finally:
        _concurrency_semaphore.release()

# ---- FastAPI App ----

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"[asr] Starting ASR service on {ASR_LISTEN_HOST}:{ASR_LISTEN_PORT}")
    print(f"[asr] Model: {ASR_MODEL_SIZE}, Device: {ASR_DEVICE}, Max concurrency: {ASR_MAX_CONCURRENCY}")
    print(f"[asr] Postprocess: enabled={ASR_POSTPROCESS_ENABLED} provider={ASR_POSTPROCESS_PROVIDER}")
    for warning in _validate_runtime_config():
        print(f"[asr] config warning: {warning}")
    if _gpu_watch_enabled():
        threading.Thread(target=_llm_watcher_loop, daemon=True, name="llm-watcher").start()
    yield
    # Shutdown
    _stop_whisper_worker("ASR service shutdown")
    _stop_qwen_worker("ASR service shutdown")
    unload_model()
    _stop_llm_server("ASR service shutdown")
    print("[asr] Shutdown complete")

app = FastAPI(
    title="Portable ASR",
    description="OpenAI-compatible speech-to-text microservice",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
async def health():
    return {"status": "ok", "model": ASR_MODEL_SIZE, "device": ASR_DEVICE}

@app.get("/v1/models")
async def models(token: str = Depends(verify_token)):
    model_ids = ["whisper-1"]
    if ASR_QWEN_ENABLED:
        model_ids.append(ASR_QWEN_MODEL_ID)
    model_ids.append(ASR_POSTPROCESS_MODEL)
    return {
        "object": "list",
        "data": [{"id": model_id, "object": "model", "owned_by": "local"} for model_id in model_ids],
    }

@app.post("/v1/text/postprocess")
async def text_postprocess(payload: PostprocessRequest, token: str = Depends(verify_token)):
    try:
        corrected = await _run_postprocess_locked(payload.text, payload.instruction, fail_closed=True)
    except PostprocessUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "text": corrected,
        "model": ASR_POSTPROCESS_MODEL,
    }

@app.post("/v1/llm/chat/completions")
async def llm_chat_completions(payload: ChatCompletionRequest, token: str = Depends(verify_token)):
    messages = [
        {"role": message.role, "content": _message_content_to_text(message.content)}
        for message in payload.messages
    ]
    await _acquire_concurrency_slot("LLM chat")
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            _llm_chat,
            messages,
            payload.max_tokens or 4096,
            payload.temperature or 0.0,
            True if payload.thinking is None else bool(payload.thinking),
            False,
        )
    finally:
        _concurrency_semaphore.release()
    if not payload.stream:
        return result

    async def stream_result():
        choice = result.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content")
        chunk = {
            "id": result.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
            "object": "chat.completion.chunk",
            "created": result.get("created", int(time.time())),
            "model": result.get("model", ASR_POSTPROCESS_MODEL),
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None,
                }
            ],
        }
        if reasoning:
            chunk["choices"][0]["delta"]["reasoning_content"] = reasoning
        yield f"data: {json.dumps(chunk)}\n\n"
        done = {
            "id": chunk["id"],
            "object": "chat.completion.chunk",
            "created": chunk["created"],
            "model": chunk["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": choice.get("finish_reason", "stop"),
                }
            ],
        }
        yield f"data: {json.dumps(done)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_result(), media_type="text/event-stream")

@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, token: str = Depends(verify_token)):
    if payload.stream:
        raise HTTPException(status_code=400, detail="Streaming responses are not supported")
    user_messages = [
        _message_content_to_text(message.content)
        for message in payload.messages
        if message.role == "user"
    ]
    instruction_messages = [
        _message_content_to_text(message.content)
        for message in payload.messages
        if message.role in ("system", "developer")
    ]
    if not user_messages:
        raise HTTPException(status_code=400, detail="At least one user message is required")
    instruction = "\n\n".join(instruction_messages).strip() or None
    raw_text = user_messages[-1]
    if instruction is None:
        transcript_markers = list(re.finditer(r'\btranscript\s*:\s*', raw_text, flags=re.IGNORECASE))
        if transcript_markers:
            marker = transcript_markers[-1]
            instruction = raw_text[:marker.start()].strip() or None
            raw_text = raw_text[marker.end():].strip()
    corrected = await _run_postprocess_locked(raw_text, instruction)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": ASR_POSTPROCESS_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": corrected,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }

@app.post("/v1/audio/transcriptions")
async def transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(None),
    language: str = Form(None),
    prompt: str = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    token: str = Depends(verify_token),
):
    # Validate content type when meaningful. Some clients upload WAV/MP3 as
    # application/octet-stream, so do not reject generic binary uploads.
    if file.content_type and file.content_type not in ("application/octet-stream", "binary/octet-stream") and not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    # Validate file size
    request_started = time.monotonic()
    read_started = time.monotonic()
    contents = await file.read()
    read_ms = _timing_ms(read_started)
    audio_bytes = len(contents)
    transcription_prompt = _normalize_transcription_prompt(prompt)
    engine = _transcription_engine(model)
    if len(contents) > ASR_MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {ASR_MAX_FILE_SIZE // (1024*1024)} MB)",
        )

    # Acquire concurrency slot. This serializes request handling and protects GPU memory transitions.
    await _acquire_concurrency_slot("transcription")

    temp_path = None
    try:
        write_started = time.monotonic()
        # Save to a safe temp file with original extension
        ext = Path(file.filename or "audio.wav").suffix or ".wav"
        # Sanitize extension
        safe_ext = "".join(c for c in ext if c.isalnum() or c in "._-")
        if not safe_ext:
            safe_ext = ".wav"
        suffix = safe_ext if safe_ext.startswith(".") else f".{safe_ext}"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp") as tmp:
            tmp.write(contents)
            temp_path = tmp.name
        write_ms = _timing_ms(write_started)

        if ASR_STOP_LLM_FOR_TRANSCRIPTION:
            _stop_llm_server("Whisper transcription requested")

        try:
            transcribe_started = time.monotonic()
            transcribe_fn = _run_qwen_transcription if engine == "qwen" else _run_transcription
            text_parts, engine_timings = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    transcribe_fn,
                    temp_path,
                    language,
                    temperature,
                    transcription_prompt,
                ),
                timeout=ASR_REQUEST_TIMEOUT + 10,
            )
            transcribe_total_ms = _timing_ms(transcribe_started)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Transcription timed out")

        full_text = " ".join(text_parts).strip()

        # If no segments, return empty
        if not full_text:
            full_text = ""

        if engine == "whisper" and not ASR_WHISPER_KEEP_WARM:
            unload_model()
        if engine == "whisper" and ASR_STOP_LLM_FOR_TRANSCRIPTION and not ASR_WHISPER_KEEP_WARM:
            _settle_after_whisper_unload()
        postprocess_started = time.monotonic()
        full_text = await asyncio.get_event_loop().run_in_executor(
            None,
            _postprocess_transcript,
            full_text,
            None,
            False,
            ASR_AUDIO_POSTPROCESS_REQUIRED,
        )
        postprocess_ms = _timing_ms(postprocess_started)
        total_ms = _timing_ms(request_started)
        timings = {
            "request_read_ms": read_ms,
            "temp_write_ms": write_ms,
            "transcribe_total_ms": transcribe_total_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": total_ms,
            "prompt_chars": len(transcription_prompt or ""),
        }
        if engine == "whisper":
            timings["whisper_total_ms"] = transcribe_total_ms
        timings.update(engine_timings)
        print(
            "[asr] transcription timing "
            f"engine={engine} bytes={audio_bytes} language={language or 'auto'} chars={len(full_text)} "
            + " ".join(f"{key}={value}" for key, value in timings.items())
        )

        # Return OpenAI-compatible JSON
        return {
            "text": full_text,
            "timings": timings,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        # Release concurrency slot
        _concurrency_semaphore.release()
        # Cleanup request memory
        del contents
        if engine == "whisper" and not ASR_WHISPER_KEEP_WARM:
            unload_model()
        gc.collect()

# ---- Main entry point ----
if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--transcribe-once":
        if len(sys.argv) != 5:
            print("usage: asr_server.py --transcribe-once AUDIO_PATH LANGUAGE TEMPERATURE", file=sys.stderr)
            raise SystemExit(2)
        audio_path_arg = sys.argv[2]
        language_arg = sys.argv[3] or None
        temperature_arg = float(sys.argv[4])
        prompt_arg = _normalize_transcription_prompt(os.environ.get("ASR_TRANSCRIBE_ONCE_PROMPT"))
        transcribe_started_arg = time.monotonic()
        segments_arg = _transcribe_file_once(audio_path_arg, language_arg, temperature_arg, prompt_arg)
        print(
            json.dumps(
                {
                    "segments": segments_arg,
                    "timings": {
                        "whisper_transcribe_ms": _timing_ms(transcribe_started_arg),
                    },
                },
                ensure_ascii=False,
            )
        )
        raise SystemExit(0)
    if len(sys.argv) >= 2 and sys.argv[1] == "--transcribe-worker":
        _transcribe_worker_loop()
        raise SystemExit(0)

    import uvicorn
    uvicorn.run(
        "asr_server:app",
        host=ASR_LISTEN_HOST,
        port=ASR_LISTEN_PORT,
        workers=1,  # Single worker; use ASR_MAX_CONCURRENCY for internal concurrency
        log_level="info",
    )
