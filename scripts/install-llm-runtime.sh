#!/usr/bin/env bash
# install-llm-runtime.sh
# Requires: sudo
#
# Installs a local llama.cpp server for optional ASR transcript post-processing.

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: This script must be run as root (sudo)." >&2
    exit 1
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { log "FATAL: $*"; exit 1; }
ok_msg() { log "OK: $*"; }

LLM_USER="${LLM_USER:-${SUDO_USER:-${USER:-root}}}"
LLM_DIR="${LLM_DIR:-/opt/asr-llm}"
LLM_BIN_DIR="${LLM_DIR}/bin"
LLM_DOWNLOAD_DIR="${LLM_DIR}/downloads"
LLM_SOURCE_DIR="${LLM_DIR}/llama.cpp"
LLM_BUILD_DIR="${LLM_DIR}/llama.cpp-build"
LLM_CACHE_DIR="${LLM_DIR}/hf-cache"
LLM_MODEL="${LLM_MODEL:-unsloth/Qwen3.5-9B-GGUF:Q4_K_M}"
LLM_CONTEXT="${LLM_CONTEXT:-32768}"
LLM_BACKEND="${LLM_BACKEND:-cuda}"
LLM_MIN_TOTAL_VRAM_MIB="${LLM_MIN_TOTAL_VRAM_MIB:-8192}"
LLM_SKIP_DOWNLOAD="${LLM_SKIP_DOWNLOAD:-0}"
LLM_SMOKE_RAW="${LLM_SMOKE_RAW:-question how are you}"
LLM_SMOKE_SYSTEM="${LLM_SMOKE_SYSTEM:-Return only the cleaned transcript. Keep the same language. Clean spelling, capitalization, punctuation, and obvious transcript errors without changing meaning.}"
LLM_SERVICE_NAME="${LLM_SERVICE_NAME:-postable-asr}"

if [ "${LLM_BACKEND}" = "cuda" ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        die "nvidia-smi is required for LLM_BACKEND=cuda"
    fi

    GPU_TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
    GPU_FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    if [ "${GPU_TOTAL_MIB}" -lt "${LLM_MIN_TOTAL_VRAM_MIB}" ]; then
        die "${LLM_MODEL} with ${LLM_CONTEXT} context needs ${LLM_MIN_TOTAL_VRAM_MIB} MiB VRAM; found ${GPU_TOTAL_MIB} MiB"
    fi
    ok_msg "GPU detected: ${GPU_NAME}, total ${GPU_TOTAL_MIB} MiB, free ${GPU_FREE_MIB} MiB"
else
    GPU_TOTAL_MIB=0
    GPU_FREE_MIB=0
    GPU_NAME="none"
fi

log "Installing runtime dependencies..."
apt-get update -qq
apt-get install -y -qq \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    jq \
    libgomp1 \
    ninja-build \
    python3 \
    tar
if [ "${LLM_BACKEND}" = "cuda" ]; then
    apt-get install -y -qq nvidia-cuda-toolkit
fi
ok_msg "Runtime dependencies installed"

if [ "${LLM_BACKEND}" = "cuda" ] && ! command -v nvcc >/dev/null 2>&1; then
    die "nvcc was not installed; cannot build llama.cpp CUDA backend"
fi
if [ "${LLM_BACKEND}" = "cuda" ]; then
    ok_msg "CUDA compiler detected: $(nvcc --version | tail -1)"
fi

mkdir -p "${LLM_BIN_DIR}" "${LLM_DOWNLOAD_DIR}" "${LLM_CACHE_DIR}"
chown -R "${LLM_USER}:${LLM_USER}" "${LLM_DIR}"

log "Resolving latest llama.cpp release tag..."
LLAMA_TAG=$(python3 - <<'PY'
import json
import urllib.request

with urllib.request.urlopen("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest", timeout=30) as response:
    release = json.load(response)
print(release["tag_name"])
PY
)
[ -n "${LLAMA_TAG}" ] || die "Could not resolve llama.cpp release tag"
ok_msg "Release tag: ${LLAMA_TAG}"

if [ -d "${LLM_SOURCE_DIR}/.git" ]; then
    log "Updating llama.cpp source..."
    git -C "${LLM_SOURCE_DIR}" fetch --tags --prune
else
    log "Cloning llama.cpp source..."
    git clone https://github.com/ggml-org/llama.cpp.git "${LLM_SOURCE_DIR}"
fi
git -C "${LLM_SOURCE_DIR}" checkout --detach "${LLAMA_TAG}"
chown -R "${LLM_USER}:${LLM_USER}" "${LLM_SOURCE_DIR}"

log "Building llama.cpp with ${LLM_BACKEND} backend..."
rm -rf "${LLM_BUILD_DIR}"
cmake_args=(
    -S "${LLM_SOURCE_DIR}"
    -B "${LLM_BUILD_DIR}"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=ON
)
if [ "${LLM_BACKEND}" = "cuda" ]; then
    cmake_args+=(-DGGML_CUDA=ON)
fi
cmake "${cmake_args[@]}"
cmake --build "${LLM_BUILD_DIR}" --target llama-server llama-cli -j"$(nproc)"

LLAMA_SERVER=$(find "${LLM_BUILD_DIR}" -type f -name llama-server -perm /111 | head -1 || true)
LLAMA_CLI=$(find "${LLM_BUILD_DIR}" -type f -name llama-cli -perm /111 | head -1 || true)
[ -n "${LLAMA_SERVER}" ] || die "llama-server executable not found in build"
[ -n "${LLAMA_CLI}" ] || die "llama-cli executable not found in build"

log "Installing llama.cpp binaries..."
rm -rf "${LLM_BIN_DIR:?}"/*
cp -a "${LLAMA_SERVER}" "${LLAMA_CLI}" "${LLM_BIN_DIR}/"
find "${LLM_BUILD_DIR}" -maxdepth 3 -type f -name 'lib*.so*' -exec cp -a {} "${LLM_BIN_DIR}/" \;
chmod 0755 "${LLM_BIN_DIR}/llama-server" "${LLM_BIN_DIR}/llama-cli"
chown -R "${LLM_USER}:${LLM_USER}" "${LLM_DIR}"
ok_msg "llama-server installed at ${LLM_BIN_DIR}/llama-server"

if [ "${LLM_BACKEND}" = "cuda" ] && ! LD_LIBRARY_PATH="${LLM_BIN_DIR}:${LD_LIBRARY_PATH:-}" "${LLM_BIN_DIR}/llama-server" --list-devices | grep -qi 'cuda'; then
    die "llama-server CUDA build did not report a CUDA device"
fi
if [ "${LLM_BACKEND}" = "cuda" ]; then
    ok_msg "llama-server CUDA devices detected"
fi

if systemctl list-unit-files "${LLM_SERVICE_NAME}-postprocess.service" >/dev/null 2>&1; then
    log "Disabling old persistent post-processing service so it does not hold VRAM..."
    systemctl disable --now "${LLM_SERVICE_NAME}-postprocess.service" 2>/dev/null || true
    systemctl reset-failed "${LLM_SERVICE_NAME}-postprocess.service" 2>/dev/null || true
fi
systemctl daemon-reload

ngl_arg=()
if [ "${LLM_BACKEND}" = "cuda" ]; then
    ngl_arg=(-ngl all --split-mode none)
fi

if [ "${LLM_SKIP_DOWNLOAD}" != "1" ]; then
    log "Priming llama.cpp model cache..."
    HF_HOME="${LLM_CACHE_DIR}" \
    HUGGINGFACE_HUB_CACHE="${LLM_CACHE_DIR}/hub" \
    LLAMA_CACHE="${LLM_CACHE_DIR}" \
    LD_LIBRARY_PATH="${LLM_BIN_DIR}" \
    "${LLM_BIN_DIR}/llama-cli" \
        -hf "${LLM_MODEL}" \
        -c "${LLM_CONTEXT}" \
        "${ngl_arg[@]}" \
        --jinja \
        --reasoning off \
        --chat-template-kwargs '{"enable_thinking":false}' \
        --simple-io \
        --no-display-prompt \
        --no-show-timings \
        --log-disable \
        -n 8 \
        --temp 0 \
        -sys 'Return only OK.' \
        -p 'OK' \
        -st >/dev/null
    ok_msg "Model cache primed"
else
    log "Skipping model cache priming because LLM_SKIP_DOWNLOAD=1"
fi

log "Running transcript cleanup smoke test..."
SMOKE_PROMPT="Raw transcript:
${LLM_SMOKE_RAW}

Corrected transcript:"
SMOKE_OUTPUT=$(HF_HOME="${LLM_CACHE_DIR}" \
    HUGGINGFACE_HUB_CACHE="${LLM_CACHE_DIR}/hub" \
    LLAMA_CACHE="${LLM_CACHE_DIR}" \
    LD_LIBRARY_PATH="${LLM_BIN_DIR}" \
    "${LLM_BIN_DIR}/llama-cli" \
        -hf "${LLM_MODEL}" \
        --offline \
        -c "${LLM_CONTEXT}" \
        "${ngl_arg[@]}" \
        --jinja \
        --reasoning off \
        --chat-template-kwargs '{"enable_thinking":false}' \
        --simple-io \
        --no-display-prompt \
        --no-show-timings \
        --log-disable \
        -n 64 \
        --temp 0 \
        -sys "${LLM_SMOKE_SYSTEM}" \
        -p "${SMOKE_PROMPT}" \
        -st)
SMOKE_RESPONSE=$(SMOKE_OUTPUT="${SMOKE_OUTPUT}" python3 - <<'PY'
import os

text = os.environ.get("SMOKE_OUTPUT", "")
if "Corrected transcript:" in text:
    text = text.rsplit("Corrected transcript:", 1)[1]
if "Exiting..." in text:
    text = text.split("Exiting...", 1)[0]
lines = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith(">")]
print(lines[-1] if lines else "")
PY
)
[ -n "${SMOKE_RESPONSE}" ] || die "Smoke test returned an empty response"
ok_msg "Smoke response: ${SMOKE_RESPONSE}"

if [ "${LLM_BACKEND}" = "cuda" ]; then
    GPU_USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    GPU_FREE_AFTER_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    ok_msg "VRAM after model load: used ${GPU_USED_MIB} MiB, free ${GPU_FREE_AFTER_MIB} MiB"
fi

echo ""
echo "========================================"
echo " ASR Post-processing LLM Installed"
echo "========================================"
echo " Runtime:  one-shot llama-cli (no resident LLM service)"
echo " Model:    ${LLM_MODEL}"
echo " Context:  ${LLM_CONTEXT}"
echo " Backend:  llama.cpp ${LLM_BACKEND}"
echo ""
echo "Restart ASR after this install so it picks up post-processing env/code:"
echo "  sudo systemctl restart ${LLM_SERVICE_NAME}"
echo "========================================"
