#!/usr/bin/env bash
set -euo pipefail

ASR_DIR="${ASR_DIR:-/opt/asr-service}"
ASR_ENV="${ASR_ENV:-${ASR_DIR}/asr.env}"

if [ ! -f "${ASR_ENV}" ]; then
    echo "ERROR: ASR env file not found: ${ASR_ENV}" >&2
    echo "Set ASR_ENV=/path/to/asr.env if the service uses a different path." >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
. "${ASR_ENV}"
set +a

: "${ASR_POSTPROCESS_MODEL:?ASR_POSTPROCESS_MODEL is required}"
: "${ASR_POSTPROCESS_CONTEXT:?ASR_POSTPROCESS_CONTEXT is required}"
: "${ASR_POSTPROCESS_CACHE_DIR:?ASR_POSTPROCESS_CACHE_DIR is required}"
: "${ASR_POSTPROCESS_LIBRARY_PATH:?ASR_POSTPROCESS_LIBRARY_PATH is required}"
: "${ASR_LLM_SERVER_HOST:=127.0.0.1}"
: "${ASR_LLM_SERVER_PORT:=9100}"

LLAMA_SERVER="${ASR_POSTPROCESS_LIBRARY_PATH}/llama-server"
if [ ! -x "${LLAMA_SERVER}" ]; then
    echo "ERROR: llama-server is not executable: ${LLAMA_SERVER}" >&2
    exit 1
fi

export HF_HOME="${ASR_POSTPROCESS_CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${ASR_POSTPROCESS_CACHE_DIR}/hub"
export LLAMA_CACHE="${ASR_POSTPROCESS_CACHE_DIR}"
export LD_LIBRARY_PATH="${ASR_POSTPROCESS_LIBRARY_PATH}"

echo "GPU before llama-server:"
nvidia-smi || true
echo
echo "Starting llama-server with service-equivalent post-processing settings..."
echo "Model:   ${ASR_POSTPROCESS_MODEL}"
echo "Context: ${ASR_POSTPROCESS_CONTEXT}"
echo "Host:    ${ASR_LLM_SERVER_HOST}:${ASR_LLM_SERVER_PORT}"
echo

exec "${LLAMA_SERVER}" \
    -hf "${ASR_POSTPROCESS_MODEL}" \
    --offline \
    --host "${ASR_LLM_SERVER_HOST}" \
    --port "${ASR_LLM_SERVER_PORT}" \
    -c "${ASR_POSTPROCESS_CONTEXT}" \
    -ctk q8_0 \
    -ctv q8_0 \
    -ngl all \
    -np 1 \
    -fa on \
    --jinja \
    --reasoning off \
    --reasoning-format deepseek \
    --no-webui \
    --no-slots
