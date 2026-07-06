#!/usr/bin/env bash
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: run with sudo." >&2
    exit 1
fi

SERVICE_USER="${SERVICE_USER:-${SUDO_USER:-${USER:-root}}}"
SERVICE_GROUP="${SERVICE_GROUP:-${SERVICE_USER}}"
ASR_DIR="${ASR_DIR:-/opt/asr-service}"
SERVICE_NAME="${SERVICE_NAME:-asr-service}"
ASR_ENV_PATH="${ASR_ENV_PATH:-${ASR_DIR}/asr.env}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASR_ENV_SOURCE="${ASR_ENV_SOURCE:-${PROJECT_DIR}/config/asr.env.example}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ASR_LISTEN_HOST_OVERRIDE="${ASR_LISTEN_HOST:-}"
ASR_LISTEN_PORT_OVERRIDE="${ASR_LISTEN_PORT:-}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

install -d -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" "${ASR_DIR}"
install -m 755 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" "${PROJECT_DIR}/asr_server.py" "${ASR_DIR}/asr_server.py"
install -m 644 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" "${PROJECT_DIR}/requirements.txt" "${ASR_DIR}/requirements.txt"

if [ ! -f "${ASR_ENV_PATH}" ]; then
    install -m 600 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" "${ASR_ENV_SOURCE}" "${ASR_ENV_PATH}"
fi

if [ -n "${ASR_LISTEN_HOST_OVERRIDE}" ] && grep -q '^ASR_LISTEN_HOST=' "${ASR_ENV_PATH}"; then
    sed -i "s/^ASR_LISTEN_HOST=.*/ASR_LISTEN_HOST=${ASR_LISTEN_HOST_OVERRIDE}/" "${ASR_ENV_PATH}"
fi
if [ -n "${ASR_LISTEN_PORT_OVERRIDE}" ] && grep -q '^ASR_LISTEN_PORT=' "${ASR_ENV_PATH}"; then
    sed -i "s/^ASR_LISTEN_PORT=.*/ASR_LISTEN_PORT=${ASR_LISTEN_PORT_OVERRIDE}/" "${ASR_ENV_PATH}"
fi

if grep -q '^ASR_BEARER_TOKEN=change-me$' "${ASR_ENV_PATH}"; then
    token="$("${PYTHON_BIN}" -c 'import secrets; print(secrets.token_urlsafe(32))')"
    sed -i "s/^ASR_BEARER_TOKEN=.*/ASR_BEARER_TOKEN=${token}/" "${ASR_ENV_PATH}"
fi

chmod 600 "${ASR_ENV_PATH}"
chown "${SERVICE_USER}:${SERVICE_GROUP}" "${ASR_ENV_PATH}"

if [ ! -d "${ASR_DIR}/venv" ]; then
    log "Creating virtualenv at ${ASR_DIR}/venv"
    "${PYTHON_BIN}" -m venv "${ASR_DIR}/venv"
    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${ASR_DIR}/venv"
fi

log "Installing Python requirements"
"${ASR_DIR}/venv/bin/pip" install --upgrade pip
"${ASR_DIR}/venv/bin/pip" install -r "${ASR_DIR}/requirements.txt"
chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${ASR_DIR}/venv"

site_packages="$("${ASR_DIR}/venv/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

cat > "/etc/systemd/system/${SERVICE_NAME}.service" <<SERVICE
[Unit]
Description=OpenAI-compatible ASR Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${ASR_DIR}
EnvironmentFile=${ASR_ENV_PATH}
Environment=LD_LIBRARY_PATH=${site_packages}/nvidia/cublas/lib:${site_packages}/nvidia/cudnn/lib:${site_packages}/nvidia/cuda_runtime/lib
ExecStart=${ASR_DIR}/venv/bin/python ${ASR_DIR}/asr_server.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
LimitNOFILE=65536
TimeoutStopSec=30

NoNewPrivileges=yes
PrivateDevices=no
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
ReadWritePaths=${ASR_DIR} /tmp

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}"

log "Installed ${SERVICE_NAME}."
log "Health: curl http://127.0.0.1:9000/health"
