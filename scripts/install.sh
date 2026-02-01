#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEVA_ROOT="${HOME}/.openclaw/seva"
VENV="${SEVA_ROOT}/venv"

mkdir -p "${SEVA_ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found" >&2
  exit 1
fi

python3 -m venv "${VENV}"
"${VENV}/bin/pip" install --upgrade pip

# Base deps (CPU-first). Users can install GPU-specific torch separately.
"${VENV}/bin/pip" install \
  aiohttp \
  numpy \
  networkx \
  sympy \
  chromadb \
  scikit-learn \
  sentence-transformers \
  transformers

# Torch strategy:
# - Default to CPU-only wheels on Linux to avoid CUDA bloat.
# - On macOS, regular torch supports MPS if available.
UNAME="$(uname -s)"
if [[ "${UNAME}" == "Linux" ]]; then
  "${VENV}/bin/pip" install --index-url https://download.pytorch.org/whl/cpu torch
else
  "${VENV}/bin/pip" install torch
fi

echo "Installing SEVA server..."
cp -f "${REPO_DIR}/seva/seva_server.py" "${SEVA_ROOT}/seva_server.py"
chmod +x "${SEVA_ROOT}/seva_server.py"

if [[ "${UNAME}" == "Linux" ]]; then
  mkdir -p "${HOME}/.config/systemd/user"
  cp -f "${REPO_DIR}/service/systemd/openclaw-seva.service" "${HOME}/.config/systemd/user/openclaw-seva.service"
  systemctl --user daemon-reload
  systemctl --user enable --now openclaw-seva.service
  echo "SEVA service started (systemd user service)."
else
  # macOS LaunchAgent (user scope). Replace %HOME% placeholders.
  mkdir -p "${HOME}/Library/LaunchAgents"
  sed "s|%HOME%|${HOME}|g" "${REPO_DIR}/service/macos/com.openclaw.seva.plist" > "${HOME}/Library/LaunchAgents/com.openclaw.seva.plist"
  launchctl unload "${HOME}/Library/LaunchAgents/com.openclaw.seva.plist" >/dev/null 2>&1 || true
  launchctl load "${HOME}/Library/LaunchAgents/com.openclaw.seva.plist"
  echo "SEVA service loaded (LaunchAgent)."
fi

echo "Link/install hook pack into OpenClaw..."
openclaw hooks install -l "${REPO_DIR}"

echo "Done. Restart the gateway so hooks reload:"

echo "  openclaw gateway restart"
