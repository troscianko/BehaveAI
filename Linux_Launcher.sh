#!/usr/bin/env bash
# Linux_Launcher.sh -- self-bootstrapping launcher for BehaveAI
# Usage:
#   ./Linux_Launcher.sh             # launch BehaveAI GUI
#   ./Linux_Launcher.sh [args...]   # pass args to behaveai CLI
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${HOME}/ultralytics-venv"
PYTHON_BIN="/usr/bin/python3"
APT_PKGS=(python3-venv python3-pip build-essential git wget curl ffmpeg \
          libglib2.0-0 libsm6 libxrender1 libxext6 libjpeg-dev zlib1g-dev \
          python3-opencv)
MARKER="${VENV_DIR}/.behaveai_ready"

is_ready() {
  [ -f "${MARKER}" ] && return 0
  if [ -x "${VENV_DIR}/bin/python" ]; then
    "${VENV_DIR}/bin/python" -c "import behaveai" >/dev/null 2>&1 && return 0
  fi
  return 1
}

bootstrap() {
  echo "== BehaveAI bootstrap: installing system & python dependencies =="
  echo "You may be asked for your sudo password to install apt packages."

  sudo apt update
  sudo apt install -y "${APT_PKGS[@]}"

  if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtualenv at ${VENV_DIR} (with --system-site-packages)..."
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
  else
    echo "Virtualenv already exists at ${VENV_DIR} - reusing."
  fi

  (
    set -e
    # shellcheck disable=SC1090
    source "${VENV_DIR}/bin/activate"
    python -m pip install --upgrade pip setuptools wheel

    # Install torch (CPU build; CUDA users can reinstall manually afterwards)
    python -m pip install torch torchvision

    # Install BehaveAI from the extracted repo directory (where this script lives)
    echo "Installing BehaveAI from ${SCRIPT_DIR}..."
    python -m pip install "${SCRIPT_DIR}"
  )

  if ! "${VENV_DIR}/bin/python" -c "import behaveai" >/dev/null 2>&1; then
    echo "ERROR: behaveai import failed after install." >&2
    exit 1
  fi

  touch "${MARKER}"
  echo "Bootstrap complete."
  echo
}

if ! is_ready; then
  bootstrap
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

exec behaveai "$@"