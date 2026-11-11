#!/usr/bin/env bash
# run-yolo.sh -- self-bootstrapping wrapper for Ultralytics + apt OpenCV on Raspberry Pi OS
# Usage:
#   ./run-yolo.sh             # if BehaveAI.py exists in CWD, runs it
#   ./run-yolo.sh script.py [args...]  # runs a specific script with args
set -euo pipefail

# If anything fails, keep the terminal window open so the error can be read
# (otherwise a double-clicked launcher's window closes immediately).
pause_on_error() {
  local status=$?
  if [ "${status}" -ne 0 ] && [ -t 0 ]; then
    echo
    echo "Launcher failed (exit code ${status}). Press Enter to close this window."
    read -r _ || true
  fi
}
trap pause_on_error EXIT

# --- Config ---
VENV_DIR="${HOME}/ultralytics-venv"
PYTHON_BIN="/usr/bin/python3"
# List of apt packages we may need
APT_PKGS=(python3-venv python3-pip build-essential git wget curl ffmpeg \
          libglib2.0-0 libsm6 libxrender1 libxext6 libjpeg-dev zlib1g-dev \
          python3-opencv python3-tk python3-pil.imagetk)
# NOTE: deliberately NOT "ultralytics[export]" - that extra pulls in every export backend
# (TFLite/litert-torch, ai-edge-quantizer, onnx, ...) whose pins don't resolve on newer Python
# (e.g. 3.14), sending pip into endless backtracking. BehaveAI only exports to NCNN, which
# needs just ncnn + pnnx.
PIP_PKGS=( ultralytics ncnn pnnx numpy scipy tqdm pillow )
# A tiny marker file to indicate successful install (optional but nice)
MARKER="${VENV_DIR}/.ultralytics_ready"

# Helper: run python inside venv (without permanently activating)
venv_python() {
  # use the venv python if present
  if [ -x "${VENV_DIR}/bin/python" ]; then
    "${VENV_DIR}/bin/python" "$@"
  else
    "${PYTHON_BIN}" "$@"
  fi
}

# Check whether venv + ultralytics are already installed & usable
is_ready() {
  if [ -f "${MARKER}" ]; then
    return 0
  fi
  # if venv exists, try to import ultralytics from it
  if [ -x "${VENV_DIR}/bin/python" ]; then
    if "${VENV_DIR}/bin/python" -c "import ultralytics, sys; sys.exit(0)" >/dev/null 2>&1; then
      # also check cv2 is importable (should be via system-site-packages)
      if "${VENV_DIR}/bin/python" -c "import cv2, sys; sys.exit(0)" >/dev/null 2>&1; then
        return 0
      fi
    fi
  fi
  return 1
}

bootstrap() {
  echo "== Ultralytics bootstrap: installing system & python dependencies =="
  echo "You may be asked for your sudo password to install apt packages."
  # Update and install apt packages
  sudo apt update
  sudo apt install -y "${APT_PKGS[@]}"

  # Create venv (with system site packages so apt OpenCV is visible)
  if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtualenv at ${VENV_DIR} (with --system-site-packages)..."
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
  else
    echo "Virtualenv already exists at ${VENV_DIR} - reusing."
  fi

  # Ensure pip in venv is up-to-date and install pip packages
  echo "Upgrading pip and installing Python packages inside venv..."
  # shellcheck disable=SC1090
  # Use a subshell so activation doesn't pollute caller environment
  (
    set -e
    source "${VENV_DIR}/bin/activate"
    python -m pip install --upgrade pip setuptools wheel
    # install ultralytics and extras; do NOT install opencv-python (we use apt's python3-opencv)
    python -m pip install "${PIP_PKGS[@]}"
  )

  # final sanity checks
  if ! "${VENV_DIR}/bin/python" -c "import ultralytics" >/dev/null 2>&1; then
    echo "ERROR: ultralytics import failed after pip install." >&2
    exit 1
  fi
  if ! "${VENV_DIR}/bin/python" -c "import cv2" >/dev/null 2>&1; then
    echo "WARNING: OpenCV (cv2) not importable inside venv. You may need to install python3-opencv via apt." >&2
    # We continue, since apt install was attempted above; user can re-run
  fi

  # create marker
  mkdir -p "${VENV_DIR}"
  touch "${MARKER}"
  echo "Bootstrap complete."
  echo
}

# If not ready, bootstrap (this will be performed only the first time or if something missing)
if ! is_ready; then
  bootstrap
fi

# Activate venv for the remainder of this script (so python uses venv)
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# OpenCV's Qt5 highgui backend can fail to show the Live preview window (no error, camera
# just never appears) under a native Wayland session (e.g. GNOME on Wayland) - forcing the
# X11/XWayland Qt platform plugin fixes it. This is already the default on X11 systems, so
# it's a no-op there.
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

# Now run the requested python script.
if [ "$#" -ge 1 ]; then
  # Run args as python command
  exec python "$@"
else
  # No args: if BehaveAI.py exists in the current directory, run it
  if [ -f "./BehaveAI.py" ]; then
    echo "Running ./BehaveAI.py in $(pwd)"
    exec python ./BehaveAI.py
  else
    cat <<EOF
No script passed and ./BehaveAI.py not found in $(pwd).

Usage:
  ${0} path/to/script.py [args...]
  # or from a folder containing BehaveAI.py:
  cd ~/yolo-projects/cam1
  ${0}

EOF
    exit 2
  fi
fi
