#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-g1"

sudo apt-get update
sudo apt-get install -y \
  python3 \
  python3-venv \
  python3-pip \
  ffmpeg \
  libsndfile1 \
  portaudio19-dev \
  alsa-utils \
  curl

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/requirements.txt"
python -m pip install onnxruntime

echo "Ubuntu G1 runtime installed."
echo "Activate with: source ${VENV_DIR}/bin/activate"
