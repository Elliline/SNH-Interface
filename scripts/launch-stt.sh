#!/usr/bin/env bash
#
# launch-stt.sh — (re)create the sparky-stt Parakeet STT container.
#
# Source of truth for how SNH's speech-to-text engine is launched, matching the
# launch-brain.sh pattern. Model is NVIDIA Parakeet TDT 0.6B v3 served through
# NeMo, wrapped by /home/ellie/sparky-stt/parakeet_stt.py in a thin FastAPI app
# that exposes BOTH SNH's native POST /transcribe (multipart field "audio") and
# the OpenAI-compatible POST /v1/audio/transcriptions (field "file"), each
# returning {"text": ...}. The NeMo NGC image is natively multi-arch (aarch64 /
# Grace), so no aarch64 workaround is needed. Parakeet 0.6B is small and shares
# the 128GB unified pool with the brain (which holds 0.85).
#
# The container installs the light web deps on top of the image's bundled NeMo,
# ensures ffmpeg is present (browser audio is webm/opus — decoded to 16 kHz mono
# before inference), then runs the wrapper. First boot downloads the model
# (~2.4GB) into the mounted HF cache, so allow a couple of minutes.
#
# Usage:
#   scripts/launch-stt.sh --print
#   scripts/launch-stt.sh
set -euo pipefail

IMAGE="nvcr.io/nvidia/nemo:25.07"
NAME="sparky-stt"
PORT=5051

INNER='set -e
command -v ffmpeg >/dev/null 2>&1 || (apt-get update -q && apt-get install -y -q ffmpeg)
pip install -q fastapi uvicorn python-multipart
cd /app && exec python3 parakeet_stt.py'

if [[ "${1:-}" == "--print" ]]; then
  echo "IMAGE: ${IMAGE}"
  echo "NAME:  ${NAME}"
  echo "PORT:  ${PORT}"
  echo "---- inner ----"
  echo "${INNER}"
  exit 0
fi

echo "[launch-stt] stopping and removing existing ${NAME} (if any)…"
docker stop "${NAME}" >/dev/null 2>&1 || true
docker rm "${NAME}"   >/dev/null 2>&1 || true

echo "[launch-stt] starting ${NAME}…"
docker run -d \
  --name "${NAME}" \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p ${PORT}:${PORT} \
  -v /home/ellie/sparky-stt:/app \
  -v /home/ellie/.cache/huggingface:/root/.cache/huggingface \
  -v /home/ellie/.cache/torch:/root/.cache/torch \
  --restart unless-stopped \
  --health-cmd "python3 -c \"import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:${PORT}/health',timeout=5).status==200 else 1)\"" \
  --health-interval 30s \
  --health-timeout 8s \
  --health-start-period 240s \
  --health-retries 3 \
  "${IMAGE}" \
  bash -c "${INNER}"

echo "[launch-stt] started. Health: curl http://localhost:${PORT}/health"
echo "[launch-stt] logs:   docker logs -f ${NAME}"
