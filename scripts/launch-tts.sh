#!/usr/bin/env bash
#
# launch-tts.sh — (re)create the sparky-tts Kokoro TTS container.
#
# Source of truth for how SNH's text-to-speech engine is launched, matching the
# launch-brain.sh pattern. Kokoro is served by the maintained OpenAI-compatible
# server (remsky/kokoro-fastapi), GPU variant, which exposes POST /v1/audio/speech
# — exactly what SNH's TTS proxy calls. The GPU image ships a native arm64 (Grace)
# manifest, so no aarch64 workaround is needed.
#
# GPU sharing: no memory cap flag exists for this image; Kokoro is a small model
# and shares the 128GB unified pool with the brain (which holds 0.85). Default
# request voice is af_heart (SNH sends the voice per request; this is only the
# fallback). Container listens on 8880; we publish it as 5050.
#
# Usage:
#   scripts/launch-tts.sh --print   # show the command, change nothing
#   scripts/launch-tts.sh           # stop+remove old container and recreate
set -euo pipefail

IMAGE="ghcr.io/remsky/kokoro-fastapi-gpu:latest"
NAME="sparky-tts"
HOST_PORT=5050
CONTAINER_PORT=8880

if [[ "${1:-}" == "--print" ]]; then
  echo "IMAGE: ${IMAGE}"
  echo "NAME:  ${NAME}"
  echo "PORT:  ${HOST_PORT} -> ${CONTAINER_PORT}"
  exit 0
fi

echo "[launch-tts] stopping and removing existing ${NAME} (if any)…"
docker stop "${NAME}" >/dev/null 2>&1 || true
docker rm "${NAME}"   >/dev/null 2>&1 || true

echo "[launch-tts] starting ${NAME}…"
docker run -d \
  --name "${NAME}" \
  --gpus all \
  --ipc=host \
  -p ${HOST_PORT}:${CONTAINER_PORT} \
  -v /home/ellie/.cache/huggingface:/root/.cache/huggingface \
  --restart unless-stopped \
  --health-cmd "python -c \"import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:${CONTAINER_PORT}/health',timeout=5).status==200 else 1)\"" \
  --health-interval 30s \
  --health-timeout 8s \
  --health-start-period 90s \
  --health-retries 3 \
  "${IMAGE}"

echo "[launch-tts] started. Health: curl http://localhost:${HOST_PORT}/health"
echo "[launch-tts] logs:   docker logs -f ${NAME}"
