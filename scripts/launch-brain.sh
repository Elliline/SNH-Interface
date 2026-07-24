#!/usr/bin/env bash
#
# launch-brain.sh — (re)create the sparky-brain vLLM container.
#
# This is the source of truth for how the local model engine is launched. There
# was no launch script/compose file before — the container was started by a bare
# `docker run`, so its args lived only in the running container. This captures
# them, and applies the root fix for the GB10 / SM 12.1 silent-hang wedge.
#
# ROOT FIX (2026-07-23), vllm-project/vllm#40969: under sustained load the engine
# would wedge at 0.0 tok/s with the default cudagraph_mode (FULL_AND_PIECEWISE).
# The confirmed upstream workaround is cudagraph_mode PIECEWISE (clean over 200+
# requests). We also drop gpu-memory-utilization 0.85 -> 0.80 (separate GB10
# reports of hard-locks above 0.8 with Gemma 4). The brain-watchdog stays as the
# smoke alarm; with this fix in place, a future wedge means something NEW.
#
# Previous (wedge-prone) serve args, for reference / rollback:
#   --gpu-memory-utilization 0.85   (no --compilation-config -> default cudagraph)
#
# Usage:
#   scripts/launch-brain.sh --print   # show the exact command, change nothing
#   scripts/launch-brain.sh           # stop+remove the old container and recreate
set -euo pipefail

IMAGE="nvcr.io/nvidia/vllm:26.06-py3"
NAME="sparky-brain"

# The vLLM serve invocation. The compilation-config JSON is single-quoted so the
# container's shell passes it to vllm intact.
SERVE="vllm serve nvidia/Gemma-4-26B-A4B-NVFP4 \
--tensor-parallel-size 1 \
--tool-call-parser gemma4 \
--enable-auto-tool-choice \
--max-model-len 131072 \
--gpu-memory-utilization 0.80 \
--compilation-config '{\"cudagraph_mode\": \"PIECEWISE\"}'"

# The image needs a fastapi pin before serving (preserved from the original run).
INNER="pip install 'fastapi<0.137' --quiet && ${SERVE}"

if [[ "${1:-}" == "--print" ]]; then
  echo "IMAGE: ${IMAGE}"
  echo "NAME:  ${NAME}"
  echo "---- exact command the container's bash -c will run ----"
  echo "${INNER}"
  exit 0
fi

echo "[launch-brain] stopping and removing existing ${NAME} (if any)…"
docker stop "${NAME}" >/dev/null 2>&1 || true
docker rm "${NAME}"   >/dev/null 2>&1 || true

echo "[launch-brain] starting ${NAME}…"
docker run -d \
  --name "${NAME}" \
  --gpus all \
  --ipc=host \
  -p 7070:8000 \
  -v /home/ellie/.cache/huggingface:/root/.cache/huggingface \
  --restart unless-stopped \
  "${IMAGE}" \
  bash -c "${INNER}"

echo "[launch-brain] started. Follow load with:  docker logs -f ${NAME}"
