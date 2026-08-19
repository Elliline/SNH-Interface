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
# The confirmed upstream workaround is cudagraph_mode PIECEWISE. We also drop
# gpu-memory-utilization 0.85 -> 0.80 (separate GB10 reports of hard-locks above
# 0.8 with Gemma 4). The brain-watchdog stays as the smoke alarm; with this fix
# in place, a future wedge means something NEW.
#
# THE FIX HAS A CEILING, and it is now measured rather than assumed (2026-08-14).
# "Clean over 200+ requests" was the upstream report, carried here without a
# bound. Swept for real against this engine (scripts/measure-brain-concurrency.js),
# 256 output tokens per request, ignore_eos so every request costs the same:
#
#     1 req   31.8 tok/s aggregate    128 req  1,615.9 tok/s   ttft p50 0.17s
#    64 req  1,031.6 tok/s            192 req  2,040.6 tok/s   ttft p50 0.18s
#   256 req  WEDGED — 0 of 256 completed, all read-timed-out, 609s wall
#
# Clean to 192: no errors, no preemptions, KV never above 7.5%. At 256 the engine
# went to 0 tok/s with the historical signature — and it was not KV pressure or
# preemption, because all 256 were ADMITTED (peak_running 256, peak_waiting 1) at
# 4.5% KV. A full batch is what stops it, not a full cache.
#
# So the batch is now pinned below the ceiling instead of left to the engine's
# default, which turns an over-subscription into a QUEUE instead of a wedge:
# past --max-num-seqs, vLLM holds requests in `waiting` rather than admitting
# them into a batch that stops. 128 is one clean step below the last good level,
# and roughly 1.6k tok/s aggregate — far above anything SNH generates, since
# background work is capped at agentPool.concurrency 3 plus the live chat turn.
# The exact break between 192 and 256 is deliberately NOT bisected: each failing
# level costs ~15 minutes of a wedged brain, and pinning the batch makes the
# number academic.
#
# Previous (wedge-prone) serve args, for reference / rollback:
#   --gpu-memory-utilization 0.85   (no --compilation-config -> default cudagraph)
#   (no --max-num-seqs -> engine default, which admits a batch large enough to wedge)
#
# Usage:
#   scripts/launch-brain.sh --print   # show the exact command, change nothing
#   scripts/launch-brain.sh           # stop+remove the old container and recreate
set -euo pipefail

IMAGE="nvcr.io/nvidia/vllm:26.06-py3"
NAME="sparky-brain"

# PREFIX CACHING IS ON, and not by accident of omission (verified 2026-08-12).
# vLLM's V1 engine defaults enable_prefix_caching to true; the running engine
# reports `enable_prefix_caching=True, enable_chunked_prefill=True` in its
# startup config, and the counters back it up
# (vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total on :7070/metrics).
# So no --enable-prefix-caching flag is passed and none is needed. Recorded here
# because "the flag is absent" reads as "the feature is off", and it is the
# thing prompt ordering in server.js is built to exploit — see the ordered
# assembly there. If a future image ever defaults it off, this is where to add it.
#
# 8-BIT WEIGHTS AS OF 2026-08-18. Was nvidia/Gemma-4-26B-A4B-NVFP4 (4-bit).
# Rollback: git checkout pre-fp8-swap-2026-08-18 -- scripts/launch-brain.sh
# The 4-bit weights are NOT deleted (18G still in ~/.cache/huggingface/hub), so
# rolling back is the checkout plus this script — no re-download.
#
# TWO FLAGS BELOW EXIST ONLY TO KEEP THE SWAP A ONE-VARIABLE EXPERIMENT:
#
# --served-model-name pins the wire name to the OLD id. data/config.json names
#   the model in four places and data/ is off limits for this test, so the alias
#   keeps every app-side string, the memory store and the request path byte-identical
#   while the weights underneath change. The cost is that /v1/models now reports a
#   name that does not describe what is loaded — this comment is the disambiguator,
#   and the alias should be dropped (and data/config.json updated) if 8-bit stays.
#
# --kv-cache-dtype fp8_e4m3 restores what the checkpoint used to supply. The NVFP4
#   checkpoint carries kv_cache_scheme {num_bits: 8, type: float}, so the engine has
#   been running an fp8_e4m3 KV cache all along (at scaling factor 1.0 — it warns).
#   The RedHatAI FP8 checkpoint has kv_cache_scheme: null, so without this flag the
#   KV cache silently becomes bf16: 2x the bytes per token, and a SECOND variable
#   changed in a test that is supposed to change only the weights. Setting it
#   explicitly reproduces the 4-bit KV path exactly (also unscaled, same warning).
#
# UNCHANGED AND DELIBERATELY SO: cudagraph_mode PIECEWISE (the 7/23 wedge fix),
# --max-num-seqs 128 (the 8/14 batch ceiling), --gpu-memory-utilization 0.80,
# --max-model-len 131072, --tool-call-parser gemma4. None of them are quantization-
# dependent, and the 128 pin only gets safer as each sequence costs more.
#
# The vLLM serve invocation. The compilation-config JSON is single-quoted so the
# container's shell passes it to vllm intact.
SERVE="vllm serve RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic \
--served-model-name nvidia/Gemma-4-26B-A4B-NVFP4 \
--tensor-parallel-size 1 \
--tool-call-parser gemma4 \
--enable-auto-tool-choice \
--max-model-len 131072 \
--max-num-seqs 128 \
--gpu-memory-utilization 0.80 \
--kv-cache-dtype fp8_e4m3 \
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
