#!/usr/bin/env bash
# Start 8 Ollama workers, each pinned to a single GPU and a unique port.
# - Uses CUDA_VISIBLE_DEVICES to isolate GPUs per worker
# - Writes logs to logs/ollama-gpu{idx}.log and PID files to logs/pids/worker_{idx}.pid
# - Performs health checks on each started worker
# Usage: bash scripts/start_ollama_workers.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
PID_DIR="${LOG_DIR}/pids"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

PORT_BASE=11434
NUM_WORKERS=8

echo "===== BEGIN start_ollama_workers $(date -Iseconds) ====="

# Sanity: nvidia-smi summary
if command -v nvidia-smi &>/dev/null; then
  echo "GPU inventory:"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true
else
  echo "WARNING: nvidia-smi not found; proceeding blindly."
fi

for i in $(seq 0 $((NUM_WORKERS-1))); do
  PORT=$((PORT_BASE + i))
  LOG_FILE="${LOG_DIR}/ollama-gpu${i}.log"
  PID_FILE="${PID_DIR}/worker_${i}.pid"

  # Skip if port already in use or pid exists and process alive
  if lsof -i TCP:${PORT} -sTCP:LISTEN -t &>/dev/null; then
    echo "Worker ${i} appears to be running on port ${PORT}; skipping start."
    continue
  fi

  # If stale PID file, remove
  if [[ -f "${PID_FILE}" ]]; then
    OLD_PID="$(cat "${PID_FILE}" || true)"
    if [[ -n "${OLD_PID}" ]] && ps -p "${OLD_PID}" &>/dev/null; then
      echo "Worker ${i} has PID ${OLD_PID} still alive; skipping."
      continue
    else
      rm -f "${PID_FILE}"
    fi
  fi

  echo "Starting worker ${i} on port ${PORT} (GPU ${i})..."
  MODELS_DIR="${ROOT_DIR}/ollama_models"
  mkdir -p "${MODELS_DIR}"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=${i} OLLAMA_HOST=127.0.0.1:${PORT} OLLAMA_MODELS="${MODELS_DIR}" nohup ollama serve \
    > "${LOG_FILE}" 2>&1 &

  echo $! > "${PID_FILE}"

  # Health check (max 20s)
  ATTEMPTS=20
  OK=0
  for _ in $(seq 1 ${ATTEMPTS}); do
    if curl -sS "http://127.0.0.1:${PORT}/api/tags" >/dev/null 2>&1; then
      OK=1
      break
    fi
    sleep 1
  done
  if [[ "${OK}" -ne 1 ]]; then
    echo "ERROR: worker ${i} on port ${PORT} failed health check; see ${LOG_FILE}"
    exit 1
  fi
  echo "Worker ${i} healthy on port ${PORT}"
done

echo "===== END start_ollama_workers $(date -Iseconds) ====="
