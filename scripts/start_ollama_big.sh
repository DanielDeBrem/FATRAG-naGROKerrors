#!/usr/bin/env bash
# Start a single "big" Ollama server intended to use all available GPUs.
# - No CUDA_VISIBLE_DEVICES restriction (so it can see all GPUs)
# - Binds to a dedicated port (default 11450)
# - Writes logs to logs/ollama-big.log and PID to logs/pids/ollama_big.pid
# - Health check after start
# Usage: bash scripts/start_ollama_big.sh [PORT]
# Example: bash scripts/start_ollama_big.sh 11450

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
PID_DIR="${LOG_DIR}/pids"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

PORT="${1:-11450}"
LOG_FILE="${LOG_DIR}/ollama-big.log"
PID_FILE="${PID_DIR}/ollama_big.pid"

echo "===== BEGIN start_ollama_big $(date -Iseconds) (port=${PORT}) ====="

# Info: GPU inventory
if command -v nvidia-smi &>/dev/null; then
  echo "GPU inventory (all GPUs will be visible to this process):"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true
else
  echo "WARNING: nvidia-smi not found; proceeding blindly."
fi

# Skip if port already in use or pid exists and process alive
if lsof -i TCP:${PORT} -sTCP:LISTEN -t &>/dev/null; then
  echo "Big Ollama appears to be running on port ${PORT}; skipping start."
  exit 0
fi

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}" || true)"
  if [[ -n "${OLD_PID}" ]] && ps -p "${OLD_PID}" &>/dev/null; then
    echo "Big Ollama PID ${OLD_PID} still alive; skipping."
    exit 0
  else
    rm -f "${PID_FILE}"
  fi
fi

echo "Starting big Ollama on port ${PORT} (all GPUs visible)..."
# Use project-local models dir to avoid permission issues
MODELS_DIR="${ROOT_DIR}/ollama_models"
mkdir -p "${MODELS_DIR}"
# Note: no CUDA_VISIBLE_DEVICES; server should be able to allocate across GPUs if supported
OLLAMA_HOST=127.0.0.1:${PORT} OLLAMA_MODELS="${MODELS_DIR}" nohup ollama serve > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

# Health check (max 30s)
ATTEMPTS=30
OK=0
for _ in $(seq 1 ${ATTEMPTS}); do
  if curl -sS "http://127.0.0.1:${PORT}/api/tags" >/dev/null 2>&1; then
    OK=1
    break
  fi
  sleep 1
done
if [[ "${OK}" -ne 1 ]]; then
  echo "ERROR: big Ollama failed health check on port ${PORT}; see ${LOG_FILE}"
  exit 1
fi

echo "Big Ollama healthy on port ${PORT}"
echo "Note: ensure required models are present (e.g., llama3.1:70b)."
echo "===== END start_ollama_big $(date -Iseconds) ====="
