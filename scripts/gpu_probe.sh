#!/usr/bin/env bash
# GPU utilization probe for FATRAG + Ollama workers
# - Starts nvidia-smi sampler per test
# - Triggers workload on each path
# - Prints max GPU utilization observed
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8020}"
LOGDIR="logs"
mkdir -p "$LOGDIR"

say() { printf "%s\n" "$*"; }

monitor_start() {
  local label="$1"
  if command -v nvidia-smi >/dev/null 2>&1; then
    rm -f "${LOGDIR}/gpu_${label}.csv" ".gpu_${label}.pid" 2>/dev/null || true
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,power.draw --format=csv -l 1 > "${LOGDIR}/gpu_${label}.csv" 2>&1 & echo $! > ".gpu_${label}.pid"
  else
    echo "nvidia-smi not available" > "${LOGDIR}/gpu_${label}.csv"
  fi
}

monitor_stop() {
  local label="$1"
  if [ -f ".gpu_${label}.pid" ]; then
    kill "$(cat ".gpu_${label}.pid")" 2>/dev/null || true
    rm -f ".gpu_${label}.pid"
  fi
  say "--- GPU tail [${label}] ---"
  tail -n 15 "${LOGDIR}/gpu_${label}.csv" || true
  awk -F, 'NR>1{gsub(/ %/,"",$4); if(($4+0)>m) m=$4+0} END{printf "Max GPU util [%s]: %s%%\n", "'"${label}"'", (m==""?0:m)}' "${LOGDIR}/gpu_${label}.csv" || true
  echo
}

wait_health() {
  for i in $(seq 1 60); do
    code="$(curl -s -o /dev/null -w "%{http_code}" "${BASE}/health" || true)"
    if [ "$code" = "200" ]; then say "Health OK at ${i}s"; return 0; fi
    sleep 1
  done
  say "Health NOT OK after 60s"; return 1
}

resolve_project_id() {
  local pj tmpfile
  tmpfile="$(mktemp)"
  curl -sS "${BASE}/admin/projects" -o "$tmpfile" || true
  pj="$(python3 - <<'PY' "$tmpfile"
import sys, json
try:
  with open(sys.argv[1],'r',encoding='utf-8') as f:
    obj=json.load(f)
  prj=(obj.get("projects") or [])
  pid=prj[0].get("project_id") or prj[0].get("id") or ""
  print(pid)
except Exception:
  print("")
PY
)"
  rm -f "$tmpfile"
  if [ -n "$pj" ]; then printf "%s" "$pj"; return 0; fi

  # Create client + project
  local cli_json cid proj_json pid tmp
  tmp="$(mktemp)"; printf '{"name":"GPU Smoke Client","type":"individual"}' > "$tmp"
  cli_json="$(curl -sS -X POST "${BASE}/admin/clients" -H "Content-Type: application/json" --data-binary @"$tmp" || true)"
  rm -f "$tmp"
  cid="$(python3 - <<'PY'
import sys,json,io
s=sys.stdin.read()
try:
  obj=json.loads(s); c=obj.get("client") or {}
  print(c.get("client_id") or c.get("id") or "")
except Exception:
  print("")
PY <<<"$cli_json")"

  tmp="$(mktemp)"; printf '{"client_id":"%s","name":"GPU Smoke Project","type":"general"}' "$cid" > "$tmp"
  proj_json="$(curl -sS -X POST "${BASE}/admin/projects" -H "Content-Type: application/json" --data-binary @"$tmp" || true)"
  rm -f "$tmp"
  pid="$(python3 - <<'PY'
import sys,json,io
s=sys.stdin.read()
try:
  obj=json.loads(s); p=obj.get("project") or {}
  print(p.get("project_id") or p.get("id") or "")
except Exception:
  print("")
PY <<<"$proj_json")"
  printf "%s" "$pid"
}

print_api_env() {
  say "=== API process OLLAMA env ==="
  for pid in $(pgrep -f "python.*main.py" || true); do
    echo "-- PID $pid"
    tr '\0' '\n' < "/proc/$pid/environ" | grep -E "OLLAMA|WORKER_PORTS" || echo "(no OLLAMA_* vars found)"
  done
  echo
}

# Main
say "=== Checking health ==="
wait_health

PROJECT_ID="$(resolve_project_id)"
if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" = "null" ]; then
  say "Could not resolve project_id"; exit 1
fi
say "Using PROJECT_ID=$PROJECT_ID"
print_api_env

# Test 1: /query (parallel x8) to drive round-robin routing
say "=== Test 1: /query parallel x8 ==="
body1="$(mktemp)"
# Long prompt to ensure GPU work
printf '{"question":"Schrijf 1200 woorden analyse over BTW facturatiecomplexiteit met voorbeelden en berekeningen. Voeg bullets toe en rekentabellen.","project_id":"%s"}' "$PROJECT_ID" > "$body1"
monitor_start "query"
pids=()
for i in $(seq 1 8); do
  curl -sS -X POST "${BASE}/query" -H "Content-Type: application/json" --data-binary @"$body1" >/dev/null 2>&1 & pids+=($!)
done
wait "${pids[@]}" || true
rm -f "$body1"
sleep 8
monitor_stop "query"

# Test 2: Flash Analysis
say "=== Test 2: Flash Analysis ==="
monitor_start "flash"
curl -sS -X POST "${BASE}/admin/projects/${PROJECT_ID}/flash-analysis" >/dev/null 2>&1 || true
sleep 20
monitor_stop "flash"

# Test 3: Analyze All (async)
say "=== Test 3: Analyze All (async) ==="
monitor_start "analyze"
curl -sS -X POST "${BASE}/admin/projects/${PROJECT_ID}/analyze-all/async" >/dev/null 2>&1 || true
sleep 20
monitor_stop "analyze"

# Test 4: Template Report
say "=== Test 4: Template Report ==="
body4="$(mktemp)"
printf '{"template_key":"holding_analysis"}' > "$body4"
monitor_start "template"
curl -sS -X POST "${BASE}/admin/projects/${PROJECT_ID}/generate-from-template" -H "Content-Type: application/json" --data-binary @"$body4" >/dev/null 2>&1 || true
rm -f "$body4"
sleep 12
monitor_stop "template"

# Test 5: FATRAG pipeline
say "=== Test 5: FATRAG Pipeline ==="
body5="$(mktemp)"
printf '{"research_question":"Korte synthese van alle documenten met nadruk op getallen/percentages."}' > "$body5"
monitor_start "pipeline"
curl -sS -X POST "${BASE}/admin/projects/${PROJECT_ID}/fatrag-pipeline" -H "Content-Type: application/json" --data-binary @"$body5" >/dev/null 2>&1 || true
rm -f "$body5"
sleep 25
monitor_stop "pipeline"

# Summaries
say "=== Summary of GPU logs ==="
ls -l "${LOGDIR}"/gpu_*.csv 2>/dev/null || true
say "Done."
