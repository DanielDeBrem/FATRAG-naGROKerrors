#!/usr/bin/env bash
# Warm up all 8 Ollama workers by loading llama3.1:8b into GPU memory
# This prevents the 120 sec model loading delay on first inference

set -euo pipefail

MODEL="llama3.1:8b"
PORTS=(11434 11435 11436 11437 11438 11439 11440 11441)

echo "🔥 Warming up 8 Ollama workers with $MODEL..."
echo ""

# Start all warmup requests in parallel
for port in "${PORTS[@]}"; do
  (
    echo "   Worker port $port: loading model..."
    start_time=$(date +%s)
    
    response=$(curl -s -X POST "http://localhost:$port/api/generate" \
      -d "{\"model\":\"$MODEL\",\"prompt\":\"warmup\",\"stream\":false,\"options\":{\"num_predict\":1}}" \
      --max-time 180 2>&1)
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if echo "$response" | grep -q '"response"'; then
      echo "   ✓ Worker port $port ready (${duration}s)"
    else
      echo "   ✗ Worker port $port FAILED"
    fi
  ) &
done

# Wait for all to complete
wait

echo ""
echo "✅ All workers warmed up! Model loaded in GPU memory."
echo ""
echo "You can now run analysis:"
echo "  python scripts/flash_analysis.py --project-name 'Test' --files your-file.pdf"
