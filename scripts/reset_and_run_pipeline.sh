#!/usr/bin/env bash
# Complete GPU Reset + Warmup + Original Pipeline
# Run with: sudo bash scripts/reset_and_run_pipeline.sh

set -euo pipefail

echo "🔥 Step 1: Killing all Ollama processes..."
pkill -9 ollama || true
sleep 3

echo "✅ Ollama processes killed"
echo ""

echo "🧹 Step 2: Clearing GPU memory..."
nvidia-smi --gpu-reset || echo "GPU reset not available, skipping"
sleep 2

echo "✅ GPU memory cleared"
echo ""

echo "🚀 Step 3: Starting 8 Ollama workers..."
bash scripts/start_ollama_workers.sh
sleep 5

echo "✅ Workers started"
echo ""

echo "🌟 Step 4: Starting big Ollama server..."
bash scripts/start_ollama_big.sh
sleep 5

echo "✅ Big server started"
echo ""

echo "🔥 Step 5: Warming up all workers with llama3.1:8b..."
bash scripts/warmup_workers.sh

echo "✅ Warmup complete"
echo ""

echo "📊 Step 6: Running Original Pipeline..."

# Activate venv if it exists
if [ -d ".venv" ]; then
  echo "Activating virtual environment..."
  source .venv/bin/activate
fi

python analysis_pipeline.py \
  --project-name "De Brem Taxatie" \
  --files fatrag_data/uploads/20251031_concept_taxatierapport_De_Brem.pdf

echo ""
echo "✅ Analysis complete! Check outputs/job-<timestamp>/ for results"
