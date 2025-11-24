#!/usr/bin/env bash
set -e

# FATRAG Fresh Run - Complete Pipeline with Vector DB Reset
# This script orchestrates a full FATRAG analysis from scratch:
# 1. Check/start Ollama workers
# 2. Reset and reindex vector database
# 3. Run complete FATRAG pipeline
# 4. Generate final reports and visualizations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment if available
if [ -f .env ]; then
    echo "[INFO] Loading .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Default research question for financial/fiscal analysis
DEFAULT_QUESTION="Algemene financiële en fiscale synthese van alle beschikbare documenten"
RESEARCH_QUESTION="${1:-$DEFAULT_QUESTION}"

echo "================================================================================"
echo "FATRAG FRESH RUN - Autonomous AI Orchestrator"
echo "================================================================================"
echo "Project: $(pwd)"
echo "Research Question: $RESEARCH_QUESTION"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Step 1: Check Python dependencies
echo "[STEP 1/5] Checking Python dependencies..."
if ! python3 -c "import chromadb, langchain_ollama, langchain_community" 2>/dev/null; then
    echo "ERROR: Missing required Python packages"
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi
echo "✓ Python dependencies OK"
echo ""

# Step 2: Check Ollama installation
echo "[STEP 2/5] Checking Ollama installation..."
if ! command -v ollama &> /dev/null; then
    echo "ERROR: ollama command not found"
    echo "Please install Ollama from https://ollama.ai"
    exit 1
fi

# Check if any models are available
MODEL_COUNT=$(ollama list 2>/dev/null | tail -n +2 | wc -l)
if [ "$MODEL_COUNT" -lt 1 ]; then
    echo "ERROR: No Ollama models found"
    echo "Please pull at least one model, e.g.: ollama pull llama3.1:8b"
    exit 1
fi
echo "✓ Ollama installed with $MODEL_COUNT model(s)"
echo ""

# Step 3: Start Ollama workers (if needed)
echo "[STEP 3/5] Checking Ollama workers..."
if [ -f scripts/start_ollama_workers.sh ]; then
    bash scripts/start_ollama_workers.sh
    echo "✓ Ollama workers started/verified"
else
    echo "WARNING: scripts/start_ollama_workers.sh not found"
    echo "Assuming Ollama is already running on port 11434"
fi
echo ""

# Step 4: Reset and reindex vector database
echo "[STEP 4/5] Resetting and reindexing vector database..."
echo "This will clear the existing Chroma collection and re-embed all documents."
python3 scripts/reset_and_reindex.py
if [ $? -ne 0 ]; then
    echo "ERROR: Vector database reset failed"
    exit 1
fi
echo "✓ Vector database reset and reindex complete"
echo ""

# Step 5: Run FATRAG analysis pipeline
echo "[STEP 5/5] Running FATRAG analysis pipeline..."
echo "Research Question: $RESEARCH_QUESTION"
echo ""

python3 scripts/fatrag_auto.py --question "$RESEARCH_QUESTION"
EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ FATRAG PIPELINE COMPLETE"
    echo "================================================================================"
    echo ""
    echo "Output locations:"
    echo "  - Final report (MD):  ./report/final/final.md"
    echo "  - Final report (PDF): ./report/final/final.pdf"
    echo "  - Organogram:         ./report/assets/org.mmd"
    echo "  - Charts:             ./report/assets/*.png"
    echo "  - L1 Analysis:        ./outputs/l1/*.jsonl"
    echo "  - L2 Synthesis:       ./outputs/l2/l2.json"
    echo "  - Evidence CSV:       ./outputs/evidence.csv"
    echo "  - Status log:         ./logs/status.log"
    echo ""
    echo "Configuration:"
    echo "  - config.yaml:        $(pwd)/config.yaml"
    echo ""
else
    echo "❌ FATRAG PIPELINE FAILED (exit code: $EXIT_CODE)"
    echo "================================================================================"
    echo ""
    echo "Check logs for details:"
    echo "  - Status log:  ./logs/status.log"
    echo "  - Ollama logs: ./logs/ollama_gpu*.log"
    echo ""
fi

exit $EXIT_CODE
