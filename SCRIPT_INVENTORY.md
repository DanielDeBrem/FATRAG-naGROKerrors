# FATRAG Script Inventory & Cleanup Plan

Gegenereerd: 2025-11-12 06:10

## ✅ SCRIPTS OM TE BEHOUDEN (Core Infrastructure)

### Core RAG Infrastructure
- **ingestion.py** - PDF/text ingestion, chunking, ChromaDB embedding
- **rag_diagnostics.py** - RAG queries, evidence extraction, synthesis
- **build_index.py** - Index building
- **analysis.py** - Core analysis utilities

### Configuration & Storage
- **config_store.py** / **config_store_mysql.py** - Config management
- **db_models.py** - Database models
- **chunk_store_mysql.py** - Chunk storage
- **job_store_mysql.py** - Job tracking
- **metrics_store.py** - Metrics storage

### Main Application
- **main.py** - FastAPI application
- **clients_projects.py** - Client/project management
- **rag_enhancements.py** - RAG enhancements

### Scripts - Keep
- **scripts/reset_and_reindex.py** - Utility for re-indexing
- **scripts/detect_vdb.py** - VDB detection

## 🆕 NIEUWE SCRIPTS TE MAKEN

### Voor RAG-based Document Analysis
1. **scripts/rag_per_document_analyzer.py**
   - Uses: `ingestion.py` functions
   - Uses: `rag_diagnostics.py` query patterns
   - Purpose: Analyze each document via RAG queries
   - Output: JSON summary per document

2. **scripts/combine_business_structure.py**
   - Input: All document JSONs
   - Purpose: Cross-document synthesis
   - Output: Complete business structure report

## ❌ SCRIPTS TE VERWIJDEREN (Failed Approaches)

### One-Shot Analysis Attempts (Failed - Context too large)
- **scripts/oneshot_analysis.py** (8.1K) - Direct one-shot attempt
- **scripts/filtered_oneshot_analysis.py** (12K) - Failed filtering
- **scripts/batch_oneshot_analysis.py** (8.0K) - Batch one-shot
- **scripts/check_document_sizes.py** (4.1K) - Size checker for one-shot

### Map-Reduce Attempts (Failed - Reduce phase issues)
- **scripts/chunked_progressive_analysis.py** (16K) - Map-reduce single GPU
- **scripts/chunked_progressive_analysis_8gpu.py** (15K) - Map-reduce 8 GPUs
- **analysis_pipeline.py** - Map-reduce pipeline
- **scripts/reduce_resume.py** (5.6K) - Resume reduce
- **scripts/finalize_from_summaries.py** (5.3K) - Map-reduce finalize

### Test/Development Scripts
- **scripts/flash_analysis.py** (23K) - Flash analysis (not relevant)
- **scripts/configurable_analysis.py** (6.7K) - Configurable approach
- **scripts/auto_test_analyses.py** (14K) - Auto testing
- **scripts/smoke_configurable_analysis.py** (2.5K) - Smoke test
- **scripts/smoke_tuner.py** (3.0K) - Tuner test
- **scripts/test_background_progressive.py** (4.0K) - Background test
- **scripts/tuner.py** (16K) - Parameter tuning

### Utility Scripts (Keep for now, maybe cleanup later)
- **scripts/check_jobs.py** (4.5K) - Job checker
- **scripts/fatrag_auto.py** (41K) - Auto runner
- **scripts/warmup_workers.sh** - GPU warmup
- **scripts/start_ollama_*.sh** - Ollama startup scripts

## 📊 STATISTICS

**Total Scripts:** 19 Python scripts
**To Keep:** 4 core + 3 new = 7 relevant scripts
**To Remove:** 15 failed/test scripts
**Space to Reclaim:** ~170KB of dead code

## 🎯 NEW APPROACH: RAG-Based Per-Document Analysis

### Why This Works
1. **No context limits** - Each document processed separately via RAG
2. **Better quality** - RAG retrieves only relevant chunks
3. **Proven infrastructure** - Uses existing working code
4. **Scalable** - Can handle documents of any size

### Workflow
```
For each PDF:
  1. Ingest → Chunk → Embed (ingestion.py)
  2. Query with targeted questions (rag_diagnostics.py patterns)
  3. Extract structured info (entities, transactions, amounts)
  4. Save JSON summary

Then:
  5. Load all JSONs
  6. Cross-document synthesis
  7. Generate business structure diagram
```

### Components Used
- **ingestion.py:**
  - `read_pdf_file()` - Extract PDF
  - `chunk_texts()` - Chunk text
  - `ingest_files()` - Embed in ChromaDB

- **rag_diagnostics.py:**
  - `retrieve()` - Similarity search
  - `extract_evidence_l1()` - Per-chunk extraction
  - `synthesize_l2()` - Cross-chunk synthesis

## 🧹 CLEANUP COMMANDS (After Verification)

```bash
# Remove failed one-shot attempts
rm scripts/oneshot_analysis.py
rm scripts/filtered_oneshot_analysis.py
rm scripts/batch_oneshot_analysis.py
rm scripts/check_document_sizes.py

# Remove failed map-reduce attempts
rm scripts/chunked_progressive_analysis.py
rm scripts/chunked_progressive_analysis_8gpu.py
rm analysis_pipeline.py
rm scripts/reduce_resume.py
rm scripts/finalize_from_summaries.py

# Remove test/dev scripts
rm scripts/flash_analysis.py
rm scripts/configurable_analysis.py
rm scripts/auto_test_analyses.py
rm scripts/smoke_configurable_analysis.py
rm scripts/smoke_tuner.py
rm scripts/test_background_progressive.py
rm scripts/tuner.py
```

**WAIT** until new RAG approach is verified working before cleanup!
