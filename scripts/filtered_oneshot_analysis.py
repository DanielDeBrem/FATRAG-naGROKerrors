#!/usr/bin/env python3
"""
Two-Stage Filtered One-Shot Analysis
Stage 1: Filter document with fast model (llama3.1:8b) - extract only financially relevant content
Stage 2: Deep analysis with large model (llama3.1:70b) - comprehensive analysis of filtered content
"""

import os
import json
import time
import urllib.request
import urllib.error
from datetime import datetime

# Configuration
FILTER_MODEL = "llama3.1:8b"
FILTER_PORT = 11434  # Worker on first GPU
ANALYSIS_MODEL = "llama3.1:70b"
ANALYSIS_PORT = 11450  # Big server with all GPUs
TEMPERATURE = 0.2
FILTER_NUM_CTX = 32000  # Smaller for filtering
ANALYSIS_NUM_CTX = 128000  # Full context for analysis
TIMEOUT = 900

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(ROOT_DIR, "outputs")


def read_pdf_file(filepath: str) -> str:
    """Extract text from PDF using pdfplumber."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except Exception as e:
        return f"[PDF extraction error: {str(e)}]"


def ollama_generate(prompt: str, model: str, port: int, num_ctx: int) -> str:
    """Call Ollama /api/generate endpoint."""
    url = f"http://127.0.0.1:{port}/api/generate"
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx": num_ctx,
        },
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            text = resp.read().decode("utf-8")
            j = json.loads(text)
            return j.get("response", "") if isinstance(j, dict) else str(j)
    except Exception as e:
        raise RuntimeError(f"Ollama error on port {port}: {str(e)}")


def build_filter_prompt(text: str) -> str:
    """Build prompt for filtering stage - extract only relevant financial content."""
    return f"""
TAAK: Extract ALLEEN financieel relevante informatie uit het taxatierapport.

VERWIJDER / NEGEER:
- Voorwaarden en algemene bepalingen van de taxateur
- Uitgebreide object beschrijvingen (architectuur, indeling, afmetingen)
- Standard disclaimers en wettelijke teksten
- Redundante herhaling van informatie
- Namen en adressen van natuurlijke personen (privacy)

BEHOUD EN EXTRACT:
- Alle bedragen en prijzen (€)
- Alle percentages, rentes, indices
- Taxatiewaarden en waarderingsmethoden
- Financiële structuren en verhoudingen
- Partijen (bedrijven, rechtspersonen)
- Eigendomsstructuur
- Datums van belang
- Fiscale informatie
- Contractuele verhoudingen
- Belangrijke voorwaarden met financiële impact

DOCUMENT:
\"\"\"
{text}
\"\"\"

Geef ALLEEN de relevante geëxtraheerde content terug (Nederlands).
Structureer in secties maar wees zeer beknopt.
Focus op data die nodig is voor financiële/fiscale analyse.
""".strip()


def build_analysis_prompt(project_name: str, filename: str, filtered_text: str) -> str:
    """Build prompt for analysis stage - comprehensive analysis of filtered content."""
    return f"""
STRICT_LANGUAGE_MODE: Nederlands

Je bent "FinAdviseur-NL", senior financieel specialist (NL, 2025).

TAAK: Analyseer de gefilterde financiële informatie volledig en grondig.

PROJECT: {project_name}
DOCUMENT: {filename}

GEFILTERDE FINANCIËLE DATA:
\"\"\"
{filtered_text}
\"\"\"

GEVRAAGD EINDRESULTAAT (allemaal bullets, Nederlands):

**TL;DR** (4-6 bullets, kern financieel/fiscaal)
- ...

**Financiële kernpunten**
- Alle relevante bedragen, percentages, waarderingen
- Vermeld exacte cijfers uit de data
- Financiële structuur en verhoudingen
- ...

**Fiscale impact en aandachtspunten**
- BTW, VPB, overdrachtsbelasting, etc.
- Specifieke aandachtspunten voor 2025
- ...

**Juridische structuur & partijen**
- Alle betrokken partijen en hun rollen
- Eigendomsstructuur
- Contractuele verhoudingen
- ...

**Taxatiemethode en waardering**
- Gebruikte waarderingsmethoden
- Onderbouwing waardering
- Belangrijke aannames
- ...

**Risico's, aannames, onzekerheden**
- Identificeer alle risico's
- Expliciete aannames
- Waar data ontbreekt: schrijf "onvoldoende data"
- ...

**Aanpak / stappenplan**
- Concrete, uitvoerbare stappen
- Prioritering
- Benodigde vervolgacties
- ...

**Open vragen** (max 5)
- Kritische vragen voor definitief advies
- ...

REGELS:
- Taal: ALLEEN Nederlands
- Stijl: Professioneel, beknopt, bullet-points
- Geen PII
- Gebruik ALLEEN informatie uit de gefilterde data
- Exacte cijfers en data vermelden
""".strip()


def run_filtered_oneshot_analysis(pdf_path: str, project_name: str = "Analyse") -> dict:
    """
    Run two-stage filtered analysis:
    1. Filter with small model (extract relevant financial content)
    2. Analyze with large model (comprehensive analysis)
    """
    print("=" * 60)
    print("🚀 TWO-STAGE FILTERED ANALYSIS")
    print("=" * 60)
    print(f"📄 Document: {pdf_path}")
    print(f"🔧 Filter Model: {FILTER_MODEL} on port {FILTER_PORT}")
    print(f"🧠 Analysis Model: {ANALYSIS_MODEL} on port {ANALYSIS_PORT}")
    print()
    
    # Create output directory
    job_id = time.strftime("filtered-%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUT_ROOT, job_id)
    os.makedirs(out_dir, exist_ok=True)
    
    filename = os.path.basename(pdf_path)
    
    # Step 1: Extract PDF text
    print("📖 STEP 1: Extracting PDF text...")
    start_time = time.time()
    raw_text = read_pdf_file(pdf_path)
    extract_time = time.time() - start_time
    
    if raw_text.startswith("[PDF extraction error"):
        raise RuntimeError(f"PDF extraction failed: {raw_text}")
    
    raw_char_count = len(raw_text)
    raw_token_estimate = raw_char_count // 4
    
    print(f"   ✓ Extracted {raw_char_count:,} characters (~{raw_token_estimate:,} tokens)")
    print(f"   ⏱️  Time: {extract_time:.1f}s")
    print()
    
    # Save raw text
    raw_text_path = os.path.join(out_dir, "01_raw_text.txt")
    with open(raw_text_path, "w", encoding="utf-8") as f:
        f.write(raw_text)
    print(f"   💾 Saved: {raw_text_path}")
    print()
    
    # Step 2: Filter content (Stage 1)
    print("🔧 STEP 2: Filtering content (Stage 1)...")
    print(f"   📊 Using {FILTER_MODEL} to extract relevant financial data...")
    print(f"   ⏱️  This may take 3-5 minutes...")
    print()
    
    filter_start = time.time()
    filter_prompt = build_filter_prompt(raw_text)
    
    try:
        filtered_text = ollama_generate(filter_prompt, FILTER_MODEL, FILTER_PORT, FILTER_NUM_CTX)
    except Exception as e:
        raise RuntimeError(f"Filtering failed: {str(e)}")
    
    filter_time = time.time() - filter_start
    
    filtered_char_count = len(filtered_text)
    filtered_token_estimate = filtered_char_count // 4
    reduction_pct = 100 * (1 - filtered_token_estimate / raw_token_estimate)
    
    print(f"   ✓ Filtering complete!")
    print(f"   📊 Original: {raw_token_estimate:,} tokens")
    print(f"   📊 Filtered: {filtered_token_estimate:,} tokens")
    print(f"   📉 Reduction: {reduction_pct:.1f}%")
    print(f"   ⏱️  Time: {filter_time/60:.1f} minutes")
    print()
    
    # Check if filtered content fits in analysis window
    if filtered_token_estimate > ANALYSIS_NUM_CTX * 0.7:
        print(f"   ⚠️  WARNING: Filtered content still large ({filtered_token_estimate:,} tokens)")
        print(f"   ⚠️  May not fit in {ANALYSIS_NUM_CTX:,} context window")
        print()
    
    # Save filtered text
    filtered_text_path = os.path.join(out_dir, "02_filtered_text.txt")
    with open(filtered_text_path, "w", encoding="utf-8") as f:
        f.write(filtered_text)
    print(f"   💾 Saved: {filtered_text_path}")
    print()
    
    # Step 3: Deep analysis (Stage 2)
    print("🧠 STEP 3: Deep analysis (Stage 2)...")
    print(f"   📊 Using {ANALYSIS_MODEL} for comprehensive analysis...")
    print(f"   ⏱️  This may take 8-12 minutes...")
    print()
    
    analysis_start = time.time()
    analysis_prompt = build_analysis_prompt(project_name, filename, filtered_text)
    
    try:
        result = ollama_generate(analysis_prompt, ANALYSIS_MODEL, ANALYSIS_PORT, ANALYSIS_NUM_CTX)
    except Exception as e:
        raise RuntimeError(f"Analysis failed: {str(e)}")
    
    analysis_time = time.time() - analysis_start
    
    print(f"   ✓ Analysis complete!")
    print(f"   ⏱️  Time: {analysis_time/60:.1f} minutes")
    print()
    
    # Step 4: Save outputs
    print("💾 STEP 4: Saving outputs...")
    
    # Save main result
    result_path = os.path.join(out_dir, "03_final_analysis.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"   ✓ Saved: {result_path}")
    
    # Save metadata
    total_time = extract_time + filter_time + analysis_time
    metadata = {
        "job_id": job_id,
        "timestamp": datetime.now().isoformat(),
        "project_name": project_name,
        "document": filename,
        "models": {
            "filter": FILTER_MODEL,
            "analysis": ANALYSIS_MODEL,
        },
        "stats": {
            "raw_chars": raw_char_count,
            "raw_tokens_estimate": raw_token_estimate,
            "filtered_chars": filtered_char_count,
            "filtered_tokens_estimate": filtered_token_estimate,
            "reduction_percentage": round(reduction_pct, 2),
            "extraction_time_seconds": round(extract_time, 2),
            "filtering_time_seconds": round(filter_time, 2),
            "analysis_time_seconds": round(analysis_time, 2),
            "total_time_seconds": round(total_time, 2),
        }
    }
    
    meta_path = os.path.join(out_dir, "00_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved: {meta_path}")
    print()
    
    # Summary
    print("=" * 60)
    print("✅ TWO-STAGE FILTERED ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"📁 Output directory: {out_dir}")
    print(f"📄 Main result: {result_path}")
    print(f"📊 Token reduction: {raw_token_estimate:,} → {filtered_token_estimate:,} ({reduction_pct:.1f}%)")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print()
    print("📂 Files created:")
    print(f"   00_metadata.json       - Stats and timing info")
    print(f"   01_raw_text.txt        - Original PDF extraction")
    print(f"   02_filtered_text.txt   - Filtered financial content")
    print(f"   03_final_analysis.txt  - Complete analysis (MAIN RESULT)")
    print()
    
    return metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Two-stage filtered analysis with llama3.1:8b + llama3.1:70b"
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to PDF file to analyze"
    )
    parser.add_argument(
        "--project-name",
        default="Analyse",
        help="Name of the project for the analysis"
    )
    
    args = parser.parse_args()
    
    # Validate file exists
    if not os.path.isfile(args.file):
        print(f"❌ Error: File not found: {args.file}")
        exit(1)
    
    # Run analysis
    try:
        result = run_filtered_oneshot_analysis(args.file, args.project_name)
        print("🎉 Success!")
        exit(0)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
