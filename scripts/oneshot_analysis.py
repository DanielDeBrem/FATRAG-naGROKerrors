#!/usr/bin/env python3
"""
One-Shot Document Analysis with llama3.1:70b
Extracts PDF, sends complete text to large model, gets analysis in one go.
Much faster and more reliable than map-reduce pipeline.
"""

import os
import json
import time
import urllib.request
import urllib.error
from datetime import datetime

# Configuration
OLLAMA_BIG_PORT = 11450
MODEL = "llama3.1:70b"
TEMPERATURE = 0.2
NUM_CTX = 128000  # 128K context window
TIMEOUT = 900  # 15 minutes max

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS_DIR = os.path.join(ROOT_DIR, "fatrag_data", "uploads")
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


def ollama_generate(prompt: str, model: str, port: int) -> str:
    """Call Ollama /api/generate endpoint."""
    url = f"http://127.0.0.1:{port}/api/generate"
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
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


def build_analysis_prompt(project_name: str, filename: str, text: str) -> str:
    """Build comprehensive analysis prompt."""
    return f"""
STRICT_LANGUAGE_MODE: Nederlands

Je bent "FinAdviseur-NL", senior financieel specialist (NL, 2025).

TAAK: Analyseer het volgende document volledig en grondig. Lever een professionele, complete Nederlandse analyse.

PROJECT: {project_name}
DOCUMENT: {filename}

DOCUMENT TEKST:
\"\"\"
{text}
\"\"\"

GEVRAAGD EINDRESULTAAT (allemaal bullets, Nederlands):

**TL;DR** (4-6 bullets, kern financieel/fiscaal)
- ...

**Financiële kernpunten**
- Alle relevante bedragen, percentages, waarderingen
- Vermeld exacte cijfers uit het document
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

**Risico's, aannames, onzekerheden**
- Identificeer alle risico's
- Expliciete aannames in het document
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
- Gebruik ALLEEN informatie uit het document
- Geen speculatie of externe informatie
- Exacte cijfers en data uit document vermelden
""".strip()


def run_oneshot_analysis(pdf_path: str, project_name: str = "Analyse") -> dict:
    """
    Run complete one-shot analysis on a single PDF document.
    Returns dict with output paths and metadata.
    """
    print(f"🚀 Starting one-shot analysis...")
    print(f"📄 Document: {pdf_path}")
    print(f"🤖 Model: {MODEL} on port {OLLAMA_BIG_PORT}")
    print()
    
    # Create output directory
    job_id = time.strftime("oneshot-%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUT_ROOT, job_id)
    os.makedirs(out_dir, exist_ok=True)
    
    filename = os.path.basename(pdf_path)
    
    # Step 1: Extract PDF text
    print("📖 Step 1: Extracting PDF text...")
    start_time = time.time()
    text = read_pdf_file(pdf_path)
    extract_time = time.time() - start_time
    
    if text.startswith("[PDF extraction error"):
        raise RuntimeError(f"PDF extraction failed: {text}")
    
    char_count = len(text)
    token_estimate = char_count // 4  # Rough estimate: 4 chars per token
    
    print(f"   ✓ Extracted {char_count:,} characters (~{token_estimate:,} tokens)")
    print(f"   ⏱️  Time: {extract_time:.1f}s")
    print()
    
    # Check if within context window
    if token_estimate > NUM_CTX * 0.8:  # Use 80% to leave room for response
        print(f"   ⚠️  WARNING: Document may be too large ({token_estimate:,} tokens)")
        print(f"   ℹ️  Context window: {NUM_CTX:,} tokens")
        print()
    
    # Step 2: Build prompt
    print("📝 Step 2: Building analysis prompt...")
    prompt = build_analysis_prompt(project_name, filename, text)
    prompt_tokens = len(prompt) // 4
    print(f"   ✓ Prompt ready (~{prompt_tokens:,} tokens)")
    print()
    
    # Step 3: Call Ollama
    print(f"🧠 Step 3: Calling {MODEL}...")
    print(f"   ⚠️  This may take 10-15 minutes...")
    print(f"   💡 Model will load into GPU VRAM if not already warm")
    print()
    
    analysis_start = time.time()
    try:
        result = ollama_generate(prompt, MODEL, OLLAMA_BIG_PORT)
    except Exception as e:
        raise RuntimeError(f"Analysis failed: {str(e)}")
    
    analysis_time = time.time() - analysis_start
    
    print(f"   ✓ Analysis complete!")
    print(f"   ⏱️  Inference time: {analysis_time/60:.1f} minutes")
    print()
    
    # Step 4: Save outputs
    print("💾 Step 4: Saving outputs...")
    
    # Save main result
    result_path = os.path.join(out_dir, "analysis.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"   ✓ Saved: {result_path}")
    
    # Save metadata
    metadata = {
        "job_id": job_id,
        "timestamp": datetime.now().isoformat(),
        "project_name": project_name,
        "document": filename,
        "model": MODEL,
        "port": OLLAMA_BIG_PORT,
        "stats": {
            "document_chars": char_count,
            "document_tokens_estimate": token_estimate,
            "extraction_time_seconds": round(extract_time, 2),
            "analysis_time_seconds": round(analysis_time, 2),
            "total_time_seconds": round(extract_time + analysis_time, 2),
        }
    }
    
    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved: {meta_path}")
    
    # Save extracted text for reference
    text_path = os.path.join(out_dir, "extracted_text.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"   ✓ Saved: {text_path}")
    print()
    
    # Summary
    print("=" * 60)
    print("✅ ONE-SHOT ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"📁 Output directory: {out_dir}")
    print(f"📄 Main result: {result_path}")
    print(f"⏱️  Total time: {(extract_time + analysis_time)/60:.1f} minutes")
    print()
    
    return metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="One-shot document analysis with llama3.1:70b"
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
        result = run_oneshot_analysis(args.file, args.project_name)
        print("🎉 Success!")
        exit(0)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
