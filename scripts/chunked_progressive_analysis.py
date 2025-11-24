#!/usr/bin/env python3
"""
Chunked Progressive Analysis - RTX 3060 Ti Optimized

Splits documents into small chunks, processes them in parallel on RTX 3060 Ti,
then hierarchically aggregates results. Much faster than full-document analysis.

Usage:
    python scripts/chunked_progressive_analysis.py --project-id project-123 --doc-path path/to/doc.pdf
"""

import os
import sys
import json
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ingestion as ing
import chunk_store_mysql as cs

# Configuration
CHUNK_SIZE = 450  # tokens - optimal for financial docs
CHUNK_OVERLAP = 75  # tokens - 17% overlap for continuity
MODEL_NAME = "llama3.1:8b"  # RTX 3060 Ti friendly
WORKER_COUNT = 2  # Parallel workers
MAX_RETRIES = 3
OLLAMA_BASE_PORT = int(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").split(":")[-1])

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def ollama_generate(prompt: str, model: str, port: int = OLLAMA_BASE_PORT, timeout: int = 60) -> str:
    """Call Ollama API"""
    import urllib.request
    import urllib.error
    
    url = f"http://127.0.0.1:{port}/api/generate"
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.15,
            "num_ctx": 2048
        }
    }).encode("utf-8")
    
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
            j = json.loads(text)
            return j.get("response", "") if isinstance(j, dict) else str(j)
    except Exception as e:
        raise RuntimeError(f"Ollama error on port {port}: {str(e)}")


def build_chunk_analysis_prompt(doc_name: str, chunk_index: int, total_chunks: int, chunk_text: str) -> str:
    """Build prompt for chunk analysis"""
    return f"""Je analyseert chunk {chunk_index+1}/{total_chunks} uit {doc_name}.

Extract ALLEEN feitelijke informatie, Nederlands, max 12 bullets:

## Financieel
- Bedragen (EUR/€)
- Percentages, rentes
- Berekeningen

## Juridisch
- Partijen/entiteiten
- Contractvoorwaarden
- Rechten/verplichtingen

## Temporeel
- Data/deadlines
- Termijnen

## Risico's
- Onzekerheden
- Aannames
- "onvoldoende data" waar info ontbreekt

GEEN interpretatie, alleen extractie. Gebruik exacte cijfers uit tekst.

TEKST:
\"\"\"
{chunk_text[:2000]}
\"\"\"

Bullets:"""


def build_aggregation_prompt(section_name: str, chunk_results: List[str]) -> str:
    """Build prompt for aggregating chunk results"""
    joined = "\n\n".join(chunk_results[:50])  # Limit for context
    
    return f"""Combineer chunk-analyses tot een samenvatting voor sectie: {section_name}

CHUNK RESULTATEN:
{joined}

Maak een Dutch, bullet-pointed samenvatting (max 20 bullets) met:
- Consolideer duplicaten
- Groepeer gerelateerde items
- Behoud exacte cijfers/percentages
- Mark "onvoldoende data" waar bewijs ontbreekt
- Focus op financieel + juridisch + risico

Bullets:"""


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF"""
    return ing.read_pdf_file(pdf_path)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks"""
    return ing.chunk_texts([text], chunk_size=chunk_size, chunk_overlap=overlap)


def process_chunk(
    chunk_id: str,
    job_id: str,
    doc_name: str,
    chunk_index: int,
    total_chunks: int,
    chunk_text: str,
    model: str,
    port: int
) -> Dict[str, Any]:
    """Process a single chunk"""
    start_time = time.time()
    
    try:
        # Mark as processing
        cs.update_chunk(chunk_id, status='processing')
        
        # Build prompt and call LLM
        prompt = build_chunk_analysis_prompt(doc_name, chunk_index, total_chunks, chunk_text)
        result = ollama_generate(prompt, model, port=port, timeout=60)
        
        processing_time = time.time() - start_time
        
        # Store result
        result_data = {
            "chunk_index": chunk_index,
            "analysis": result,
            "timestamp": datetime.now().isoformat()
        }
        
        cs.update_chunk(
            chunk_id,
            status='completed',
            result_json=json.dumps(result_data, ensure_ascii=False),
            processing_time_sec=processing_time
        )
        
        return {
            "success": True,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "result": result,
            "processing_time": processing_time
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        
        cs.update_chunk(
            chunk_id,
            status='failed',
            error_message=error_msg,
            processing_time_sec=processing_time
        )
        
        return {
            "success": False,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "error": error_msg,
            "processing_time": processing_time
        }


def aggregate_chunks_hierarchical(
    job_id: str,
    chunk_results: List[Dict[str, Any]],
    doc_name: str,
    model: str,
    port: int
) -> str:
    """Hierarchically aggregate chunk results"""
    
    # Sort by chunk index
    sorted_results = sorted(chunk_results, key=lambda x: x.get('chunk_index', 0))
    
    # Group into sections (every 20 chunks)
    section_size = 20
    sections = []
    
    for i in range(0, len(sorted_results), section_size):
        section_chunks = sorted_results[i:i+section_size]
        section_texts = [r['result'] for r in section_chunks if r.get('result')]
        
        if section_texts:
            section_name = f"Sectie {i//section_size + 1}"
            prompt = build_aggregation_prompt(section_name, section_texts)
            
            try:
                section_summary = ollama_generate(prompt, model, port=port, timeout=90)
                sections.append({
                    'name': section_name,
                    'summary': section_summary,
                    'chunk_range': f"{i}-{i+len(section_chunks)-1}"
                })
            except Exception as e:
                sections.append({
                    'name': section_name,
                    'summary': f"⚠️ Aggregatie mislukt: {str(e)}",
                    'chunk_range': f"{i}-{i+len(section_chunks)-1}"
                })
    
    # Final aggregation if multiple sections
    if len(sections) > 1:
        final_prompt = f"""Combineer sectie-samenvattingen tot finale analyse voor: {doc_name}

SECTIES:
{chr(10).join([f"## {s['name']}{chr(10)}{s['summary']}" for s in sections])}

Maak een Dutch, gestructureerde finale analyse met:

## Samenvatting (TL;DR)
- Kern bevindingen (3-5 bullets)

## Financieel Detail
- Bedragen, waarderingen
- Percentages, rentes
- Berekeningen

## Juridische Structuur
- Partijen, rollen
- Contractvoorwaarden
- Rechten/plichten

## Temporele Aspecten
- Data, deadlines
- Termijnen, looptijden

## Risico's & Aannames
- Onzekerheden
- Aannames
- "onvoldoende data" waar relevant

Max 40 bullets totaal. Gebruik exacte cijfers."""
        
        try:
            final_analysis = ollama_generate(final_prompt, model, port=port, timeout=120)
        except Exception as e:
            final_analysis = f"⚠️ Finale aggregatie mislukt: {str(e)}\n\n" + \
                           "\n\n".join([f"## {s['name']}\n{s['summary']}" for s in sections])
    else:
        final_analysis = sections[0]['summary'] if sections else "onvoldoende data"
    
    return final_analysis


def run_chunked_analysis(
    project_id: str,
    doc_path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    model: str = MODEL_NAME,
    worker_count: int = WORKER_COUNT
) -> Dict[str, Any]:
    """Run complete chunked analysis pipeline"""
    
    # Generate job ID
    job_id = f"chunked-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(OUTPUT_ROOT, job_id)
    os.makedirs(output_dir, exist_ok=True)
    
    doc_name = os.path.basename(doc_path)
    
    print(f"🚀 Starting chunked analysis: {job_id}")
    print(f"   Document: {doc_name}")
    print(f"   Model: {model}")
    print(f"   Workers: {worker_count}")
    print()
    
    # Phase 1: Extract & Chunk
    print("📄 Phase 1: Extracting & chunking...")
    text = extract_text_from_pdf(doc_path)
    
    if not text or text.startswith("["):
        raise RuntimeError(f"Failed to extract text from {doc_path}")
    
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
    total_chunks = len(chunks)
    
    print(f"   ✓ Extracted {len(text)} chars")
    print(f"   ✓ Created {total_chunks} chunks")
    print()
    
    # Create job in DB
    cs.create_chunk_job(
        job_id=job_id,
        project_id=project_id,
        doc_path=doc_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        total_chunks=total_chunks,
        model_name=model,
        worker_count=worker_count,
        output_dir=output_dir
    )
    
    # Store chunks in DB
    print("💾 Storing chunks in database...")
    for i, chunk in enumerate(chunks):
        chunk_id = cs.generate_chunk_id(job_id, i)
        cs.create_chunk(
            chunk_id=chunk_id,
            job_id=job_id,
            doc_name=doc_name,
            chunk_index=i,
            total_chunks=total_chunks,
            chunk_text=chunk
        )
    print(f"   ✓ Stored {total_chunks} chunks")
    print()
    
    # Phase 2: Process chunks in parallel
    print(f"⚡ Phase 2: Processing {total_chunks} chunks with {worker_count} workers...")
    cs.update_chunk_job(job_id, status='processing')
    
    start_time = time.time()
    completed = 0
    failed = 0
    chunk_results = []
    
    # Determine ports to use (alternate between workers)
    ports = [OLLAMA_BASE_PORT + i for i in range(worker_count)]
    
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {}
        
        for i, chunk in enumerate(chunks):
            chunk_id = cs.generate_chunk_id(job_id, i)
            port = ports[i % len(ports)]
            
            future = executor.submit(
                process_chunk,
                chunk_id, job_id, doc_name, i, total_chunks,
                chunk, model, port
            )
            futures[future] = i
        
        for future in as_completed(futures):
            result = future.result()
            chunk_results.append(result)
            
            if result['success']:
                completed += 1
            else:
                failed += 1
            
            # Update progress
            elapsed = time.time() - start_time
            throughput = completed / (elapsed / 60) if elapsed > 0 else 0
            remaining = total_chunks - completed - failed
            eta = (remaining / throughput) if throughput > 0 else 0
            
            cs.update_chunk_job(
                job_id,
                chunks_completed=completed,
                chunks_failed=failed,
                throughput=round(throughput, 2),
                eta_minutes=round(eta, 1)
            )
            
            # Progress output
            pct = round((completed + failed) / total_chunks * 100, 1)
            print(f"   Progress: {completed}/{total_chunks} ({pct}%) | "
                  f"Throughput: {throughput:.1f} chunks/min | ETA: {eta:.1f} min")
    
    processing_time = time.time() - start_time
    print(f"\n   ✓ Completed in {processing_time/60:.1f} minutes")
    print(f"   ✓ Success: {completed}, Failed: {failed}")
    print()
    
    # Phase 3: Aggregate results
    print("🔄 Phase 3: Aggregating results...")
    cs.update_chunk_job(job_id, status='aggregating')
    
    successful_results = [r for r in chunk_results if r['success']]
    
    if not successful_results:
        error_msg = "All chunks failed processing"
        cs.update_chunk_job(job_id, status='failed', error_message=error_msg)
        raise RuntimeError(error_msg)
    
    final_analysis = aggregate_chunks_hierarchical(
        job_id, successful_results, doc_name, model, OLLAMA_BASE_PORT
    )
    
    print("   ✓ Aggregation complete")
    print()
    
    # Phase 4: Save outputs
    print("💾 Saving outputs...")
    
    # Save final report
    report_lines = [
        f"# Chunked Analysis: {doc_name}",
        f"Gegenereerd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Job ID: {job_id}",
        "",
        f"## Metadata",
        f"- Chunks verwerkt: {completed}/{total_chunks}",
        f"- Processing tijd: {processing_time/60:.1f} minuten",
        f"- Model: {model}",
        f"- Chunk size: {chunk_size} tokens",
        "",
        "## Analyse",
        "",
        final_analysis,
        ""
    ]
    
    report_path = os.path.join(output_dir, "final_analysis.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    # Save metadata
    metadata = {
        "job_id": job_id,
        "project_id": project_id,
        "doc_path": doc_path,
        "doc_name": doc_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "total_chunks": total_chunks,
        "chunks_completed": completed,
        "chunks_failed": failed,
        "model": model,
        "worker_count": worker_count,
        "processing_time_minutes": round(processing_time / 60, 2),
        "throughput_chunks_per_min": round(completed / (processing_time / 60), 2),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    cs.update_chunk_job(job_id, status='completed')
    
    print(f"   ✓ Report saved to: {report_path}")
    print()
    print("✅ Analysis complete!")
    
    return {
        "job_id": job_id,
        "output_dir": output_dir,
        "report_path": report_path,
        "metadata": metadata
    }


if __name__ == "__main__":
    import argparse
    
    p = argparse.ArgumentParser(description="Chunked Progressive Analysis")
    p.add_argument("--project-id", required=True, help="Project ID")
    p.add_argument("--doc-path", required=True, help="Path to PDF document")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Chunk size in tokens")
    p.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help="Chunk overlap in tokens")
    p.add_argument("--model", default=MODEL_NAME, help="Model name")
    p.add_argument("--workers", type=int, default=WORKER_COUNT, help="Number of parallel workers")
    
    args = p.parse_args()
    
    try:
        result = run_chunked_analysis(
            project_id=args.project_id,
            doc_path=args.doc_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            model=args.model,
            worker_count=args.workers
        )
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
