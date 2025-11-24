#!/usr/bin/env python3
"""
Chunked Progressive Analysis - 8-GPU MAXIMUM THROUGHPUT

Optimized for SuperMicro server with 8× RTX 3060 Ti GPUs.
Uses 16 parallel workers (2 per GPU) for maximum speed.

Expected performance: ~17 minutes for large documents (vs hours with 2 workers)

Usage:
    python scripts/chunked_progressive_analysis_8gpu.py --project-id test --doc-path path/to/doc.pdf
"""

import os
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ingestion as ing
import chunk_store_mysql as cs

# 8-GPU OPTIMIZED CONFIGURATION
CHUNK_SIZE = 350        # Smaller for faster processing
CHUNK_OVERLAP = 50      # Minimal but sufficient
MODEL_NAME = "llama3.1:8b"
WORKER_COUNT = 16       # 2 workers per GPU for max throughput
CONTEXT_WINDOW = 1536   # Smaller = faster inference

# All 8 Ollama workers (ports 11434-11441)
OLLAMA_PORTS = list(range(11434, 11442))

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def ollama_generate(prompt: str, model: str, port: int, timeout: int = 45) -> str:
    """Call Ollama API with optimized settings"""
    import urllib.request
    
    url = f"http://127.0.0.1:{port}/api/generate"
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,      # Lower = faster
            "num_ctx": CONTEXT_WINDOW,
            "num_predict": 256       # Limit output for speed
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


def build_fast_chunk_prompt(doc_name: str, chunk_index: int, total_chunks: int, chunk_text: str) -> str:
    """Ultra-compact prompt for maximum speed"""
    return f"""Chunk {chunk_index+1}/{total_chunks} uit {doc_name}

Extract max 8 bullets:
- Bedragen (EUR)
- Partijen
- Data
- Percentages/rentes

TEKST:
{chunk_text[:1400]}

Bullets:"""


def build_fast_aggregation_prompt(section_name: str, chunk_results: List[str]) -> str:
    """Fast aggregation prompt"""
    joined = "\n\n".join(chunk_results[:40])
    
    return f"""Consolideer {section_name}:

{joined}

Max 16 bullets, Nederlands, feitelijk:"""


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
        cs.update_chunk(chunk_id, status='processing')
        
        prompt = build_fast_chunk_prompt(doc_name, chunk_index, total_chunks, chunk_text)
        result = ollama_generate(prompt, model, port=port, timeout=45)
        
        processing_time = time.time() - start_time
        
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


def aggregate_chunks_fast(
    job_id: str,
    chunk_results: List[Dict[str, Any]],
    doc_name: str,
    model: str,
    port: int
) -> str:
    """Fast hierarchical aggregation"""
    
    sorted_results = sorted(chunk_results, key=lambda x: x.get('chunk_index', 0))
    
    # Group into sections (25 chunks each for speed)
    section_size = 25
    sections = []
    
    for i in range(0, len(sorted_results), section_size):
        section_chunks = sorted_results[i:i+section_size]
        section_texts = [r['result'] for r in section_chunks if r.get('result')]
        
        if section_texts:
            section_name = f"Sectie {i//section_size + 1}"
            prompt = build_fast_aggregation_prompt(section_name, section_texts)
            
            try:
                section_summary = ollama_generate(prompt, model, port=port, timeout=60)
                sections.append({
                    'name': section_name,
                    'summary': section_summary
                })
            except Exception as e:
                sections.append({
                    'name': section_name,
                    'summary': f"⚠️ Aggregatie mislukt: {str(e)}"
                })
    
    # Final synthesis if multiple sections
    if len(sections) > 1:
        final_prompt = f"""Finale analyse {doc_name}:

{chr(10).join([f"## {s['name']}{chr(10)}{s['summary']}" for s in sections[:10]])}

Maak Nederlandse samenvatting (max 30 bullets):

## TL;DR

## Financieel

## Juridisch

## Risico's"""
        
        try:
            final_analysis = ollama_generate(final_prompt, model, port=port, timeout=90)
        except Exception as e:
            final_analysis = f"⚠️ Finale synthese mislukt: {str(e)}\n\n" + \
                           "\n\n".join([f"## {s['name']}\n{s['summary']}" for s in sections])
    else:
        final_analysis = sections[0]['summary'] if sections else "onvoldoende data"
    
    return final_analysis


def run_8gpu_analysis(
    project_id: str,
    doc_path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    model: str = MODEL_NAME,
    worker_count: int = WORKER_COUNT
) -> Dict[str, Any]:
    """Run 8-GPU optimized analysis"""
    
    job_id = f"chunked8gpu-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(OUTPUT_ROOT, job_id)
    os.makedirs(output_dir, exist_ok=True)
    
    doc_name = os.path.basename(doc_path)
    
    print(f"🚀 8-GPU Analysis: {job_id}")
    print(f"   Document: {doc_name}")
    print(f"   Model: {model}")
    print(f"   Workers: {worker_count} (across 8 GPUs)")
    print(f"   Chunk size: {chunk_size} tokens")
    print()
    
    # Phase 1: Extract & Chunk
    print("📄 Phase 1: Extracting & chunking...")
    text = extract_text_from_pdf(doc_path)
    
    if not text or text.startswith("["):
        raise RuntimeError(f"Failed to extract text from {doc_path}")
    
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
    total_chunks = len(chunks)
    
    print(f"   ✓ Extracted {len(text):,} chars")
    print(f"   ✓ Created {total_chunks:,} chunks")
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
    
    # Store chunks
    print("💾 Storing chunks...")
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
    print(f"   ✓ Stored {total_chunks:,} chunks")
    print()
    
    # Phase 2: Process with 8-GPU power
    print(f"⚡ Phase 2: Processing {total_chunks:,} chunks with {worker_count} workers...")
    cs.update_chunk_job(job_id, status='processing')
    
    start_time = time.time()
    completed = 0
    failed = 0
    chunk_results = []
    
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {}
        
        for i, chunk in enumerate(chunks):
            chunk_id = cs.generate_chunk_id(job_id, i)
            # Distribute across all 8 ports cyclically
            port = OLLAMA_PORTS[i % len(OLLAMA_PORTS)]
            
            future = executor.submit(
                process_chunk,
                chunk_id, job_id, doc_name, i, total_chunks,
                chunk, model, port
            )
            futures[future] = i
        
        # Progress tracking
        last_update = time.time()
        
        for future in as_completed(futures):
            result = future.result()
            chunk_results.append(result)
            
            if result['success']:
                completed += 1
            else:
                failed += 1
            
            # Update progress every  2 seconds
            now = time.time()
            if now - last_update >= 2.0:
                elapsed = now - start_time
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
                
                pct = round((completed + failed) / total_chunks * 100, 1)
                print(f"   Progress: {completed:,}/{total_chunks:,} ({pct}%) | "
                      f"{throughput:.1f} chunks/min | ETA: {eta:.1f} min")
                
                last_update = now
    
    processing_time = time.time() - start_time
    print(f"\n   ✓ Completed in {processing_time/60:.1f} minutes")
    print(f"   ✓ Success: {completed:,}, Failed: {failed}")
    print(f"   ✓ Throughput: {completed/(processing_time/60):.1f} chunks/min")
    print()
    
    # Phase 3: Aggregate
    print("🔄 Phase 3: Aggregating...")
    cs.update_chunk_job(job_id, status='aggregating')
    
    successful_results = [r for r in chunk_results if r['success']]
    
    if not successful_results:
        error_msg = "All chunks failed"
        cs.update_chunk_job(job_id, status='failed', error_message=error_msg)
        raise RuntimeError(error_msg)
    
    final_analysis = aggregate_chunks_fast(
        job_id, successful_results, doc_name, model, OLLAMA_PORTS[0]
    )
    
    print("   ✓ Aggregation complete")
    print()
    
    # Save outputs
    print("💾 Saving...")
    
    report_lines = [
        f"# 8-GPU Chunked Analysis: {doc_name}",
        f"Gegenereerd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Job ID: {job_id}",
        "",
        f"## Performance Metrics",
        f"- Chunks processed: {completed:,}/{total_chunks:,}",
        f"- Processing time: {processing_time/60:.1f} minutes",
        f"- Throughput: {completed/(processing_time/60):.1f} chunks/min",
        f"- Workers: {worker_count} (8 GPUs)",
        f"- Model: {model}",
        "",
        "## Analyse",
        "",
        final_analysis,
        ""
    ]
    
    report_path = os.path.join(output_dir, "final_analysis.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
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
        "gpu_count": len(OLLAMA_PORTS),
        "processing_time_minutes": round(processing_time / 60, 2),
        "throughput_chunks_per_min": round(completed / (processing_time / 60), 2),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    cs.update_chunk_job(job_id, status='completed')
    
    print(f"   ✓ Report: {report_path}")
    print()
    print("✅ 8-GPU Analysis complete!")
    
    return {
        "job_id": job_id,
        "output_dir": output_dir,
        "report_path": report_path,
        "metadata": metadata
    }


if __name__ == "__main__":
    import argparse
    
    p = argparse.ArgumentParser(description="8-GPU Chunked Progressive Analysis")
    p.add_argument("--project-id", required=True, help="Project ID")
    p.add_argument("--doc-path", required=True, help="Path to PDF document")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Chunk size")
    p.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help="Overlap")
    p.add_argument("--model", default=MODEL_NAME, help="Model name")
    p.add_argument("--workers", type=int, default=WORKER_COUNT, help="Worker count")
    
    args = p.parse_args()
    
    try:
        result = run_8gpu_analysis(
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
