#!/usr/bin/env python3
"""
Flash Analysis - Ultra-fast financial document scanning (30-90 seconds).

Strategy:
  - Map stage: llama3.1:8b on 8 GPU workers (parallel)
  - Reduce stage: llama3.1:8b per document (NO gemma2:27b)
  - NO Final stage (skip cross-doc synthesis)
  - Aggressive chunking: smaller chunks, less overlap
  - Output: Compact bullet report (1-2 pages)

Usage:
  python scripts/flash_analysis.py --project-name "Project X" --files path/to/doc1.pdf path/to/doc2.pdf
  
  Or via API:
  POST /admin/projects/{id}/flash-analysis
"""

from __future__ import annotations

import os
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ingestion as ing
import config_store as cfg_store


# ---------- Configuration ----------

# Load config from config.json (respects LLM settings from admin UI)
try:
    APP_CONFIG = cfg_store.load_config()
except Exception:
    APP_CONFIG = {}

WORKER_PORTS = [11434, 11435, 11436, 11437, 11438, 11439, 11440, 11441]  # 8 GPUs

# Models - use config.json setting or env fallback
FLASH_MODEL = APP_CONFIG.get("LLM_MODEL") or os.getenv("FLASH_MODEL") or os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")

# Settings - use config.json or fallback
TEMPERATURE = APP_CONFIG.get("TEMPERATURE", 0.15)  # Configurable via LLM config page
FLASH_NUM_CTX = APP_CONFIG.get("NUM_CTX", 3072)  # Smaller context = faster
CHUNK_SIZE = 800  # Smaller chunks
CHUNK_OVERLAP = 50  # Minimal overlap
# Concurrency per GPU port (can be overridden by caller)
FLASH_CONCURRENCY = int(os.getenv("FLASH_CONCURRENCY", "1"))
# Temporarily force routing to a single port to stabilize multi-worker environments
FORCE_SINGLE_PORT = os.getenv("FORCE_SINGLE_PORT", "true").lower() in ("1", "true", "yes")

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ---------- HTTP client ----------

def ollama_generate(
    prompt: str,
    model: str,
    port: int,
    options: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> str:
    """Call Ollama /api/generate (non-streaming)"""
    import urllib.request
    import urllib.error

    url = f"http://127.0.0.1:{port}/api/generate"
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options or {},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
            j = json.loads(text)
            return j.get("response", "") if isinstance(j, dict) else str(j)
    except Exception as e:
        raise RuntimeError(f"Ollama error on port {port}: {str(e)}")


# ---------- Worker health preflight (short timeout, friendly failure) ----------

def healthy_worker_ports(ports: List[int], per_port_timeout: int = 5) -> List[int]:
    """
    Robust preflight: perform a 1-token /api/generate smoke test on each candidate port.
    Select only ports that can generate within 3–5 seconds.
    Falls back to OLLAMA_BASE_URL port if none pass.
    """
    healthy: List[int] = []
    import time as _time
    try:
        import urllib.request
        import urllib.error
        from urllib.parse import urlparse
    except Exception:
        return []

    for p in ports:
        try:
            url = f"http://127.0.0.1:{p}/api/generate"
            body = json.dumps({
                "model": FLASH_MODEL,
                "prompt": "ok",
                "stream": False,
                "options": {"num_predict": 1}
            }).encode("utf-8")
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            t0 = _time.time()
            with urllib.request.urlopen(req, timeout=per_port_timeout) as resp:
                _ = resp.read()
            dt = _time.time() - t0
            if dt <= 5.0:
                healthy.append(p)
        except Exception:
            continue

    if healthy:
        return healthy

    # Fallback: force single port from OLLAMA_BASE_URL
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    try:
        parsed = urlparse(base)
        p = parsed.port or 11434
        # Probe fallback port
        try:
            url = f"http://127.0.0.1:{p}/api/generate"
            body = json.dumps({"model": FLASH_MODEL, "prompt": "ok", "stream": False, "options": {"num_predict": 1}}).encode("utf-8")
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            _ = urllib.request.urlopen(req, timeout=per_port_timeout).read()
            return [int(p)]
        except Exception:
            pass
    except Exception:
        pass

    return []

# Warm selected model on each port with tiny prompt; retry with exponential backoff
def warmup_model_on_ports(ports: List[int], model: str = FLASH_MODEL) -> None:
    backoffs = [0.5, 1.0, 2.0]
    for p in ports:
        for attempt, delay in enumerate([0.0] + backoffs):
            if delay:
                time.sleep(delay)
            try:
                _ = ollama_generate("ping", model, p, options={"num_predict": 1}, timeout=10)
                break
            except RuntimeError as e:
                # Retry on transient load errors
                if attempt == len(backoffs):
                    with open(os.path.join(OUTPUT_ROOT, "flash_warmup_errors.log"), "a") as ef:
                        ef.write(f"[WARMUP-ERROR] port {p}: {str(e)}\n")
                continue

# Deterministic fallback summary using evidence.csv and simple rules
def build_deterministic_fallback_report(project_name: str, out_dir: str) -> str:
    import csv
    from glob import glob

    def load_latest_evidence() -> List[Dict[str, str]]:
        # Try job-local first
        ev_path = os.path.join(out_dir, "evidence.csv")
        if os.path.isfile(ev_path):
            with open(ev_path, "r", encoding="utf-8") as cf:
                return list(csv.DictReader(cf))
        # Else try most recent evidence.csv in outputs/
        candidates = sorted(glob(os.path.join(OUTPUT_ROOT, "job-*", "evidence.csv")), key=os.path.getmtime, reverse=True)
        if candidates:
            with open(candidates[0], "r", encoding="utf-8") as cf:
                return list(csv.DictReader(cf))
        # Else try root outputs/evidence.csv
        root_ev = os.path.join(OUTPUT_ROOT, "evidence.csv")
        if os.path.isfile(root_ev):
            with open(root_ev, "r", encoding="utf-8") as cf:
                return list(csv.DictReader(cf))
        return []

    rows = load_latest_evidence()
    amounts, percentages, rates, entities, dates = [], [], [], [], []
    import re

    for r in rows:
        t = (r.get("type") or "").lower()
        v = r.get("value") or ""
        if t == "amount":
            amounts.append(v)
        elif t == "percentage":
            percentages.append(v)
        elif t == "rate":
            rates.append(v)
        elif t == "entity":
            entities.append(v)
        elif t == "date":
            dates.append(v)

    # Top-N by magnitude (parse EUR amounts when possible)
    def parse_amount(a: str) -> float:
        m = re.search(r'([0-9][0-9\.\,]*)', a)
        if not m:
            return 0.0
        s = m.group(1).replace('.', '').replace(',', '.')
        try:
            return float(s)
        except:
            return 0.0

    amounts_sorted = sorted(amounts, key=parse_amount, reverse=True)[:15]
    perc_sorted = percentages[:15]
    rates_sorted = rates[:10]
    entities_sorted = list(dict.fromkeys(entities))[:20]
    years = sorted(set(re.findall(r'\b(?:19|20)\d{2}\b', " ".join(dates))))[:15]

    lines = [
        f"# Flash Analyse (Deterministische fallback) – {project_name}",
        f"Gegenereerd: {time.strftime('%Y-%m-%d %H:%M:%S')} (LLM fallback actief)",
        "",
        "## TL;DR: kern financieel",
        "- onvoldoende data: LLM-generatie mislukt; onderstaande is gebaseerd op geaggregeerd bewijs.",
        "",
        "## Financiële kernpunten",
        "- Top bedragen:",
    ]
    for a in amounts_sorted:
        lines.append(f"  - {a}")
    lines.extend([
        "- Percentages:",
    ])
    for p in perc_sorted:
        lines.append(f"  - {p}")
    lines.extend([
        "- Indicatieve rentes/tarieven:",
    ])
    for r in rates_sorted:
        lines.append(f"  - {r}")
    lines.extend([
        "",
        "## Fiscale aandachtspunten (indicatief, 2025)",
        "- Let op IB/VPB/BOR/BTW; beoordeel vrijstellingen en tarieven op basis van context.",
        "- Markeer 'onvoldoende data' waar bedragen/feiten ontbreken.",
        "",
        "## Juridische structuur & partijen",
    ])
    for e in entities_sorted:
        lines.append(f"- {e}")
    lines.extend([
        "",
        "## Risico’s en aannames",
        "- onvoldoende data voor volledige duiding; verifieer bronnen en context.",
        "",
        "## Tijdlijn (jaartallen genoemd)",
    ])
    if years:
        lines.append("- " + ", ".join(sorted(set(years))))
    else:
        lines.append("- onvoldoende data")
    lines.append("")
    return "\n".join(lines)

# ---------- Flash Prompts (ultra-compact) ----------

def build_flash_map_prompt(filename: str, chunk_index: int, chunk_total: int, chunk_text: str) -> str:
    """Ultra-compact map prompt - max 6 bullets per chunk"""
    return f"""Financiële scan chunk {chunk_index+1}/{chunk_total} uit {filename}.

ALLEEN bullets (max 6):
- Bedragen (EUR/€)
- Partijen/entiteiten
- Data/termijnen
- Transacties
- Risico's

Geen uitleg, alleen feiten.

TEKST:
\"\"\"
{chunk_text}
\"\"\"

Bullets:""".strip()


def build_flash_reduce_prompt(filename: str, bullets: List[str]) -> str:
    """Compact reduce with fiscal specifics"""
    joined = "\n\n".join(bullets[:3000])  # cap for speed
    
    return f"""Combineer chunk-bullets tot een compacte samenvatting van {filename}.

BEWIJS:
{joined}

Maak ALLEEN bullets (max 18 totaal), Nederlands, feitelijk, zonder PII:

## TL;DR (kern financieel: bedragen/waarderingen)

## Financieel (topbedragen, percentages, rentes)

## Fiscaal (IB, VPB, BOR, BTW - tarieven/vrijstellingen waar mogelijk)

## Juridische structuur & partijen (entiteiten/rollen)

## Termijnen & tijdlijn (jaartallen/data)

## Risico’s & aannames (“onvoldoende data” waar bewijs ontbreekt)

Geen extra tekst. Alleen bullets. Gebruik exacte cijfers/percentages uit bewijs.""".strip()


# ---------- Helpers ----------

def list_project_files(upload_dir: str) -> List[str]:
    """List all supported documents in upload directory"""
    files = []
    if not os.path.isdir(upload_dir):
        return files
    
    for fname in sorted(os.listdir(upload_dir)):
        fpath = os.path.join(upload_dir, fname)
        if os.path.isfile(fpath):
            ext = os.path.splitext(fname)[1].lower()
            if ext in [".pdf", ".txt", ".md", ".xlsx", ".xls"]:
                files.append(fpath)
    return files


def extract_text(fpath: str) -> str:
    """Extract text from file"""
    ext = os.path.splitext(fpath)[1].lower()
    if ext == ".pdf":
        return ing.read_pdf_file(fpath)
    elif ext in [".txt", ".md"]:
        return ing.read_text_file(fpath)
    elif ext in [".xlsx", ".xls"]:
        return ing.read_excel_file(fpath)
    return ""


def sample_chunks(chunks: List[str], max_chunks: int = 20) -> List[str]:
    """
    For large documents, sample chunks strategically:
    - First 5 (intro/summary)
    - Random sample from middle
    - Last 3 (conclusion)
    """
    if len(chunks) <= max_chunks:
        return chunks
    
    start = chunks[:5]
    end = chunks[-3:]
    middle_size = max_chunks - len(start) - len(end)
    
    if middle_size > 0:
        import random
        middle = chunks[5:-3]
        sampled_middle = random.sample(middle, min(middle_size, len(middle)))
        return start + sampled_middle + end
    else:
        return start + end


# ---------- Pipeline ----------

def run_flash_analysis(
    project_name: str = "Project",
    upload_dir: Optional[str] = None,
    selected_files: Optional[List[str]] = None,
    worker_ports: Optional[List[int]] = None,
    concurrency: Optional[int] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run flash analysis pipeline.
    Returns output directory and metadata.
    """

    # Resolve effective parameters (do NOT mutate module globals to stay thread-safe)
    ports = worker_ports if worker_ports else WORKER_PORTS
    # Simplify routing in unstable environments: route all to single base port when FORCE_SINGLE_PORT=true
    if FORCE_SINGLE_PORT:
        from urllib.parse import urlparse
        base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        try:
            parsed = urlparse(base)
            base_port = parsed.port or 11434
        except Exception:
            base_port = 11434
        ports = [int(base_port)]
    threads_per_port = 1 if FORCE_SINGLE_PORT else (concurrency if (isinstance(concurrency, int) and concurrency > 0) else FLASH_CONCURRENCY)
    eff_chunk_size = chunk_size if (isinstance(chunk_size, int) and chunk_size > 0) else CHUNK_SIZE
    eff_chunk_overlap = chunk_overlap if (isinstance(chunk_overlap, int) and chunk_overlap >= 0) else CHUNK_OVERLAP

    # Setup output directory
    job_id = time.strftime("flash-%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUT_ROOT, job_id)
    os.makedirs(out_dir, exist_ok=True)
    
    # Get files
    if selected_files:
        files = selected_files
    elif upload_dir:
        files = list_project_files(upload_dir)
    else:
        files = list_project_files(os.path.join(ROOT_DIR, "fatrag_data", "uploads"))
    
    if not files:
        raise RuntimeError("Geen documenten gevonden")
    
    # Phase 1: Extract & chunk
    doc_chunks: Dict[str, List[str]] = {}
    
    for fpath in files:
        fname = os.path.basename(fpath)
        text = extract_text(fpath)
        
        if not text or text.startswith("["):  # error messages
            continue
        
        # Chunk with aggressive settings
        chunks = ing.chunk_texts([text], chunk_size=eff_chunk_size, chunk_overlap=eff_chunk_overlap)
        
        # Sample if too many chunks
        chunks = sample_chunks(chunks, max_chunks=25)
        
        doc_chunks[fname] = chunks
    
    if not doc_chunks:
        raise RuntimeError("Geen bruikbare tekst uit documenten")
    
    # Preflight: 1-token /api/generate smoke per port; select only ports that respond within 3–5s
    ports = healthy_worker_ports(ports)
    if ports:
        # Warm selected model once per port with tiny prompt and backoff
        try:
            warmup_model_on_ports(ports, model=FLASH_MODEL)
        except Exception:
            pass
    else:
        # No LLM available: fall back to deterministic summary
        final_report = build_deterministic_fallback_report(project_name, out_dir)
        for fn in ("flash_report.txt", "flash_report.md"):
            with open(os.path.join(out_dir, fn), "w", encoding="utf-8") as f:
                f.write(final_report)
        metadata = {
            "job_id": job_id,
            "project_name": project_name,
            "mode": "flash",
            "model": FLASH_MODEL,
            "documents_analyzed": 0,
            "total_chunks": 0,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "deterministic_fallback": True
        }
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return {
            "job_dir": out_dir,
            "job_id": job_id,
            "model": FLASH_MODEL,
            "documents": [],
            "report_path": os.path.join(out_dir, "flash_report.md"),
        }

    # Phase 2: Map (parallel)
    map_outputs: Dict[str, List[str]] = {k: [] for k in doc_chunks.keys()}
    
    def submit_map_task(fname: str, idx: int, chunk: str, port: int):
        prompt = build_flash_map_prompt(fname, idx, len(doc_chunks[fname]), chunk)
        options = {"temperature": TEMPERATURE, "num_ctx": FLASH_NUM_CTX}
        resp = ollama_generate(prompt, FLASH_MODEL, port, options=options, timeout=30)
        return fname, idx, resp
    
    futures = []
    with ThreadPoolExecutor(max_workers=max(1, len(ports) * threads_per_port)) as ex:
        port_iter = 0
        for fname, chunks in doc_chunks.items():
            for idx, chunk in enumerate(chunks):
                port = ports[port_iter % len(ports)]
                port_iter += 1
                futures.append(ex.submit(submit_map_task, fname, idx, chunk, port))
        
        for fut in as_completed(futures):
            try:
                fname, idx, resp = fut.result()
                map_outputs[fname].append(resp)
            except Exception as e:
                # Continue on errors
                with open(os.path.join(out_dir, "errors.log"), "a") as ef:
                    ef.write(f"[MAP-ERROR] {fname} chunk {idx}: {str(e)}\n")
    
    # Phase 3: Reduce per document (fast)
    doc_summaries: Dict[str, str] = {}
    
    def submit_reduce_task(fname: str, bullets: List[str], port: int):
        prompt = build_flash_reduce_prompt(fname, bullets)
        options = {"temperature": TEMPERATURE, "num_ctx": FLASH_NUM_CTX}
        resp = ollama_generate(prompt, FLASH_MODEL, port, options=options, timeout=30)
        return fname, resp
    
    futures = []
    with ThreadPoolExecutor(max_workers=max(1, len(ports) * threads_per_port)) as ex:
        for i, fname in enumerate(doc_chunks.keys()):
            port = ports[i % len(ports)]
            bullets = map_outputs.get(fname, [])
            futures.append(ex.submit(submit_reduce_task, fname, bullets, port))
        
        for fut in as_completed(futures):
            try:
                fname, resp = fut.result()
                doc_summaries[fname] = resp
            except Exception as e:
                with open(os.path.join(out_dir, "errors.log"), "a") as ef:
                    ef.write(f"[REDUCE-ERROR] {fname}: {str(e)}\n")
    
    # If reduce failed for all docs, produce deterministic fallback summary
    if not doc_summaries:
        final_report = build_deterministic_fallback_report(project_name, out_dir)
        with open(os.path.join(out_dir, "flash_report.txt"), "w", encoding="utf-8") as f:
            f.write(final_report)
        with open(os.path.join(out_dir, "flash_report.md"), "w", encoding="utf-8") as f:
            f.write(final_report)
        metadata = {
            "job_id": job_id,
            "project_name": project_name,
            "mode": "flash",
            "model": FLASH_MODEL,
            "documents_analyzed": 0,
            "total_chunks": sum(len(c) for c in doc_chunks.values()),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "deterministic_fallback": True
        }
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return {
            "job_dir": out_dir,
            "job_id": job_id,
            "model": FLASH_MODEL,
            "documents": [],
            "report_path": os.path.join(out_dir, "flash_report.md"),
        }

    # Phase 4: Generate final report (no LLM, just format)
    report_lines = [
        f"# Flash Analyse: {project_name}",
        f"Gegenereerd: {time.strftime('%Y-%m-%d %H:%M:%S')} (Flash Mode)",
        "",
        "## Overview",
        f"- Documenten: {len(doc_summaries)}",
        f"- Mode: Flash (30-90s scan)",
        f"- Model: {FLASH_MODEL}",
        "",
    ]
    
    # Add TL;DR (extract first bullets from each doc)
    report_lines.append("## 🔍 Key Findings")
    for fname, summary in doc_summaries.items():
        lines = [l.strip() for l in summary.split('\n') if l.strip().startswith('-')]
        if lines:
            report_lines.append(f"**{fname}:** {lines[0].lstrip('- ')}")
    report_lines.append("")
    
    # Add per-document summaries
    report_lines.append("## 📄 Per Document")
    report_lines.append("")
    
    for fname, summary in doc_summaries.items():
        report_lines.append(f"### {fname}")
        report_lines.append(summary)
        report_lines.append("")
    
    # Add disclaimer
    report_lines.extend([
        "---",
        "",
        "**Note:** Dit is een Flash Analyse (30-90s). Voor uitgebreide analyse, gebruik 'Snelle Analyse' of 'FATRAG Pipeline'.",
        ""
    ])
    
    final_report = "\n".join(report_lines)
    
    # Save outputs
    with open(os.path.join(out_dir, "flash_report.txt"), "w", encoding="utf-8") as f:
        f.write(final_report)
    
    with open(os.path.join(out_dir, "flash_report.md"), "w", encoding="utf-8") as f:
        f.write(final_report)
    
    # Save metadata
    metadata = {
        "job_id": job_id,
        "project_name": project_name,
        "mode": "flash",
        "model": FLASH_MODEL,
        "documents_analyzed": len(doc_summaries),
        "total_chunks": sum(len(c) for c in doc_chunks.values()),
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return {
        "job_dir": out_dir,
        "job_id": job_id,
        "model": FLASH_MODEL,
        "documents": list(doc_summaries.keys()),
        "report_path": os.path.join(out_dir, "flash_report.md"),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Flash Analysis - Ultra-fast document scan")
    p.add_argument("--project-name", default="Project", help="Project name")
    p.add_argument("--upload-dir", help="Upload directory path")
    p.add_argument("--files", nargs="*", help="Specific files to analyze")
    p.add_argument("--worker-ports", help="Comma-separated list of worker ports (e.g. 11434,11435)")
    p.add_argument("--concurrency", type=int, help="Threads per port (default: env FLASH_CONCURRENCY or 1)")
    args = p.parse_args()

    ports = None
    if args.worker_ports:
        try:
            ports = [int(x.strip()) for x in args.worker_ports.split(",") if x.strip().isdigit()]
        except Exception:
            ports = None
    
    result = run_flash_analysis(
        project_name=args.project_name,
        upload_dir=args.upload_dir,
        selected_files=args.files,
        worker_ports=ports,
        concurrency=args.concurrency
    )
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
