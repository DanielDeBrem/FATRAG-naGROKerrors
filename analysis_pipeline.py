#!/usr/bin/env python3
"""
Multi-GPU, multi-layer analysis pipeline for financial/fiscal documents using Ollama.

Modes:
  - Fresh run (default):
      0) Preprocess (extract text, evidence, chunk)
      1) Map (per chunk) on 8 GPU workers (ports 11434..11441) with a small/fast model
      2) Reduce (per document) on worker pool with a mid-size model (default: gemma2:27b)
         - Fallback on model-not-found: llama3.1:8b
      3) Final cross-document synthesis on a "big" Ollama at port 11450 with a large model (e.g. llama3.1:70b)
  - Resume run (pass --resume-dir <existing job dir>):
      - Skips preprocess & map
      - Loads map_chunk bullets from outputs/<job>/map_chunks/*.md
      - Loads evidence from outputs/<job>/evidence.csv (if present)
      - Runs Reduce + Final only, writing to the same job directory

Outputs (job_dir = outputs/{job_id} or --resume-dir):
  job_dir/
    - final.txt                (complete analysis, NL, bullets)
    - final.json               (structured sections)
    - evidence.csv             (aggregated evidence; reused if --resume-dir)
    - doc_summaries/{doc}.md   (per-document summary)
    - map_chunks/{doc}_{i}.md  (per-chunk map outputs)

Assumptions:
  - Ollama servers are started via:
      bash scripts/start_ollama_workers.sh
      bash scripts/start_ollama_big.sh
  - Models present (or pullable) locally in a shared OLLAMA_MODELS dir.
  - Uploaded files in: fatrag_data/uploads
"""

from __future__ import annotations

import os
import json
import time
import csv
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

import ingestion as ing


# ---------- Configuration ----------

WORKER_PORTS = [11434, 11435, 11436, 11437, 11438, 11439, 11440, 11441]  # 8 GPUs
BIG_PORT = 11450  # big server (all GPUs visible)

# Models
MAP_MODEL = os.getenv("MAP_MODEL", "llama3.1:8b")
# Use gemma2:27b as mid-size reduce model (Option B). This model must exist in the local registry.
REDUCE_MODEL = os.getenv("REDUCE_MODEL", "gemma2:27b")
# Fallback reduce model if the requested reduce model is not found on a worker.
FALLBACK_REDUCE_MODEL = os.getenv("FALLBACK_REDUCE_MODEL", "llama3.1:8b")
FINAL_MODEL = os.getenv("FINAL_MODEL", "llama3.1:70b")

# Prompt style & options
TEMPERATURE = float(os.getenv("ANALYSIS_TEMPERATURE", "0.2"))
MAP_NUM_CTX = int(os.getenv("MAP_NUM_CTX", "4096"))
REDUCE_NUM_CTX = int(os.getenv("REDUCE_NUM_CTX", "8192"))
FINAL_NUM_CTX = int(os.getenv("FINAL_NUM_CTX", "16384"))

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(ROOT_DIR, "fatrag_data", "uploads")
OUTPUT_ROOT = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ---------- HTTP client (Ollama) ----------

def ollama_generate(
    prompt: str,
    model: str,
    port: int,
    options: Optional[Dict[str, Any]] = None,
    timeout: int = 180,
) -> str:
    """
    Call Ollama /api/generate with non-streaming.
    Uses urllib to avoid extra dependencies.
    """
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
            # Ollama's /api/generate returns {"response": "...", ...}
            return j.get("response", "") if isinstance(j, dict) else str(j)
    except urllib.error.HTTPError as e:
        # Surface the original body so callers can decide on fallback
        raise RuntimeError(f"Ollama HTTPError {e.code}: {e.read().decode('utf-8', errors='ignore')}")
    except Exception as e:
        raise RuntimeError(f"Ollama request error on port {port}: {str(e)}")


# ---------- Prompts ----------

def build_map_prompt(
    project_name: str,
    filename: str,
    chunk_index: int,
    chunk_total: int,
    chunk_text: str,
    evidence: Dict[str, List[str]],
) -> str:
    ev_amounts = ", ".join((evidence.get("amounts") or [])[:20]) or "geen"
    ev_perc = ", ".join((evidence.get("percentages") or [])[:20]) or "geen"
    ev_rates = ", ".join((evidence.get("rates") or [])[:20]) or "geen"
    ev_dates = ", ".join((evidence.get("dates") or [])[:20]) or "geen"

    return f"""
Je bent FinAdviseur-NL. Analyseer onderstaande chunk beknopt en in het Nederlands. 
- Focus op bedragen (EUR/€), percentages, rentes, waarderingen, termijnen.
- Negeer footers/headers/meta. 
- Wees feitelijk; als data ontbreekt: schrijf 'onvoldoende data'.
- Lever uitsluitend puntsgewijze bullets (geen extra tekst).

Project: {project_name}
Document: {filename} (chunk {chunk_index+1}/{chunk_total})

Bekende financiële signalen (globaal uit document):
- Bedragen: {ev_amounts}
- Percentages: {ev_perc}
- Rentes: {ev_rates}
- Datums/Jaren: {ev_dates}

TEKST:
\"\"\"
{chunk_text}
\"\"\" 

Geef uitsluitend bullets:
- Bedragen/valuta:
- Percentages/rentes:
- Waarderingen/schattingen:
- Termijnen/data:
- Partijen/entiteiten:
- Belangrijkste financiële observaties:
""".strip()


def build_reduce_prompt(project_name: str, filename: str, bullets: List[str]) -> str:
    joined = "\n\n".join(bullets[:5000])  # safety cap
    return f"""
Je bent FinAdviseur-NL. Combineer onderstaande 'bewijsregels' tot één compacte samenvatting per document.
- taal: Nederlands
- vorm: puntsgewijs/bullet points, professioneel
- geen PII
- gebruik bedragen/percentages exact zoals genoemd
- als data ontbreekt: schrijf expliciet 'onvoldoende data'

Project: {project_name}
Document: {filename}

BEWIJS (samengevat per chunk):
{joined}

Maak de volgende secties (bullets):
- Financiële kernpunten
- Fiscale aspecten
- Juridische structuur & partijen
- Risico's, aannames, onzekerheden ('onvoldoende data' waar relevant)
- Aanpak / stappenplan
""".strip()


def build_final_prompt(project_name: str, doc_summaries: Dict[str, str], evidence_rows: List[Dict[str, str]]) -> str:
    docs = "\n\n".join([f"=== {name} ===\n{summary}" for name, summary in doc_summaries.items()])
    # compact evidence table-like view
    ev_lines = []
    for r in evidence_rows[:2000]:
        ev_lines.append(f"{r.get('document','')} | {r.get('type','')} | {r.get('value','')}")
    ev_text = "\n".join(ev_lines)

    return f"""
STRICT_LANGUAGE_MODE: Nederlands

Je bent "FinAdviseur-NL", senior financieel specialist (NL, 2025).
Doel: lever een professionele, volledig Nederlandse analyse over alle documenten samen. 
Stijl: kort-bondige bullets, geen PII, gebruik 'onvoldoende data' wanneer nodig.

Project: {project_name}

Bewijs (per document):
{docs}

Geaggregeerde signalen (document | type | waarde):
{ev_text}

Gevraagd eindresultaat (allemaal bullets, NL):
- TL;DR (3–6 bullets, kern financieel/fiscaal)
- Financiële kernpunten (bedragen/percentages/rentes/waarderingen)
- Fiscale impact en aandachtspunten
- Juridische structuur & partijen
- Risico's, aannames, onzekerheden ('onvoldoende data' waar relevant)
- Aanpak / stappenplan (actiegericht, meteen uitvoerbaar)
- Open vragen (max 5) om tot definitief advies te komen
""".strip()


# ---------- Helpers ----------

def list_upload_files() -> List[str]:
    files = []
    if not os.path.isdir(UPLOADS_DIR):
        return files
    for fname in sorted(os.listdir(UPLOADS_DIR)):
        fpath = os.path.join(UPLOADS_DIR, fname)
        if os.path.isfile(fpath):
            ext = os.path.splitext(fname)[1].lower()
            if ext in [".pdf", ".txt", ".md", ".xlsx", ".xls"]:
                files.append(fpath)
    return files


def extract_text_for_file(fpath: str) -> str:
    ext = os.path.splitext(fpath)[1].lower()
    if ext == ".pdf":
        return ing.read_pdf_file(fpath)
    elif ext in [".txt", ".md"]:
        return ing.read_text_file(fpath)
    elif ext in [".xlsx", ".xls"]:
        return ing.read_excel_file(fpath)
    else:
        return ""


def aggregate_evidence(doc_evidence: Dict[str, Dict[str, List[str]]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for doc, ev in doc_evidence.items():
        for a in ev.get("amounts", []):
            rows.append({"document": doc, "type": "amount", "value": a})
        for p in ev.get("percentages", []):
            rows.append({"document": doc, "type": "percentage", "value": p})
        for r in ev.get("rates", []):
            rows.append({"document": doc, "type": "rate", "value": r})
        for d in ev.get("dates", []):
            rows.append({"document": doc, "type": "date", "value": d})
        for e in ev.get("entities", []):
            rows.append({"document": doc, "type": "entity", "value": e})
    return rows


def load_evidence_csv(job_dir: str) -> List[Dict[str, str]]:
    """Load evidence.csv from an existing job directory, best-effort."""
    ev_path = os.path.join(job_dir, "evidence.csv")
    rows: List[Dict[str, str]] = []
    if os.path.isfile(ev_path):
        with open(ev_path, "r", encoding="utf-8") as cf:
            reader = csv.DictReader(cf)
            for r in reader:
                rows.append({"document": r.get("document", ""), "type": r.get("type", ""), "value": r.get("value", "")})
    return rows


def load_map_bullets_from_dir(job_dir: str) -> Dict[str, List[str]]:
    """
    Load map outputs from job_dir/map_chunks/*.md and group them per original document.
    Filenames are expected as {docname}.{ext}_{index:04d}.md — we group by the portion before the last underscore.
    """
    map_dir = os.path.join(job_dir, "map_chunks")
    grouped: Dict[str, List[str]] = {}
    if not os.path.isdir(map_dir):
        return grouped

    for fname in sorted(os.listdir(map_dir)):
        if not fname.endswith(".md"):
            continue
        # Split by last underscore, remove .md
        stem = fname[:-3]
        if "_" not in stem:
            # fallback: treat entire stem as doc name
            doc_key = stem
        else:
            base, _idx = stem.rsplit("_", 1)
            doc_key = base  # includes original extension in the name

        fpath = os.path.join(map_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                text = fh.read().strip()
        except Exception:
            text = ""
        grouped.setdefault(doc_key, []).append(text)
    return grouped


# ---------- Pipeline ----------

def run_pipeline(
    project_name: str = "Project",
    selected_files: Optional[List[str]] = None,
    resume_dir: Optional[str] = None,
) -> Dict[str, Any]:
    # Directories
    if resume_dir:
        out_dir = os.path.abspath(resume_dir)
        if not os.path.isdir(out_dir):
            raise RuntimeError(f"Resume directory not found: {resume_dir}")
        map_dir = os.path.join(out_dir, "map_chunks")
        sum_dir = os.path.join(out_dir, "doc_summaries")
        os.makedirs(sum_dir, exist_ok=True)
    else:
        job_id = time.strftime("job-%Y%m%d_%H%M%S")
        out_dir = os.path.join(OUTPUT_ROOT, job_id)
        map_dir = os.path.join(out_dir, "map_chunks")
        sum_dir = os.path.join(out_dir, "doc_summaries")
        os.makedirs(map_dir, exist_ok=True)
        os.makedirs(sum_dir, exist_ok=True)

    # Fresh run branch
    if not resume_dir:
        files = selected_files or list_upload_files()
        if not files:
            raise RuntimeError("Geen documenten gevonden in fatrag_data/uploads")

        # 0) Preprocess
        doc_texts: Dict[str, str] = {}
        doc_evidence: Dict[str, Dict[str, List[str]]] = {}
        doc_chunks: Dict[str, List[str]] = {}

        for fpath in files:
            fname = os.path.basename(fpath)
            text = extract_text_for_file(fpath)
            if not text or text.startswith("[PDF extraction error") or text.startswith("[Excel extraction error"):
                continue
            doc_texts[fname] = text
            ev = ing.extract_financial_evidence(text)
            doc_evidence[fname] = ev
            chunks = ing.chunk_texts([text], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            doc_chunks[fname] = chunks

        if not doc_chunks:
            raise RuntimeError("Geen bruikbare tekst uit documenten gehaald.")

        # 1) Map stage (parallel over WORKER_PORTS)
        map_outputs: Dict[str, List[str]] = {k: [] for k in doc_chunks.keys()}

        def submit_map_task(fname: str, idx: int, chunk: str, port: int) -> Tuple[str, int, str]:
            prompt = build_map_prompt(project_name, fname, idx, len(doc_chunks[fname]), chunk, doc_evidence.get(fname, {}))
            options = {"temperature": TEMPERATURE, "num_ctx": MAP_NUM_CTX}
            resp = ollama_generate(prompt, MAP_MODEL, port, options=options, timeout=240)
            return fname, idx, resp

        futures = []
        with ThreadPoolExecutor(max_workers=min(32, len(WORKER_PORTS) * 4)) as ex:
            port_iter = 0
            for fname, chunks in doc_chunks.items():
                for idx, chunk in enumerate(chunks):
                    port = WORKER_PORTS[port_iter % len(WORKER_PORTS)]
                    port_iter += 1
                    futures.append(ex.submit(submit_map_task, fname, idx, chunk, port))

            for fut in as_completed(futures):
                try:
                    fname, idx, resp = fut.result()
                    map_outputs[fname].append(f"Bron: {fname} | Chunk {idx+1}/{len(doc_chunks[fname])}\n{resp}")
                    # persist chunk output
                    with open(os.path.join(map_dir, f"{fname}_{idx:04d}.md"), "w", encoding="utf-8") as f:
                        f.write(resp)
                except Exception as e:
                    # continue best-effort
                    with open(os.path.join(map_dir, "errors.log"), "a", encoding="utf-8") as ef:
                        ef.write(f"[MAP-ERROR] {str(e)}\n{traceback.format_exc()}\n")

        # 2) Reduce per document
        doc_summaries: Dict[str, str] = {}

        def submit_reduce_task(fname: str, bullets: List[str], port: int) -> Tuple[str, str]:
            prompt = build_reduce_prompt(project_name, fname, bullets)
            options = {"temperature": TEMPERATURE, "num_ctx": REDUCE_NUM_CTX}
            try:
                resp = ollama_generate(prompt, REDUCE_MODEL, port, options=options, timeout=300)
            except RuntimeError as e:
                # Fallback on model-not-found (HTTP 404)
                if "404" in str(e) or ("model" in str(e).lower() and "not found" in str(e).lower()):
                    resp = ollama_generate(prompt, FALLBACK_REDUCE_MODEL, port, options=options, timeout=300)
                else:
                    raise
            return fname, resp

        futures = []
        with ThreadPoolExecutor(max_workers=len(WORKER_PORTS)) as ex:
            for i, fname in enumerate(map_outputs.keys()):
                port = WORKER_PORTS[i % len(WORKER_PORTS)]
                bullets = map_outputs.get(fname, []) or []
                futures.append(ex.submit(submit_reduce_task, fname, bullets, port))
            for fut in as_completed(futures):
                try:
                    fname, resp = fut.result()
                    doc_summaries[fname] = resp
                    # persist
                    with open(os.path.join(sum_dir, f"{fname}.md"), "w", encoding="utf-8") as f:
                        f.write(resp)
                except Exception as e:
                    with open(os.path.join(sum_dir, "errors.log"), "a", encoding="utf-8") as ef:
                        ef.write(f"[REDUCE-ERROR] {str(e)}\n{traceback.format_exc()}\n")

        # 3) Final cross-document synthesis
        evidence_rows = aggregate_evidence(doc_evidence)
        # Save evidence CSV
        with open(os.path.join(out_dir, "evidence.csv"), "w", newline="", encoding="utf-8") as cf:
            w = csv.DictWriter(cf, fieldnames=["document", "type", "value"])
            w.writeheader()
            for r in evidence_rows:
                w.writerow(r)

        final_prompt = build_final_prompt(project_name, doc_summaries, evidence_rows)
        final_options = {"temperature": TEMPERATURE, "num_ctx": FINAL_NUM_CTX}
        final_text = ollama_generate(final_prompt, FINAL_MODEL, BIG_PORT, options=final_options, timeout=900)

        # Save final artifacts
        with open(os.path.join(out_dir, "final.txt"), "w", encoding="utf-8") as f:
            f.write(final_text)

    # Resume branch: reuse map_chunks + evidence.csv; run Reduce + Final only
    else:
        # Load bullets per document from map_chunks
        map_outputs = load_map_bullets_from_dir(out_dir)
        if not map_outputs:
            raise RuntimeError(f"Geen map_chunks gevonden in {out_dir}/map_chunks")

        # Reduce
        doc_summaries: Dict[str, str] = {}

        def submit_reduce_task_resume(fname: str, bullets: List[str], port: int) -> Tuple[str, str]:
            prompt = build_reduce_prompt(project_name, fname, bullets)
            options = {"temperature": TEMPERATURE, "num_ctx": REDUCE_NUM_CTX}
            try:
                resp = ollama_generate(prompt, REDUCE_MODEL, port, options=options, timeout=300)
            except RuntimeError as e:
                if "404" in str(e) or ("model" in str(e).lower() and "not found" in str(e).lower()):
                    resp = ollama_generate(prompt, FALLBACK_REDUCE_MODEL, port, options=options, timeout=300)
                else:
                    raise
            return fname, resp

        futures = []
        with ThreadPoolExecutor(max_workers=len(WORKER_PORTS)) as ex:
            keys = sorted(map_outputs.keys())
            for i, fname in enumerate(keys):
                port = WORKER_PORTS[i % len(WORKER_PORTS)]
                bullets = map_outputs.get(fname, []) or []
                futures.append(ex.submit(submit_reduce_task_resume, fname, bullets, port))
            for fut in as_completed(futures):
                try:
                    fname, resp = fut.result()
                    doc_summaries[fname] = resp
                    with open(os.path.join(sum_dir, f"{fname}.md"), "w", encoding="utf-8") as f:
                        f.write(resp)
                except Exception as e:
                    with open(os.path.join(sum_dir, "errors.log"), "a", encoding="utf-8") as ef:
                        ef.write(f"[REDUCE-ERROR] {str(e)}\n{traceback.format_exc()}\n")

        # Final
        evidence_rows = load_evidence_csv(out_dir)
        final_prompt = build_final_prompt(project_name, doc_summaries, evidence_rows)
        final_options = {"temperature": TEMPERATURE, "num_ctx": FINAL_NUM_CTX}
        final_text = ollama_generate(final_prompt, FINAL_MODEL, BIG_PORT, options=final_options, timeout=900)
        with open(os.path.join(out_dir, "final.txt"), "w", encoding="utf-8") as f:
            f.write(final_text)

    # Also store a basic JSON structure (best-effort section split)
    sections = {
        "tldr": [],
        "financiele_kernpunten": [],
        "fiscale_impact": [],
        "juridische_structuur_partijen": [],
        "risicos_aannames_onvoldoende_data": [],
        "aanpak_stappenplan": [],
        "open_vragen": [],
        "raw": final_text,
    }
    # naive parse bullet lines by section headers
    cur_key = None
    for line in final_text.splitlines():
        l = line.strip()
        tag_map = {
            "tl;dr": "tldr",
            "financiële kernpunten": "financiele_kernpunten",
            "fiscale impact": "fiscale_impact",
            "juridische structuur": "juridische_structuur_partijen",
            "risico's": "risicos_aannames_onvoldoende_data",
            "aanpak": "aanpak_stappenplan",
            "open vragen": "open_vragen",
        }
        low = l.lower()
        for k, v in tag_map.items():
            if low.startswith(k):
                cur_key = v
                break
        if l.startswith("-") and cur_key:
            sections[cur_key].append(l)

    with open(os.path.join(out_dir, "final.json"), "w", encoding="utf-8") as jf:
        json.dump(sections, jf, ensure_ascii=False, indent=2)

    return {
        "job_dir": out_dir,
        "models": {
            "map": MAP_MODEL,
            "reduce": REDUCE_MODEL,
            "reduce_fallback": FALLBACK_REDUCE_MODEL,
            "final": FINAL_MODEL,
        },
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run multi-GPU multi-layer analysis via Ollama")
    p.add_argument("--project-name", default="Project", help="Naam van het project voor prompts")
    p.add_argument("--files", nargs="*", help="Specifieke bestanden (pad) i.p.v. alle uit uploads/")
    p.add_argument("--resume-dir", default=None, help="Pad naar bestaande job-dir om Reduce + Final te hervatten")
    args = p.parse_args()

    result = run_pipeline(project_name=args.project_name, selected_files=args.files, resume_dir=args.resume_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
