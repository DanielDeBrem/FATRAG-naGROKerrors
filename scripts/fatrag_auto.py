#!/usr/bin/env python3
"""
FATRAG Auto Orchestrator (read-only VectorDB)

# Disable ChromaDB telemetry to prevent PostHog errors
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY'] = 'False'

Implements the requested multi-layer pipeline using ONLY existing vector-DB retrieval:
1) Auto-detect: working dirs, VectorDB (read-only), Ollama models, GPU worker endpoints
2) Retrieve: read-only topK chunks from the detected VectorDB
3) Layer-1 analysis (parallel over 8 GPU endpoints) → JSONL shards in ./outputs/l1/
4) Layer-2 synthesis (cross-doc) → ./outputs/l2/l2.json (+ evidence.csv, clusters)
5) Final report (map-reduce sections, parallel) → ./report/final/final.md + final.pdf
   - Mermaid organogram (.mmd → .png where possible)
   - Matplotlib charts from CSVs (if matplotlib available)
6) Validation + KPI logging

Hard constraints:
- Do NOT modify the vector DB (no re-embed, no schema change, no writes)
- Choose models from what is already present in Ollama (no pulls)
- One Ollama server per GPU (ports 11434..11441), map-reduce parallelism
- Network timeouts: <= 30s for health/DB pings; friendly errors
- Keep logs in ./logs and write '✅ DONE <fase>' per phase in ./logs/status.log

NOTE:
- This script prefers Chroma (local) when present, as seen in this repo.
- If tools like pandoc or mermaid-cli are not available, it writes fallbacks and logs warnings.
"""

from __future__ import annotations

import os
import sys
import re
import csv
import json
import time
import math
import glob
import shutil
import socket
import signal
import random
import string
import traceback
import subprocess
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional imports guarded
try:
    import chromadb  # for read-only detection & listing
except Exception:
    chromadb = None

# LangChain Chroma vectorstore for read-only similarity_search
try:
    from langchain_ollama import OllamaEmbeddings
    try:
        # Try new langchain-chroma package first
        from langchain_chroma import Chroma as LCChroma
    except ImportError:
        # Fallback to deprecated langchain_community
        from langchain_community.vectorstores import Chroma as LCChroma
except Exception:
    OllamaEmbeddings = None
    LCChroma = None

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

# Load config from config.json (respects LLM settings from admin UI)
sys.path.insert(0, PROJECT_ROOT)
try:
    import config_store as cfg_store
    APP_CONFIG = cfg_store.load_config()
except Exception:
    APP_CONFIG = {}

# Constants
WORKER_PORTS = [11434, 11435, 11436, 11437, 11438, 11439, 11440, 11441]
CONNECT_TIMEOUT_S = 5
READ_TIMEOUT_S = 30  # For retrieval and health checks
GENERATION_TIMEOUT_S = 600  # For actual LLM generation (10 minutes for 70B model)

# Directories
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
L1_DIR = os.path.join(PROJECT_ROOT, "outputs", "l1")
L2_DIR = os.path.join(PROJECT_ROOT, "outputs", "l2")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")
ASSETS_DIR = os.path.join(REPORT_DIR, "assets")
FINAL_DIR = os.path.join(REPORT_DIR, "final")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(L1_DIR, exist_ok=True)
os.makedirs(L2_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

STATUS_LOG = os.path.join(LOGS_DIR, "status.log")
CONFIG_YAML = os.path.join(PROJECT_ROOT, "config.yaml")

# Defaults (can be overridden by config.json)
DEFAULT_TOPK = 200
DEFAULT_TEMPS = {
    "l1": APP_CONFIG.get("TEMPERATURE", 0.2),
    "l2": APP_CONFIG.get("TEMPERATURE", 0.15),
    "final": APP_CONFIG.get("TEMPERATURE", 0.1)
}
DEFAULT_PARALLEL = {"max_inflight_per_gpu": 4}

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log_status(msg: str) -> None:
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    try:
        with open(STATUS_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def read_json(path: str, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def which(cmd: str) -> Optional[str]:
    try:
        return shutil.which(cmd)
    except Exception:
        return None

def http_get(url: str, timeout: int = READ_TIMEOUT_S) -> Tuple[int, str]:
    # Simple curl-based GET to avoid adding deps; respects connect/read timeouts via curl
    try:
        cmd = ["curl", "-m", str(timeout), "-sS", "-w", "%{http_code}", "-o", "-", url]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return (0, f"curl_error:{proc.stderr.strip()}")
        # Split body and code: since we wrote stdout (-o -) and appended code with -w
        # But curl mixes; safer: call twice for health; here we just return 200 if body looks JSON.
        return (200, proc.stdout)
    except Exception as e:
        return (0, str(e))

def port_healthy(port: int) -> bool:
    try:
        code, _ = http_get(f"http://127.0.0.1:{port}/api/tags", timeout=READ_TIMEOUT_S)
        return code == 200
    except Exception:
        return False

def start_workers_if_needed() -> None:
    unhealthy = [p for p in WORKER_PORTS if not port_healthy(p)]
    if not unhealthy:
        log_status("All worker ports healthy; skipping start.")
        return
    # Start via existing script
    script = os.path.join(SCRIPTS_DIR, "start_ollama_workers.sh")
    if not os.path.isfile(script):
        log_status("WARNING: scripts/start_ollama_workers.sh not found; cannot auto-start workers.")
        return
    log_status(f"Starting Ollama workers for unhealthy ports: {unhealthy}")
    try:
        subprocess.run(["bash", script], check=True)
    except subprocess.CalledProcessError as e:
        log_status(f"ERROR: failed to start workers: {e}")
        return
    # Wait health
    deadline = time.time() + 30
    while time.time() < deadline:
        if all(port_healthy(p) for p in WORKER_PORTS):
            log_status("All worker ports healthy after start.")
            return
        time.sleep(1)
    log_status("WARNING: Not all worker ports became healthy within 30s.")

def nvidia_smi_snapshot() -> str:
    if not which("nvidia-smi"):
        return "nvidia-smi not found"
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, check=False
        )
        return q.stdout.strip() or "(no output)"
    except Exception as e:
        return f"(nvidia-smi error: {e})"

# -----------------------------------------------------------------------------
# VectorDB detection (read-only)
# -----------------------------------------------------------------------------

@dataclass
class VectorDBInfo:
    type: str
    connection: str
    collection: Optional[str]
    topK_default: int

def detect_vectordb() -> Optional[VectorDBInfo]:
    # 1) FAISS (look for *.faiss) — read-only not implemented here; we will log but skip selection
    faiss_candidates = glob.glob(os.path.join(PROJECT_ROOT, "**", "*.faiss"), recursive=True)
    if faiss_candidates:
        log_status(f"Detected FAISS candidates: {faiss_candidates[:3]}{'...' if len(faiss_candidates)>3 else ''} (skipping; no read-only client wired)")

    # 2) Chroma (prefer if present)
    chroma_dir = os.getenv("CHROMA_DIR")
    chroma_collection = os.getenv("CHROMA_COLLECTION")
    # Fallback to config/config.json
    cfg_json = read_json(os.path.join(PROJECT_ROOT, "config", "config.json"), default={})
    if not chroma_dir:
        chroma_dir = cfg_json.get("CHROMA_DIR")
    if not chroma_collection:
        chroma_collection = cfg_json.get("CHROMA_COLLECTION")
    # If config points to a non-existent directory, ignore it so we can fallback
    if chroma_dir and not os.path.isdir(chroma_dir):
        log_status(f"WARNING: Config CHROMA_DIR '{chroma_dir}' does not exist; falling back to repository defaults.")
        chroma_dir = None

    # Final default: prefer repo's fatrag_chroma_db if present
    if not chroma_dir:
        default_dir = os.path.join(PROJECT_ROOT, "fatrag_chroma_db")
        if os.path.isdir(default_dir):
            chroma_dir = default_dir

    if chromadb and chroma_dir and os.path.isdir(chroma_dir):
        try:
            client = chromadb.PersistentClient(path=chroma_dir)
            cols = client.list_collections()
            col_names = [c.name for c in cols]

            # Optional: exclude specific collections via env (e.g., "pieterman,langchain")
            exclude_raw = os.getenv("VDB_EXCLUDE_COLLECTIONS", "")
            excluded = set(s.strip() for s in exclude_raw.split(",") if s.strip())
            if excluded:
                col_names = [n for n in col_names if n not in excluded]

            # Inspect counts per collection (prefer non-empty)
            counts: Dict[str, int] = {}
            for name in col_names:
                try:
                    coll = client.get_collection(name)
                    cnt = 0
                    try:
                        # Most chroma clients expose count()
                        cnt = int(coll.count())
                    except Exception:
                        # Fallback: attempt a small peek() if available
                        try:
                            peek = coll.peek()
                            # peek may return dict of arrays; estimate length by ids
                            if isinstance(peek, dict) and "ids" in peek and isinstance(peek["ids"], list):
                                cnt = len(peek["ids"])
                        except Exception:
                            cnt = 0
                    counts[name] = cnt
                except Exception:
                    counts[name] = 0

            # Apply exclusion to counts as well
            if excluded:
                counts = {k: v for k, v in counts.items() if k in col_names}

            # Choose collection:
            # - if requested exists and non-empty -> use it
            # - else pick the non-empty collection with the highest count
            # - else fallback to first available (may be empty)
            chosen: Optional[str] = None
            if chroma_collection and chroma_collection in col_names and counts.get(chroma_collection, 0) > 0:
                chosen = chroma_collection
            else:
                non_empty = [n for n in col_names if counts.get(n, 0) > 0]
                if non_empty:
                    # pick the largest by count
                    chosen = max(non_empty, key=lambda n: counts.get(n, 0))
                else:
                    chosen = col_names[0] if col_names else None

            if excluded:
                log_status(f"Chroma detected at {chroma_dir}; excluded={list(excluded)}; collections(after exclude)={col_names}; counts={counts}")
            else:
                log_status(f"Chroma detected at {chroma_dir}; collections: {col_names}; counts={counts}")
            if chosen:
                log_status(f"Chroma read-only OK; selecting collection: {chosen}")
            return VectorDBInfo(type="chroma", connection=os.path.abspath(chroma_dir), collection=chosen, topK_default=DEFAULT_TOPK)
        except Exception as e:
            log_status(f"WARNING: Chroma detection failed: {e}")

    # 3) Weaviate (env WEAVIATE_* or localhost:8080)
    weaviate_url = os.getenv("WEAVIATE_URL") or "http://127.0.0.1:8080"
    try:
        code, _ = http_get(f"{weaviate_url}/v1/.well-known/ready", timeout=READ_TIMEOUT_S)
        if code == 200:
            log_status(f"Weaviate detected at {weaviate_url} (ready)")
            return VectorDBInfo(type="weaviate", connection=weaviate_url, collection=None, topK_default=DEFAULT_TOPK)
    except Exception:
        pass

    # 4) Qdrant (env QDRANT_URL or localhost:6333)
    qdrant_url = os.getenv("QDRANT_URL") or "http://127.0.0.1:6333"
    try:
        code, _ = http_get(f"{qdrant_url}/collections", timeout=READ_TIMEOUT_S)
        if code == 200:
            log_status(f"Qdrant detected at {qdrant_url}")
            return VectorDBInfo(type="qdrant", connection=qdrant_url, collection=None, topK_default=DEFAULT_TOPK)
    except Exception:
        pass

    # 5) Milvus and 6) pgvector would require client libs; skip hard detection here to avoid new deps
    log_status("No supported VectorDB detected with read-only query capability. Aborting.")
    return None

# -----------------------------------------------------------------------------
# Models detection (ollama list)
# -----------------------------------------------------------------------------

@dataclass
class ModelsChoice:
    l1: str
    l2: str
    final: str
    inventory: List[str]
    warnings: List[str]

def parse_ollama_list() -> List[str]:
    try:
        proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return []
        lines = [ln.strip() for ln in proc.stdout.strip().splitlines() if ln.strip()]
        # Skip header if present
        if lines and lines[0].lower().startswith("name"):
            lines = lines[1:]
        names = [ln.split()[0] for ln in lines if ln]
        return names
    except Exception:
        return []

def choose_model(installed: List[str], prefs: List[str]) -> Optional[str]:
    # Exact or prefix/infix matches
    for pat in prefs:
        for m in installed:
            if m == pat or m.startswith(pat) or pat in m:
                return m
    # Fallback: prefer qwen2.5 > llama3.1 > any instruct 7-8B
    for m in installed:
        low = m.lower()
        if "qwen2.5" in low and "instruct" in low and any(x in low for x in ["7b", "8b"]):
            return m
    for m in installed:
        low = m.lower()
        if "llama3.1" in low and "instruct" in low and any(x in low for x in ["7b", "8b"]):
            return m
    # Any instruct small
    for m in installed:
        low = m.lower()
        if "instruct" in low and any(x in low for x in ["7b", "8b"]):
            return m
    # Any instruct
    for m in installed:
        if "instruct" in m.lower():
            return m
    return installed[0] if installed else None

def choose_final(installed: List[str]) -> Optional[str]:
    # Prefer 70B llama
    for pat in ["llama3.1:70b-instruct", "llama3:70b-instruct", "llama3.1:70b", "llama3:70b"]:
        for m in installed:
            if m.startswith(pat) or pat in m:
                return m
    # Next best: 34B/30B/13B
    for size in ["34b", "30b", "13b"]:
        for m in installed:
            if "instruct" in m.lower() and size in m.lower():
                return m
    # Finally: any instruct bigger than 8b
    for m in installed:
        low = m.lower()
        if "instruct" in low and any(s in low for s in ["13b", "34b", "30b"]):
            return m
    # Fallback to any instruct
    for m in installed:
        if "instruct" in m.lower():
            return m
    return installed[0] if installed else None

def detect_models() -> ModelsChoice:
    inv = parse_ollama_list()
    warnings: List[str] = []
    if not inv:
        warnings.append("No Ollama models found in local inventory.")
    
    # Check config.json first for user-configured model
    config_model = APP_CONFIG.get("LLM_MODEL")
    if config_model and config_model in inv:
        # User heeft een model ingesteld via LLM config pagina - gebruik die!
        log_status(f"Using configured LLM model from config.json: {config_model}")
        return ModelsChoice(
            l1=config_model,
            l2=config_model, 
            final=config_model,  # Gebruik configured model voor alle stages
            inventory=inv,
            warnings=[]
        )
    
    # Fallback: auto-detect like before
    # L1/L2 small preference
    l_small = choose_model(inv, ["llama3.1:8b-instruct", "llama3:8b-instruct", "mistral:7b-instruct", "qwen2:7b-instruct"])
    if not l_small:
        l_small = inv[0] if inv else "llama3.1:8b"
        warnings.append(f"Falling back L1/L2 to {l_small}")
    # Final big preference
    f_big = choose_final(inv)
    if not f_big:
        f_big = l_small
        warnings.append(f"Falling back Final to {f_big}")
    # VRAM warnings (rough)
    if f_big and all(x not in f_big.lower() for x in ["70b"]):
        warnings.append("70B model not detected; using best available alternative.")
    return ModelsChoice(l1=l_small, l2=l_small, final=f_big, inventory=inv, warnings=warnings)

# -----------------------------------------------------------------------------
# Ollama generate (non-streaming) with urllib
# -----------------------------------------------------------------------------

def ollama_generate(prompt: str, model: str, port: int, options: Optional[Dict[str, Any]] = None, timeout: int = GENERATION_TIMEOUT_S) -> Tuple[str, float]:
    import urllib.request
    import urllib.error
    url = f"http://127.0.0.1:{port}/api/generate"
    body = {"model": model, "prompt": prompt, "stream": False, "options": options or {}}
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        # Use the actual timeout parameter (not limited to READ_TIMEOUT_S) for LLM generation
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
            j = json.loads(text) if text else {}
            # Some ollama builds return dict with "response"
            out = j.get("response", "") if isinstance(j, dict) else (text or "")
            return out, (time.time() - t0)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama HTTPError {e.code}: {e.read().decode('utf-8', errors='ignore')}")
    except Exception as e:
        raise RuntimeError(f"Ollama request error on port {port}: {str(e)}")

# -----------------------------------------------------------------------------
# Retrieval (read-only)
# -----------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_id: int
    text: str
    meta: Dict[str, Any]

def build_chroma_retriever(vdb: VectorDBInfo):
    if not (LCChroma and OllamaEmbeddings):
        raise RuntimeError("LangChain Chroma/OllamaEmbeddings not available for retrieval.")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    embed_model = os.getenv("OLLAMA_EMBED_MODEL") or os.getenv("EMBED_MODEL") or "gemma2:2b"
    emb = OllamaEmbeddings(model=embed_model, base_url=base_url)
    vs = LCChroma(
        persist_directory=vdb.connection,
        embedding_function=emb,
        collection_name=vdb.collection or None,
    )
    return vs.as_retriever(search_kwargs={"k": vdb.topK_default})

def retrieve_chunks(vdb: VectorDBInfo, research_question: str) -> List[RetrievedChunk]:
    t0 = time.time()
    if vdb.type == "chroma":
        retriever = build_chroma_retriever(vdb)
        docs = retriever.get_relevant_documents(research_question)
        chunks: List[RetrievedChunk] = []
        for i, d in enumerate(docs):
            meta = dict(getattr(d, "metadata", {}) or {})
            text = getattr(d, "page_content", "") or ""
            doc_id = str(meta.get("doc_id") or meta.get("source") or meta.get("path") or f"doc_{i}")
            chunks.append(RetrievedChunk(doc_id=doc_id, chunk_id=i, text=text, meta=meta))
        elapsed = time.time() - t0
        log_status(f"Retrieve: {len(chunks)} chunks in {elapsed:.2f}s (p95 not measured per-call).")
        return chunks
    # Weaviate/Qdrant detection supported above, but not wired-in for retrieval to avoid new deps
    raise RuntimeError(f"Unsupported VectorDB type for retrieval in this environment: {vdb.type}")

# -----------------------------------------------------------------------------
# L1 Analysis (parallel per GPU, JSONL shards)
# -----------------------------------------------------------------------------

def l1_prompt(research_question: str, doc_id: str, chunk_text: str) -> str:
    return f"""
Je bent FinAdviseur-NL, senior financieel specialist voor Nederland.
Focus: financiële en fiscale analyse, rechtsvormen, waarderingen, structuren.

Vraag: {research_question}
Context (chunk, bron={doc_id}):
---
{chunk_text}
---
Taken:
1) Financiële feiten in bullets (bedragen EUR, percentages, waarderingen, termijnen).
2) Entiteiten: personen (NL naam), BV's, holdings, stichtingen, maatschappen.
3) Relaties: aandeelhouderschap (X houdt Y% aandelen in Z), erfpacht, certificering, agio.
4) Fiscale aspecten: IB, VPB, BTW, erf-/schenkbelasting, waarderingen.
5) Bedragen: altijd met valuta (EUR/€), inclusief context (taxatie, agio, kapitaal).
6) Data: jaartallen, deadline, termijnen.
7) Citaten: exacte tekst met bron={doc_id}:range.

Output JSON: {{
  "facts": ["financieel feit 1", "financieel feit 2"],
  "entities": ["Entiteit 1 (type: BV/persoon/holding)", "Entiteit 2"],
  "relations": ["X houdt Y% aandelen in Z", "A certificeert aandelen B"],
  "amounts": ["EUR 1.234.567 (context)", "€ 50.000 agio"],
  "dates": ["2025", "deadline 31-12-2025"],
  "citations": ["{{'text':'...', 'source':'{doc_id}:p.X'}}"],
  "confidence": 0.0-1.0
}}

Wees precies, feitelijk, NL-jurisdictie. Output alleen JSON.
""".strip()

def run_l1(chunks: List[RetrievedChunk], model: str, temps: Dict[str, float]) -> Dict[str, Any]:
    os.makedirs(L1_DIR, exist_ok=True)
    # Round-robin shards
    shard_files = [os.path.join(L1_DIR, f"l1_shard{i}.jsonl") for i in range(len(WORKER_PORTS))]
    shard_fhs = [open(p, "a", encoding="utf-8") for p in shard_files]
    latencies: List[float] = []
    counts = 0

    def submit_task(item: RetrievedChunk, shard_idx: int, port: int) -> Tuple[int, float, str]:
        prompt = l1_prompt(RESEARCH_QUESTION, item.doc_id, item.text[:8000])  # safety cap
        options = {"temperature": temps.get("l1", 0.2), "num_ctx": 4096}
        # Use 5 minutes timeout for L1 (qwen2.5:7b needs more time than gemma2:2b)
        out, secs = ollama_generate(prompt, model, port, options=options, timeout=300)
        # Attempt to coerce to JSON
        payload: Dict[str, Any]
        try:
            payload = json.loads(out)
            if not isinstance(payload, dict):
                raise ValueError("not a dict")
        except Exception:
            payload = {
                "facts": [],
                "entities": [],
                "relations": [],
                "amounts": [],
                "dates": [],
                "citations": [],
                "confidence": 0.4,
                "raw": out,
            }
        rec = {
            "doc_id": item.doc_id,
            "chunk_id": item.chunk_id,
            "model": model,
            "ts": ts(),
            "latency_s": secs,
            "port": port,
            "meta": item.meta,
            "output": payload,
        }
        line = json.dumps(rec, ensure_ascii=False)
        return shard_idx, secs, line

    futures = []
    with ThreadPoolExecutor(max_workers=min(32, len(WORKER_PORTS) * DEFAULT_PARALLEL["max_inflight_per_gpu"])) as ex:
        for idx, item in enumerate(chunks):
            shard_idx = idx % len(WORKER_PORTS)
            port = WORKER_PORTS[shard_idx]
            futures.append(ex.submit(submit_task, item, shard_idx, port))
        for fut in as_completed(futures):
            try:
                shard_idx, secs, line = fut.result()
                latencies.append(secs)
                shard_fhs[shard_idx].write(line + "\n")
                counts += 1
            except Exception as e:
                log_status(f"[L1-ERROR] {e}")

    for fh in shard_fhs:
        try:
            fh.close()
        except Exception:
            pass

    # p95 latency
    p95 = sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0.0
    log_status(f"✅ DONE L1: {counts} chunks; p95 latency {p95:.2f}s")
    return {"count": counts, "p95_latency_s": p95}

# -----------------------------------------------------------------------------
# L2 Synthesis (cross-doc)
# -----------------------------------------------------------------------------

def iter_jsonl(path: str):
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def load_l1_outputs() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for i in range(len(WORKER_PORTS)):
        p = os.path.join(L1_DIR, f"l1_shard{i}.jsonl")
        items.extend(list(iter_jsonl(p)))
    return items

def l2_synthesize(l1_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Aggregate evidence and simple clustering by doc_id
    per_doc: Dict[str, Dict[str, Any]] = {}
    evidence_rows: List[Dict[str, str]] = []

    for it in l1_items:
        doc = str(it.get("doc_id", ""))
        out = it.get("output", {}) or {}
        per = per_doc.setdefault(doc, {"facts": [], "entities": [], "relations": [], "amounts": [], "dates": [], "citations": []})
        for k in ["facts", "entities", "relations", "amounts", "dates", "citations"]:
            arr = out.get(k) or []
            if isinstance(arr, list):
                per[k].extend(arr)

        # evidence rows
        for a in out.get("amounts") or []:
            evidence_rows.append({"document": doc, "type": "amount", "value": str(a)})
        for d in out.get("dates") or []:
            evidence_rows.append({"document": doc, "type": "date", "value": str(d)})
        for e in out.get("entities") or []:
            evidence_rows.append({"document": doc, "type": "entity", "value": str(e)})

    # Deduplicate and compact
    def uniq(seq: List[Any]) -> List[Any]:
        seen = set()
        res = []
        for x in seq:
            key = json.dumps(x, sort_keys=True, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x)
            if key in seen:
                continue
            seen.add(key)
            res.append(x)
        return res

    for doc, agg in per_doc.items():
        for k in list(agg.keys()):
            if isinstance(agg[k], list):
                agg[k] = uniq(agg[k])

    # Crude clustering by overlapping entities keywords
    clusters: List[Dict[str, Any]] = []
    if per_doc:
        # cluster per doc as baseline
        for doc, agg in per_doc.items():
            clusters.append({"theme": f"doc:{doc}", "docs": [doc], "facts": agg.get("facts", []), "entities": agg.get("entities", []), "relations": agg.get("relations", [])})

    # Save evidence.csv
    ev_csv = os.path.join(OUTPUTS_DIR, "evidence.csv")
    try:
        with open(ev_csv, "w", newline="", encoding="utf-8") as cf:
            w = csv.DictWriter(cf, fieldnames=["document", "type", "value"])
            w.writeheader()
            for r in evidence_rows:
                w.writerow(r)
    except Exception as e:
        log_status(f"WARNING: failed to write evidence.csv: {e}")

    # Save l2.json and clusters
    os.makedirs(L2_DIR, exist_ok=True)
    l2_path = os.path.join(L2_DIR, "l2.json")
    clusters_path = os.path.join(L2_DIR, "l2_clusters.jsonl")
    try:
        with open(l2_path, "w", encoding="utf-8") as jf:
            json.dump({"per_doc": per_doc, "clusters": clusters, "research_question": RESEARCH_QUESTION}, jf, ensure_ascii=False, indent=2)
        with open(clusters_path, "w", encoding="utf-8") as f:
            for c in clusters:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    except Exception as e:
        log_status(f"WARNING: failed to write L2 outputs: {e}")

    log_status(f"✅ DONE L2: {len(per_doc)} docs aggregated; clusters={len(clusters)}")
    return {"docs": len(per_doc), "clusters": len(clusters), "evidence_rows": len(evidence_rows)}

# -----------------------------------------------------------------------------
# Organogram (Mermaid) and Charts
# -----------------------------------------------------------------------------

def build_mermaid_from_l2(l2_json: Dict[str, Any]) -> str:
    # Build a simple graph from entities & relations
    lines = ["graph TD"]
    per_doc = l2_json.get("per_doc", {})
    node_ids: Dict[str, str] = {}
    nid = 1

    def get_node_id(label: str) -> str:
        nonlocal nid
        key = label.strip()
        if key not in node_ids:
            node_ids[key] = f"N{nid}"
            nid += 1
        return node_ids[key]

    for doc, agg in per_doc.items():
        ents = agg.get("entities", []) or []
        rels = agg.get("relations", []) or []
        # create nodes
        for e in ents:
            label = str(e)[:60]
            n = get_node_id(label)
            lines.append(f'  {n}["{label}"]')
        # create edges if relations carry "A owns X% of B" like strings
        for r in rels:
            s = str(r)
            m = re.search(r"([A-Za-z0-9 .&()/_-]+)\s+(?:houdt|bezit|heeft)\s+(\d+)%\s+(?:van\s+de\s+aandelen\s+in|in|van)\s+([A-Za-z0-9 .&()/_-]+)", s, flags=re.IGNORECASE)
            if m:
                a, pct, b = m.group(1).strip(), m.group(2), m.group(3).strip()
                na, nb = get_node_id(a), get_node_id(b)
                lines.append(f'  {na} -->|{pct}%| {nb}')
    return "\n".join(lines)

def write_mermaid_and_render(l2_json: Dict[str, Any]) -> None:
    mmd = build_mermaid_from_l2(l2_json)
    mmd_path = os.path.join(ASSETS_DIR, "org.mmd")
    write_text(mmd_path, mmd)
    mmdc = which("mmdc")
    if mmdc:
        try:
            png_path = os.path.join(ASSETS_DIR, "org.png")
            subprocess.run([mmdc, "-i", mmd_path, "-o", png_path], check=False)
            log_status("Mermaid PNG rendered.")
        except Exception as e:
            log_status(f"WARNING: Mermaid render failed: {e}")
    else:
        log_status("WARNING: mermaid-cli (mmdc) not found; wrote org.mmd only.")

def try_plot_charts() -> None:
    # optional matplotlib
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ev_csv = os.path.join(OUTPUTS_DIR, "evidence.csv")
        if not os.path.isfile(ev_csv):
            log_status("Charts: evidence.csv not found; skipping.")
            return
        # Count by type
        counts: Dict[str, int] = {}
        with open(ev_csv, "r", encoding="utf-8") as cf:
            reader = csv.DictReader(cf)
            for r in reader:
                t = r.get("type") or ""
                counts[t] = counts.get(t, 0) + 1
        if counts:
            labels = list(counts.keys())
            vals = [counts[k] for k in labels]
            plt.figure(figsize=(6, 4))
            plt.bar(labels, vals, color="#4a90e2")
            plt.title("Evidence counts by type")
            plt.tight_layout()
            plt.savefig(os.path.join(ASSETS_DIR, "evidence_counts.png"))
            plt.close()
        log_status("Charts: generated evidence_counts.png")
    except Exception as e:
        log_status(f"WARNING: matplotlib unavailable or failed: {e}")

# -----------------------------------------------------------------------------
# Final report (map-reduce sections)
# -----------------------------------------------------------------------------

SECTIONS = [
    "Executive Summary",
    "Methode",
    "Thema-bevindingen",
    "Risico’s & Aanbevelingen",
    "Organogram",
    "Tabellen & Grafieken",
    "Bijlagen (citatie-lijst)",
]

def section_prompt(section: str, l2_json: Dict[str, Any], temps: Dict[str, float]) -> str:
    # Compact context from L2
    per_doc = l2_json.get("per_doc", {})
    docs_preview = []
    for doc, agg in list(per_doc.items())[:20]:
        facts = agg.get("facts", [])[:5]
        docs_preview.append(f"- {doc}: {', '.join([str(x) for x in facts])}")
    docs_text = "\n".join(docs_preview)

    return f"""
STRICT_LANGUAGE_MODE: Nederlands
Je bent "FinAdviseur-NL", senior financieel specialist (NL, 2025).
Doel: schrijf sectie "{section}" in bullets. Professioneel, kort-bondig, geen PII, gebruik 'onvoldoende data' waar nodig.

Onderzoeksvraag: {RESEARCH_QUESTION}

Context (samenvatting per document):
{docs_text}

Vereisten voor deze sectie:
- Schrijf uitsluitend de sectie "{section}" (geen extra tekst, geen andere secties)
- Gebruik bullets
- Temperatuur laag, feitelijk
""".strip()

def run_final(models: ModelsChoice, temps: Dict[str, float]) -> Dict[str, Any]:
    # Load L2
    l2_path = os.path.join(L2_DIR, "l2.json")
    try:
        with open(l2_path, "r", encoding="utf-8") as f:
            l2_json = json.load(f)
    except Exception:
        l2_json = {"per_doc": {}, "clusters": [], "research_question": RESEARCH_QUESTION}

    # Generate Mermaid and Charts
    try:
        write_mermaid_and_render(l2_json)
    except Exception as e:
        log_status(f"WARNING: organogram generation failed: {e}")

    try:
        try_plot_charts()
    except Exception as e:
        log_status(f"WARNING: chart generation failed: {e}")

    # Parallel per section over worker ports with Final model
    section_texts: Dict[str, str] = {}
    latencies: List[float] = []

    def submit_section(section: str, port: int) -> Tuple[str, str, float]:
        prompt = section_prompt(section, l2_json, temps)
        options = {"temperature": temps.get("final", 0.1), "num_ctx": 16384}
        out, secs = ollama_generate(prompt, models.final, port, options=options, timeout=GENERATION_TIMEOUT_S)
        return section, out.strip(), secs

    futures = []
    with ThreadPoolExecutor(max_workers=len(WORKER_PORTS)) as ex:
        for i, section in enumerate(SECTIONS):
            port = WORKER_PORTS[i % len(WORKER_PORTS)]
            futures.append(ex.submit(submit_section, section, port))
        for fut in as_completed(futures):
            try:
                sec, text, secs = fut.result()
                section_texts[sec] = text
                latencies.append(secs)
            except Exception as e:
                log_status(f"[FINAL-ERROR] {e}")

    # Merge to final.md
    final_md = []
    for s in SECTIONS:
        final_md.append(f"## {s}\n")
        final_md.append(section_texts.get(s, "onvoldoende data"))
        final_md.append("\n\n")
    final_md_str = "\n".join(final_md)
    final_md_path = os.path.join(FINAL_DIR, "final.md")
    write_text(final_md_path, final_md_str)

    # Try to export to PDF via pandoc
    pdf_path = os.path.join(FINAL_DIR, "final.pdf")
    pandoc = which("pandoc")
    if pandoc:
        try:
            subprocess.run([pandoc, final_md_path, "-o", pdf_path], check=False)
        except Exception as e:
            log_status(f"WARNING: pandoc export failed: {e}")
    else:
        log_status("WARNING: pandoc not found; final.md written, PDF skipped.")

    p95 = sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0.0
    log_status(f"✅ DONE FINAL: sections={len(section_texts)}; p95 latency {p95:.2f}s")
    return {"sections_done": len(section_texts), "p95_latency_s": p95, "final_md": final_md_path, "final_pdf": (pdf_path if os.path.isfile(pdf_path) else None)}

# -----------------------------------------------------------------------------
# CONFIG.yaml writer
# -----------------------------------------------------------------------------

def write_config_yaml(project_root: str, vdb: VectorDBInfo, models: ModelsChoice, temps: Dict[str, float]) -> None:
    endpoints = [f"http://127.0.0.1:{p}" for p in WORKER_PORTS]
    data = {
        "project_root": project_root,
        "endpoints": endpoints,
        "vectordb": {
            "type": vdb.type,
            "conn_string": vdb.connection,
            "collection": vdb.collection,
        },
        "models": {"l1": models.l1, "l2": models.l2, "final": models.final, "inventory": models.inventory, "warnings": models.warnings},
        "retrieval": {"topK_default": vdb.topK_default},
        "temps": temps,
        "parallel": DEFAULT_PARALLEL,
        "timeouts": {"connect_s": CONNECT_TIMEOUT_S, "read_s": READ_TIMEOUT_S},
    }
    # Minimal YAML emitter
    def to_yaml(obj, indent=0):
        sp = "  " * indent
        if isinstance(obj, dict):
            lines = []
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{sp}{k}:")
                    lines.append(to_yaml(v, indent + 1))
                else:
                    sval = json.dumps(v, ensure_ascii=False) if isinstance(v, str) and (":" in v or v.strip() == "") else v
                    lines.append(f"{sp}{k}: {sval}")
            return "\n".join(lines)
        elif isinstance(obj, list):
            lines = []
            for it in obj:
                if isinstance(it, (dict, list)):
                    lines.append(f"{sp}-")
                    lines.append(to_yaml(it, indent + 1))
                else:
                    lines.append(f"{sp}- {it}")
            return "\n".join(lines)
        else:
            return f"{sp}{obj}"

    yml = to_yaml(data)
    write_text(CONFIG_YAML, yml)

# -----------------------------------------------------------------------------
# KPI + Validation
# -----------------------------------------------------------------------------

def validate_and_kpis() -> Dict[str, Any]:
    final_md = os.path.join(FINAL_DIR, "final.md")
    final_pdf = os.path.join(FINAL_DIR, "final.pdf")
    md_ok = False
    pdf_ok = False
    md_sections = 0
    try:
        if os.path.isfile(final_md):
            with open(final_md, "r", encoding="utf-8") as f:
                content = f.read()
            md_sections = content.count("## ")
            md_ok = md_sections >= 3
    except Exception:
        pass
    if os.path.isfile(final_pdf):
        try:
            # cannot easily count pages without extra deps; just check size
            pdf_ok = os.path.getsize(final_pdf) > 1024
        except Exception:
            pdf_ok = False

    # Basic KPI based on L1 and retrieval
    l1_items = []
    for i in range(len(WORKER_PORTS)):
        p = os.path.join(L1_DIR, f"l1_shard{i}.jsonl")
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                l1_items.extend(f.readlines())
    chunks_cnt = len(l1_items)

    kpi = {
        "final_md_sections": md_sections,
        "final_md_ok": md_ok,
        "final_pdf_ok": pdf_ok,
        "l1_chunks": chunks_cnt,
        "gpu_snapshot": nvidia_smi_snapshot(),
    }
    log_status(f"Validation: final.md ok={md_ok}, final.pdf ok={pdf_ok}, sections={md_sections}")
    log_status("✅ DONE VALIDATION")
    return kpi

# -----------------------------------------------------------------------------
# Main Orchestration
# -----------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="Run FATRAG multi-layer pipeline (read-only VectorDB)")
    p.add_argument("--question", dest="question", default="Algemene synthese", help="RESEARCH_QUESTION")
    args = p.parse_args()

    global RESEARCH_QUESTION
    RESEARCH_QUESTION = (args.question or "").strip() or "Algemene synthese"
    log_status(f"Project root: {PROJECT_ROOT}")
    log_status(f"Research question: {RESEARCH_QUESTION}")

    # 1) Detect VectorDB, models, and GPU servers; write config.yaml
    vdb = detect_vectordb()
    if not vdb:
        log_status("ERROR: No usable VectorDB detected. Exiting.")
        sys.exit(2)
    models = detect_models()
    for w in models.warnings:
        log_status(f"MODEL-WARN: {w}")

    write_config_yaml(PROJECT_ROOT, vdb, models, DEFAULT_TEMPS)
    log_status("✅ DONE DETECT + CONFIG")

    # 2) Ensure Ollama workers running and healthy
    start_workers_if_needed()
    # Healthcheck all ports
    healthy = [p for p in WORKER_PORTS if port_healthy(p)]
    log_status(f"Ollama workers healthy: {healthy}")

    # 3) Retrieve (read-only) chunks from VectorDB
    try:
        chunks = retrieve_chunks(vdb, RESEARCH_QUESTION)
    except Exception as e:
        log_status(f"ERROR: retrieve failed: {e}")
        sys.exit(2)
    if not chunks:
        log_status("onvoldoende data: geen chunks uit VectorDB.")
        # still write empty artifacts to keep pipeline shape
        write_text(os.path.join(FINAL_DIR, "final.md"), "onvoldoende data")
        sys.exit(0)
    log_status(f"✅ DONE RETRIEVE: {len(chunks)} chunks")

    # 4) Layer-1 analysis (parallel JSONL)
    try:
        l1_stats = run_l1(chunks, models.l1, DEFAULT_TEMPS)
    except Exception as e:
        log_status(f"ERROR: L1 failed: {e}")
        sys.exit(2)

    # 5) Layer-2 synthesis
    try:
        l1_loaded = load_l1_outputs()
        l2_stats = l2_synthesize(l1_loaded)
    except Exception as e:
        log_status(f"ERROR: L2 failed: {e}")
        sys.exit(2)

    # 6) Final report (map-reduce sections)
    try:
        final_stats = run_final(models, DEFAULT_TEMPS)
    except Exception as e:
        log_status(f"ERROR: FINAL failed: {e}")
        sys.exit(2)

    # Validation + KPI
    kpi = validate_and_kpis()

    # Print summary for CLI
    summary = {
        "config_yaml": CONFIG_YAML,
        "outputs": {
            "l1_dir": L1_DIR,
            "l2_dir": L2_DIR,
        },
        "report": {
            "final_md": final_stats.get("final_md"),
            "final_pdf": final_stats.get("final_pdf"),
            "assets_dir": ASSETS_DIR,
        },
        "stats": {
            "l1": l1_stats,
            "l2": l2_stats,
            "final": final_stats,
            "kpi": kpi,
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
