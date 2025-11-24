from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse as FastAPIFileResponse
from pydantic import BaseModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.middleware.base import BaseHTTPMiddleware
import socket
from document_comparison import compute_diff, get_similarity_score, version_tracker
from tax_calculator import tax_calculator
from free_ports import find_free_ports
from typing import Optional, Dict, Any, List
import document_classification as doc_class
from config_store_mysql import load_config, save_config, update_runtime_from_env, backup_config, list_backups, restore_backup
import feedback_store_mysql as fb_store
import job_store_mysql as js
import ingestion as ing
import asyncio
from datetime import datetime
from langdetect import detect, LangDetectException
import subprocess
import signal
import time
import shutil
import json
import logging
from itertools import cycle
import rag_enhancements as rag_enh

# ---------- Rate Limiting Setup ----------
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import os

# Import modular API components
from app.router_init import include_routers


# ========== FASTAPI APP SETUP ==========
app = FastAPI()

# Include modular API routers (documents, files, etc.)
try:
    include_routers(app)
except Exception as e:
    # Fail-soft: log but do not crash app startup
    print(f"Warning: failed to include API routers: {e}")

# ---------- Authentication Setup ----------
import secrets
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import RedirectResponse
from passlib.context import CryptContext

security = HTTPBasic()
# Use pbkdf2_sha256 to avoid bcrypt backend issues (72-byte limit, buggy self-tests) and keep auth robust
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# Simple user database (in production, use proper database)
USERS_DB = {
    "admin": {
        "username": "admin",
        "password": pwd_context.hash("admin"),
        "role": "admin",
        "full_name": "Administrator",
        "description": "Full system access"
    },
    "client": {
        "username": "client",
        "password": pwd_context.hash("client"),
        "role": "client",
        "full_name": "Client User",
        "description": "Limited access for client users"
    }
}

def authenticate_user(credentials: HTTPBasicCredentials):
    """Authenticate user against our user database."""
    user = USERS_DB.get(credentials.username)
    if not user:
        return None

    if not pwd_context.verify(credentials.password, user["password"]):
        return None

    return user

def admin_required(credentials: HTTPBasicCredentials = Depends(security)):
    """Dependency that requires admin level access."""
    user = authenticate_user(credentials)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

    if user["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required",
        )

    return user

# ---------- Rate Limiting Setup ----------
limiter = Limiter(key_func=get_remote_address)

# ---------- Global Exception Handlers ----------
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": "Te veel aanvragen. Probeer het over een paar minuten opnieuw.",
                "user_message": "De server heeft te veel verzoeken ontvangen. Wacht even en probeer het opnieuw.",
                "limit": exc.detail,
                "retry_after": exc.retry_after
            }
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Don't override HTTPExceptions that are already well-formed
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "VALIDATION_ERROR" if exc.status_code == 400 else "INTERNAL_ERROR",
                    "message": exc.detail,
                    "user_message": exc.detail
                }
            }
        )

    # Generate error ID for tracking
    import uuid
    error_id = str(uuid.uuid4())[:8]

    # Log the actual error with ID for debugging
    print(f"[ERROR {error_id}] {str(exc)}")

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "Een technisch probleem is opgetreden. Probeer het opnieuw.",
                "user_message": "Er ging iets mis bij het verwerken van uw verzoek. Neem contact op met support indien het probleem aanhoudt.",
                "reference_id": error_id
            }
        }
    )

# ---------- Timeout Middleware ----------
class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, default_timeout: int = 300):
        super().__init__(app)
        self.default_timeout = default_timeout

    async def dispatch(self, request: Request, call_next):
        # Allow per-request timeout override via header
        timeout_seconds = request.headers.get("X-Timeout-Seconds")
        if timeout_seconds:
            try:
                timeout_seconds = int(timeout_seconds)
            except ValueError:
                timeout_seconds = self.default_timeout
        else:
            timeout_seconds = self.default_timeout

        try:
            return await asyncio.wait_for(call_next(request), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=408,
                content={
                    "error": {
                        "code": "TIMEOUT",
                        "message": "De aanvraag duurde te lang. Probeer het opnieuw.",
                        "user_message": "De server kon de aanvraag niet tijdig verwerken. Probeer het opnieuw of neem contact op met support."
                    }
                }
            )

# Add rate limiting middleware
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
# Add timeout middleware (default 5 minutes, adjustable per request)
app.add_middleware(TimeoutMiddleware, default_timeout=300)

# Static file mounts
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="fatrag_data"), name="assets")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Config state
app.state.config = update_runtime_from_env(load_config())

LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "gemma2:2b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./fatrag_chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "fatrag")

# Multi-GPU worker routing for Ollama
# Configure ports via OLLAMA_WORKER_PORTS="11434,11435,11436,11437,11438,11439,11440,11441"
WORKER_PORTS = [int(p.strip()) for p in os.getenv("OLLAMA_WORKER_PORTS", "11434").split(",") if p.strip().isdigit()]
if not WORKER_PORTS:
    WORKER_PORTS = [11434]
_app_ports_cycle = cycle(WORKER_PORTS)

def pick_worker_base_url() -> str:
    """
    Pick an Ollama worker base URL.
    If FORCE_SINGLE_PORT=true, route all traffic to OLLAMA_BASE_URL (default :11434).
    Otherwise, round-robin over WORKER_PORTS.
    """
    force = os.getenv("FORCE_SINGLE_PORT", "true").lower() in ("1", "true", "yes")
    if force:
        return os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    try:
        port = next(_app_ports_cycle)
    except Exception:
        port = WORKER_PORTS[0]
    return f"http://127.0.0.1:{port}"

# Cloud routing detection (Ollama Cloud via local base URL)
def is_cloud_model(cfg: Dict[str, Any], model_name: Optional[str]) -> bool:
    """
    Determine if the current model is routed to cloud even if base_url is local.
    Precedence:
      1) Env OLLAMA_CLOUD_ROUTED=true
      2) Config OLLAMA_CLOUD_ROUTED=true
      3) Config CLOUD_MODELS includes model name (exact match)
      4) Heuristics on model name (contains ':cloud' or endswith '-cloud')
    """
    try:
        if str(os.getenv("OLLAMA_CLOUD_ROUTED", "")).lower() in ("1", "true", "yes"):
            return True
        if str((cfg or {}).get("OLLAMA_CLOUD_ROUTED", "")).lower() in ("1", "true", "yes"):
            return True
        clouds = (cfg or {}).get("CLOUD_MODELS") or []
        if isinstance(clouds, str):
            try:
                clouds = json.loads(clouds)
            except Exception:
                clouds = [c.strip() for c in clouds.split(",") if c.strip()]
        if isinstance(clouds, list) and model_name and model_name in clouds:
            return True
        m = (model_name or "").lower()
        if ":cloud" in m or m.endswith("-cloud"):
            return True
    except Exception:
        pass
    return False

def cloud_provider(cfg: Dict[str, Any]) -> str:
    """
    Returns cloud provider name if known (from config/env), else 'cloud'.
    """
    prov = (cfg or {}).get("OLLAMA_CLOUD_PROVIDER") or os.getenv("OLLAMA_CLOUD_PROVIDER") or (cfg or {}).get("LLM_PROVIDER")
    if isinstance(prov, str) and prov.strip():
        return prov.strip()
    return "cloud"

# Professional financial advisory tone phrases per language
TONE_PHRASES = {
    "nl": [
        "Juridisch correct en praktisch toepasbaar",
        "Transparant over aannames en onzekerheden",
        "Concrete stappen met documentatie",
    ],
    "en": [
        "Legally correct and practically applicable",
        "Transparent about assumptions and uncertainties",
        "Concrete steps with documentation",
    ],
}

def get_tone_phrases(code: str) -> str:
    # Default to Dutch for financial advisory (NL jurisdiction)
    key = "nl" if code == "nl" else "en"
    return " • ".join(TONE_PHRASES[key])

class Query(BaseModel):
    question: str
    project_id: Optional[str] = None
    client_id: Optional[str] = None

# LLM will be built per request from app.state.config (see helpers below)

# Embeddings (gemma2:2b voor efficiency)
embed_model = OllamaEmbeddings(
    model=EMBED_MODEL,
    base_url=pick_worker_base_url()
)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embed_model,
    collection_name=CHROMA_COLLECTION
)

# FinAdviseur-NL prompt – Professional Financial & Tax Advisory
prompt_template = """
LANGUAGE POLICY:
- You MUST answer only in: {lang_hint}.
- Do not use any other language. Do not add translations or bilingual output.
- If the question mixes languages, choose the predominant language and stick to it.

IDENTITY & EXPERTISE:
Je bent "FinAdviseur-NL", een senior financieel specialist voor Nederland. 
Je expertise omvat: inkomstenbelasting (IB), vennootschapsbelasting (VPB), 
omzetbelasting (BTW), loonheffingen, overdrachtsbelasting, schenk- en 
erfbelasting; rechtsvormen (eenmanszaak, VOF, maatschap, BV, NV, stichting, 
coöperatie), herstructureringen (inbreng, activa/passiva, aandelentransacties, 
juridische splitsing/fusie), waarderingen (DCF, multiples, intrinsieke waarde), 
financiering (bank, achtergesteld, convertibles), en notariële akten 
(oprichting BV, statutenwijziging, certificering/aandelen, levenstestament, 
schenkings- en huwelijkse voorwaarden).

DOEL:
- Lever praktisch, correct en to-the-point NL-advies met concrete stappen.
- Markeer onzekerheden expliciet en benoem vereiste aannames.
- Geef altijd NL-jurisdictie-context en jaartal/regime (2025 tenzij anders).

STIJL & FORMAT:
- Antwoord in het Nederlands, professioneel maar begrijpelijk.
- Begin met een korte TL;DR in 2–4 bullets.
- Gebruik daarna secties: 
  1) Situatie & aannames 
  2) Opties met pro/contra & fiscale impact 
  3) Aanpak stap-voor-stap (to-do) 
  4) Risico's & randvoorwaarden 
  5) Documenten/akten & betrokken partijen (notaris, Belastingdienst, bank) 
  6) Indicatieve bedragen/bandbreedtes (alleen indien verantwoord)

BELEID & VEILIGHEID:
- Jurisdictie = Nederland (tenzij user anders vraagt). Noem expliciet wanneer 
  regels per gemeente/provincie of uitzonderingen gelden.
- Als data ontbreekt: stel maximaal 3 gerichte vragen die direct beslissend zijn. 
  Doe tijdelijke aannames en ga verder (label: "Aannames").
- Geen speculatie buiten expertise; geen persoonlijke beleggingsaanbevelingen 
  zonder risicoprofiel/termijn/doelen. 
- Geef geen juridisch bindende verklaring; verwijs voor definitieve stap naar 
  notaris/adviseur en noem relevante formulieren/acties (bv. KVK-wijziging, 
  modelakten, aangifteformulieren, vooroverleg Belastingdienst).
- Wees streng op anti-hallucinatie: als je het niet zeker weet, zeg het en 
  benoem waar te verifiëren (wet, besluit, HR-jurisprudentie, Belastingdienst).

REASONING:
- Toon geen interne chain-of-thought; geef alleen conclusies, berekeningen, 
  en redeneerstappen die nodig zijn voor het antwoord.
- Bij berekeningen: laat formule, input, en uitkomst zien. Toon bandbreedtes 
  en gevoeligheid (scenario laag/basis/hoog) indien relevant.

Professional standards: {tone_phrases}

Remember: Answer ONLY in {lang_hint}. Use Nederlandse jurisdictie en jaartal 2025.

Context: {context}
Question: {question}

Geef je antwoord in FinAdviseur-NL stijl met TL;DR en gestructureerde secties:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question", "lang_hint", "tone_phrases"])

strict_prompt_template = """
STRICT_LANGUAGE_MODE: true
- You MUST answer only in: {lang_hint}.
- Remove any other language. Do not include translations, parenthetical notes, or dual-language content.

Je bent "FinAdviseur-NL", senior financieel specialist voor Nederland.
Expertise: IB, VPB, BTW, loonheffingen, rechtsvormen, waarderingen, notariële akten.
Jurisdictie: Nederland, jaartal 2025.

FORMAT: Geef kort TL;DR + secties (Situatie, Opties, Stappen, Risico's, Documenten, Bedragen).
Professional standards: {tone_phrases}

Context: {context}
Question: {question}

Antwoord UITSLUITEND in {lang_hint}, FinAdviseur-NL stijl:
"""
STRICT_PROMPT = PromptTemplate(template=strict_prompt_template, input_variables=["context", "question", "lang_hint", "tone_phrases"])

def build_llm_from_config(cfg: Dict[str, Any]) -> ChatOllama:
    model = cfg.get("LLM_MODEL") or os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
    base_url = cfg.get("OLLAMA_BASE_URL") or pick_worker_base_url()
    temperature = cfg.get("TEMPERATURE", 0.7)
    # Adaptive timeout: longer for cloud/large models; override with LLM_TIMEOUT if set
    try:
        timeout = int(cfg.get("LLM_TIMEOUT", 30) or 30)
    except Exception:
        timeout = 30
    try:
        if is_cloud_model(cfg, model) and timeout < 90:
            timeout = 90
    except Exception:
        pass
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        timeout=timeout,
    )


def build_qa_chain(cfg: Dict[str, Any], lang_hint: str = "", tone_phrases: str = "", prompt: PromptTemplate = PROMPT, search_filter: Optional[Dict[str, Any]] = None) -> RetrievalQA:
    k = int(cfg.get("RETRIEVER_K", 5) or 5)
    search_kwargs = {"k": k}
    if search_filter:
        search_kwargs["filter"] = search_filter
    return RetrievalQA.from_chain_type(
        llm=build_llm_from_config(cfg),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs=search_kwargs),
        chain_type_kwargs={"prompt": prompt.partial(lang_hint=lang_hint, tone_phrases=tone_phrases)},
    )


async def warmup_llm(cfg: Dict[str, Any]) -> None:
    """
    Best-effort warmup with exponential backoff to ensure the selected model loads.
    """
    try:
        llm = build_llm_from_config(cfg)
        delays = [0.0, 0.5, 1.0, 2.0]
        for d in delays:
            if d:
                await asyncio.sleep(d)
            try:
                await llm.ainvoke("ping")
                break
            except Exception:
                continue
    except Exception:
        # Warmup failures should not break the API
        pass

@app.post("/query")
async def query(q: Query):
    try:
        # Special handling for contact questions
        lower_q = (q.question or "").strip().lower()
        if "met wie moet ik contact opnemen" in lower_q or ("contact" in lower_q and "opnemen" in lower_q):
            return {"response": "Voor contact: ga naar www.fatrag.com — daar vind je alle mogelijkheden om met ons in gesprek te gaan."}

        cfg = getattr(app.state, "config", {}) or {}

        # Language detection for response language hint
        raw_q = q.question or ""
        try:
            code = detect(raw_q)
        except LangDetectException:
            code = ""
        lang_map = {"nl": "Nederlands", "de": "Deutsch", "fr": "Français", "en": "English"}
        lang_hint = lang_map.get(code, "")
        tone_phrases = get_tone_phrases(code if code in lang_map else "en")

        # First attempt with standard prompt
        search_filter = None
        try:
            pid = getattr(q, "project_id", None)
            cid = getattr(q, "client_id", None)
            if pid or cid:
                sf = {}
                if pid:
                    sf["project_id"] = pid
                if cid:
                    sf["client_id"] = cid
                search_filter = sf
        except Exception:
            search_filter = None

        qa = build_qa_chain(cfg, lang_hint=lang_hint, tone_phrases=tone_phrases, prompt=PROMPT, search_filter=search_filter)
        response = qa.invoke({"query": q.question})
        answer = response["result"] if isinstance(response, dict) and "result" in response else response
        answer = answer or ""

        # Optional post-check: if model drifted to a wrong language, retry once with strict prompt
        try:
            ans_code = detect(answer) if answer else ""
        except LangDetectException:
            ans_code = ""
        if code in lang_map and ans_code and ans_code != code:
            qa2 = build_qa_chain(cfg, lang_hint=lang_hint, tone_phrases=tone_phrases, prompt=STRICT_PROMPT, search_filter=search_filter)
            response2 = qa2.invoke({"query": q.question})
            answer2 = response2["result"] if isinstance(response2, dict) and "result" in response2 else response2
            if answer2:
                answer = answer2

        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/login")
async def login_page():
    """Serve the login page."""
    return FileResponse("static/login.html")

@app.get("/logout")
async def logout():
    """Logout endpoint - returns 401 to trigger browser's basic auth logout."""
    return JSONResponse(
        status_code=401,
        content={"message": "Logged out successfully"},
        headers={"WWW-Authenticate": "Basic"}
    )

@app.get("/")
async def root():
    """Root path redirects to login page."""
    return RedirectResponse(url="/login", status_code=302)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/login")
async def login_page():
    """Serve the login page."""
    return FileResponse("static/login.html")

@app.get("/logout")
async def logout():
    """Logout endpoint - returns 401 to trigger browser's basic auth logout."""
    return JSONResponse(
        status_code=401,
        content={"message": "Logged out successfully"},
        headers={"WWW-Authenticate": "Basic"}
    )

@app.get("/")
async def root():
    """Root path redirects to login page."""
    return RedirectResponse(url="/login", status_code=302)

def is_port_free(port: int) -> bool:
    # Robust port check: allow bind even when previous socket is in TIME_WAIT
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False

# @app.get("/admin/classification/categories")
# async def get_classification_categories(_: bool = Depends(admin_required)):
#     """
#     Get list of all available document categories.
#     """
#     try:
#         return {"categories": doc_class.list_categories()}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

def _job_set_stage(job_id: str, stage: str, progress: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> None:
    """
    Best-effort stage/telemetry updater:
    - Merges metadata JSON (adds current_stage and stages[] timeline)
    - Optionally updates progress
    - Accepts extra fields (e.g., model, base_url, cloud, started_at, result_filename)
    """
    try:
        job = js.get_job(job_id) or {}
        meta = job.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        stages = meta.get("stages") or []
        stages.append({"stage": stage, "ts": datetime.now().isoformat()})
        meta["stages"] = stages
        meta["current_stage"] = stage
        if extra and isinstance(extra, dict):
            for k, v in extra.items():
                meta[k] = v
        js.update_job(job_id, progress=progress if progress is not None else job.get("progress"), metadata=meta)
    except Exception:
        # Never crash callers
        pass

# ----- Admin endpoints -----
@app.get("/admin/health")
async def admin_health(_: bool = Depends(admin_required)):
    return {"status": "ok"}

@app.get("/admin/config")
async def get_config(_: bool = Depends(admin_required)):
    return app.state.config

@app.get("/admin/config/debug")
async def debug_config(_: bool = Depends(admin_required)):
    """Debug endpoint to show both app.state.config and config.json on disk"""
    import config_store
    disk_config = config_store.load_config()
    return {
        "app_state_config": app.state.config,
        "disk_config": disk_config,
        "match": app.state.config == disk_config,
        "llm_model_in_state": app.state.config.get("LLM_MODEL"),
        "llm_model_on_disk": disk_config.get("LLM_MODEL"),
        "temperature_in_state": app.state.config.get("TEMPERATURE"),
        "temperature_on_disk": disk_config.get("TEMPERATURE"),
    }

@app.put("/admin/config")
async def put_config(cfg: Dict[str, Any], _: bool = Depends(admin_required)):
    try:
        # Take a backup before modifying config
        try:
            backup_config("pre-change")
        except Exception:
            pass

        save_config(cfg)
        app.state.config = update_runtime_from_env(load_config())

        # Warm up the newly selected model asynchronously (skip for cloud models)
        model_name = cfg.get("LLM_MODEL", "").lower()
        is_cloud_model = "cloud" in model_name or "api" in model_name or "gpt" in model_name
        
        if not is_cloud_model:
            try:
                asyncio.create_task(warmup_llm(app.state.config))
            except Exception:
                pass

        return {
            "status": "saved", 
            "config": app.state.config,
            "warmup_skipped": is_cloud_model,
            "message": "Cloud model detected - warmup skipped" if is_cloud_model else "Model warming up"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/config/backups")
async def admin_list_backups(_: bool = Depends(admin_required)):
    try:
        return {"items": list_backups()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/config/backup")
async def admin_make_backup(body: Optional[Dict[str, Any]] = None, _: bool = Depends(admin_required)):
    try:
        label = ""
        if body and isinstance(body, dict):
            label = (body.get("label") or "").strip()
        name = backup_config(label if label else None)
        return {"status": "ok", "name": name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/config/rollback")
async def admin_rollback_backup(body: Dict[str, Any], _: bool = Depends(admin_required)):
    name = (body or {}).get("name") or ""
    if not name:
        raise HTTPException(status_code=400, detail="Field 'name' is required")
    try:
        restore_backup(name)
        app.state.config = update_runtime_from_env(load_config())
        try:
            asyncio.create_task(warmup_llm(app.state.config))
        except Exception:
            pass
        return {"status": "rolled_back", "config": app.state.config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Documents
@app.get("/admin/docs")
async def admin_docs(_: bool = Depends(admin_required)):
    try:
        return {"items": ing.list_uploaded_files()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/docs/text")
async def admin_docs_text(payload: Dict[str, Any], _: bool = Depends(admin_required)):
    name = (payload.get("filename") or "document.txt").strip()
    text = payload.get("text") or ""
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' is required")
    # Save to uploads dir
    ing.ensure_dirs()
    uploads_dir = os.path.join(os.path.dirname(__file__), "fatrag_data", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    fpath = os.path.join(uploads_dir, name)
    try:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(text)
        chunks = ing.chunk_texts([text], 500, 100)
        meta = {"source": name, "doc_id": name, "kind": "upload", "uploaded_by": "admin"}
        ing.ingest_texts(vectorstore, chunks, meta, persist=True)
        return {"status": "ingested", "filename": name, "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/admin/docs/{filename}")
async def admin_docs_delete(filename: str, _: bool = Depends(admin_required)):
    try:
        ing.delete_by_source(vectorstore, filename, persist=True)
    except Exception:
        pass
    try:
        uploads_dir = os.path.join(os.path.dirname(__file__), "fatrag_data", "uploads")
        fpath = os.path.join(uploads_dir, filename)
        if os.path.isfile(fpath):
            os.remove(fpath)
    except Exception:
        pass
    return {"status": "deleted", "filename": filename}

@app.delete("/admin/projects/{project_id}/documents/{doc_id}")
async def delete_project_document(project_id: str, doc_id: str, _: bool = Depends(admin_required)):
    """
    Delete a document from a project.
    Removes from database and optionally from filesystem and vectorstore.
    """
    try:
        import pymysql
        conn = cp.get_db_connection()
        
        try:
            # Get document info first
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT filename, file_path FROM documents WHERE doc_id = %s AND project_id = %s",
                    (doc_id, project_id)
                )
                result = cursor.fetchone()
                
                if not result:
                    raise HTTPException(status_code=404, detail="Document not found")
                
                filename = result.get("filename")
                file_path = result.get("file_path")
            
            # Delete from vectorstore
            if filename:
                try:
                    ing.delete_by_source(vectorstore, filename, persist=True)
                except Exception as e:
                    # Log but continue - vectorstore delete is best effort
                    print(f"Warning: Could not delete from vectorstore: {e}")
            
            # Delete physical file
            if file_path and os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Could not delete file: {e}")
            
            # Delete from database
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM documents WHERE doc_id = %s AND project_id = %s",
                    (doc_id, project_id)
                )
            conn.commit()
            
            return {
                "status": "deleted",
                "doc_id": doc_id,
                "filename": filename
            }
        
        finally:
            conn.close()
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/admin/api/outputs/{dirname}")
async def delete_output_directory(dirname: str, _: bool = Depends(admin_required)):
    """
    Delete a specific outputs subdirectory (recursive).
    Safety:
      - Only allows known analysis prefixes (job-, flash-, rag_analysis_, oneshot-, batch-, chunked, filtered-, structure-)
      - Blocks reserved/system dirs (l1, l2, diagnostics) and path traversal
    """
    try:
        safe_name = os.path.basename((dirname or "").strip())
        if not safe_name or safe_name in (".", ".."):
            raise HTTPException(status_code=400, detail="Invalid directory name")

        # Allow-list to avoid accidental deletion of reserved dirs
        allowed_prefixes = ("job-", "flash-", "rag_analysis_", "oneshot-", "batch-", "chunked", "filtered-", "structure-")
        if not any(safe_name.startswith(p) for p in allowed_prefixes):
            raise HTTPException(status_code=400, detail="Deletion not allowed for this directory")

        outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
        target_dir = os.path.join(outputs_dir, safe_name)

        # Ensure target is inside outputs_dir
        real_outputs = os.path.realpath(outputs_dir)
        real_target = os.path.realpath(target_dir)
        if not real_target.startswith(real_outputs + os.sep) and real_target != real_outputs:
            raise HTTPException(status_code=400, detail="Unsafe path")

        if not os.path.isdir(real_target):
            raise HTTPException(status_code=404, detail="Directory not found")

        shutil.rmtree(real_target)
        return {"status": "deleted", "name": safe_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Feedback
class FeedbackIn(BaseModel):
    question: str
    answer: str
    rating: Optional[str] = None  # "up" | "down"
    corrected_answer: Optional[str] = None
    tags: Optional[list[str]] = None
    user_role: Optional[str] = None

@app.post("/feedback")
async def submit_user_feedback(item: FeedbackIn):
    cfg = getattr(app.state, "config", {}) or {}
    if not cfg.get("FEEDBACK_ENABLED", True):
        raise HTTPException(status_code=403, detail="Feedback disabled")
    try:
        rec = fb_store.submit_feedback(
            question=item.question,
            answer=item.answer,
            rating=item.rating,
            corrected_answer=item.corrected_answer,
            tags=item.tags,
            user_role=item.user_role,
        )
        return {"status": "ok", "id": rec["id"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/feedback")
async def admin_list_feedback(status: Optional[str] = None, _: bool = Depends(admin_required)):
    try:
        return {"items": fb_store.list_feedback(status=status)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ModerationIn(BaseModel):
    corrected_answer: Optional[str] = None

@app.post("/admin/feedback/{fid}/approve")
async def admin_approve_feedback(fid: str, body: ModerationIn, _: bool = Depends(admin_required)):
    rec = fb_store.update_status(fid, "approved", corrected_answer=body.corrected_answer)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    content = body.corrected_answer or rec.get("corrected_answer") or rec.get("answer") or ""
    if content:
        chunks = ing.chunk_texts([content], 500, 100)
        meta = {"source": f"feedback:{fid}", "doc_id": f"feedback:{fid}", "kind": "feedback", "uploaded_by": "moderator"}
        try:
            ing.ingest_texts(vectorstore, chunks, meta, persist=True)
        except Exception:
            # still return success on moderation even if ingest fails
            pass
    return {"status": "approved", "id": fid}

@app.post("/admin/feedback/{fid}/reject")
async def admin_reject_feedback(fid: str, _: bool = Depends(admin_required)):
    rec = fb_store.update_status(fid, "rejected")
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    return {"status": "rejected", "id": fid}

# Admin UI
@app.get("/admin")
async def admin_root(_: bool = Depends(admin_required)):
    # Serve a minimal static admin SPA
    return FileResponse("static/admin/index.html")

# Also serve /admin/index.html explicitly to avoid 404s in links
@app.get("/admin/index.html")
async def admin_index_html(_: bool = Depends(admin_required)):
    return FileResponse("static/admin/index.html")

@app.get("/admin/clients.html")
async def admin_clients(_: bool = Depends(admin_required)):
    return FileResponse("static/admin/clients.html")

@app.get("/admin/projects.html")
async def admin_projects(_: bool = Depends(admin_required)):
    return FileResponse("static/admin/projects.html")

@app.get("/admin/project-detail.html")
async def admin_project_detail(_: bool = Depends(admin_required)):
    return FileResponse("static/admin/project-detail.html")

# Serve additional Admin static pages (ensure links in menu do not 404)
@app.get("/admin/monitor.html")
async def admin_monitor(_: bool = Depends(admin_required)):
    return FileResponse("static/admin/monitor.html")

@app.get("/admin/progressive-test.html")
async def admin_progressive_test(_: bool = Depends(admin_required)):
    return FileResponse("static/admin/progressive-test.html")

@app.get("/admin/review-output.html")
async def admin_review_output(_: bool = Depends(admin_required)):
    return FileResponse("static/admin/review-output.html")

@app.get("/admin/analyses.html")
async def admin_analyses(_: bool = Depends(admin_required)):
    return FileResponse("static/admin/analyses.html")

@app.get("/admin/llm-config.html")
async def admin_llm_config(_: bool = Depends(admin_required)):
    return FileResponse("static/admin/llm-config.html")

@app.get("/admin/upload-progress.html")
async def admin_upload_progress(_: bool = Depends(admin_required)):
    return FileResponse("static/admin/upload-progress.html")

@app.get("/admin/ollama/models")
async def list_ollama_models(_: bool = Depends(admin_required)):
    """
    List all available Ollama models on the system.
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail="Failed to list Ollama models")
        
        # Parse ollama list output
        lines = result.stdout.strip().split('\n')
        models = []
        
        # Skip header line
        for line in lines[1:]:
            if not line.strip():
                continue
            
            parts = line.split()
            if len(parts) >= 1:
                model_name = parts[0]
                # Get size if available
                size = parts[1] if len(parts) > 1 else ""
                models.append({
                    "name": model_name,
                    "size": size
                })
        
        return {"models": models}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Timeout while listing models")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Ollama not found. Is it installed?")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/api/outputs")
async def list_outputs(_: bool = Depends(admin_required)):
    """List all directories in outputs folder"""
    try:
        outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
        if not os.path.isdir(outputs_dir):
            return {"directories": []}
        
        dirs = []
        for item in os.listdir(outputs_dir):
            item_path = os.path.join(outputs_dir, item)
            if os.path.isdir(item_path):
                # Get directory stats
                stat = os.stat(item_path)
                
                # List files in directory
                files = []
                try:
                    for f in os.listdir(item_path):
                        f_path = os.path.join(item_path, f)
                        if os.path.isfile(f_path):
                            files.append({
                                "name": f,
                                "size": os.path.getsize(f_path)
                            })
                except Exception:
                    pass
                
                dirs.append({
                    "name": item,
                    "mtime": stat.st_mtime,
                    "files": files
                })
        
        return {"directories": dirs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/api/outputs/{dirname}")
async def get_output_directory(dirname: str, _: bool = Depends(admin_required)):
    """
    Get details for a specific outputs subdirectory, including file list with direct URLs.
    """
    try:
        # Prevent path traversal
        safe_name = os.path.basename(dirname.strip())
        if not safe_name or safe_name in (".", ".."):
            raise HTTPException(status_code=400, detail="Invalid directory name")
        
        outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
        target_dir = os.path.join(outputs_dir, safe_name)
        if not os.path.isdir(target_dir):
            raise HTTPException(status_code=404, detail="Directory not found")
        
        stat = os.stat(target_dir)
        files = []
        try:
            for f in os.listdir(target_dir):
                f_path = os.path.join(target_dir, f)
                if os.path.isfile(f_path):
                    files.append({
                        "name": f,
                        "size": os.path.getsize(f_path),
                        "url": f"/outputs/{safe_name}/{f}"
                    })
        except Exception:
            pass
        
        return {
            "name": safe_name,
            "mtime": stat.st_mtime,
            "files": files,
            "path": f"/outputs/{safe_name}/"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== CLIENT MANAGEMENT ENDPOINTS ==========
import clients_projects as cp

@app.post("/admin/clients")
async def create_client(body: Dict[str, Any], _: bool = Depends(admin_required)):
    try:
        client = cp.create_client(
            name=body.get("name"),
            type=body.get("type", "individual"),
            tax_id=body.get("tax_id"),
            contact_info=body.get("contact_info"),
            notes=body.get("notes"),
            metadata=body.get("metadata"),
        )
        return {"status": "created", "client": client}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/clients")
async def list_clients(archived: bool = False, _: bool = Depends(admin_required)):
    try:
        clients = cp.list_clients(archived=archived)
        return {"clients": clients}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/clients/{client_id}")
async def get_client(client_id: str, _: bool = Depends(admin_required)):
    try:
        client = cp.get_client(client_id)
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        return {"client": client}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/admin/clients/{client_id}")
async def update_client(client_id: str, body: Dict[str, Any], _: bool = Depends(admin_required)):
    try:
        client = cp.update_client(
            client_id=client_id,
            name=body.get("name"),
            type=body.get("type"),
            tax_id=body.get("tax_id"),
            contact_info=body.get("contact_info"),
            notes=body.get("notes"),
            metadata=body.get("metadata"),
        )
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        return {"status": "updated", "client": client}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/admin/clients/{client_id}")
async def archive_client(client_id: str, _: bool = Depends(admin_required)):
    try:
        client = cp.archive_client(client_id)
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        return {"status": "archived", "client": client}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== PROJECT MANAGEMENT ENDPOINTS ==========

@app.post("/admin/projects")
async def create_project(body: Dict[str, Any], _: bool = Depends(admin_required)):
    try:
        project = cp.create_project(
            client_id=body.get("client_id"),
            name=body.get("name"),
            type=body.get("type", "general"),
            description=body.get("description"),
            metadata=body.get("metadata"),
        )
        return {"status": "created", "project": project}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/projects")
async def list_projects(
    client_id: Optional[str] = None,
    status: Optional[str] = None,
    archived: bool = False,
    _: bool = Depends(admin_required)
):
    try:
        projects = cp.list_projects(client_id=client_id, status=status, archived=archived)
        return {"projects": projects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/projects/{project_id}")
async def get_project(project_id: str, _: bool = Depends(admin_required)):
    try:
        project = cp.get_project_with_documents(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return {"project": project}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/admin/projects/{project_id}")
async def update_project(project_id: str, body: Dict[str, Any], _: bool = Depends(admin_required)):
    try:
        project = cp.update_project(
            project_id=project_id,
            name=body.get("name"),
            type=body.get("type"),
            status=body.get("status"),
            description=body.get("description"),
            metadata=body.get("metadata"),
        )
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return {"status": "updated", "project": project}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/admin/projects/{project_id}")
async def archive_project(project_id: str, _: bool = Depends(admin_required)):
    try:
        project = cp.archive_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return {"status": "archived", "project": project}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== DOCUMENT UPLOAD & ANALYSIS ENDPOINTS ==========

@app.post("/admin/projects/{project_id}/upload")
async def upload_project_documents(
    project_id: str,
    files: list[UploadFile] = File(...),
    _: bool = Depends(admin_required)
):
    try:
        # Verify project exists
        project = cp.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Defensive: ensure files present
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files uploaded. Field 'files' is required (multipart/form-data).")
        
        # Log basic request info
        try:
            print(f"[UPLOAD] project={project_id} files={len(files)}")
        except Exception:
            pass

        client_id = project.get("client_id")
        uploaded_files = []
        
        ing.ensure_dirs()
        uploads_dir = os.path.join(os.path.dirname(__file__), "fatrag_data", "uploads")
        
        import pymysql
        conn = cp.get_db_connection()
        
        try:
            for file in files:
                try:
                    fname = (file.filename or "").strip()
                    if not fname:
                        uploaded_files.append({
                            "filename": fname,
                            "status": "skipped",
                            "reason": "Empty filename"
                        })
                        continue

                    # Save file
                    file_path = os.path.join(uploads_dir, fname)
                    content = await file.read()
                    if not content:
                        uploaded_files.append({
                            "filename": fname,
                            "status": "skipped",
                            "reason": "Empty file content"
                        })
                        continue

                    with open(file_path, "wb") as f:
                        f.write(content)
                    
                    # Ingest to vectorstore with project context
                    result = ing.ingest_files(
                        vectorstore,
                        [file_path],
                        user="admin",
                        kind="project_upload",
                        extra_metadata={"project_id": project_id, "client_id": client_id},
                        persist=False,
                    )
                    
                    # Save to documents table
                    doc_id = cp.generate_id("doc-")
                    source_type = "project_upload"
                    
                    with conn.cursor() as cursor:
                        sql = """
                            INSERT INTO documents 
                            (doc_id, project_id, client_id, filename, source_type, file_path, file_size, status)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        cursor.execute(sql, (
                            doc_id,
                            project_id,
                            client_id,
                            fname,
                            source_type,
                            file_path,
                            len(content),
                            "indexed"
                        ))
                    conn.commit()
                    
                    uploaded_files.append({
                        "doc_id": doc_id,
                        "filename": fname,
                        "size": len(content),
                        "status": "uploaded",
                        "ingestion": result,
                    })
                except Exception as fe:
                    # Log and continue with next file
                    try:
                        print(f"[UPLOAD-ERROR] project={project_id} file={getattr(file,'filename','?')} err={fe}")
                    except Exception:
                        pass
                    uploaded_files.append({
                        "filename": getattr(file, "filename", ""),
                        "status": "error",
                        "error": str(fe)
                    })
            
            # Persist vectorstore once after all files
            vectorstore.persist()
            
            return {"status": "success", "project_id": project_id, "files": uploaded_files}
        finally:
            conn.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Helper: Map-Reduce LLM analysis over project documents ----------
import asyncio as _asyncio

async def analyze_project_documents_map_reduce(project: Dict[str, Any], documents: list[Dict[str, Any]]) -> str:
    """
    Map-Reduce analyse:
    - Map: per chunk korte, financiële bullets (NL, zonder rommel/footers)
    - Reduce: combineer tot één coherente analyse voor financieel professional
    - Regels: NL, bullets, geen PII, 'onvoldoende data' wanneer nodig
    """
    uploads_dir = os.path.join(os.path.dirname(__file__), "fatrag_data", "uploads")
    cfg = getattr(app.state, "config", {}) or {}
    llm = build_llm_from_config(cfg)

    map_summaries: list[str] = []
    doc_summaries: list[str] = []

    for doc in documents:
        filename = doc.get("filename", "")
        if not filename:
            continue
        file_path = os.path.join(uploads_dir, filename)
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(filename)[1].lower()
        # Extract cleaned text using ingestion helpers
        if ext == ".pdf":
            text = ing.read_pdf_file(file_path)
        elif ext in [".txt", ".md"]:
            text = ing.read_text_file(file_path)
        elif ext in [".xlsx", ".xls"]:
            text = ing.read_excel_file(file_path)
        else:
            text = ""

        if not text or text.startswith("[PDF extraction error"):
            continue

        # Per-document summary line
        doc_summaries.append(f"- {filename} ({len(text)} chars)")

        # Financial evidence to anchor the LLM
        try:
            evidence = ing.extract_financial_evidence(text)  # type: ignore[attr-defined]
        except Exception:
            evidence = {"amounts": [], "percentages": [], "rates": [], "dates": [], "entities": []}

        # Chunk for map stage (larger chunks for finance context)
        chunks = ing.chunk_texts([text], chunk_size=1500, chunk_overlap=200)

        for idx, chunk in enumerate(chunks):
            ev_amounts = ", ".join((evidence.get("amounts") or [])[:20]) or "geen"
            ev_perc = ", ".join((evidence.get("percentages") or [])[:20]) or "geen"
            ev_rates = ", ".join((evidence.get("rates") or [])[:20]) or "geen"
            ev_dates = ", ".join((evidence.get("dates") or [])[:20]) or "geen"

            map_prompt = f"""
Je bent FinAdviseur-NL. Analyseer onderstaande chunk beknopt en in het Nederlands. 
- Focus op bedragen (EUR/€), percentages, rentes, waarderingen, termijnen.
- Negeer footers/headers/meta. 
- Wees feitelijk; als data ontbreekt: schrijf 'onvoldoende data'.
- Lever uitsluitend puntsgewijze bullets (geen extra tekst).

Project: {project.get('name','')}
Document: {filename} (chunk {idx+1}/{len(chunks)})

Bekende financiële signalen:
- Bedragen: {ev_amounts}
- Percentages: {ev_perc}
- Rentes: {ev_rates}
- Datums/Jaren: {ev_dates}

TEKST:
\"\"\"
{chunk}
\"\"\" 

Geef uitsluitend bullets:
- Bedragen/valuta:
- Percentages/rentes:
- Waarderingen/schattingen:
- Termijnen/data:
- Partijen/entiteiten:
- Belangrijkste financiële observaties:
"""
            try:
                res = await llm.ainvoke(map_prompt)
                content = getattr(res, "content", None) or str(res)
                content = content.strip()
                if content:
                    map_summaries.append(f"Bron: {filename} | Chunk {idx+1}/{len(chunks)}\n{content}")
            except Exception:
                # Skip failed chunk maps but continue
                continue

    # If nothing mapped, early signal
    if not map_summaries:
        return "onvoldoende data: geen financiële informatie uit documenten kunnen extraheren."

    # Reduce: produce thorough but structured professional analysis (NL, bullets)
    # To respect 30s per-call timeouts, do a batched two-step reduce if the evidence is large.
    async def _llm_call(prompt: str) -> Optional[str]:
        try:
            res = await llm.ainvoke(prompt)
            return (getattr(res, "content", None) or str(res)).strip()
        except Exception:
            return None

    # If many map_summaries, first create partial reduces, then synthesize
    batch_size = 40
    partials: list[str] = []
    if len(map_summaries) > batch_size:
        for i in range(0, len(map_summaries), batch_size):
            batch = map_summaries[i:i+batch_size]
            part_prompt = f"""
Je bent FinAdviseur-NL. Combineer onderstaande 'bewijsregels' tot een compacte deel-samenvatting.
- taal: Nederlands, bullets, geen PII
- gebruik bedragen/percentages exact zoals in bewijs
- 'onvoldoende data' waar relevant

Project: {project.get('name','')}

DEEL-BEWIJS:
{chr(10).join(batch)}

Maak bullets:
- Kernbedragen en percentages
- Partijen/structuur
- Fiscale punten (IB/VPB/BOR/BTW indien aanwezig)
- Termijnen/data
- Risico's/aannames
""".strip()
            out = await _llm_call(part_prompt)
            if out:
                partials.append(out)
            await asyncio.sleep(0)  # yield
    # Synthesis prompt (either from partials or directly from all evidence)
    if partials:
        synth_input = "\n\n".join(partials)
    else:
        synth_input = (chr(10)*2).join(map_summaries)

    reduce_prompt = f"""
Je bent FinAdviseur-NL. Combineer onderstaande bewijsregels tot één grondige, coherente analyse.
- taal: Nederlands
- vorm: puntsgewijs/bullet points, professioneel
- geen PII
- gebruik bedragen/percentages exact zoals in bewijs
- als data ontbreekt: schrijf expliciet 'onvoldoende data'

Project: {project.get('name','')}

Documenten:
{chr(10).join(doc_summaries)}

BEWIJS/SAMENVATTING:
{chr(10)}{synth_input}

Maak de volgende secties (allemaal bullets):
- TL;DR (3-6 bullets, kern financieel)
- Financiële kernpunten (bedragen/percentages/rentes/waarderingen)
- Juridische structuur & partijen
- Fiscale aandachtspunten (IB/VPB/BOR/BTW, tarieven/vrijstellingen indien mogelijk)
- Risico's, aannames, onzekerheden (markeer 'onvoldoende data' waar relevant)
- Aanpak / stappenplan (actiegericht)
- Open vragen (max 5) die nodig zijn om advies te finaliseren
""".strip()
    out = await _llm_call(reduce_prompt)
    if out:
        return out
    return "onvoldoende data: reduce-fase mislukt (timeout of model niet beschikbaar)"

@app.post("/admin/projects/{project_id}/analyze-all")
async def analyze_all_project_documents(project_id: str, _: bool = Depends(admin_required)):
    """
    Analyze all documents in a project using LLM and save the analysis as a new document
    """
    try:
        from datetime import datetime
        
        # Get project with documents
        project = cp.get_project_with_documents(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        documents = project.get("documents", [])
        if not documents:
            raise HTTPException(status_code=400, detail="No documents to analyze")
        
        # Extract text from all documents
        uploads_dir = os.path.join(os.path.dirname(__file__), "fatrag_data", "uploads")
        all_texts = []
        doc_summaries = []
        
        for doc in documents:
            filename = doc.get("filename", "")
            if not filename:
                continue
                
            file_path = os.path.join(uploads_dir, filename)
            if not os.path.isfile(file_path):
                continue
            
            ext = os.path.splitext(filename)[1].lower()
            text = ""
            
            if ext == ".pdf":
                text = ing.read_pdf_file(file_path)
            elif ext in [".txt", ".md"]:
                text = ing.read_text_file(file_path)
            
            if text:
                all_texts.append(f"=== Document: {filename} ===\n{text}\n")
                doc_summaries.append(f"- {filename} ({len(text)} characters)")
        
        if not all_texts:
            raise HTTPException(status_code=400, detail="Could not extract text from documents")
        
        # Combine all texts
        combined_text = "\n\n".join(all_texts)
        
        # Create comprehensive analysis prompt
        analysis_prompt = f"""
Je bent een senior financieel analist. Analyseer alle documenten voor dit project grondig en systematisch.

PROJECT: {project.get('name', 'Unnamed')}
DOCUMENTEN ({len(doc_summaries)}):
{chr(10).join(doc_summaries)}

VOLLEDIGE DOCUMENTINHOUD:
{combined_text}

OPDRACHT:
Maak een grondige, gestructureerde analyse met:

1. **EXECUTIVE SUMMARY** (2-3 alinea's)
   - Kern van het project
   - Belangrijkste bevindingen
   - Kritische aandachtspunten

2. **DOCUMENT OVERZICHT**
   - Per document: doel, type, belangrijkste inhoud
   - Samenhang tussen documenten
   - Ontbrekende informatie

3. **FINANCIËLE ANALYSE**
   - Bedragen en waarderingen
   - Cashflows en financiering
   - Fiscale aspecten
   - Risico's en onzekerheden

4. **JURIDISCHE STRUCTUUR**
   - Betrokken entiteiten
   - Eigendomsverhoudingen
   - Contractuele afspraken
   - Relevante wet- en regelgeving

5. **CONCLUSIES & AANBEVELINGEN**
   - Belangrijkste conclusies
   - Concrete next steps
   - Aandachtspunten voor advisering
   - Vragen voor verdere uitdieping

6. **BIJLAGEN**
   - Lijst van alle acroniemen/afkortingen
   - Belangrijke datums
   - Contactpersonen/partijen

Wees:
- Concreet en feitelijk
- Transparant over aannames
- Volledig in Nederlands
- Professioneel maar toegankelijk
"""
        
        # Map-Reduce LLM analysis with evidence-first
        analysis_text = await analyze_project_documents_map_reduce(project, documents)

        # Save analysis as new document (requested naming)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_filename = f"gemaakte analyse - {timestamp}.txt"
        analysis_path = os.path.join(uploads_dir, analysis_filename)
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write(f"GRONDIGE ANALYSE - {project.get('name', 'Project')}\n")
            f.write(f"Gegenereerd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Geanalyseerde documenten: {len(documents)}\n")
            f.write("=" * 80 + "\n\n")
            f.write(analysis_text)
        
        # Save to database
        doc_id = cp.generate_id("doc-")
        file_size = os.path.getsize(analysis_path)
        
        import pymysql
        conn = cp.get_db_connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO documents 
                    (doc_id, project_id, client_id, filename, source_type, file_path, file_size, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    doc_id,
                    project_id,
                    project.get("client_id"),
                    analysis_filename,
                    "llm_analysis",
                    analysis_path,
                    file_size,
                    "indexed"
                ))
            conn.commit()
        finally:
            conn.close()
        
        return {
            "status": "success",
            "analysis_id": doc_id,
            "filename": analysis_filename,
            "analyzed_documents": len(documents),
            "analysis_length": len(analysis_text)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- GPU Preemption Helper ----------
def preempt_gpu_processes() -> None:
    """
    Best-effort: stop all non-essential GPU processes before starting analysis.
    - Requires nvidia-smi to be installed.
    - Skips our own PID and 'ollama' so LLM service remains available.
    """
    try:
        # Check nvidia-smi presence
        proc = subprocess.run(
            ["which", "nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            return  # No NVIDIA GPUs or driver
        
        # Query compute PIDs
        q = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if q.returncode != 0:
            return
        
        pids = []
        for line in q.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                pids.append(int(line))
            except ValueError:
                continue
        
        me = os.getpid()
        for pid in pids:
            if pid == me:
                continue
            # Inspect cmdline to avoid killing ollama (keep LLM available)
            cmdline = ""
            try:
                with open(f"/proc/{pid}/cmdline", "r") as f:
                    cmdline = f.read().replace("\x00", " ")
            except Exception:
                pass
            lower = cmdline.lower()
            if "ollama" in lower:
                continue
            # Be gentle first
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
        # Wait a moment and SIGKILL stubborn ones
        time.sleep(2)
        for pid in pids:
            if pid == me:
                continue
            try:
                # If still alive, send SIGKILL
                os.kill(pid, 0)
            except ProcessLookupError:
                continue  # already gone
            except Exception:
                continue
            # Re-check cmdline, skip ollama
            cmdline = ""
            try:
                with open(f"/proc/{pid}/cmdline", "r") as f:
                    cmdline = f.read().replace("\x00", " ")
            except Exception:
                pass
            if "ollama" in cmdline.lower():
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
    except Exception:
        # Never crash caller due to preemption failures
        pass

# ---------- Background job runner (no request timeout) ----------
async def _background_analyze_job(job_id: str, project_id: str) -> None:
    try:
        # Initialize telemetry
        try:
            cfg = getattr(app.state, "config", {}) or {}
            base_url = cfg.get("OLLAMA_BASE_URL") or os.getenv("OLLAMA_BASE_URL", "")
            cloud = is_cloud_model(cfg, cfg.get("LLM_MODEL"))
            _job_set_stage(job_id, "starting", progress=0, extra={
                "job_type": "analysis_all",
                "model": cfg.get("LLM_MODEL"),
                "base_url": base_url,
                "cloud": cloud,
                "cloud_provider": cloud_provider(cfg),
                "started_at": datetime.now().isoformat(),
            })
        except Exception:
            pass
        # Preempt GPUs and mark running
        try:
            js.update_job(job_id, status="preempting_gpu", progress=0)
        except Exception:
            pass
        preempt_gpu_processes()
        try:
            js.update_job(job_id, status="running", progress=5)
        except Exception:
            pass

        # Load project and docs
        project = cp.get_project_with_documents(project_id)
        if not project:
            js.update_job(job_id, status="failed", error_message="Project not found")
            _job_set_stage(job_id, "failed", progress=0, extra={"error_message": "Project not found"})
            return
        documents = project.get("documents", [])
        if not documents:
            js.update_job(job_id, status="failed", error_message="No documents to analyze")
            _job_set_stage(job_id, "failed", progress=0, extra={"error_message": "No documents to analyze"})
            return

        # Run long analysis (map-reduce, update progress during run)
        _job_set_stage(job_id, "analyzing", progress=40)
        try:
            js.update_job(job_id, status="running", progress=60)
            _job_set_stage(job_id, "analyzing", progress=60)
        except Exception:
            pass
        analysis_text = await analyze_project_documents_map_reduce(project, documents)

        # Save as new document
        uploads_dir = os.path.join(os.path.dirname(__file__), "fatrag_data", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_filename = f"gemaakte analyse - {timestamp}.txt"
        analysis_path = os.path.join(uploads_dir, analysis_filename)

        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write(f"GRONDIGE ANALYSE - {project.get('name', 'Project')}\n")
            f.write(f"Gegenereerd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Geanalyseerde documenten: {len(documents)}\n")
            f.write("=" * 80 + "\n\n")
            f.write(analysis_text)

        # Insert into documents table
        doc_id = cp.generate_id("doc-")
        file_size = os.path.getsize(analysis_path)

        import pymysql
        conn = cp.get_db_connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO documents 
                    (doc_id, project_id, client_id, filename, source_type, file_path, file_size, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    doc_id,
                    project_id,
                    project.get("client_id"),
                    analysis_filename,
                    "llm_analysis",
                    analysis_path,
                    file_size,
                    "indexed"
                ))
            conn.commit()
        finally:
            conn.close()

        # Mark completed
        try:
            js.update_job(job_id, status="completed", progress=100, result_filename=analysis_filename)
            _job_set_stage(job_id, "completed", progress=100, extra={"result_filename": analysis_filename})
        except Exception:
            pass
    except Exception as e:
        try:
            js.update_job(job_id, status="failed", error_message=str(e))
            _job_set_stage(job_id, "failed", progress=0, extra={"error_message": str(e)})
        except Exception:
            pass

# Enqueue long-running analysis (returns immediately)
@app.post("/admin/projects/{project_id}/analyze-all/async")
async def analyze_all_project_documents_async(project_id: str, background_tasks: BackgroundTasks, _: bool = Depends(admin_required)):
    try:
        job_id = cp.generate_id("job-")
        job = js.create_job(job_id, "analysis_all", project_id=project_id, metadata={"source": "admin_ui"})
        if not job:
            raise HTTPException(status_code=500, detail="Failed to create analysis job")
        background_tasks.add_task(_background_analyze_job, job_id, project_id)
        return {"status": "queued", "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Flash Analysis (30-90s ultra-fast scan) ----------
async def _background_flash_analysis(job_id: str, project_id: str) -> None:
    """
    Run flash analysis using scripts/flash_analysis.py
    - Ultra-fast: only llama3.1:8b, aggressive chunking
    - Map → Reduce only (no Final stage)
    - Output: compact bullet report (1-2 pages)
    """
    try:
        # Initialize telemetry as early as possible
        try:
            cfg = getattr(app.state, "config", {}) or {}
            base_url = cfg.get("OLLAMA_BASE_URL") or os.getenv("OLLAMA_BASE_URL", "")
            flash_model = cfg.get("FLASH_MODEL") or cfg.get("LLM_MODEL") or os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
            cloud = is_cloud_model(cfg, flash_model)
            _job_set_stage(job_id, "starting", progress=0, extra={
                "job_type": "flash_analysis",
                "model": flash_model,
                "base_url": base_url,
                "cloud": cloud,
                "cloud_provider": cloud_provider(cfg),
                "started_at": datetime.now().isoformat(),
            })
        except Exception:
            pass
        try:
            js.update_job(job_id, status="running", progress=10)
            js.update_job(job_id, status="running", progress=10)
            _job_set_stage(job_id, "initializing", progress=10)
        except Exception:
            pass

        # Load project
        project = cp.get_project_with_documents(project_id)
        if not project:
            js.update_job(job_id, status="failed", error_message="Project not found")
            return

        documents = project.get("documents", [])
        if not documents:
            js.update_job(job_id, status="failed", error_message="No documents to analyze")
            return

        # Prepare upload directory for flash_analysis.py
        uploads_dir = os.path.join(os.path.dirname(__file__), "fatrag_data", "uploads")

        try:
            js.update_job(job_id, status="running", progress=20)
            _job_set_stage(job_id, "preparing", progress=20)
        except Exception:
            pass

        # Run flash_analysis script
        flash_script = os.path.join(os.path.dirname(__file__), "scripts", "flash_analysis.py")
        if not os.path.isfile(flash_script):
            js.update_job(job_id, status="failed", error_message="flash_analysis.py script not found")
            return

        cmd = [
            "python3",
            flash_script,
            "--project-name",
            project.get('name', 'Project'),
            "--upload-dir",
            uploads_dir
        ]

        try:
            js.update_job(job_id, status="running", progress=40)
            _job_set_stage(job_id, "running_script", progress=40)
        except Exception:
            pass

        # Run flash analysis (should take 30-90s)
        env = os.environ.copy()
        env["OLLAMA_LLM_MODEL"] = flash_model

        proc = subprocess.run(
            cmd,
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            timeout=300,  # 5 min max (conservative)
            env=env
        )

        if proc.returncode != 0:
            error_msg = f"Flash analysis failed: {proc.stderr[:500]}"
            js.update_job(job_id, status="failed", error_message=error_msg)
            return

        try:
            js.update_job(job_id, status="saving", progress=85)
            _job_set_stage(job_id, "saving", progress=85)
        except Exception:
            pass

        # Parse result
        try:
            result = json.loads(proc.stdout.strip().split("\n")[-1])
        except Exception:
            result = {}

        # Copy flash report to uploads as project document
        report_path = result.get("report_path")
        if report_path and os.path.isfile(report_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            flash_filename = f"Flash_Analyse_{timestamp}.md"
            flash_dest = os.path.join(uploads_dir, flash_filename)
            shutil.copy(report_path, flash_dest)

            # Save to database
            doc_id = cp.generate_id("doc-")
            file_size = os.path.getsize(flash_dest)

            import pymysql
            conn = cp.get_db_connection()
            try:
                with conn.cursor() as cursor:
                    sql = """
                        INSERT INTO documents 
                        (doc_id, project_id, client_id, filename, source_type, file_path, file_size, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(sql, (
                        doc_id,
                        project_id,
                        project.get("client_id"),
                        flash_filename,
                        "flash_analysis",
                        flash_dest,
                        file_size,
                        "completed"
                    ))
                conn.commit()
            finally:
                conn.close()

            # Mark completed
            try:
                js.update_job(
                    job_id,
                    status="completed",
                    progress=100,
                    result_filename=flash_filename,
                    metadata={"model": result.get("model"), "documents": len(result.get("documents", []))}
                )
                _job_set_stage(job_id, "completed", progress=100, extra={"result_filename": flash_filename})
            except Exception:
                pass
        else:
            js.update_job(job_id, status="failed", error_message="No report generated")
            _job_set_stage(job_id, "failed", progress=0, extra={"error_message": "No report generated"})

    except subprocess.TimeoutExpired:
        try:
            js.update_job(job_id, status="failed", error_message="Flash analysis timeout")
            _job_set_stage(job_id, "failed", progress=0, extra={"error_message": "Flash analysis timeout"})
        except Exception:
            pass
    except Exception as e:
        try:
            js.update_job(job_id, status="failed", error_message=str(e))
            _job_set_stage(job_id, "failed", progress=0, extra={"error_message": str(e)})
        except Exception:
            pass

@app.post("/admin/projects/{project_id}/flash-analysis")
async def run_flash_analysis(
    project_id: str,
    background_tasks: BackgroundTasks,
    _: bool = Depends(admin_required)
):
    """
    Run ultra-fast Flash Analysis (30-90 seconds).
    
    This endpoint:
    - Uses only llama3.1:8b for speed
    - Aggressive chunking (smaller chunks, less overlap)
    - Map → Reduce only (skips Final stage)
    - Outputs compact 1-2 page bullet report
    
    Perfect for quick document scanning before meetings or initial reviews.
    Returns immediately with job_id for tracking progress.
    """
    try:
        # Verify project exists
        project = cp.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Create background job
        job_id = cp.generate_id("job-")
        cfg = getattr(app.state, "config", {}) or {}
        flash_model = cfg.get("FLASH_MODEL") or cfg.get("LLM_MODEL") or os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
        job = js.create_job(
            job_id,
            "flash_analysis",
            project_id=project_id,
            metadata={
                "source": "admin_ui",
                "mode": "flash",
                "model": flash_model
            }
        )

        if not job:
            raise HTTPException(status_code=500, detail="Failed to create flash analysis job")

        # Start background task
        background_tasks.add_task(_background_flash_analysis, job_id, project_id)

        return {
            "status": "queued",
            "job_id": job_id,
            "message": "Flash Analysis started. Expected completion in 30-90 seconds.",
            "model": flash_model,
            "track_progress": f"/admin/jobs/{job_id}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== PROGRESSIVE TESTING ENDPOINTS ==========

import quality_ratings as qr

async def _background_progressive_test(run_id: str, config: Dict[str, Any], project_id: str) -> None:
    """
    Run progressive test in background with configurable parameters.
    Updates run status and stores metrics in database.
    """
    # Setup file logger for progressive tests
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "progressive_test.log")
    logger = logging.getLogger("progressive_test")
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == os.path.abspath(log_path) for h in logger.handlers):
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.info(f"[{run_id}] Starting background progressive test for project {project_id}")

    try:
        # Prepare QR store early for status updates
        qr_store = qr.QualityRatings()

        # Import configurable analysis with explicit path
        import sys
        scripts_dir = os.path.join(os.path.dirname(__file__), "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        logger.info(f"[{run_id}] sys.path[0]={sys.path[0]}; attempting import of configurable_analysis from {scripts_dir}")
        try:
            import configurable_analysis as ca  # type: ignore
        except Exception as ie:
            logger.exception(f"[{run_id}] Import error loading configurable_analysis: {ie}")
            qr_store.update_test_run_status(run_id, "failed", progress=100, current_stage="error")
            return
        logger.info(f"[{run_id}] Imported configurable_analysis: file={getattr(ca, '__file__', 'unknown')}, has run_configurable_analysis={hasattr(ca, 'run_configurable_analysis')}")

        if not hasattr(ca, "run_configurable_analysis"):
            msg = "run_configurable_analysis not found in configurable_analysis"
            logger.error(f"[{run_id}] {msg}")
            qr_store.update_test_run_status(run_id, "failed", progress=100, current_stage="error")
            return

        # Update status: starting
        qr_store.update_test_run_status(run_id, "running", progress=10, current_stage="initializing")
        
        # Get project info
        project = cp.get_project_with_documents(project_id)
        if not project:
            qr_store.update_test_run_status(run_id, "failed", progress=100, current_stage="error")
            return
        
        # Prepare config
        analysis_config = ca.AnalysisConfig(
            model=config.get("model", "llama3.1:8b"),
            temperature=config.get("temperature", 0.15),
            max_tokens=config.get("max_tokens", 2500),
            max_chunks=config.get("max_chunks", 25),
        )
        
        # Update: analyzing
        qr_store.update_test_run_status(run_id, "running", progress=30, current_stage="analyzing")
        
        # Run analysis
        uploads_dir = os.path.join(os.path.dirname(__file__), "fatrag_data", "uploads")
        logger.info(f"[{run_id}] Calling ca.run_configurable_analysis(model={analysis_config.model}, temp={analysis_config.temperature}, max_tokens={analysis_config.max_tokens}, max_chunks={analysis_config.max_chunks})")
        try:
            # Optional GPU/throughput overrides from config for tuner-driven runs
            w_ports = config.get("worker_ports")
            conc = config.get("concurrency")
            csize = config.get("chunk_size")
            coverlap = config.get("chunk_overlap")

            result = ca.run_configurable_analysis(
                config=analysis_config,
                project_id=project_id,
                project_name=project.get("name", "Project"),
                upload_dir=uploads_dir,
                run_id=run_id,
                worker_ports=w_ports,
                concurrency=conc,
                chunk_size=csize,
                chunk_overlap=coverlap,
            )
        except Exception as re:
            logger.exception(f"[{run_id}] Exception in run_configurable_analysis: {re}")
            qr_store.update_test_run_status(run_id, "failed", progress=100, current_stage="error")
            return
        
        logger.info(f"[{run_id}] run_configurable_analysis returned success={result.get('success')} job_id={result.get('job_id')} documents={len(result.get('documents', []))}")
        
        if not result.get("success"):
            error_msg = result.get("error", "Unknown error")
            logger.error(f"[{run_id}] Analysis returned error: {error_msg}")
            qr_store.update_test_run_status(run_id, "failed", progress=100, current_stage="error")
            return
        
        # Update: completed
        qr_store.update_test_run_status(
            run_id,
            "completed",
            progress=100,
            current_stage="done"
        )
        
    except Exception as e:
        try:
            logging.exception(f"[{run_id}] Unhandled exception in _background_progressive_test: {e}")
            qr_store = qr.QualityRatings()
            qr_store.update_test_run_status(run_id, "failed", progress=100, current_stage="error")
        except:
            pass

@app.post("/api/progressive-test/start")
async def start_progressive_test(
    body: Dict[str, Any],
    background_tasks: BackgroundTasks,
    _: bool = Depends(admin_required)
):
    """
    Start a progressive test run with configurable parameters.
    
    Body:
    {
        "project_id": "project-xxx",
        "config": {
            "model": "llama3.1:8b",
            "temperature": 0.15,
            "max_tokens": 2500,
            "max_chunks": 25
        },
        "notes": "Testing level 2 config"
    }
    
    Returns: {"run_id": "run-xxx", "status": "queued"}
    """
    try:
        project_id = body.get("project_id")
        if not project_id:
            raise HTTPException(status_code=400, detail="project_id required")
        
        # Verify project exists
        project = cp.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        config = body.get("config", {})
        notes = body.get("notes", "")

        # Create test run with unique ID
        import time
        run_id = f"run-{int(time.time())}"
        qr_store = qr.QualityRatings()
        
        # Create test run with proper parameters
        qr_store.create_test_run(
            run_id=run_id,
            project_id=project_id,
            analysis_type="flash",  # Default to flash
            level=1,  # Start with level 1
            config=config
        )

        # Start background task
        background_tasks.add_task(_background_progressive_test, run_id, config, project_id)
        
        return {
            "run_id": run_id,
            "status": "queued",
            "message": "Progressive test started"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/progressive-test/status/{run_id}")
async def get_progressive_test_status(run_id: str, _: bool = Depends(admin_required)):
    """
    Get current status of a progressive test run.
    
    Returns:
    {
        "run_id": "run-xxx",
        "status": "running",  # queued|running|completed|failed
        "progress": 50,
        "stage": "analyzing",
        "preview": "First 500 chars of output...",
        "created_at": "2025-11-10T23:00:00"
    }
    """
    try:
        qr_store = qr.QualityRatings()
        run = qr_store.get_test_run(run_id)
        
        if not run:
            raise HTTPException(status_code=404, detail="Test run not found")
        
        return {
            "run_id": run.get("run_id"),
            "status": run.get("status"),
            "progress": run.get("progress", 0),
            "stage": run.get("stage", ""),
            "preview": run.get("output_preview", ""),
            "created_at": run.get("created_at"),
            "error_message": run.get("error_message"),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/progressive-test/result/{run_id}")
async def get_progressive_test_result(run_id: str, _: bool = Depends(admin_required)):
    """
    Get complete result and metrics for a completed test run.
    
    Returns full output text, configuration, and performance metrics.
    """
    try:
        qr_store = qr.QualityRatings()
        run = qr_store.get_test_run(run_id)
        
        if not run:
            raise HTTPException(status_code=404, detail="Test run not found")
        
        if run.get("status") != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Test run not completed yet (status: {run.get('status')})"
            )
        
        # Try to load full output from file
        metrics = run.get("metrics", {})
        job_dir = metrics.get("job_dir") if isinstance(metrics, dict) else None
        full_output = run.get("output_preview", "")
        
        if job_dir and os.path.isdir(job_dir):
            report_path = os.path.join(job_dir, "flash_report.md")
            if os.path.isfile(report_path):
                with open(report_path, "r", encoding="utf-8") as f:
                    full_output = f.read()
        
        return {
            "run_id": run.get("run_id"),
            "project_id": run.get("project_id"),
            "status": run.get("status"),
            "config": run.get("config_json"),
            "output_text": full_output,
            "metrics": run.get("metrics"),
            "created_at": run.get("created_at"),
            "completed_at": run.get("updated_at"),
            "notes": run.get("notes"),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/progressive-test/rate/{run_id}")
async def rate_progressive_test(run_id: str, body: Dict[str, Any], _: bool = Depends(admin_required)):
    """
    Save quality rating for a test run.
    
    Body:
    {
        "accuracy": 4,      # 1-5 stars
        "completeness": 5,  # 1-5 stars
        "relevance": 4,     # 1-5 stars
        "clarity": 3,       # 1-5 stars
        "notes": "Good overall but missed some details"
    }
    """
    try:
        qr_store = qr.QualityRatings()
        run = qr_store.get_test_run(run_id)
        
        if not run:
            raise HTTPException(status_code=404, detail="Test run not found")
        
        # Validate ratings
        for dimension in ["accuracy", "completeness", "relevance", "clarity"]:
            rating = body.get(dimension)
            if rating is not None:
                if not isinstance(rating, int) or rating < 1 or rating > 5:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"{dimension} must be an integer between 1 and 5"
                    )
        
        # Save rating
        rating_id = qr_store.save_quality_rating(
            run_id=run_id,
            accuracy=body.get("accuracy"),
            completeness=body.get("completeness"),
            relevance=body.get("relevance"),
            clarity=body.get("clarity"),
            notes=body.get("notes", "")
        )
        
        return {
            "status": "saved",
            "rating_id": rating_id,
            "run_id": run_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/progressive-test/history")
async def get_progressive_test_history(
    project_id: Optional[str] = None,
    limit: int = 50,
    _: bool = Depends(admin_required)
):
    """
    Get history of progressive test runs, optionally filtered by project.
    """
    try:
        qr_store = qr.QualityRatings()
        runs = qr_store.list_test_runs(project_id=project_id, limit=limit)
        
        return {
            "runs": runs,
            "count": len(runs)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== PARAMETER TUNING ENDPOINT ==========
@app.post("/api/progressive-test/tune")
async def tune_progressive_params(
    body: Dict[str, Any],
    _: bool = Depends(admin_required)
):
    """
    Run multi-GPU parameter tuning using Progressive Tests and persist best configuration.

    Body:
    {
      "project_id": "project-xxx",
      "search_space": {
        "model": ["llama3.1:8b", "qwen2.5:7b"],
        "temperature": [0.1, 0.15, 0.2],
        "max_tokens": [1536, 2048, 3072],
        "max_chunks": [15, 25, 35],
        "chunk_size": [600, 800, 1000, 1200],
        "chunk_overlap": [25, 50, 100, 200],
        "concurrency": [1, 2]
      },
      "objective": "maximize_chunks_per_second",
      "budget": { "max_trials": 8, "max_total_runtime_seconds": 1800, "early_stopping_rounds": 3 },
      "persist": true
    }
    """
    try:
        project_id = (body or {}).get("project_id")
        if not project_id:
            raise HTTPException(status_code=400, detail="project_id required")

        # Verify project exists
        project = cp.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        search_space = (body or {}).get("search_space") or {}
        objective = (body or {}).get("objective") or "maximize_chunks_per_second"
        budget = (body or {}).get("budget") or {}
        persist = bool((body or {}).get("persist", True))

        # Import tuner lazily and via scripts path
        import sys as _sys
        _scripts_dir = os.path.join(os.path.dirname(__file__), "scripts")
        if _scripts_dir not in _sys.path:
            _sys.path.insert(0, _scripts_dir)
        import tuner as _tuner  # type: ignore

        # Base URL for self API
        base_url = "http://127.0.0.1:8020"

        # Run tuning (GPU-aware) and optionally persist
        result = await _tuner.tune_fatrag_pipeline_params(
            project_id=project_id,
            search_space=search_space,
            objective=objective,
            budget=budget,
            base_url=base_url,
        )

        if persist:
            try:
                await _tuner.persist_best_to_config(result.get("best_config", {}))
                # Reload runtime config and warm the model (non-blocking)
                app.state.config = update_runtime_from_env(load_config())
                try:
                    asyncio.create_task(warmup_llm(app.state.config))
                except Exception:
                    pass
            except Exception as pe:
                # Attach but do not fail the request
                result["persist_error"] = str(pe)

        return {
            "status": "completed",
            **result
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- FATRAG Auto Pipeline Integration (Feature 1) ----------
async def _background_fatrag_pipeline(job_id: str, project_id: str, research_question: str) -> None:
    """
    Run full FATRAG Auto pipeline (L1 → L2 → Final) using scripts/fatrag_auto.py
    - Resets vector DB with project documents
    - Runs 8-GPU parallel analysis
    - Generates final.md + final.pdf + organogram
    - Saves all outputs as new project documents
    """
    try:
        # Initialize telemetry
        try:
            cfg = getattr(app.state, "config", {}) or {}
            base_url = cfg.get("OLLAMA_BASE_URL") or os.getenv("OLLAMA_BASE_URL", "")
            pipeline_model = cfg.get("PIPELINE_MODEL") or cfg.get("LLM_MODEL") or os.getenv("OLLAMA_LLM_MODEL")
            cloud = is_cloud_model(cfg, pipeline_model)
            _job_set_stage(job_id, "starting", progress=0, extra={
                "job_type": "fatrag_pipeline",
                "model": pipeline_model,
                "base_url": base_url,
                "cloud": cloud,
                "cloud_provider": cloud_provider(cfg),
                "started_at": datetime.now().isoformat(),
                "research_question": research_question,
            })
        except Exception:
            pass
        # Update job status
        try:
            js.update_job(job_id, status="initializing", progress=0)
            _job_set_stage(job_id, "initializing", progress=0)
        except Exception:
            pass

        # Get project
        project = cp.get_project_with_documents(project_id)
        if not project:
            js.update_job(job_id, status="failed", error_message="Project not found")
            return
        
        documents = project.get("documents", [])
        if not documents:
            js.update_job(job_id, status="failed", error_message="No documents to analyze")
            return

        # Create job-specific output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_output_dir = os.path.join(os.path.dirname(__file__), "outputs", f"job-{timestamp}")
        os.makedirs(job_output_dir, exist_ok=True)

        try:
            js.update_job(job_id, status="running_pipeline", progress=10)
            _job_set_stage(job_id, "running_pipeline", progress=10)
        except Exception:
            pass

        # Run FATRAG Auto pipeline
        fatrag_script = os.path.join(os.path.dirname(__file__), "scripts", "fatrag_auto.py")
        if not os.path.isfile(fatrag_script):
            js.update_job(job_id, status="failed", error_message="fatrag_auto.py script not found")
            return

        # Execute pipeline with project-specific question
        cmd = [
            "python3",
            fatrag_script,
            "--question",
            research_question or f"Algemene financiële en fiscale synthese van project: {project.get('name', 'Unnamed')}"
        ]
        
        try:
            js.update_job(job_id, status="running_l1_analysis", progress=20)
            _job_set_stage(job_id, "running_l1_analysis", progress=20)
        except Exception:
            pass

        # Run pipeline (this takes time)
        env = os.environ.copy()
        env["OLLAMA_LLM_MODEL"] = pipeline_model

        proc = subprocess.run(
            cmd,
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
            env=env
        )

        if proc.returncode != 0:
            error_msg = f"Pipeline failed: {proc.stderr[:500]}"
            js.update_job(job_id, status="failed", error_message=error_msg)
            return

        try:
            js.update_job(job_id, status="saving_outputs", progress=90)
            _job_set_stage(job_id, "saving_outputs", progress=90)
        except Exception:
            pass

        # Parse pipeline output (JSON summary)
        try:
            pipeline_result = json.loads(proc.stdout.strip().split("\n")[-1])
        except Exception:
            pipeline_result = {}

        # Save generated files as project documents
        uploads_dir = os.path.join(os.path.dirname(__file__), "fatrag_data", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        
        import pymysql
        conn = cp.get_db_connection()
        saved_docs = []

        try:
            # 1. Save final.md
            final_md = pipeline_result.get("report", {}).get("final_md")
            if final_md and os.path.isfile(final_md):
                md_filename = f"FATRAG_Report_{timestamp}.md"
                md_path = os.path.join(uploads_dir, md_filename)
                shutil.copy(final_md, md_path)
                
                doc_id = cp.generate_id("doc-")
                with conn.cursor() as cursor:
                    sql = """
                        INSERT INTO documents 
                        (doc_id, project_id, client_id, filename, source_type, file_path, file_size, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(sql, (
                        doc_id, project_id, project.get("client_id"),
                        md_filename, "fatrag_report", md_path,
                        os.path.getsize(md_path), "completed"
                    ))
                conn.commit()
                saved_docs.append(md_filename)

            # 2. Save final.pdf
            final_pdf = pipeline_result.get("report", {}).get("final_pdf")
            if final_pdf and os.path.isfile(final_pdf):
                pdf_filename = f"FATRAG_Report_{timestamp}.pdf"
                pdf_path = os.path.join(uploads_dir, pdf_filename)
                shutil.copy(final_pdf, pdf_path)
                
                doc_id = cp.generate_id("doc-")
                with conn.cursor() as cursor:
                    sql = """
                        INSERT INTO documents 
                        (doc_id, project_id, client_id, filename, source_type, file_path, file_size, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(sql, (
                        doc_id, project_id, project.get("client_id"),
                        pdf_filename, "fatrag_report", pdf_path,
                        os.path.getsize(pdf_path), "completed"
                    ))
                conn.commit()
                saved_docs.append(pdf_filename)

            # 3. Save L2 evidence CSV
            evidence_csv = os.path.join(os.path.dirname(__file__), "outputs", "evidence.csv")
            if os.path.isfile(evidence_csv):
                csv_filename = f"FATRAG_Evidence_{timestamp}.csv"
                csv_path = os.path.join(uploads_dir, csv_filename)
                shutil.copy(evidence_csv, csv_path)
                
                doc_id = cp.generate_id("doc-")
                with conn.cursor() as cursor:
                    sql = """
                        INSERT INTO documents 
                        (doc_id, project_id, client_id, filename, source_type, file_path, file_size, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(sql, (
                        doc_id, project_id, project.get("client_id"),
                        csv_filename, "fatrag_evidence", csv_path,
                        os.path.getsize(csv_path), "completed"
                    ))
                conn.commit()
                saved_docs.append(csv_filename)

            # 4. Save organogram if available
            org_mmd = os.path.join(os.path.dirname(__file__), "report", "assets", "org.mmd")
            if os.path.isfile(org_mmd):
                mmd_filename = f"FATRAG_Organogram_{timestamp}.mmd"
                mmd_path = os.path.join(uploads_dir, mmd_filename)
                shutil.copy(org_mmd, mmd_path)
                
                doc_id = cp.generate_id("doc-")
                with conn.cursor() as cursor:
                    sql = """
                        INSERT INTO documents 
                        (doc_id, project_id, client_id, filename, source_type, file_path, file_size, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(sql, (
                        doc_id, project_id, project.get("client_id"),
                        mmd_filename, "fatrag_organogram", mmd_path,
                        os.path.getsize(mmd_path), "completed"
                    ))
                conn.commit()
                saved_docs.append(mmd_filename)

        finally:
            conn.close()

        # Mark job completed
        result_summary = {
            "pipeline_stats": pipeline_result.get("stats", {}),
            "saved_documents": saved_docs,
            "timestamp": timestamp
        }
        
        try:
            js.update_job(
                job_id, 
                status="completed", 
                progress=100,
                result_filename=f"FATRAG_Report_{timestamp}.pdf",
                metadata=result_summary
            )
            _job_set_stage(job_id, "completed", progress=100, extra={"result_filename": f"FATRAG_Report_{timestamp}.pdf"})
        except Exception:
            pass

    except subprocess.TimeoutExpired:
        try:
            js.update_job(job_id, status="failed", error_message="Pipeline timeout after 1 hour")
            _job_set_stage(job_id, "failed", progress=0, extra={"error_message": "Pipeline timeout after 1 hour"})
        except Exception:
            pass
        except Exception:
            pass
        preempt_gpu_processes()
        try:
            js.update_job(job_id, status="running", progress=5)
            _job_set_stage(job_id, "running", progress=5)
        except Exception:
            pass

@app.post("/admin/projects/{project_id}/fatrag-pipeline")
async def run_fatrag_pipeline(
    project_id: str, 
    background_tasks: BackgroundTasks,
    body: Optional[Dict[str, Any]] = None,
    _: bool = Depends(admin_required)
):
    """
    Run full FATRAG Auto pipeline (L1 → L2 → Final) on all project documents.
    
    This endpoint:
    1. Retrieves all project documents from vector DB
    2. Runs parallel L1 analysis across 8 GPU workers (qwen2.5:7b)
    3. Synthesizes L2 cross-document evidence
    4. Generates final report with llama3.1:70b (map-reduce)
    5. Creates organogram and charts
    6. Saves all outputs as new project documents
    
    Returns immediately with job_id for tracking progress.
    """
    try:
        # Verify project exists
        project = cp.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Get research question from body or use default
        research_question = ""
        if body and isinstance(body, dict):
            research_question = body.get("research_question", "")
        
        if not research_question:
            research_question = f"Algemene financiële en fiscale synthese van project: {project.get('name', 'Unnamed')}"

        # Create background job
        job_id = cp.generate_id("job-")
        job = js.create_job(
            job_id, 
            "fatrag_pipeline",
            project_id=project_id,
            metadata={
                "source": "admin_ui",
                "research_question": research_question,
                "pipeline": "L1→L2→Final",
                "gpu_workers": 8
            }
        )
        
        if not job:
            raise HTTPException(status_code=500, detail="Failed to create pipeline job")

        # Start background task
        background_tasks.add_task(_background_fatrag_pipeline, job_id, project_id, research_question)

        return {
            "status": "queued",
            "job_id": job_id,
            "message": "FATRAG Auto pipeline started. This will take 10-30 minutes depending on document count.",
            "research_question": research_question,
            "track_progress": f"/admin/jobs/{job_id}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Job status endpoint
@app.get("/admin/jobs")
async def admin_list_jobs(
    status: Optional[str] = None,
    job_type: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 50,
    _: bool = Depends(admin_required)
):
    try:
        items = js.list_jobs(status=status, job_type=job_type, project_id=project_id, limit=limit)
        return {"jobs": items, "count": len(items)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/jobs/{job_id}")
async def admin_get_job(job_id: str, _: bool = Depends(admin_required)):
    try:
        job = js.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, _: bool = Depends(admin_required)):
    """
    Soft-cancel a running/queued job.
    - Marks the job as failed with 'Canceled by user'
    - Frontend will remove it from tracked jobs and refresh UI
    Note: Long-running subprocesses may continue in the background if already spawned.
    """
    try:
        job = js.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        js.update_job(job_id, status="failed", progress=0, error_message="Canceled by user")
        return {"status": "canceled", "job_id": job_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/jobs/cleanup-stuck")
async def cleanup_stuck_jobs(body: Optional[Dict[str, Any]] = None, _: bool = Depends(admin_required)):
    """
    Clean up jobs that are stuck in running/queued status for too long.
    Body: {"max_age_hours": 2} (optional, defaults to 2 hours)
    """
    try:
        from datetime import timedelta
        import pymysql
        
        max_age_hours = 2
        if body and "max_age_hours" in body:
            max_age_hours = int(body["max_age_hours"])
        
        conn = cp.get_db_connection()
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        try:
            # Find stuck jobs
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT job_id, job_type, project_id, status, created_at, updated_at
                    FROM jobs
                    WHERE status IN ('running', 'queued', 'initializing', 'preempting_gpu', 
                                    'running_pipeline', 'running_l1_analysis', 'saving_outputs',
                                    'generating', 'saving', 'analyzing')
                    AND updated_at < %s
                """, (cutoff_time,))
                
                stuck_jobs = cursor.fetchall()
                
                if not stuck_jobs:
                    return {
                        "status": "success",
                        "cleaned": 0,
                        "message": "No stuck jobs found"
                    }
                
                # Mark as failed
                cursor.execute("""
                    UPDATE jobs
                    SET status = 'failed',
                        error_message = 'Job timed out / stuck (cleaned up automatically)',
                        progress = 0,
                        updated_at = NOW()
                    WHERE status IN ('running', 'queued', 'initializing', 'preempting_gpu',
                                    'running_pipeline', 'running_l1_analysis', 'saving_outputs',
                                    'generating', 'saving', 'analyzing')
                    AND updated_at < %s
                """, (cutoff_time,))
                
                conn.commit()
                
                return {
                    "status": "success",
                    "cleaned": len(stuck_jobs),
                    "max_age_hours": max_age_hours,
                    "jobs": [
                        {
                            "job_id": j["job_id"],
                            "type": j["job_type"],
                            "project_id": j["project_id"],
                            "status": j["status"]
                        }
                        for j in stuck_jobs
                    ]
                }
        finally:
            conn.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== FEATURE 2: FINANCIËLE DASHBOARD ==========

def extract_dashboard_metrics(project_id: str) -> Dict[str, Any]:
    """
    Extract key financial metrics from project documents and analyses.
    Returns dashboard-ready data with amounts, entities, timeline, risks.
    """
    import re
    from collections import defaultdict
    
    project = cp.get_project_with_documents(project_id)
    if not project:
        return {"error": "Project not found"}
    
    uploads_dir = os.path.join(os.path.dirname(__file__), "fatrag_data", "uploads")
    documents = project.get("documents", [])
    
    # Initialize metrics
    metrics = {
        "summary": {
            "total_documents": len(documents),
            "total_analyses": 0,
            "entities_count": 0,
            "relations_count": 0,
        },
        "financial": {
            "amounts": [],
            "total_value": 0,
            "currency_breakdown": defaultdict(int),
            "largest_transactions": [],
        },
        "entities": {
            "companies": [],
            "persons": [],
            "holdings": [],
        },
        "timeline": [],
        "risks": [],
        "analysis_status": {
            "completed": 0,
            "in_progress": 0,
            "failed": 0,
        }
    }
    
    # Parse evidence CSV if exists
    evidence_files = [d for d in documents if "evidence" in d.get("filename", "").lower() and d.get("filename", "").endswith(".csv")]
    for doc in evidence_files:
        try:
            filepath = os.path.join(uploads_dir, doc.get("filename", ""))
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    import csv
                    reader = csv.DictReader(f)
                    for row in reader:
                        doc_type = row.get("type", "")
                        value = row.get("value", "")
                        
                        if doc_type == "amount":
                            # Extract EUR amounts
                            amount_match = re.search(r'[€EUR]\s*([0-9.,]+)', value)
                            if amount_match:
                                amount_str = amount_match.group(1).replace('.', '').replace(',', '.')
                                try:
                                    amount = float(amount_str)
                                    metrics["financial"]["amounts"].append({
                                        "value": amount,
                                        "formatted": value,
                                        "source": row.get("document", "")
                                    })
                                    metrics["financial"]["total_value"] += amount
                                    metrics["financial"]["currency_breakdown"]["EUR"] += 1
                                except:
                                    pass
                        
                        elif doc_type == "entity":
                            entity_lower = value.lower()
                            if "b.v." in entity_lower or "bv" in entity_lower:
                                metrics["entities"]["companies"].append(value)
                            elif "holding" in entity_lower:
                                metrics["entities"]["holdings"].append(value)
                            else:
                                metrics["entities"]["persons"].append(value)
                        
                        elif doc_type == "date":
                            metrics["timeline"].append(value)
        except Exception:
            pass
    
    # Sort largest transactions
    if metrics["financial"]["amounts"]:
        sorted_amounts = sorted(metrics["financial"]["amounts"], key=lambda x: x["value"], reverse=True)
        metrics["financial"]["largest_transactions"] = sorted_amounts[:10]
    
    # Count unique entities
    metrics["summary"]["entities_count"] = (
        len(set(metrics["entities"]["companies"])) +
        len(set(metrics["entities"]["persons"])) +
        len(set(metrics["entities"]["holdings"]))
    )
    
    # Parse analysis documents for risks
    analysis_docs = [d for d in documents if "analyse" in d.get("filename", "").lower()]
    metrics["summary"]["total_analyses"] = len(analysis_docs)
    
    for doc in analysis_docs:
        try:
            filepath = os.path.join(uploads_dir, doc.get("filename", ""))
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Extract risks (look for risk/risico sections)
                    risk_section = re.search(r'(?:risico|risk)(?:\'s|s)?\s*[:\-]?\s*(.{0,500})', content, re.IGNORECASE)
                    if risk_section:
                        risk_text = risk_section.group(1)
                        # Split by bullets or newlines
                        risk_items = [r.strip() for r in re.split(r'[\n•\-]', risk_text) if r.strip() and len(r.strip()) > 20]
                        metrics["risks"].extend(risk_items[:5])
        except Exception:
            pass
    
    # Deduplicate entities
    metrics["entities"]["companies"] = list(set(metrics["entities"]["companies"]))[:20]
    metrics["entities"]["persons"] = list(set(metrics["entities"]["persons"]))[:20]
    metrics["entities"]["holdings"] = list(set(metrics["entities"]["holdings"]))[:20]
    metrics["timeline"] = list(set(metrics["timeline"]))[:20]
    metrics["risks"] = list(set(metrics["risks"]))[:5]
    
    return metrics

@app.get("/admin/projects/{project_id}/dashboard")
async def get_project_dashboard(project_id: str, _: bool = Depends(admin_required)):
    """
    Get dashboard metrics for a project.
    Returns financial KPIs, entities, timeline, and risks.
    """
    try:
        project = cp.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        metrics = extract_dashboard_metrics(project_id)
        
        return {
            "project_id": project_id,
            "project_name": project.get("name"),
            "metrics": metrics,
            "generated_at": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== FEATURE 3: SMART TEMPLATES ==========

# Pre-defined report templates
REPORT_TEMPLATES = {
    "holding_analysis": {
        "name": "Holding Structuur Analyse",
        "description": "Analyse van een holding structuur met aandachtspunten voor VPB, IB en optimalisatie",
        "sections": [
            {
                "title": "Executive Summary",
                "prompt": "Geef een beknopte samenvatting (max 250 woorden) van de holding structuur. Focus op: aantal entiteiten, totale waarde, belangrijkste aandachtspunten, urgente acties."
            },
            {
                "title": "Structuuroverzicht",
                "prompt": "Beschrijf de juridische structuur: welke holdings, welke BV's, aandeelhoudersstructuur, certificering indien van toepassing. Gebruik bullets."
            },
            {
                "title": "Fiscale Positie",
                "prompt": "Analyseer de fiscale positie: VPB-plicht per entiteit, deelnemingsvrijstelling, fiscale eenheid mogelijk/aanwezig, carry-back/forward verliezen, buitenlandse structuren. Geef concrete bedragen waar bekend."
            },
            {
                "title": "Waardering & Vermogen",
                "prompt": "Geef overzicht van waarderingen: boekwaarden, taxaties, goodwill, stille reserves. Bereken totale waarde van de structuur indien mogelijk."
            },
            {
                "title": "Risico's & Compliance",
                "prompt": "Identificeer fiscale en juridische risico's: transfer pricing, substance-eisen, informele kapitaalstortingen, aandelenleningconstructies, UBO-registratie, jaarrekeningenplicht. Geef per risico een rating (laag/medium/hoog)."
            },
            {
                "title": "Optimalisatiemogelijkheden",
                "prompt": "Geef concrete optimalisatiemogelijkheden: herstructurering, interne financiering, toekomstige schenkingen/bedrijfsopvolging, belastinglatentie. Schat baten per optie."
            },
            {
                "title": "Actieplan",
                "prompt": "Formuleer een concreet actieplan met prioriteiten, deadlines, betrokken partijen (notaris, accountant, adviseur), en geschatte kosten."
            }
        ]
    },
    "estate_planning": {
        "name": "Erfbelasting & Schenking Analyse",
        "description": "Analyse van vermogensoverdracht met focus op erfbelasting, vrijstellingen en optimalisatie",
        "sections": [
            {
                "title": "Executive Summary",
                "prompt": "Vat samen: wie zijn erflater en erfgenamen, totaal vermogen, verwachte erfbelasting, belangrijkste aanbevelingen."
            },
            {
                "title": "Vermogensoverzicht",
                "prompt": "Geef overzicht van alle vermogensbestanddelen: onroerend goed, BV-aandelen, banktegoeden, overige bezittingen. Markeer wat box 1/2/3 vermogen is. Noem waarderingen en waarderingsgrondslagen."
            },
            {
                "title": "Erfbelastingberekening",
                "prompt": "Bereken verwachte erfbelasting per erfgenaam (kinderen, partner, overige). Gebruik actuele tarieven 2025. Toon berekening stapsgewijs met vrijstellingen."
            },
            {
                "title": "Schenkingsmogelijkheden",
                "prompt": "Analyseer schenkingsmogelijkheden: jaarlijkse vrijstelling (€6.633 / €29.484 eenmalig voor eigen woning), BOR schenking, bedrijfsopvolgingsregeling. Bereken impact op erfbelasting."
            },
            {
                "title": "Optimalisatiestrategieën",
                "prompt": "Geef strategieën voor erfbelastingbesparing: tijdig schenken, testament aanpassing, splitsing eigendom/vruchtgebruik, trust/certificering, bedrijfsopvolging binnen BOR/BOSA. Reken impact door per strategie."
            },
            {
                "title": "Juridische Aspecten",
                "prompt": "Behandel: testament aanwezig?, legitieme portie, langstlevende regeling, executeurstestament, huwelijkse voorwaarden, samenlevingscontract. Geef actiepunten."
            },
            {
                "title": "Tijdlijn & Actie",
                "prompt": "Maak een tijdlijn met mijlpalen: notarisafspraken, schenkingen plannen, eventuele herstructurering. Wees concreet in data en deadlines."
            }
        ]
    },
    "business_valuation": {
        "name": "Bedrijfswaardering Rapport",
        "description": "Waardering van een onderneming/BV voor (ver)koop, inbreng of fiscale doeleinden",
        "sections": [
            {
                "title": "Executive Summary",
                "prompt": "Vat samen: wat wordt gewaardeerd, doel van waardering, conclusie (range), belangrijkste value drivers."
            },
            {
                "title": "Bedrijfsbeschrijving",
                "prompt": "Beschrijf de onderneming: activiteiten, marktpositie, geschiedenis, organisatie, personeel, USP's. Gebruik feitelijke data uit documenten."
            },
            {
                "title": "Financiële Analyse",
                "prompt": "Analyseer financiële performance laatste 3-5 jaar: omzet, EBITDA, nettowinst, kasstroom, balans (EV, werkkapitaal). Bereken ratios (marges, ROA, ROE, solvabiliteit). Toon trends."
            },
            {
                "title": "Waarderingsmethoden",
                "prompt": "Pas toe: DCF (3 scenario's: laag/basis/hoog), multiples (omzet/EBITDA vergelijking met peers), intrinsieke waarde. Leg keuzes uit. Geef range per methode."
            },
            {
                "title": "Normalisaties & Correcties",
                "prompt": "Identificeer en kwantificeer normalisaties: eenmalige kosten, directeursalaris marktconform, privégebruik activa, achterstallig onderhoud, goodwill. Toon impact op waardering."
            },
            {
                "title": "Value Drivers & Risico's",
                "prompt": "Identificeer value drivers (klantrelaties, IP, contracten, personeel, marktpositie) en risico's (klantconcentratie, personeelsafhankelijkheid, marktdynamiek). Gewogen scoring."
            },
            {
                "title": "Conclusie & Bandbredte",
                "prompt": "Concludeer met definitieve waardebepaling (bereik) met onderbouwing. Geef aanbevelingen voor onderhandelingen. Noem fiscale implicaties (overdrachtsbelasting, vennootschapsbelasting, inkomstenbelasting)."
            }
        ]
    },
    "tax_optimization": {
        "name": "Fiscale Optimalisatie Scan",
        "description": "Identificatie van fiscale optimalisatiemogelijkheden voor ondernemers",
        "sections": [
            {
                "title": "Executive Summary",
                "prompt": "Vat samen: huidige fiscale situatie, belangrijkste kansen, verwachte besparing, implementatie-inspanning."
            },
            {
                "title": "Box 1 Optimalisatie",
                "prompt": "Analyseer IB box 1: MKB-winstvrijstelling, zelfstandigenaftrek, startersaftrek, investeringsaftrek (KIA/MIA/EIA), oudedagsreserve. Bereken optimaal af te trekken bedrag per jaar."
            },
            {
                "title": "Box 2 & Dividendplanning",
                "prompt": "Behandel box 2 aanmerkelijk belang: dividenduitkering vs salary, lijfrente-premie, stamrechtverplichting, doorschuiffaciliteit. Reken impact per optie (netto na belasting)."
            },
            {
                "title": "VPB & Deelnemingen",
                "prompt": "Analyseer VPB: deelnemingsvrijstelling toepasbaar?, fiscale eenheid zinvol?, innovatiebox mogelijk?, liquidatiereserve/stakingsverlies. Kwantificeer voordeel per maatregel."
            },
            {
                "title": "Auto & Privégebruik",
                "prompt": "Optimaliseer auto van de zaak: bijtelling, elektrisch rijden, lease vs koop, kilometers, privégebruik anders dan auto. Bereken fiscaal voordeligste optie."
            },
            {
                "title": "Pensioen & Lijfrente",
                "prompt": "Adviseer pensioenopbouw: lijfrentepremie eigen beheer vs extern, banksparen, oudedagsreserve omzetten naar lijfrente, FOR (fiscale oudedagsreserve) maximering. Bereken optimale jaarlijkse premie."
            },
            {
                "title": "Implementatieplan",
                "prompt": "Maak een concreet implementatieplan per optimalisatie: wat, wanneer, wie doet wat (adviseur/notaris/accountant), kosten, baten, risico's. Prioriteer op ROI."
            }
        ]
    }
}

async def generate_template_report(project_id: str, template_key: str) -> str:
    """
    Generate a report for a project using a specific template.
    Uses LLM to fill in each section based on project documents.
    """
    if template_key not in REPORT_TEMPLATES:
        raise ValueError(f"Template '{template_key}' not found")
    
    template = REPORT_TEMPLATES[template_key]
    project = cp.get_project_with_documents(project_id)
    if not project:
        raise ValueError("Project not found")
    
    # Gather project context
    uploads_dir = os.path.join(os.path.dirname(__file__), "fatrag_data", "uploads")
    documents = project.get("documents", [])
    
    # Extract text from all documents for context
    doc_texts = []
    for doc in documents:
        filename = doc.get("filename", "")
        if not filename or "analyse" in filename.lower() or "fatrag" in filename.lower():
            continue  # Skip analysis outputs to avoid circular references
        
        file_path = os.path.join(uploads_dir, filename)
        if not os.path.isfile(file_path):
            continue
        
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pdf":
            text = ing.read_pdf_file(file_path)
        elif ext in [".txt", ".md"]:
            text = ing.read_text_file(file_path)
        else:
            continue
        
        if text and not text.startswith("[PDF extraction error"):
            doc_texts.append(f"Document: {filename}\n{text[:5000]}")  # Cap per doc
    
    combined_context = "\n\n---\n\n".join(doc_texts[:10])  # Max 10 docs
    
    # Generate each section using LLM
    cfg = getattr(app.state, "config", {}) or {}
    llm = build_llm_from_config(cfg)
    
    report_sections = []
    report_sections.append(f"# {template['name']}\n")
    report_sections.append(f"**Project:** {project.get('name', 'Unnamed')}\n")
    report_sections.append(f"**Gegenereerd:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_sections.append(f"**Template:** {template['description']}\n")
    report_sections.append("\n---\n\n")
    
    for section in template["sections"]:
        section_title = section["title"]
        section_prompt = section["prompt"]
        
        full_prompt = f"""
Je bent FinAdviseur-NL, senior financieel specialist.

OPDRACHT: Schrijf de sectie "{section_title}" voor het project "{project.get('name', '')}"

INSTRUCTIE:
{section_prompt}

REGELS:
- Taal: Nederlands
- Gebruik concrete data uit onderstaande documenten
- Wees specifiek met bedragen (EUR/€), percentages, datums
- Als data ontbreekt: schrijf "onvoldoende data" en benoem wat nodig is
- Gebruik bullets waar mogelijk
- Geen hallucinaties: alleen feiten uit documenten

PROJECTCONTEXT:
{combined_context[:8000]}

Schrijf ALLEEN de inhoud voor sectie "{section_title}" (geen intro, geen meta-commentaar):
"""
        
        try:
            res = await llm.ainvoke(full_prompt)
            content = getattr(res, "content", None) or str(res)
            content = content.strip()
            
            report_sections.append(f"## {section_title}\n\n")
            report_sections.append(content)
            report_sections.append("\n\n")
        except Exception as e:
            report_sections.append(f"## {section_title}\n\n")
            report_sections.append(f"**Error:** Kon sectie niet genereren ({str(e)})\n\n")
    
    return "".join(report_sections)

@app.get("/admin/templates")
async def list_report_templates(_: bool = Depends(admin_required)):
    """
    List all available report templates.
    """
    templates_list = []
    for key, template in REPORT_TEMPLATES.items():
        templates_list.append({
            "key": key,
            "name": template["name"],
            "description": template["description"],
            "sections_count": len(template["sections"])
        })
    return {"templates": templates_list}

@app.get("/admin/templates/{template_key}")
async def get_report_template(template_key: str, _: bool = Depends(admin_required)):
    """
    Get details of a specific template.
    """
    if template_key not in REPORT_TEMPLATES:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return {"template": REPORT_TEMPLATES[template_key]}

@app.post("/admin/projects/{project_id}/generate-from-template")
async def generate_project_report_from_template(
    project_id: str,
    body: Dict[str, Any],
    background_tasks: BackgroundTasks,
    _: bool = Depends(admin_required)
):
    """
    Generate a report for a project using a template.
    Runs in background and saves the result as a new document.
    """
    template_key = body.get("template_key")
    if not template_key:
        raise HTTPException(status_code=400, detail="Field 'template_key' is required")
    
    if template_key not in REPORT_TEMPLATES:
        raise HTTPException(status_code=404, detail="Template not found")
    
    try:
        # Verify project exists
        project = cp.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Create background job
        job_id = cp.generate_id("job-")
        template_name = REPORT_TEMPLATES[template_key]["name"]
        job = js.create_job(
            job_id,
            "template_report",
            project_id=project_id,
            metadata={
                "source": "admin_ui",
                "template_key": template_key,
                "template_name": template_name
            }
        )
        
        if not job:
            raise HTTPException(status_code=500, detail="Failed to create template job")
        
        # Background task to generate report
        async def _background_template_generation():
            try:
                js.update_job(job_id, status="generating", progress=10)
                
                # Generate report
                report_content = await generate_template_report(project_id, template_key)
                
                js.update_job(job_id, status="saving", progress=90)
                
                # Save as new document
                uploads_dir = os.path.join(os.path.dirname(__file__), "fatrag_data", "uploads")
                os.makedirs(uploads_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{template_name.replace(' ', '_')}_{timestamp}.md"
                filepath = os.path.join(uploads_dir, filename)
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(report_content)
                
                # Save to database
                doc_id = cp.generate_id("doc-")
                file_size = os.path.getsize(filepath)
                
                import pymysql
                conn = cp.get_db_connection()
                try:
                    with conn.cursor() as cursor:
                        sql = """
                            INSERT INTO documents 
                            (doc_id, project_id, client_id, filename, source_type, file_path, file_size, status)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        cursor.execute(sql, (
                            doc_id,
                            project_id,
                            project.get("client_id"),
                            filename,
                            "template_report",
                            filepath,
                            file_size,
                            "completed"
                        ))
                    conn.commit()
                finally:
                    conn.close()
                
                js.update_job(job_id, status="completed", progress=100, result_filename=filename)
            except Exception as e:
                js.update_job(job_id, status="failed", error_message=str(e))
        
        background_tasks.add_task(_background_template_generation)
        
        return {
            "status": "queued",
            "job_id": job_id,
            "template_name": template_name,
            "message": f"Generating {template_name} report. This will take 2-5 minutes.",
            "track_progress": f"/admin/jobs/{job_id}"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/docs/{filename}/analyze")
async def analyze_document(filename: str, _: bool = Depends(admin_required)):
    try:
        uploads_dir = os.path.join(os.path.dirname(__file__), "fatrag_data", "uploads")
        file_path = os.path.join(uploads_dir, filename)
        
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        ext = os.path.splitext(filename)[1].lower()
        
        import analysis
        if ext == ".pdf":
            result = analysis.analyze_pdf(file_path)
        elif ext in [".xlsx", ".xls"]:
            result = analysis.analyze_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type for analysis")
        
        return {"filename": filename, "analysis": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/docs/{filename}/download")
async def download_document(filename: str, _: bool = Depends(admin_required)):
    try:
        uploads_dir = os.path.join(os.path.dirname(__file__), "fatrag_data", "uploads")
        file_path = os.path.join(uploads_dir, filename)
        
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FastAPIFileResponse(file_path, filename=filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== UPLOAD PROGRESS TRACKING ENDPOINTS ==========

import upload_progress_store as ups
import ingestion_with_progress as ing_progress

@app.get("/admin/uploads/progress/{upload_id}")
async def get_upload_progress(upload_id: str, _: bool = Depends(admin_required)):
    """Get real-time progress for a file upload"""
    try:
        tracker = ups.UploadProgressTracker()
        upload = tracker.get_upload(upload_id)
        if not upload:
            raise HTTPException(status_code=404, detail="Upload not found")
        return {"upload": upload}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/uploads/batch/{batch_id}")
async def get_batch_progress(batch_id: str, _: bool = Depends(admin_required)):
    """Get progress for a batch upload"""
    try:
        tracker = ups.UploadProgressTracker()
        batch = tracker.get_batch(batch_id)
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        # Get all uploads in this batch
        uploads = tracker.get_batch_uploads(batch_id)
        
        return {
            "batch": batch,
            "uploads": uploads,
            "total": len(uploads)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/projects/{project_id}/uploads")
async def list_project_uploads(
    project_id: str,
    status: Optional[str] = None,
    limit: int = 50,
    _: bool = Depends(admin_required)
):
    """List all uploads for a project"""
    try:
        tracker = ups.UploadProgressTracker()
        uploads = tracker.list_uploads(project_id=project_id, status=status, limit=limit)
        return {"uploads": uploads, "count": len(uploads)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/projects/{project_id}/upload-with-progress")
async def upload_project_documents_with_progress(
    project_id: str,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    _: bool = Depends(admin_required)
):
    """
    Upload documents with live progress tracking.
    Returns batch_id for tracking progress via WebSocket or polling.
    """
    try:
        # Verify project exists
        project = cp.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        client_id = project.get("client_id")
        
        # Generate batch ID
        batch_id = ups.generate_id("batch-")
        
        # Save files to disk first
        ing.ensure_dirs()
        uploads_dir = os.path.join(os.path.dirname(__file__), "fatrag_data", "uploads")
        file_paths = []
        
        for file in files:
            fname = (file.filename or "").strip()
            if not fname:
                continue
            
            file_path = os.path.join(uploads_dir, fname)
            content = await file.read()
            if not content:
                continue
            
            with open(file_path, "wb") as f:
                f.write(content)
            
            file_paths.append(file_path)
        
        if not file_paths:
            raise HTTPException(status_code=400, detail="No valid files to process")
        
        # Start background ingestion with progress tracking
        async def _background_ingest_with_progress():
            try:
                result = ing_progress.ingest_files_batch_with_progress(
                    vectorstore=vectorstore,
                    file_paths=file_paths,
                    batch_id=batch_id,
                    project_id=project_id,
                    client_id=client_id,
                    user="admin",
                    kind="project_upload",
                    extra_metadata={"project_id": project_id, "client_id": client_id},
                )
                
                # Also save to documents table
                import pymysql
                conn = cp.get_db_connection()
                try:
                    for res in result.get("results", []):
                        if res.get("status") == "completed":
                            doc_id = cp.generate_id("doc-")
                            filename = res.get("filename")
                            file_path = os.path.join(uploads_dir, filename)
                            
                            if os.path.isfile(file_path):
                                with conn.cursor() as cursor:
                                    sql = """
                                        INSERT INTO documents 
                                        (doc_id, project_id, client_id, filename, source_type, 
                                         file_path, file_size, status)
                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                    """
                                    cursor.execute(sql, (
                                        doc_id, project_id, client_id,
                                        filename, "project_upload",
                                        file_path, os.path.getsize(file_path),
                                        "indexed"
                                    ))
                    conn.commit()
                finally:
                    conn.close()
            except Exception as e:
                print(f"Background ingestion error: {e}")
        
        background_tasks.add_task(_background_ingest_with_progress)
        
        return {
            "status": "queued",
            "batch_id": batch_id,
            "total_files": len(file_paths),
            "message": "Upload queued. Track progress via /admin/uploads/batch/{batch_id}",
            "track_url": f"/admin/uploads/batch/{batch_id}"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== ORGANOGRAM ENDPOINTS (FEATURE 4: INTERACTIVE EDITOR) ==========

import organogram_service as org_svc

@app.post("/admin/organograms")
async def create_organogram(body: Dict[str, Any], _: bool = Depends(admin_required)):
    try:
        organogram = org_svc.create_organogram(
            project_id=body.get("project_id"),
            name=body.get("name"),
            structure_data=body.get("structure_data"),
            notes=body.get("notes"),
        )
        return {"status": "created", "organogram": organogram}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Feature 4: Interactive organogram manipulation endpoints
@app.post("/admin/organograms/{organogram_id}/nodes")
async def add_organogram_node(organogram_id: str, body: Dict[str, Any], _: bool = Depends(admin_required)):
    """
    Add a node to an organogram.
    Body: {"id": "node1", "label": "Company A", "type": "bv", "metadata": {...}}
    """
    try:
        organogram = org_svc.get_organogram(organogram_id)
        if not organogram:
            raise HTTPException(status_code=404, detail="Organogram not found")
        
        structure = organogram.get("structure_data", {}) or {}
        nodes = structure.get("nodes", [])
        
        # Add new node
        node = {
            "id": body.get("id") or f"node_{len(nodes)+1}",
            "label": body.get("label", "New Node"),
            "type": body.get("type", "company"),
            "metadata": body.get("metadata", {})
        }
        nodes.append(node)
        structure["nodes"] = nodes
        
        # Update organogram
        updated = org_svc.update_organogram(
            organogram_id=organogram_id,
            structure_data=structure,
            increment_version=True
        )
        
        return {"status": "success", "node": node, "organogram": updated}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/admin/organograms/{organogram_id}/nodes/{node_id}")
async def update_organogram_node(organogram_id: str, node_id: str, body: Dict[str, Any], _: bool = Depends(admin_required)):
    """
    Update a node in an organogram.
    Body: {"label": "Company A Updated", "type": "holding", "metadata": {...}}
    """
    try:
        organogram = org_svc.get_organogram(organogram_id)
        if not organogram:
            raise HTTPException(status_code=404, detail="Organogram not found")
        
        structure = organogram.get("structure_data", {}) or {}
        nodes = structure.get("nodes", [])
        
        # Find and update node
        found = False
        for node in nodes:
            if node.get("id") == node_id:
                if "label" in body:
                    node["label"] = body["label"]
                if "type" in body:
                    node["type"] = body["type"]
                if "metadata" in body:
                    node["metadata"] = body["metadata"]
                found = True
                break
        
        if not found:
            raise HTTPException(status_code=404, detail="Node not found")
        
        structure["nodes"] = nodes
        
        # Update organogram
        updated = org_svc.update_organogram(
            organogram_id=organogram_id,
            structure_data=structure,
            increment_version=True
        )
        
        return {"status": "success", "organogram": updated}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/admin/organograms/{organogram_id}/nodes/{node_id}")
async def delete_organogram_node(organogram_id: str, node_id: str, _: bool = Depends(admin_required)):
    """
    Delete a node from an organogram (and all edges connected to it).
    """
    try:
        organogram = org_svc.get_organogram(organogram_id)
        if not organogram:
            raise HTTPException(status_code=404, detail="Organogram not found")
        
        structure = organogram.get("structure_data", {}) or {}
        nodes = structure.get("nodes", [])
        edges = structure.get("edges", [])
        
        # Remove node
        nodes = [n for n in nodes if n.get("id") != node_id]
        
        # Remove edges connected to this node
        edges = [e for e in edges if e.get("from") != node_id and e.get("to") != node_id]
        
        structure["nodes"] = nodes
        structure["edges"] = edges
        
        # Update organogram
        updated = org_svc.update_organogram(
            organogram_id=organogram_id,
            structure_data=structure,
            increment_version=True
        )
        
        return {"status": "success", "organogram": updated}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/organograms/{organogram_id}/edges")
async def add_organogram_edge(organogram_id: str, body: Dict[str, Any], _: bool = Depends(admin_required)):
    """
    Add an edge between nodes in an organogram.
    Body: {"from": "node1", "to": "node2", "label": "100%", "type": "ownership"}
    """
    try:
        organogram = org_svc.get_organogram(organogram_id)
        if not organogram:
            raise HTTPException(status_code=404, detail="Organogram not found")
        
        structure = organogram.get("structure_data", {}) or {}
        edges = structure.get("edges", [])
        nodes = structure.get("nodes", [])
        
        # Validate that nodes exist
        node_ids = {n.get("id") for n in nodes}
        from_id = body.get("from")
        to_id = body.get("to")
        
        if from_id not in node_ids or to_id not in node_ids:
            raise HTTPException(status_code=400, detail="Source or target node not found")
        
        # Add edge
        edge = {
            "from": from_id,
            "to": to_id,
            "label": body.get("label", ""),
            "type": body.get("type", "relation")
        }
        edges.append(edge)
        structure["edges"] = edges
        
        # Update organogram
        updated = org_svc.update_organogram(
            organogram_id=organogram_id,
            structure_data=structure,
            increment_version=True
        )
        
        return {"status": "success", "edge": edge, "organogram": updated}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/admin/organograms/{organogram_id}/edges")
async def delete_organogram_edge(organogram_id: str, body: Dict[str, Any], _: bool = Depends(admin_required)):
    """
    Delete an edge from an organogram.
    Body: {"from": "node1", "to": "node2"}
    """
    try:
        organogram = org_svc.get_organogram(organogram_id)
        if not organogram:
            raise HTTPException(status_code=404, detail="Organogram not found")
        
        structure = organogram.get("structure_data", {}) or {}
        edges = structure.get("edges", [])
        
        from_id = body.get("from")
        to_id = body.get("to")
        
        # Remove edge
        edges = [e for e in edges if not (e.get("from") == from_id and e.get("to") == to_id)]
        
        structure["edges"] = edges
        
        # Update organogram
        updated = org_svc.update_organogram(
            organogram_id=organogram_id,
            structure_data=structure,
            increment_version=True
        )
        
        return {"status": "success", "organogram": updated}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/organograms/{organogram_id}/export/mermaid")
async def export_organogram_to_mermaid(organogram_id: str, _: bool = Depends(admin_required)):
    """
    Export organogram to Mermaid diagram syntax.
    """
    try:
        organogram = org_svc.get_organogram(organogram_id)
        if not organogram:
            raise HTTPException(status_code=404, detail="Organogram not found")
        
        structure = organogram.get("structure_data", {}) or {}
        nodes = structure.get("nodes", [])
        edges = structure.get("edges", [])
        
        # Generate Mermaid syntax
        mermaid_lines = ["graph TD"]
        
        # Add nodes
        for node in nodes:
            node_id = node.get("id", "")
            label = node.get("label", "")
            node_type = node.get("type", "")
            
            # Choose shape based on type
            if node_type == "person":
                shape = f'{node_id}(("{label}"))'
            elif node_type == "holding":
                shape = f'{node_id}["{label}"]'
            else:
                shape = f'{node_id}["{label}"]'
            
            mermaid_lines.append(f"  {shape}")
        
        # Add edges
        for edge in edges:
            from_id = edge.get("from", "")
            to_id = edge.get("to", "")
            label = edge.get("label", "")
            
            if label:
                mermaid_lines.append(f"  {from_id} -->|{label}| {to_id}")
            else:
                mermaid_lines.append(f"  {from_id} --> {to_id}")
        
        mermaid_code = "\n".join(mermaid_lines)
        
        return {
            "mermaid": mermaid_code,
            "organogram_name": organogram.get("name", "")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/organograms/{organogram_id}/validate")
async def validate_organogram(organogram_id: str, _: bool = Depends(admin_required)):
    """
    Validate organogram structure (cycles, disconnected nodes, etc.).
    """
    try:
        organogram = org_svc.get_organogram(organogram_id)
        if not organogram:
            raise HTTPException(status_code=404, detail="Organogram not found")
        
        structure = organogram.get("structure_data", {}) or {}
        nodes = structure.get("nodes", [])
        edges = structure.get("edges", [])
        
        warnings = []
        errors = []
        
        # Check for duplicate node IDs
        node_ids = [n.get("id") for n in nodes]
        if len(node_ids) != len(set(node_ids)):
            errors.append("Duplicate node IDs found")
        
        # Check for edges referencing non-existent nodes
        node_id_set = set(node_ids)
        for edge in edges:
            if edge.get("from") not in node_id_set:
                errors.append(f"Edge references non-existent source node: {edge.get('from')}")
            if edge.get("to") not in node_id_set:
                errors.append(f"Edge references non-existent target node: {edge.get('to')}")
        
        # Check for cycles (basic DFS-based detection)
        def has_cycle():
            visited = set()
            rec_stack = set()
            
            def dfs(node_id):
                visited.add(node_id)
                rec_stack.add(node_id)
                
                # Find all edges from this node
                for edge in edges:
                    if edge.get("from") == node_id:
                        neighbor = edge.get("to")
                        if neighbor not in visited:
                            if dfs(neighbor):
                                return True
                        elif neighbor in rec_stack:
                            return True
                
                rec_stack.remove(node_id)
                return False
            
            for node_id in node_id_set:
                if node_id not in visited:
                    if dfs(node_id):
                        return True
            return False
        
        if has_cycle():
            warnings.append("Cycle detected in organogram (may be intentional for cross-holdings)")
        
        # Check for disconnected nodes
        connected_nodes = set()
        for edge in edges:
            connected_nodes.add(edge.get("from"))
            connected_nodes.add(edge.get("to"))
        
        disconnected = node_id_set - connected_nodes
        if disconnected:
            warnings.append(f"{len(disconnected)} disconnected node(s): {list(disconnected)[:5]}")
        
        is_valid = len(errors) == 0
        
        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "node_count": len(nodes),
            "edge_count": len(edges)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/organograms/{organogram_id}")
async def get_organogram(organogram_id: str, _: bool = Depends(admin_required)):
    try:
        organogram = org_svc.get_organogram(organogram_id)
        if not organogram:
            raise HTTPException(status_code=404, detail="Organogram not found")
        return {"organogram": organogram}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/admin/organograms/{organogram_id}")
async def update_organogram(organogram_id: str, body: Dict[str, Any], _: bool = Depends(admin_required)):
    try:
        organogram = org_svc.update_organogram(
            organogram_id=organogram_id,
            name=body.get("name"),
            structure_data=body.get("structure_data"),
            notes=body.get("notes"),
            increment_version=body.get("increment_version", False),
        )
        if not organogram:
            raise HTTPException(status_code=404, detail="Organogram not found")
        return {"status": "updated", "organogram": organogram}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/admin/organograms/{organogram_id}")
async def delete_organogram(organogram_id: str, _: bool = Depends(admin_required)):
    try:
        success = org_svc.delete_organogram(organogram_id)
        if not success:
            raise HTTPException(status_code=404, detail="Organogram not found")
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/projects/{project_id}/organograms")
async def list_project_organograms(project_id: str, _: bool = Depends(admin_required)):
    try:
        organograms = org_svc.list_organograms(project_id=project_id)
        return {"organograms": organograms}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/projects/{project_id}/organogram/generate")
async def auto_generate_organogram(project_id: str, body: Optional[Dict[str, Any]] = None, _: bool = Depends(admin_required)):
    try:
        # Get project documents
        project = cp.get_project_with_documents(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Extract text from all documents
        uploads_dir = os.path.join(os.path.dirname(__file__), "fatrag_data", "uploads")
        document_texts = []
        
        for doc in project.get("documents", []):
            filename = doc.get("filename", "")
            if filename:
                file_path = os.path.join(uploads_dir, filename)
                if os.path.isfile(file_path):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext == ".pdf":
                        document_texts.append(ing.read_pdf_file(file_path))
                    elif ext in [".txt", ".md"]:
                        document_texts.append(ing.read_text_file(file_path))
        
        # Generate organogram
        name = (body or {}).get("name", "Auto-generated Organogram")
        organogram = org_svc.auto_generate_organogram(project_id, document_texts, name)
        
        return {"status": "generated", "organogram": organogram}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/organogram/templates/{template_name}")
async def get_organogram_template(template_name: str, _: bool = Depends(admin_required)):
    try:
        if template_name == "empty":
            template = org_svc.create_empty_organogram_template()
        elif template_name == "holding":
            template = org_svc.create_simple_holding_template()
        else:
            raise HTTPException(status_code=404, detail="Template not found")
        return {"template": template}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== FEATURE 5: DOCUMENT COMPARISON TOOL ==========

@app.post("/admin/docs/compare")
async def compare_documents(body: Dict[str, Any], _: bool = Depends(admin_required)):
    """
    Compare two documents side-by-side with diff highlighting.
    Body: {"filename1": "doc1.txt", "filename2": "doc2.txt"}
    """
    try:
        filename1 = body.get("filename1")
        filename2 = body.get("filename2")
        
        if not filename1 or not filename2:
            raise HTTPException(status_code=400, detail="Both filename1 and filename2 required")
        
        uploads_dir = os.path.join(os.path.dirname(__file__), "fatrag_data", "uploads")
        file1_path = os.path.join(uploads_dir, filename1)
        file2_path = os.path.join(uploads_dir, filename2)
        
        if not os.path.isfile(file1_path):
            raise HTTPException(status_code=404, detail=f"File not found: {filename1}")
        if not os.path.isfile(file2_path):
            raise HTTPException(status_code=404, detail=f"File not found: {filename2}")
        
        # Read file contents
        with open(file1_path, 'r', encoding='utf-8') as f:
            content1 = f.read()
        with open(file2_path, 'r', encoding='utf-8') as f:
            content2 = f.read()
        
        # Compute diff
        diff_blocks = compute_diff(content1, content2)
        similarity = get_similarity_score(content1, content2)
        
        # Track versions
        version_tracker.add_version(filename1, content1, datetime.now().isoformat())
        version_tracker.add_version(filename2, content2, datetime.now().isoformat())
        
        return {
            "filename1": filename1,
            "filename2": filename2,
            "similarity": similarity,
            "similarity_percent": round(similarity * 100, 2),
            "diff_blocks": diff_blocks,
            "total_changes": len([b for b in diff_blocks if b['type'] != 'unchanged']),
            "lines_added": len([b for b in diff_blocks if b['type'] == 'added']),
            "lines_removed": len([b for b in diff_blocks if b['type'] == 'removed'])
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/docs/{filename}/versions")
async def get_document_versions(filename: str, _: bool = Depends(admin_required)):
    """
    Get version history for a document.
    """
    try:
        versions = version_tracker.get_versions(filename)
        return {"filename": filename, "versions": versions, "count": len(versions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== FEATURE 6: TAX SCENARIO CALCULATOR ==========

@app.post("/admin/tax/calculate/income")
async def calculate_income_tax(body: Dict[str, Any], _: bool = Depends(admin_required)):
    """
    Calculate Dutch income tax (IB Box 1).
    Body: {"income": 75000}
    """
    try:
        income = body.get("income", 0)
        if not isinstance(income, (int, float)) or income < 0:
            raise HTTPException(status_code=400, detail="Invalid income value")
        
        result = tax_calculator.calculate_income_tax(float(income))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/tax/calculate/corporate")
async def calculate_corporate_tax(body: Dict[str, Any], _: bool = Depends(admin_required)):
    """
    Calculate Dutch corporate tax (VPB).
    Body: {"profit": 250000}
    """
    try:
        profit = body.get("profit", 0)
        if not isinstance(profit, (int, float)) or profit < 0:
            raise HTTPException(status_code=400, detail="Invalid profit value")
        
        result = tax_calculator.calculate_corporate_tax(float(profit))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/tax/calculate/vat")
async def calculate_vat(body: Dict[str, Any], _: bool = Depends(admin_required)):
    """
    Calculate Dutch VAT (BTW).
    Body: {"net_amount": 1000, "vat_rate": "standard"}
    """
    try:
        net_amount = body.get("net_amount", 0)
        vat_rate = body.get("vat_rate", "standard")
        
        if not isinstance(net_amount, (int, float)) or net_amount < 0:
            raise HTTPException(status_code=400, detail="Invalid net_amount value")
        
        if vat_rate not in ["standard", "reduced", "zero"]:
            raise HTTPException(status_code=400, detail="vat_rate must be 'standard', 'reduced', or 'zero'")
        
        result = tax_calculator.calculate_vat(float(net_amount), vat_rate)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/tax/calculate/inheritance")
async def calculate_inheritance_tax(body: Dict[str, Any], _: bool = Depends(admin_required)):
    """
    Calculate Dutch inheritance tax (Erfbelasting).
    Body: {"inheritance": 500000, "relationship": "child"}
    """
    try:
        inheritance = body.get("inheritance", 0)
        relationship = body.get("relationship", "child")

        if not isinstance(inheritance, (int, float)) or inheritance < 0:
            raise HTTPException(status_code=400, detail="Invalid inheritance value")

        if relationship not in ["partner", "child"]:
            raise HTTPException(status_code=400, detail="relationship must be 'partner' or 'child'")

        result = tax_calculator.calculate_inheritance_tax(float(inheritance), relationship)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/tax/compare-scenarios")
async def compare_tax_scenarios(body: Dict[str, Any], _: bool = Depends(admin_required)):
    """
    Compare multiple tax scenarios.
    Body: {
        "scenarios": [
            {"name": "Scenario 1", "tax_type": "ib", "income": 75000},
            {"name": "Scenario 2", "tax_type": "vpb", "profit": 200000}
        ]
    }
    """
    try:
        scenarios = body.get("scenarios", [])
        if not scenarios or not isinstance(scenarios, list):
            raise HTTPException(status_code=400, detail="scenarios array required")

        result = tax_calculator.compare_scenarios(scenarios)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    # Prefer explicit FATRAG port (project rule: 8020), allow override via PORT env
    port_env = os.getenv("PORT")
    try:
        port = int(port_env) if port_env else 8020
    except (TypeError, ValueError):
        port = 8020

    host = os.getenv("HOST", "0.0.0.0")
    reload_flag = os.getenv("UVICORN_RELOAD", "false").lower() in ("1", "true", "yes")

    print(f"Starting FATRAG API on http://{host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=reload_flag)
