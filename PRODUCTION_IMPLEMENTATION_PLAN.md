# 🚀 Production Implementation Plan
## Progressive Testing met Echte Ollama Integratie

**Status:** READY TO BUILD
**Estimated Time:** 3-4 hours work
**Complexity:** HIGH

## 📋 WHAT YOU ASKED FOR

1. ✅ Build all backend functions (real Ollama calls)
2. ✅ Add interface to control/review output
3. ✅ Make progressive testing REAL

## 🏗️ IMPLEMENTATION STEPS

### FASE 1: Backend API Endpoints (30 min)
**File:** `main.py`

Add 3 new endpoints:
```python
POST /api/progressive-test/start
  - Params: {type, level, config, project_id}
  - Returns: {run_id, status: "started"}
  - Triggers background job

GET /api/progressive-test/status/{run_id}
  - Returns: {status, progress, current_stage, output_preview}
  - Real-time status updates

GET /api/progressive-test/result/{run_id}
  - Returns: {result, metrics, full_output}
  - Complete results + raw output for review
```

### FASE 2: Configurable Analysis Functions (45 min)
**File:** `scripts/configurable_analysis.py` (NEW)

Build 3 configurable runners:
```python
async def run_flash_with_config(config, project_id, run_id):
    # Use config.model, config.temp, config.tokens
    # Track with MetricsStore
    # Return raw output + metadata

async def run_grondige_with_config(config, project_id, run_id):
    # Configurable chunk_size, temperature
    # Real map-reduce with llama3.1:70b
    # Return structured output

async def run_template_with_config(config, template_key, project_id, run_id):
    # Parallel or sequential based on config
    # Track section timing
    # Return full report
```

### FASE 3: Output Review Interface (45 min)
**File:** `static/admin/review-output.html` (NEW)

Features:
- Show raw LLM output
- Side-by-side comparison (config A vs B)
- Rating system (1-5 stars per criterion):
  * Accuracy
  * Completeness
  * Relevance
  * Speed
- Save ratings to database
- View historical ratings

### FASE 4: Update Progressive Tester UI (30 min)
**File:** `static/admin/progressive-test.html`

Changes:
- Replace simulation with real API calls
- Show real-time progress updates
- Add "Review Output" button per level
- Display actual GPU usage (from nvidia-smi)
- Show real duration + tokens

### FASE 5: Metrics Integration (30 min)
**Files:** All analysis functions

Add to every analysis:
```python
ms = MetricsStore()
run_id = ms.log_analysis_start(...)
ms.log_stage("retrieval", duration, tokens)
ms.log_stage("generation", duration, tokens)
ms.log_analysis_complete(run_id, result, quality_scores)
```

### FASE 6: Quality Rating System (30 min)
**File:** `quality_ratings.py` (NEW)

Database table:
```sql
CREATE TABLE quality_ratings (
    id INT PRIMARY KEY AUTO_INCREMENT,
    run_id VARCHAR(100),
    analysis_type VARCHAR(50),
    config JSON,
    accuracy_score INT,      -- 1-5
    completeness_score INT,  -- 1-5
    relevance_score INT,     -- 1-5
    speed_score INT,         -- 1-5
    notes TEXT,
    created_at TIMESTAMP
);
```

## 📊 DATA FLOW

```
User clicks Start
  ↓
POST /api/progressive-test/start
  ↓
Background job starts
  ↓
run_flash_with_config()
  ├─ Metrics tracking
  ├─ Real Ollama call
  └─ Save output
  ↓
Frontend polls /status
  ├─ Shows progress
  └─ Updates UI
  ↓
Job completes
  ↓
GET /result → Review page
  ↓
User rates output
  ↓
Saves to quality_ratings
```

## 🎯 KEY FEATURES

**For You:**
- ✅ See REAL output from each config
- ✅ Rate quality on multiple dimensions
- ✅ Compare configs side-by-side
- ✅ Track which configs work best
- ✅ Export best configs for production

**Technical:**
- ✅ Real Ollama integration
- ✅ GPU usage visible
- ✅ Background jobs (FastAPI)
- ✅ Metrics tracking
- ✅ Quality database
- ✅ Historical comparison

## ⏱️ TIMELINE

- **NOW**: Build backend (1.5h)
- **THEN**: Build review UI (1h)
- **FINALLY**: Integrate & test (1h)

**Total: ~3-4 hours**

## 🚨 WHAT I NEED

Before I start building:

1. **Test Project**: Which project_id should I use for testing?
2. **Documents**: Are there documents in that project?
3. **Ollama Status**: Are your Ollama workers running?

Run:
```bash
curl http://localhost:11434/api/tags  # Check 8b model
curl http://localhost:11435/api/tags  # Check 70b model (if separate)
```

## 📁 FILES I'LL CREATE/MODIFY

**NEW:**
- `scripts/configurable_analysis.py` (core logic)
- `static/admin/review-output.html` (review UI)
- `quality_ratings.py` (rating system)
- `migrations/003_quality_ratings.sql` (DB schema)

**MODIFY:**
- `main.py` (add 3 API endpoints)
- `static/admin/progressive-test.html` (replace simulation)
- `metrics_store.py` (minor enhancements)

## ✅ READY TO START?

Reply with:
- "GO" → I start building now
- "WAIT" → You want to discuss first
- Other questions → I answer them

Let's make this REAL! 🚀

---

## 🔁 Volgende Roadmap: Ingestion Pipeline & AVG-proof Anonimisering

Dit is de stap-voor-stap route om van de huidige situatie naar jouw ideale pipeline te gaan (inclusief latere AVG-proof anonimisatie voor cloud LLM’s).

### Fase A – Basis ingest & status-flow (klein en veilig)

Doel: upload-flow consistent maken en statusvelden gebruiken zonder grote breuken.

- [ ] `documents`-tabel uitbreiden met velden (of bevestigen dat ze bestaan):
  - `status` (uploaded / normalized / classified / indexed / analyzed)
  - `metadata` (JSON) voor o.a. `document_type`, `has_financial_data`, `currency`, `total_amount`
- [ ] `DocumentIngestService` + `/admin/projects/{project_id}/upload` harmoniseren:
  - Altijd:
    - originele file → `fatrag_data/uploads`
    - entry in `documents` met `source_type = 'project_upload'`
    - `status = 'uploaded'`
- [ ] Bestaande code paden (oude ingest) aanpassen zodat ze dezelfde `status`-flow gebruiken.

**Resultaat:** elke upload heeft een nette DB-entry met een beginstatus en metadata-veld.

### Fase B – Normalisatie-laag (tekst / MD / JSON)

Doel: één consistente representatie van documenten vóór classificatie en embeddings.

- [ ] Nieuwe service `document_extractors.py`:
  - PDF → tekst
  - XLSX → JSON/tekst
  - DOCX → tekst
  - TXT/MD → tekst
- [ ] Per document:
  - `normalized_text` (of MD) opslaan (DB of file)
  - optioneel `normalized_json`
  - `status = 'normalized'`
- [ ] Integreren in ingest-pipeline als afzonderlijke stap (background job mogelijk).

**Resultaat:** alle documenten hebben een genormaliseerde vorm die klaar is voor classificatie en chunking.

### Fase C – Classificatie (soort document)

Doel: elk document typeren (jaarrekening, contract, belastingaanslag, …).

- [ ] Nieuwe service `document_classifier.py`:
  - Input: `normalized_text` + metadata
  - Output: `document_type` + extra labels (jaar, partijen, etc.)
- [ ] `documents.metadata.document_type` vullen en `status = 'classified'`.
- [ ] Eenvoudige admin-overzicht toevoegen (welke documenten welk type hebben).

**Resultaat:** je kunt in de UI/filtering per documenttype werken en straks type-specifieke analyses draaien.

### Fase D – Chunk & Embed refactor (projectbewust)

Doel: RAG-laag opschonen en laten aansluiten op nieuwe metadata.

- [ ] Chunking-service laten werken op `normalized_text`:
  - Config uit `Settings` (chunk_size, chunk_overlap).
  - Chunks annoteren met `project_id`, `client_id`, `document_type`, kern-metadata.
- [ ] Embeddings:
  - Uniforme entrypoint voor vectorstore (één adapter).
  - Filters op `project_id` en `document_type` afdwingen in retrievers.
- [ ] `status = 'indexed'` wanneer chunks + embeddings klaar zijn.

**Resultaat:** stabiele, goed gefilterde RAG-laag, per project en per documenttype.

### Fase E – Standaardanalyses per documenttype

Doel: one-click analyses per type (jaarrekening, contract, etc.).

- [ ] Template-config (bijv. in `config.json` of aparte tabel) voor:
  - `document_type` → lijst analyses (naam, prompt, model, parameters).
- [ ] Service-laag:
  - Zoekt relevante chunks op basis van type.
  - Voert LLM-call uit volgens template.
  - Slaat output op als nieuw document met `source_type` bv. `analysis:jaarrekening`.
- [ ] Admin-UI:
  - Per project en type een “Start standaardanalyse”-knop.

**Resultaat:** kwalitatief consistente, herhaalbare analyses per documentsoort.

### Fase F – AVG-proof Anonimisering (voor cloud LLM’s)

Doel: een aparte module om documenten/chunks te ontdoen van PII voordat ze naar een cloud LLM gaan.

- [ ] Nieuwe service `anonymization.py`:
  - Detecteert PII (namen, BSN, adressen, IBAN, kentekens, e-mails, etc.).
  - Vervangt met placeholders (`[PERSOON_1]`, `[ADRES_1]`, …).
  - Houdt een mapping bij zodat je (optioneel) achteraf kunt terugmappen.
- [ ] Integratiepunt in pipeline:
  - Voor cloud-routes (`is_cloud_model(...) == True`):
    - Gebruik `anonymize(text)` → LLM → des-anonimiseren in rapport (indien nodig).
  - Voor lokale modellen: optie om native (niet-geanonimiseerde) tekst te gebruiken.
- [ ] Config-vlaggen:
  - Per analyse-type/LLM-config:
    - `require_anonymization_for_cloud: bool`.

**Resultaat:** je kunt veilig cloud LLM’s gebruiken zonder AVG-risico, met een duidelijke plek in de pipeline waar anonimisering gebeurt.

Deze roadmap staat naast de bestaande Progressive Testing-plannen in `PRODUCTION_IMPLEMENTATION_PLAN.md` en maakt het mogelijk om de ingest-pipeline én AVG-proof anonimisering stap voor stap te implementeren zonder de huidige werking in één keer omver te trekken.
</result>
<task_progress>
- [x] Inspect existing production implementation plan
- [x] Add a phased, step-by-step roadmap for the new ingestion pipeline
- [x] Expliciet een latere fase voor AVG-proof anonimisering opgenomen
- [x] Fases klein en opeenvolgend gemaakt (A–F)
</task_progress>
