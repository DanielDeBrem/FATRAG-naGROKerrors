# FATRAG Architecture Overview

## Current State Analysis

### FastAPI Backend Structure
The application is built with FastAPI and runs on port 8020. The main entry point is `main.py` which creates a single FastAPI application instance with numerous endpoints:

- **Routers/Endpoints**: Currently all endpoints are defined in `main.py` (over 1000 lines) without clear separation:
  - Document management (`/admin/docs/*`)
  - Client/project management (`/admin/clients/*`, `/admin/projects/*`)
  - LLM configuration (`/admin/config`)
  - Analysis jobs (`/admin/jobs/*`)
  - Query interface (`/query`)
  - Admin UI serving

- **Middleware**: Uses SlowAPIMiddleware for rate limiting and custom TimeoutMiddleware

- **Services**: Business logic is scattered across the main file or separate files like:
  - `clients_projects.py` - Client/project CRUD
  - `organogram_service.py` - Organogram management
  - `tax_calculator.py` - Tax calculations
  - `analysis.py` - Document analysis
  - Various job processing scripts in `scripts/`

### Data Storage
- **MySQL**: Extensive use of MySQL with tables for clients, projects, documents, jobs, feedback, etc.
- **Vector Store**: ChromaDB for embeddings storage
- **File System**: Documents stored in `fatrag_data/uploads/`

### LLM Integration
- **Ollama**: Primary LLM provider with support for multiple models (llama3.1:8b, gemma2:2b)
- **Load Balancing**: Multi-GPU worker routing via OLLAMA_WORKER_PORTS
- **Cloud Routing**: Support for cloud-hosted Ollama via OLLAMA_CLOUD_ROUTED

### Frontend
- **Admin Interface**: HTML/JS interface served from `static/admin/`
- **No Streamlit**: Current frontend is pure HTML/CSS/JS, no Streamlit integration

## Weaknesses Identified

### Architecture Issues
1. **No Clear Layer Separation**: All logic mixed in `main.py`
2. **No Dependency Injection**: Global variables and direct imports
3. **No Abstraction Layer**: Direct Ollama calls scattered throughout
4. **Massive Single File**: `main.py` is over 1000+ lines

### Quality Issues
5. **No Tests**: No unit or integration tests visible
6. **Poor Error Handling**: Basic try/catch blocks, inconsistent responses
7. **No Logging Strategy**: Print statements instead of proper logging
8. **No Configuration Management**: Environment variables scattered

### Code Organization Issues
9. **Mixed Concerns**: HTTP logic, business logic, and data access in one file
10. **Hard Dependencies**: Direct DB calls in endpoints
11. **No Type Safety**: Limited Pydantic model usage
12. **Ad-hoc Patterns**: Inconsistent API design patterns

## Proposed Architecture

### Directory Structure
```
app/
├── api/                    # HTTP layer
│   ├── documents.py       # Document upload/management
│   ├── advisory.py        # Query and advice endpoints
│   ├── __init__.py
│   └── dependencies.py    # Dependency injection
├── core/                  # Cross-cutting concerns
│   ├── config.py         # Settings management
│   ├── logging.py        # Logging configuration
│   ├── exceptions.py     # Custom exceptions
│   └── utils.py          # Utility functions
├── models/                # Data models
│   ├── domain/           # Domain entities
│   │   ├── document.py
│   │   ├── advisory.py
│   │   └── __init__.py
│   ├── dto/              # Data transfer objects
│   │   ├── upload.py
│   │   └── advisory.py
│   └── __init__.py
├── services/             # Business logic layer
│   ├── document_ingest.py     # Document processing
│   ├── document_classifier.py # Content classification
│   ├── document_extractors.py # Text/data extraction
│   ├── embeddings.py          # Vector operations
│   ├── advisory.py           # Advice generation
│   ├── ollama_client.py      # LLM abstracted client
│   └── __init__.py
├── repositories/         # Data access layer
│   ├── documents.py      # Document persistence
│   ├── embeddings.py     # Vector storage
│   ├── jobs.py          # Job tracking
│   └── __init__.py
└── main.py              # FastAPI application

frontend/
└── streamlit_app.py     # Streamlit UI

scripts/                 # Existing analysis scripts
docs/                    # Documentation
tests/                   # Test suite
```

### Design Principles
1. **Dependency Injection**: Use dependency injection for services and repositories
2. **SOLID Principles**: Single responsibility, open/closed, etc.
3. **Clean Architecture**: Clear separation between layers
4. **Type Safety**: Extensive Pydantic usage
5. **Error Handling**: Custom exceptions and consistent error responses
6. **Configuration**: Centralized settings management
7. **Logging**: Structured logging throughout
8. **Testing**: Unit and integration tests for each layer

## Document Ingestion & Analysis Pipeline (Target Design)

Doel: een duidelijke, herhaalbare pipeline van ruwe upload → hoogwaardige advies-output, met per stap controle en uitbreidbaarheid.

### 1. Upload & opslag origineel bestand

**Stap:**
- User uploadt file via:
  - `/admin/projects/{project_id}/upload`
  - of een generieke `/documents/upload`-endpoint.

**Acties:**
- *Validatie* (service: `DocumentIngestService`):
  - Extensie/MIME (PDF, TXT, MD, XLSX, DOCX, etc.).
  - Bestandsgrootte (max X MB).
- *Opslag in filesystem*:
  - `fatrag_data/uploads/<doc_id>_<originele_naam>.<ext>`.
- *Opslag in SQL* (tabel `documents`):
  - `doc_id` (bijv. `doc-xxxx`).
  - `project_id`, `client_id`.
  - `filename`, `file_path`, `file_size`.
  - `source_type = 'project_upload'`.
  - `status = 'uploaded'`.
  - `metadata` (JSON) initieel leeg of minimale info.

### 2. Normalisatie naar tekst/MD/JSON

**Doel:** één consistente representatie per document, geschikt voor LLM & analyses.

**Stap:**
- Extractie:
  - PDF → platte tekst.
  - XLSX/XLS → relevante tabellen/cellen → tekst/JSON.
  - DOCX → tekst (python-docx).
  - TXT/MD → direct leesbaar.

**Acties:**
- *Normalisatie*:
  - Schoonmaken van headers/footers.
  - Wegfilteren van ruis (paginanummers, watermerken).
  - Opslaan in:
    - filesystem als `.md` of `.json`, én/of
    - SQL in aparte kolommen, bijv.:
      - `normalized_text` (LONGTEXT) voor MD/tekst.
      - `normalized_json` (JSON) voor gestructureerde data.
- *Statusupdate*:
  - `status = 'normalized'`.

### 3. Documentclassificatie (soort document)

**Doel:** elk document krijgt een semantisch type, zodat we gerichte analyses kunnen doen.

**Voorbeelden van types:**
- `jaarrekening`
- `contract`
- `belastingaanslag`
- `financiele_memo`
- `(arbeids)overeenkomst`
- `taxatie`
- `notariele_akte`
- etc.

**Stap:**
- Classificatie-service (bijv. `document_classifier.py`):
  - Input: `normalized_text` (en evt. metadata).
  - Model: rule-based + LLM (prompt-gestuurd).
- Output wordt opgeslagen in:
  - `documents.metadata.document_type`.
  - Eventueel extra velden (jaartal, tegenpartij, bedragscategorie).

**Statusupdate:**
- `status = 'classified'`.

### 4. Chunking & Embedding

**Doel:** documenten bevraagbaar maken (RAG) en hergebruik voor analyses.

**Stap:**
- Chunking:
  - Op basis van `normalized_text`.
  - Configurabel via `chunk_size`, `chunk_overlap` (zie `Settings`).
  - Iedere chunk → `DocumentChunk` (in code) met:
    - `document_id`, `index`, `text`, `metadata` (type, jaar, partijen, etc.).
- Embedding:
  - Via Ollama embeddings (`OllamaEmbeddings`).
  - Wegschrijven naar vectorstore (ChromaDB):
    - Collection per project of globale collectie met `project_id`-filter.
- Optioneel: chunk-rows in MySQL (tabel `document_chunks`) met referentie naar vectorstore-ID.

**Statusupdate:**
- `status = 'indexed'`.
- `metadata.has_financial_data`, `metadata.currency`, `metadata.total_amount` kunnen in deze stap al deels afgeleid worden uit de chunks.

### 5. Bevraagbaarheid door LLM (RAG-laag)

**Doel:** interactief vragen kunnen stellen per project/document.

**Stap:**
- RAG-keten (`build_qa_chain` in `main.py`):
  - Gebruikt vectorstore als retriever.
  - Filter op:
    - `project_id`
    - optioneel `client_id`
    - `document_type` (voor gerichte vragen).
- Antwoord prompt-gestuurd (FinAdviseur-NL stijl), met keten:
  - Query → retrieve relevante chunks → LLM synthese.

### 6. Standaard analyses per documenttype

**Doel:** één klik “standaard analyse” voor elk documenttype, zodat de adviseur altijd een consistente baseline heeft.

**Voorbeelden:**
- Jaarrekening:
  - Ratio’s (solvabiliteit, rentabiliteit).
  - Trendanalyse 3–5 jaar.
  - Cashflow-inzichten.
- Contract:
  - Belangrijkste verplichtingen.
  - Termijnen / opzegtermijnen.
  - Risico-clausules.
- Belastingaanslag:
  - Aanslagbedrag vs. verwachting.
  - Termijnen / bezwaar.
  - Kansen en risico’s.
- Notariële akte:
  - Eigendomsoverdracht / structuur.
  - Relevante fiscale regimes.
  - Latente belastingposities.

**Implementatie-idee:**
- Per `document_type` één of meer “analysis templates”:
  - Prompt-tekst + parameters (model, max tokens, temperature).
  - Opslag in configuratie (later beheerbaar via instellingenpagina).
- Service-laag:
  - Haalt relevante chunks/documents + context op.
  - Voert LLM-call uit op basis van type-specifieke prompt.
  - Slaat output op als nieuw document:
    - `source_type = 'llm_analysis'` of gedetailleerder, bijv. `'analysis:jaarrekening'`.
    - Bestandsnaam in `fatrag_data/uploads/` + entry in `documents`.

### 7. Instellingenpagina (toekomst)

**Doel:** per soort analyse het LLM-gedrag fijnregelen.

**Voor elke analyseconfiguratie:**
- Model (bijv. `llama3.1:8b`, `qwen2.5:7b`).
- Temperature, max_tokens.
- Chunking-instellingen (size/overlap).
- Prompt-tekst (met placeholders).
- Outputformaat (MD, PDF, CSV, …).

**Frontend:**
- Nieuwe admin-pagina (bijv. `static/admin/llm-config.html` uitbreiden) met:
  - Lijst van documenttypes.
  - Per type de beschikbare standaardanalyses.
  - Edit/save van prompts en LLM-parameters.

**Backend:**
- Opslag in `config.json` of aparte `analysis_templates`-tabel.
- API’s onder `/admin/config/analysis-templates`.

---

### Status t.o.v. huidige code

- **Deels aanwezig:**
  - Upload + opslag op disk (`ingestion.py`, `DocumentIngestService`, `/admin/projects/{project_id}/upload`).
  - SQL-opslag in `documents` (via `DocumentsRepository` en raw SQL in `main.py`).
  - Chunking + embeddings (vectorstore/Chroma, RAG-keten).
  - LLM-analyse pipelines (flash, full, FATRAG).

- **Nog te bouwen / te harmoniseren:**
  - Eenduidige normalisatie-laag (MD/JSON) + velden in DB (`normalized_text`, `normalized_json`).
  - Structurele documentclassificatie (`document_type` invullen voor alle documenten).
  - Consequente status-flow (`uploaded` → `normalized` → `classified` → `indexed` → geanalyseerd).
  - Standaardanalyse per type (templates + services).
  - Instellingenpagina voor LLM per analyse-type.

Deze pipelinebeschrijving is de **doelarchitectuur** waar de codebase stap voor stap naartoe refactored kan worden.
