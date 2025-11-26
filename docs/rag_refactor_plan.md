# FATRAG RAG-architectuur verbeterplan

Dit document legt het verbeterplan vast dat eerder in de chat is geformuleerd, zodat we het stap voor stap kunnen implementeren en later kunnen raadplegen.

## 1. Huidige situatie (samenvatting)

### Core RAG-pad

- Upload via legacy endpoints in `main.py` (`/admin/projects/{project_id}/upload`) met:
  - `ingestion.py` + directe Chroma-ingest (`ing.ingest_files(...)`).
- Query via `/query` met:
  - `EmbeddingsService.get_retriever(...)` (centrale retriever-adapter).
  - `build_llm_from_config` + `build_qa_chain` op basis van `app.state.config`.
  - Taaldetectie + FinAdviseur-NL prompt.

### Nieuwe gelaagde architectuur (deels geïmplementeerd)

- `app/services/document_ingest.py` (`DocumentIngestService`) implementeert een nette pipeline:
  - validatie → opslag op disk → normalisatie via `extract_for_db` → classificatie via `DocumentClassifierService` → persist via `DocumentsRepository` → index via `DocumentIndexService`.
- `app/services/embeddings.py` biedt een centrale singleton vectorstore + retrieval-API (`EmbeddingsService`).

### Jobs / analyses

- Rijke achtergrondjob-laag in `main.py`:
  - Volledige map-reduce analyses per project.
  - Flash analyse via `scripts/flash_analysis.py`.
  - Tuning/progressieve tests via `configurable_analysis.py`, `QualityRatings`, etc.
  - Volledige FATRAG pipeline via `scripts/fatrag_auto.py`.

### Admin UX

- Admin SPA onder `static/admin/*` met:
  - Projects, documents, jobs, pipeline-config, upload-progress, organograms, dashboards, templates, tax calculators.

Conclusie: veel functionaliteit, maar verdeeld over een nieuwe `app/`-laag en een nog steeds grote `main.py` waarin HTTP, DB, RAG, job orchestration en filesystem-werk door elkaar lopen.

---

## 2. Belangrijkste problemen / inconsistenties

### 2.1 Dubbele vectorstore / embedding-configuratie

- `main.py` definieert:

  ```python
  embed_model = OllamaEmbeddings(...)
  vectorstore = Chroma(...)
  ```

- `app/services/embeddings.py` definieert:

  ```python
  _embed_model = OllamaEmbeddings(...)
  _vectorstore = Chroma(...)
  ```

Gevolgen:

- Er bestaan twee Chroma-instanties met mogelijk verschillende configuratie.
- Sommige codepaden schrijven naar de ene instantie en lezen uit de andere.
- Configuratie (model/base_url) raakt versnipperd.

### 2.2 Twee ingest/index-pijplijnen

- Legacy-pad in `main.py` gebruikt:
  - `ing.ensure_dirs`, schrijf naar `fatrag_data/uploads/`.
  - `ing.ingest_files(...)` naar Chroma.
  - Handmatige SQL naar `documents` met `status="indexed"`.

- Nieuwe pad in `DocumentIngestService`:
  - Validatie, normalisatie via `extract_for_db`, classificatie, opslag via `DocumentsRepository`, index via `DocumentIndexService`.

Gevolg: het doelproces (`uploaded → normalized → classified → indexed`) is niet consequent voor alle documenten; admin UI gebruikt nog vooral het legacy-pad.

### 2.3 Gemengde layering en directe DB-toegang

- Sommige endpoints gebruiken repositories/services (bijv. `cp.create_client`, `DocumentsRepository`), andere doen directe SQL (`pymysql`) in endpoints.
- Veel endpoints accepteren `Dict[str, Any]` in plaats van Pydantic DTO’s.

### 2.4 RAG-kwaliteitsknoppen verspreid

- Chunking, overlap, en modelkeuzes zitten verdeeld over:
  - `ingestion.py`, `ingestion_with_progress.py`,
  - `analyze_project_documents_map_reduce`, flash analysis scripts, tuner scripts, etc.
- Geen eenduidige declaratieve configuratie per analyse-type/document-type.

### 2.5 Tests & observability

- Er is een globale exception handler met nette JSON, maar:
  - Externe calls (Ollama, subprocess-scripts) geven nog vaak generieke `HTTPException(500, detail=str(e))`.
  - Logging is deels `print`, deels `logging`, niet centraal geconfigureerd.
- Er is geen systematische set adapter-level unit tests voor Chroma/Ollama/scripts.

---

## 3. Verbeteringen (technisch)

### 3.1 Vectorstore-unificatie rond `EmbeddingsService`

**Doel:** één bron van waarheid voor embeddings en retrieval.

Acties:

1. Verwijder directe `OllamaEmbeddings`/`Chroma`-instantiaties in `main.py`; gebruik overal:

   ```python
   from app.services.embeddings import EmbeddingsService
   ```

2. Vervang alle gebruik van de globale `vectorstore`:
   - Voor retrieval: via `EmbeddingsService.get_retriever(...)`.
   - Voor directe Chroma-operaties: via `EmbeddingsService.raw_vectorstore()`.

3. Zorg dat alle ingest-paden chunks schrijven via `EmbeddingsService` (of via een gedeelde helper die `_vectorstore` uit `EmbeddingsService` gebruikt).

### 3.2 Centreer uploads op `DocumentIngestService`

**Doel:** één ingest/normalisatie/classificatie-pijplijn.

Acties:

1. Introduceer een (nieuwe of aangepaste) endpoint voor projectuploads die:
   - Per bestand `DocumentIngestService.ingest_document(file)` aanroept.
   - `project_id`/`client_id` koppelt via `DocumentsRepository`.

2. Laat `upload-with-progress` in een tweede stap bovenop deze service werken.

3. Hanteer consistente statusovergangen (`uploaded → normalized → classified → indexed`) en laat `DocumentIndexService` `status="indexed"` zetten na succesvolle indexering.

### 3.3 RAG-metadata & filters aligneren

**Doel:** betrouwbare filtering op project, client, documenttype en bron.

Acties:

1. Definieer een canoniek metadataschema in `EmbeddingsService.add_document_chunks`, met ten minste:

   ```json
   {
     "doc_id": "...",
     "project_id": "...",
     "client_id": "...",
     "document_type": "...",
     "source_type": "...",
     "filename": "...",
     "chunk_index": ...
   }
   ```

2. Zorg dat alle ingest-paden deze velden invullen (voor analyses bijv. `source_type="llm_analysis"`, `flash_analysis`, `fatrag_report`, etc.).

3. Breid `/query` uit met optionele filters op `document_type` en/of `source_type` en geef die door naar `EmbeddingsService.get_retriever`.

### 3.4 Query-keten verfijnen voor kwaliteit & veiligheid

Acties:

1. Voeg een guardrail toe wanneer de retriever geen of weinig relevante context vindt:
   - Antwoord dan kort met een “onvoldoende data”-stijl bericht i.p.v. hallucinaties.

2. Verbeter taalkeuze:
   - Bij detectiefouten standaard naar Nederlands (`lang_hint="Nederlands"`).
   - Optioneel: laat `Query` een expliciete `lang`-override bevatten.

3. Maak retriever-parameters (`RETRIEVER_K`, score thresholds, filters) config-gedreven via `config.json` en `/admin/config`.

### 3.5 Strakkere API-grenzen met Pydantic DTO’s

Acties:

1. Vervang `Dict[str, Any]` bodies door DTO’s in `app/models/dto/*` voor o.a.:
   - `/admin/config` (PUT),
   - client/project create/update,
   - flash/pipeline job requests,
   - progressive test en tuner endpoints,
   - tax-calculator scenario’s.

2. Houd responsmodellen eveneens consistent via DTO’s waar nuttig.

### 3.6 Job-orchestratie uit `main.py` halen

Acties:

1. Verplaats job-specifieke logica naar services, bijv.:
   - `app/services/analysis_jobs.py`,
   - `app/services/flash_analysis_service.py`,
   - `app/services/pipeline_service.py`.

2. Beperk `main.py` tot:
   - Endpoint-definities,
   - dependency-injectie van services,
   - eenvoudige aanroep van servicemethoden.

### 3.7 Beter error-handling en logging voor externe calls

Acties:

1. Introduceer een `OllamaClient`-service in `app/services/ollama_client.py` die:
   - Alle LLM- en embeddingscalls centraliseert,
   - Exceptions vertaalt naar domein-specifieke fouten (bijv. `LLMUnavailableError`),
   - Gestructureerd logt.

2. Idem voor subprocess-scripts (flash, pipeline, configurable analysis) via een `ScriptRunner`-service.

3. Voeg centrale logging-config toe in `app/core/logging.py` (structured logging / JSON).

### 3.8 Tests voor adapters en RAG-paden

Acties:

1. `EmbeddingsService`-tests met een fake embeddingfunctie en tijdelijke Chroma-dir:
   - Verifiëren dat filters (`project_id`, `client_id`, `document_type`) correct werken.

2. `DocumentIngestService`-tests:
   - Validatie, statusovergangen, classificatie-integratie (mocked), cleanup.

3. Losgekoppelde tests voor taalkeuze/promptselectie in `/query`.

4. Tests voor job-services (met fake script runner / fake Ollama).

---

## 4. Stapsgewijze migratiestrategie

1. Stap 1: `/query` + andere leespaden volledig naar `EmbeddingsService` brengen (geen DB-wijziging nodig).
2. Stap 2: nieuwe upload-endpoint via `DocumentIngestService`, deze in de admin-UI gaan gebruiken.
3. Stap 3: backfill-script voor bestaande documenten:
   - Normalisatie en classificatie aanvullen,
   - Vectorstore-metadata harmoniseren.
4. Stap 4: job-orchestratie naar services verplaatsen.
5. Stap 5: DTO’s + adapter-tests uitrollen.
