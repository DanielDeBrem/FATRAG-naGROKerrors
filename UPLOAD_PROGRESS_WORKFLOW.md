# File Upload Progress Tracking Workflow

## Overzicht

Een complete workflow voor file-upload met live voortgangsindicatie en gedetailleerde rapportage per stap. Elk bestand doorloopt de volgende stages:

1. **Upload** (0-10%) - Bestand wordt naar server gestuurd
2. **Extractie** (10-30%) - Tekst wordt geëxtraheerd uit PDF/Excel/tekstbestanden
3. **Tokenisatie** (30-50%) - Tekst wordt opgesplitst in chunks
4. **Embedding** (50-80%) - Embeddings worden gegenereerd per chunk
5. **Indexering** (80-100%) - Chunks worden opgeslagen in vectordatabase

## Database Schema

### Tabellen

#### `upload_progress`
Houdt de voortgang bij van individuele bestandsuploads door alle stages.

```sql
- upload_id (PK): Uniek ID voor deze upload
- project_id: Gekoppeld project
- filename: Bestandsnaam
- file_size: Grootte in bytes
- status: queued|uploading|uploaded|extracting|tokenizing|embedding|indexing|completed|failed
- progress_percent: 0-100%
- current_stage: Huidige stage beschrijving
- total_chunks: Aantal gegenereerde chunks
- total_tokens: Geschat aantal tokens
- extraction_time_ms: Tijd voor tekst extractie
- tokenization_time_ms: Tijd voor tokenisatie
- embedding_time_ms: Tijd voor embeddings
- indexing_time_ms: Tijd voor indexering
- error_message: Foutmelding bij falen
- created_at, completed_at: Timestamps
```

#### `upload_batches`
Houdt de voortgang bij van batch uploads (meerdere bestanden tegelijk).

```sql
- batch_id (PK): Uniek ID voor deze batch
- project_id: Gekoppeld project
- total_files: Totaal aantal bestanden
- completed_files: Aantal voltooide bestanden
- failed_files: Aantal gefaalde bestanden
- status: queued|processing|completed|partial_failure|failed
- started_at, completed_at: Timestamps
```

## API Endpoints

### Upload met Progress Tracking

**POST** `/admin/projects/{project_id}/upload-with-progress`

Upload bestanden met live progress tracking (batch mode).

**Request:**
```
Content-Type: multipart/form-data
files: [File, File, ...]
```

**Response:**
```json
{
  "status": "queued",
  "batch_id": "batch-1731330000000",
  "total_files": 3,
  "message": "Upload queued...",
  "track_url": "/admin/uploads/batch/batch-1731330000000"
}
```

### Progress Tracking Endpoints

**GET** `/admin/uploads/progress/{upload_id}`
- Haal real-time voortgang op voor een enkel bestand

**GET** `/admin/uploads/batch/{batch_id}`
- Haal voortgang op voor een batch upload
- Bevat lijst van alle individuele uploads in de batch

**GET** `/admin/projects/{project_id}/uploads?status=&limit=50`
- Lijst alle uploads voor een project
- Filter op status (optional)

## Python Modules

### `upload_progress_store.py`

Bevat `UploadProgressTracker` klasse voor database operaties:

```python
tracker = UploadProgressTracker()

# Create upload record
upload = tracker.create_upload(
    upload_id="upload-123",
    filename="document.pdf",
    project_id="proj-1",
    file_size=1024000
)

# Update progress
tracker.update_upload(
    upload_id="upload-123",
    status="extracting",
    progress_percent=25,
    extraction_time_ms=1500
)

# Create batch
batch = tracker.create_batch("batch-1", "proj-1", 5)
tracker.update_batch("batch-1", completed_files=3, failed_files=1)
```

### `ingestion_with_progress.py`

Bevat enhanced ingestion functies met progress callbacks:

```python
from ingestion_with_progress import ingest_file_with_progress

# Single file met progress
result = ingest_file_with_progress(
    vectorstore=vectorstore,
    file_path="/path/to/file.pdf",
    upload_id="upload-123",
    tracker=tracker,
    chunk_size=500,
    chunk_overlap=100
)

# Batch met progress
result = ingest_files_batch_with_progress(
    vectorstore=vectorstore,
    file_paths=["/path/1.pdf", "/path/2.pdf"],
    batch_id="batch-1",
    project_id="proj-1"
)
```

## Admin UI

### Upload Progress Pagina

**URL:** `/admin/upload-progress.html`

**Query Parameters:**
- `?batch_id=batch-xxx` - Toon voortgang voor een batch
- `?project_id=proj-xxx` - Toon alle uploads voor een project
- `?upload_id=upload-xxx` - Toon voortgang voor een enkel bestand

**Features:**
- Auto-refresh elke 1-3 seconden (afhankelijk van type)
- Live progress bars met smooth animaties
- Gedetailleerde metrics per bestand (chunks, tokens, timings)
- Batch overzicht met totalen
- Error messages bij failures
- Automatisch stoppen met refreshen bij completion

**Stage Visualisatie:**
- Kleurgecodeerde status badges
- Pulserende animatie voor actieve stages
- Gedetailleerde metrics per stage:
  - Extractie tijd
  - Tokenisatie tijd
  - Embedding tijd
  - Indexering tijd
  - Totaal chunks
  - Totaal tokens

## Workflow Details

### 1. File Upload Stage (0-10%)
- Bestand wordt via multipart/form-data naar server gestuurd
- Opgeslagen in `fatrag_data/uploads/`
- `upload_progress` record wordt aangemaakt met status "uploaded"

### 2. Text Extraction Stage (10-30%)
- **PDF**: PyMuPDF extraction met cleaning
- **Excel**: Pandas dataframe to text
- **Text**: Direct inlezen met UTF-8
- Headers/footers worden verwijderd
- Timing wordt bijgehouden in `extraction_time_ms`

### 3. Tokenization Stage (30-50%)
- RecursiveCharacterTextSplitter wordt gebruikt
- Configureerbare `chunk_size` (default 500)
- Configureerbare `chunk_overlap` (default 100)
- Token estimate: ~4 chars per token
- Metrics: `total_chunks`, `total_tokens`, `tokenization_time_ms`

### 4. Embedding Stage (50-80%)
- Embeddings worden gegenereerd per chunk
- Model: mxbai-embed-large (of geconfigureerd model)
- Batch processing met progress updates
- 10 progress updates tijdens embedding fase
- Metrics: `embedding_time_ms`, `embedding_dimensions` (768)

### 5. Indexing Stage (80-100%)
- Chunks + embeddings → ChromaDB
- Metadata wordt toegevoegd (source, doc_id, project_id, etc.)
- Vectorstore.persist() voor durability
- Metrics: `indexing_time_ms`
- Bij success: status → "completed", progress → 100%

### Error Handling
- Bij error in elke stage:
  - Status → "failed"
  - `error_message` wordt gezet
  - `error_stage` wordt bijgehouden
  - Processing stopt voor dat bestand
- Batch processing gaat door bij failures
- `retry_count` wordt bijgehouden (future: auto-retry)

## Usage Examples

### Voorbeeld 1: Upload via Admin UI

1. Ga naar Project Detail pagina
2. Klik "Upload Documents"
3. Selecteer bestanden
4. Submit → redirect naar upload-progress.html?batch_id=xxx
5. Watch live progress per bestand

### Voorbeeld 2: Programmatisch Upload

```python
import asyncio
from fastapi import BackgroundTasks
from ingestion_with_progress import ingest_files_batch_with_progress

async def upload_documents(project_id: str, files: list):
    batch_id = generate_id("batch-")
    
    # Start background processing
    background_tasks.add_task(
        ingest_files_batch_with_progress,
        vectorstore=vectorstore,
        file_paths=files,
        batch_id=batch_id,
        project_id=project_id
    )
    
    return {"batch_id": batch_id}
```

### Voorbeeld 3: Progress Polling

```javascript
// Poll for progress every 2 seconds
async function pollProgress(batchId) {
    const response = await fetch(`/admin/uploads/batch/${batchId}`);
    const data = await response.json();
    
    // Update UI
    updateProgressUI(data.batch, data.uploads);
    
    // Continue polling if not done
    if (!['completed', 'failed'].includes(data.batch.status)) {
        setTimeout(() => pollProgress(batchId), 2000);
    }
}
```

## Performance Metrics

Typische verwerkingstijden per bestand (geschat):

- **PDF (10 MB, 50 pagina's)**:
  - Extractie: 2-5 sec
  - Tokenisatie: 0.5-1 sec
  - Embedding: 10-20 sec (afhankelijk van chunks)
  - Indexering: 1-2 sec
  - **Totaal: 15-30 sec**

- **Excel (5 MB, 10 sheets)**:
  - Extractie: 1-3 sec
  - Tokenisatie: 0.3-0.5 sec
  - Embedding: 5-10 sec
  - Indexering: 0.5-1 sec
  - **Totaal: 7-15 sec**

- **Text (1 MB)**:
  - Extractie: < 0.1 sec
  - Tokenisatie: 0.2-0.5 sec
  - Embedding: 3-5 sec
  - Indexering: 0.5-1 sec
  - **Totaal: 4-7 sec**

## Configuratie

Environment variables in `.env`:

```bash
# Chunk configuratie
CHUNK_SIZE=500           # Characters per chunk
CHUNK_OVERLAP=100        # Overlap tussen chunks

# Embedding model
OLLAMA_EMBED_MODEL=mxbai-embed-large

# Database
DB_USER=fatrag
DB_PASSWORD=fatrag_pw
DB_NAME=fatrag
```

## Toekomstige Verbeteringen

- [ ] WebSocket support voor real-time updates (geen polling)
- [ ] Auto-retry bij failures met exponential backoff
- [ ] Parallel processing van meerdere bestanden
- [ ] Resume capability bij server restart
- [ ] Detailed stage substeps (bijv. per 10% van embedding)
- [ ] Upload queue met prioritering
- [ ] Bandwidth throttling voor grote batches
- [ ] Storage quota management per project
- [ ] File type validation voor upload
- [ ] Virus scanning integratie

## Troubleshooting

### Upload hangt bij 50% (embedding stage)
- Check Ollama service status: `systemctl status ollama`
- Check Ollama logs: `journalctl -u ollama -f`
- Verify embedding model is loaded: `ollama list`

### Database errors
- Check MySQL connection: `mysql -u fatrag -pfatrag_pw fatrag`
- Verify tables exist: `SHOW TABLES LIKE 'upload%';`
- Check migrations: Ensure 004_upload_progress_tracking.sql is applied

### UI not updating
- Check browser console for errors
- Verify API endpoints are responding: `curl /admin/uploads/batch/{id}`
- Check refresh interval isn't stopped early

### High memory usage
- Reduce chunk_size to process smaller chunks
- Process files sequentially instead of batch
- Enable batch size limits in upload endpoint

## Compliance

- **No PII in logs**: Alle progress logs bevatten alleen filenames, geen content
- **Secrets in .env**: Database credentials nooit commiten
- **Network timeouts**: Alle API calls hebben 30s timeout
- **Error messages**: User-friendly, geen stack traces naar client
- **Persistence**: Job state persistent in database voor auditability
