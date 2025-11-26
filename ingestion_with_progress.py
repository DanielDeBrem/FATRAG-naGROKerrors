"""
Enhanced ingestion module with live progress tracking
Extends ingestion.py with callback support for progress reporting
"""

import os

# Disable ChromaDB telemetry to prevent PostHog errors
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY'] = 'False'
import time
from typing import List, Dict, Any, Optional, Callable
from langchain_community.vectorstores import Chroma
import ingestion as ing
from upload_progress_store import UploadProgressTracker, create_progress_callback
from app.services.embeddings import EmbeddingsService


def ingest_file_with_progress(
    file_path: str,
    upload_id: str,
    tracker: UploadProgressTracker,
    user: Optional[str] = None,
    kind: str = "upload",
    extra_metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> Dict[str, Any]:
    """
    Ingest a single file with detailed progress tracking through all stages.
    
    Stages:
    1. File upload (0-10%)
    2. Text extraction (10-30%)
    3. Tokenization/chunking (30-50%)
    4. Embedding generation (50-80%)
    5. Vector indexing (80-100%)
    """
    
    filename = os.path.basename(file_path)
    callback = create_progress_callback(tracker, upload_id)
    vectorstore = EmbeddingsService.raw_vectorstore()
    
    try:
        # Stage 1: File uploaded (already done by caller)
        callback("uploaded", 10)
        
        # Stage 2: Text extraction
        callback("extracting", 15)
        ext = os.path.splitext(filename)[1].lower()
        
        extraction_start = time.time()
        
        if ext == ".pdf":
            text = ing.read_pdf_file(file_path)
        elif ext in [".xlsx", ".xls"]:
            text = ing.read_excel_file(file_path)
        else:
            text = ing.read_text_file(file_path)
        
        extraction_time_ms = int((time.time() - extraction_start) * 1000)
        
        if not text or text.startswith("[") and "error" in text.lower():
            callback("failed", 0, error_message=text or "Empty file", error_stage="extraction")
            return {
                "status": "failed",
                "filename": filename,
                "error": text or "Empty file"
            }
        
        callback("extracting", 30, extraction_time_ms=extraction_time_ms)
        
        # Stage 3: Tokenization/chunking
        callback("tokenizing", 35)
        
        tokenization_start = time.time()
        chunks = ing.chunk_texts([text], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        tokenization_time_ms = int((time.time() - tokenization_start) * 1000)
        
        # Estimate total tokens (rough approximation: 1 token ≈ 4 chars)
        total_tokens = sum(len(chunk) // 4 for chunk in chunks)
        
        callback("tokenizing", 50, 
                total_chunks=len(chunks), 
                total_tokens=total_tokens,
                tokenization_time_ms=tokenization_time_ms)
        
        # Stage 4: Embedding generation
        callback("embedding", 55)
        
        embedding_start = time.time()
        
        # Prepare metadata
        meta = {
            "source": filename,
            "doc_id": filename,
            "kind": kind,
            "uploaded_by": user or "admin",
            "path": file_path,
            "file_type": ext[1:],
        }
        if extra_metadata:
            meta.update(extra_metadata)
        
        # Add chunks with embeddings (this is the slowest part)
        # Update progress incrementally during embedding
        batch_size = max(1, len(chunks) // 10)  # 10 progress updates
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            vectorstore.add_texts(texts=batch, metadatas=[meta] * len(batch))
            
            # Update progress: 55% -> 80%
            progress = 55 + int((i + len(batch)) / len(chunks) * 25)
            callback("embedding", progress, total_chunks=len(chunks))
        
        embedding_time_ms = int((time.time() - embedding_start) * 1000)
        
        callback("embedding", 80, 
                embedding_time_ms=embedding_time_ms,
                embedding_dimensions=768)  # gemma2:2b default
        
        # Stage 5: Indexing (persist to disk)
        callback("indexing", 85)
        
        indexing_start = time.time()
        vectorstore.persist()
        indexing_time_ms = int((time.time() - indexing_start) * 1000)
        
        callback("indexing", 95, indexing_time_ms=indexing_time_ms)
        
        # Complete
        callback("completed", 100, 
                doc_id=filename,
                total_chunks=len(chunks),
                total_tokens=total_tokens)
        
        return {
            "status": "completed",
            "filename": filename,
            "chunks": len(chunks),
            "tokens": total_tokens,
            "extraction_time_ms": extraction_time_ms,
            "tokenization_time_ms": tokenization_time_ms,
            "embedding_time_ms": embedding_time_ms,
            "indexing_time_ms": indexing_time_ms,
        }
    
    except Exception as e:
        error_msg = str(e)
        callback("failed", 0, error_message=error_msg, error_stage="unknown")
        return {
            "status": "failed",
            "filename": filename,
            "error": error_msg
        }


def ingest_files_batch_with_progress(
    file_paths: List[str],
    batch_id: str,
    project_id: str,
    client_id: Optional[str] = None,
    user: Optional[str] = None,
    kind: str = "upload",
    extra_metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> Dict[str, Any]:
    """
    Ingest multiple files with batch progress tracking.
    Each file gets its own upload_progress record.
    """
    
    tracker = UploadProgressTracker()
    
    # Create batch record
    tracker.create_batch(batch_id, project_id, len(file_paths))
    tracker.update_batch(batch_id, status="processing")
    
    results = []
    completed = 0
    failed = 0
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
        
        # Create upload progress record
        upload_id = f"upload-{int(time.time() * 1000)}-{filename}"
        tracker.create_upload(
            upload_id=upload_id,
            filename=filename,
            project_id=project_id,
            client_id=client_id,
            file_size=file_size,
            file_path=file_path,
        )
        
        # Ingest with progress tracking
        result = ingest_file_with_progress(
            file_path=file_path,
            upload_id=upload_id,
            tracker=tracker,
            user=user,
            kind=kind,
            extra_metadata=extra_metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        results.append(result)
        
        if result.get("status") == "completed":
            completed += 1
        else:
            failed += 1
        
        # Update batch progress
        tracker.update_batch(
            batch_id,
            completed_files=completed,
            failed_files=failed,
        )
    
    # Finalize batch
    if failed == 0:
        final_status = "completed"
    elif completed == 0:
        final_status = "failed"
    else:
        final_status = "partial_failure"
    
    tracker.update_batch(batch_id, status=final_status)
    
    return {
        "batch_id": batch_id,
        "status": final_status,
        "total_files": len(file_paths),
        "completed": completed,
        "failed": failed,
        "results": results,
    }
