"""Domain models for documents and related entities."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Domain model representing a stored document.

    This model is used across:
    - app.services.document_ingest.DocumentIngestService
    - app.repositories.documents.DocumentsRepository
    - app.api.documents endpoints

    It intentionally matches the fields read/written in the `documents` table:
    - doc_id
    - filename
    - file_path
    - file_size
    - status
    - metadata (JSON with document_type, has_financial_data, currency, total_amount)
    - uploaded_at / updated_at
    - error_message (optional)
    """

    id: str
    filename: str
    # Bron in de DB-kolom source_type (bijv. upload, project_upload, llm_analysis, flash_analysis)
    source_type: str = "upload"
    mime_type: str = "application/octet-stream"
    file_size: int = 0

    # Raw extracted text; may be None for very large files or unsupported types
    raw_text: Optional[str] = None

    # Filesystem location
    storage_path: Optional[str] = None

    # Logical ownership / scoping
    project_id: Optional[str] = None
    client_id: Optional[str] = None

    # Lifecycle/status in the ingestion pipeline
    # Typical values: uploaded, normalized, classified, indexed, analyzed, failed
    status: str = "uploaded"
    error_message: Optional[str] = None

    # Optional semantic/financial metadata (stored in documents.metadata JSON)
    document_type: Optional[str] = None
    has_financial_data: bool = False
    currency: Optional[str] = None
    total_amount: Optional[float] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        # Allow constructing from ORM/DB rows if needed
        from_attributes = True


class DocumentChunk(BaseModel):
    """Logical chunk of a document for vector search / analysis.

    Not all parts of the system use this yet, but it is exposed via
    `app.models.__all__` and can be used by future refactors or services.
    """

    id: str
    document_id: str
    index: int  # 0-based chunk index within the document
    text: str

    # Optional embedding vector and arbitrary metadata
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        from_attributes = True
