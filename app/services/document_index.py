"""Document indexing service (Fase D).

Verantwoordelijkheden:
- Tekst van een Document chunken
- Chunks met rijke metadata in de vectorstore plaatsen via EmbeddingsService
- De bijbehorende document-row in MySQL op 'indexed' zetten met chunk_count/indexed_at
"""

from __future__ import annotations

from typing import Optional

import ingestion as ing
from app.models.domain import Document as DocumentModel
from app.repositories.documents import DocumentsRepository
from app.services.embeddings import EmbeddingsService
from app.core.config import get_settings


class DocumentIndexService:
    """Indexeer Documenten in de vectorstore op een consistente manier."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.repo = DocumentsRepository()

        # Chunk-instellingen (later uit config.json / admin-UI configureerbaar)
        # Val terug op defaults als niet aanwezig
        self.chunk_size = getattr(self.settings, "chunk_size", 500)
        self.chunk_overlap = getattr(self.settings, "chunk_overlap", 100)

    def index_document(self, document: DocumentModel) -> int:
        """Indexeer één document.

        - Verwacht dat het document al in de DB is opgeslagen.
        - Gebruikt document.raw_text als bron; als die ontbreekt, wordt niets gedaan.
        - Schrijft chunks naar de vectorstore met metadata (doc_id, project_id, client_id, document_type).
        - Zet in de DB: status='indexed', chunk_count en indexed_at.
        """
        if not document or not document.id:
            return 0

        # Zonder tekst heeft indexeren geen zin
        if not document.raw_text:
            return 0

        # Chunk de tekst
        chunks = ing.chunk_texts(
            [document.raw_text],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if not chunks:
            return 0

        # Extra metadata voor vectorstore
        extra_meta = {"filename": document.filename}

        # Voeg chunks toe aan vectorstore
        n_chunks = EmbeddingsService.add_document_chunks(
            doc_id=document.id,
            project_id=getattr(document, "project_id", None),
            client_id=getattr(document, "client_id", None),
            document_type=getattr(document, "document_type", None),
            source_type="project_upload",  # voor generieke ingest kan dit later parametriseerbaar worden
            chunks=chunks,
            extra_metadata=extra_meta,
            persist=True,
        )

        # Markeer in DB als geïndexeerd
        if n_chunks > 0:
            self.repo.mark_indexed(document.id, n_chunks)

        return n_chunks
