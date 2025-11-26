"""Service for document ingestion and processing."""

import os
import hashlib
import mimetypes
import uuid
from typing import Optional

from fastapi import UploadFile

from app.models.domain import Document as DocumentModel
from app.models.dto.upload import UploadResponse
from app.repositories.documents import DocumentsRepository
from app.core.config import get_settings
from app.services.document_classifier import DocumentClassifierService
from app.services.document_index import DocumentIndexService

# Import text extraction functions from existing ingestion module
from ingestion import (
    read_text_file,
    read_pdf_file,
    read_excel_file,
    clean_extracted_text,
    ensure_dirs,
)
from app.services.document_extractors import extract_for_db


class DocumentIngestService:
    """Service for processing document uploads."""

    def __init__(self):
        self.settings = get_settings()
        self.repository = DocumentsRepository()
        self.classifier = DocumentClassifierService()
        self.indexer = DocumentIndexService()

    def _validate_file(self, file: UploadFile) -> None:
        """Validate uploaded file."""
        # Check file size
        file_content = file.file.read()
        file.file.seek(0)  # Reset file pointer
        file_size = len(file_content)

        if file_size > self.settings.max_upload_size:
            raise ValueError(
                f"File too large: {file_size} bytes (max {self.settings.max_upload_size})"
            )

        # Check MIME type
        detected_mime = (
            mimetypes.guess_type(file.filename)[0] if file.filename else None
        )
        content_mime = (
            mimetypes.guess_type(file.filename)[0] if file.filename else None
        )

        # Allow detected MIME or fall back to filename extension check
        allowed_extensions = [".pdf", ".txt", ".docx", ".md", ".xlsx", ".xls"]
        file_extension = os.path.splitext(file.filename or "")[1].lower()

        if file_extension not in allowed_extensions:
            if detected_mime not in self.settings.allowed_file_types:
                raise ValueError(
                    f"Unsupported file type: {detected_mime} (extension: {file_extension})"
                )

        # Check for empty files
        if file_size == 0:
            raise ValueError("Empty file")

    def _save_file_to_disk(self, file: UploadFile, document_id: str) -> str:
        """Save uploaded file to disk and return the file path."""
        ensure_dirs()

        # Create filename with document ID prefix for uniqueness
        original_filename = file.filename or "unnamed"
        name, ext = os.path.splitext(original_filename)
        safe_filename = f"{document_id}_{name}{ext}"

        # Ensure upload directory exists
        os.makedirs(self.settings.upload_directory, exist_ok=True)

        file_path = os.path.join(self.settings.upload_directory, safe_filename)

        # Save file
        with open(file_path, "wb") as buffer:
            content = file.file.read()
            buffer.write(content)

        file.file.seek(0)  # Reset for potential re-reading

        return file_path

    def _extract_text_from_file(self, file_path: str) -> str:
        """Extract text content from file based on extension."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == ".pdf":
                raw_text = read_pdf_file(file_path)
            elif ext in [".xlsx", ".xls"]:
                raw_text = read_excel_file(file_path)
            elif ext in [".txt", ".md"]:
                raw_text = read_text_file(file_path)
            elif ext == ".docx":
                # For now, treat as binary - will need python-docx
                raw_text = "[DOCX file detected - text extraction pending implementation]"
            else:
                raw_text = "[Unsupported file type for text extraction]"

            return clean_extracted_text(raw_text) if raw_text else ""

        except Exception as e:
            return f"[Text extraction error: {str(e)}]"

    def _determine_mime_type(self, filename: str, file_path: str) -> str:
        """Determine MIME type of file."""
        # Try to detect from file content first
        mime_type, _ = mimetypes.guess_type(filename)

        if not mime_type and os.path.isfile(file_path):
            # Fallback: check file extension
            ext = os.path.splitext(filename)[1].lower()
            mime_map = {
                ".pdf": "application/pdf",
                ".txt": "text/plain",
                ".md": "text/markdown",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ".xls": "application/vnd.ms-excel",
            }
            mime_type = mime_map.get(ext, "application/octet-stream")

        return mime_type or "application/octet-stream"

    async def ingest_document(
        self,
        file: UploadFile,
        project_id: Optional[str] = None,
        client_id: Optional[str] = None,
        source_type: str = "upload",
    ) -> UploadResponse:
        """Process and ingest a document upload.

        Optionally attaches project_id/client_id and a specific source_type
        (e.g. 'upload', 'project_upload', 'llm_analysis').
        """

        # Validate file
        self._validate_file(file)

        # Generate document ID (unique, niet alleen op bestandsnaam gebaseerd)
        # Gebruik een random UUID-fragment om collisions bij herhaalde uploads te voorkomen.
        document_id = f"doc-{uuid.uuid4().hex[:16]}"

        try:
            # Save file to disk
            file_path = self._save_file_to_disk(file, document_id)

            # Get initial file size from disk
            file_size = os.path.getsize(file_path)

            # Determine MIME type
            mime_type = self._determine_mime_type(file.filename, file_path)

            # Normalization / extraction (can be expensive for large PDFs)
            raw_text: Optional[str] = None
            status = "uploaded"

            # Only attempt full text extraction for files < 10MB for now
            if file_size < 10 * 1024 * 1024:
                try:
                    # Use shared normalization service (Fase B)
                    normalized_text, normalized_size = extract_for_db(file_path)
                    raw_text = normalized_text
                    # Prefer normalized_size if available, otherwise keep original size
                    file_size = normalized_size or file_size
                    status = "normalized"
                except Exception:
                    # Fallback: best-effort extraction via legacy helper
                    raw_text = self._extract_text_from_file(file_path)
                    # Ook in de fallback beschouwen we dit als een genormaliseerde representatie
                    status = "normalized"

            # Optional classificatie (Fase C): alleen als we tekst hebben
            document_type: Optional[str] = None
            if raw_text:
                try:
                    cls = self.classifier.classify(raw_text, file.filename or "")
                    document_type = cls.document_type
                except Exception:
                    document_type = None

            # Create domain model
            document = DocumentModel(
                id=document_id,
                filename=file.filename,
                source_type=source_type,
                mime_type=mime_type,
                file_size=file_size,
                raw_text=raw_text,
                storage_path=file_path,
                project_id=project_id,
                client_id=client_id,
                status=status,
                document_type=document_type,
            )

            # Save to repository
            saved_document = self.repository.save_document(document)

            # Index in vectorstore (Fase D) – best effort, faalt niet de upload
            try:
                if saved_document.raw_text:
                    self.indexer.index_document(saved_document)
            except Exception:
                # Indexering mag ingest niet doen falen
                pass

            return UploadResponse(
                document_id=saved_document.id,
                filename=saved_document.filename,
                mime_type=saved_document.mime_type,
                file_size=saved_document.file_size,
                status=saved_document.status,
                message="Document uploaded and processed successfully",
            )

        except Exception as e:
            # If anything fails, try to clean up the file
            try:
                if "file_path" in locals() and os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception:
                pass

            # Update status if document was created
            if "document_id" in locals():
                self.repository.update_document_status(document_id, "failed", str(e))

            raise
