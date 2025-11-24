"""Repository for document operations with MySQL."""

import os
from typing import List, Optional
from datetime import datetime
import pymysql
from app.models.domain import Document as DocumentModel
from app.core.config import get_settings


class DocumentsRepository:
    """Repository for document CRUD operations."""

    def __init__(self):
        self.settings = get_settings()

    def _get_connection(self):
        """Get database connection."""
        return pymysql.connect(
            host=self.settings.mysql_host,
            port=self.settings.mysql_port,
            user=self.settings.mysql_user,
            password=self.settings.mysql_password,
            database=self.settings.mysql_database,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

    def save_document(self, document: DocumentModel) -> DocumentModel:
        """Save a document to the database."""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                # Insert document
                sql = """
                    INSERT INTO documents
                    (doc_id, filename, source_type, file_path, file_size, status,
                     project_id, client_id, metadata, uploaded_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """

                # Prepare metadata JSON
                metadata = {
                    "document_type": document.document_type,
                    "has_financial_data": document.has_financial_data,
                    "currency": document.currency,
                    "total_amount": document.total_amount,
                } if (document.document_type or document.has_financial_data or document.currency or document.total_amount is not None) else None

                cursor.execute(sql, (
                    document.id,
                    document.filename,
                    "upload",  # Default source type for now
                    document.storage_path,
                    document.file_size,
                    document.status,
                    document.project_id,
                    document.client_id,
                    metadata,
                    document.created_at
                ))

                conn.commit()

        return document

    def get_document_by_id(self, document_id: str) -> Optional[DocumentModel]:
        """Get a document by ID."""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT doc_id, filename, source_type, file_path, file_size, status,
                           project_id, client_id, metadata, uploaded_at
                    FROM documents
                    WHERE doc_id = %s AND deleted_at IS NULL
                """, (document_id,))

                row = cursor.fetchone()
                if not row:
                    return None

                # Parse metadata
                metadata = row.get("metadata") or {}
                document_type = metadata.get("document_type") if metadata else None
                has_financial_data = metadata.get("has_financial_data", False) if metadata else False
                currency = metadata.get("currency") if metadata else None
                total_amount = metadata.get("total_amount") if metadata else None

                return DocumentModel(
                    id=row["doc_id"],
                    filename=row["filename"],
                    mime_type="application/octet-stream",  # TODO: determine from file
                    file_size=row["file_size"],
                    raw_text=None,  # TODO: load from file if needed
                    storage_path=row["file_path"],
                    project_id=row.get("project_id"),
                    client_id=row.get("client_id"),
                    created_at=row["uploaded_at"],
                    updated_at=row["uploaded_at"],  # TODO: track updates
                    document_type=document_type,
                    has_financial_data=has_financial_data,
                    currency=currency,
                    total_amount=total_amount,
                    status=row["status"]
                )

    def update_document_status(self, document_id: str, status: str, error_message: Optional[str] = None) -> bool:
        """Update document status."""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE documents
                    SET status = %s, error_message = %s, updated_at = NOW()
                    WHERE doc_id = %s
                """, (status, error_message, document_id))

                conn.commit()
                return cursor.rowcount > 0

    def mark_indexed(self, document_id: str, chunk_count: int) -> bool:
        """Mark a document as indexed and store chunk_count + indexed_at."""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE documents
                    SET status = %s,
                        chunk_count = %s,
                        indexed_at = NOW(),
                        updated_at = NOW()
                    WHERE doc_id = %s
                """, ("indexed", chunk_count, document_id))

                conn.commit()
                return cursor.rowcount > 0

    def list_documents(self, project_id: Optional[str] = None, limit: int = 100) -> List[DocumentModel]:
        """List documents, optionally filtered by project."""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                if project_id:
                    cursor.execute("""
                        SELECT doc_id, filename, source_type, file_path, file_size, status,
                               project_id, client_id, metadata, uploaded_at
                        FROM documents
                        WHERE project_id = %s AND deleted_at IS NULL
                        ORDER BY uploaded_at DESC
                        LIMIT %s
                    """, (project_id, limit))
                else:
                    cursor.execute("""
                        SELECT doc_id, filename, source_type, file_path, file_size, status,
                               project_id, client_id, metadata, uploaded_at
                        FROM documents
                        WHERE deleted_at IS NULL
                        ORDER BY uploaded_at DESC
                        LIMIT %s
                    """, (limit,))

                rows = cursor.fetchall()

        documents = []
        for row in rows:
            metadata = row.get("metadata") or {}
            documents.append(DocumentModel(
                id=row["doc_id"],
                filename=row["filename"],
                mime_type="application/octet-stream",  # TODO: determine from file
                file_size=row["file_size"],
                storage_path=row["file_path"],
                project_id=row.get("project_id"),
                client_id=row.get("client_id"),
                created_at=row["uploaded_at"],
                status=row["status"],
                document_type=metadata.get("document_type"),
                has_financial_data=metadata.get("has_financial_data", False),
                currency=metadata.get("currency"),
                total_amount=metadata.get("total_amount")
            ))

        return documents

    def delete_document(self, document_id: str) -> bool:
        """Soft delete a document."""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE documents
                    SET deleted_at = NOW()
                    WHERE doc_id = %s AND deleted_at IS NULL
                """, (document_id,))

                conn.commit()

                # Also remove physical file if it exists
                document = self.get_document_by_id(document_id)
                if document and document.storage_path and os.path.isfile(document.storage_path):
                    try:
                        os.remove(document.storage_path)
                    except OSError:
                        pass  # Ignore file deletion errors

                return cursor.rowcount > 0
