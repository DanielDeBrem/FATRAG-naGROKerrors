"""Document upload and management endpoints."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
from app.models.dto.upload import UploadResponse
from app.services import DocumentIngestService

router = APIRouter(prefix="/documents", tags=["documents"])


def get_document_ingest_service() -> DocumentIngestService:
    """Dependency injection for DocumentIngestService."""
    return DocumentIngestService()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentIngestService = Depends(get_document_ingest_service)
) -> UploadResponse:
    """
    Upload and process a document.

    Supported file types: PDF, TXT, DOCX, MD, XLSX, XLS
    Maximum file size: 50MB
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        # Process the upload through the service
        result = await service.ingest_document(file)

        return result

    except ValueError as e:
        # Validation errors
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Other processing errors
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@router.get("/")
async def list_documents(
    service: DocumentIngestService = Depends(get_document_ingest_service)
) -> List[dict]:
    """
    List all uploaded documents (basic info).
    """
    try:
        # Get documents from repository
        documents = service.repository.list_documents()

        # Convert to response format
        return [
            {
                "id": doc.id,
                "filename": doc.filename,
                "mime_type": doc.mime_type,
                "file_size": doc.file_size,
                "status": doc.status,
                "document_type": doc.document_type,
                "created_at": doc.created_at.isoformat()
            }
            for doc in documents
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get("/{document_id}")
async def get_document(
    document_id: str,
    service: DocumentIngestService = Depends(get_document_ingest_service)
):
    """
    Get details of a specific document.
    """
    try:
        document = service.repository.get_document_by_id(document_id)

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "id": document.id,
            "filename": document.filename,
            "mime_type": document.mime_type,
            "file_size": document.file_size,
            "status": document.status,
            "document_type": document.document_type,
            "has_financial_data": document.has_financial_data,
            "currency": document.currency,
            "total_amount": document.total_amount,
            "storage_path": document.storage_path,
            "text_length": len(document.raw_text) if document.raw_text else 0,
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    service: DocumentIngestService = Depends(get_document_ingest_service)
):
    """
    Delete a document and its associated file.
    """
    try:
        success = service.repository.delete_document(document_id)

        if not success:
            raise HTTPException(status_code=404, detail="Document not found")

        return {"status": "deleted", "document_id": document_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
