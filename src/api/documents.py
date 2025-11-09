"""
Document management API endpoints for FATRAG.
"""
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import logging
import hashlib
import os
import uuid
from datetime import datetime

from src.core.database import get_db, Session
from src.core.config import settings
from src.models.database import Document, DocumentType, ProcessingStatus
from src.services.vector_db import milvus_service
from src.services.ollama_service import ollama_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_category: Optional[str] = Form(None),
    company_name: Optional[str] = Form(None),
    financial_year: Optional[int] = Form(None),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Upload a document for processing.
    
    Args:
        file: Uploaded file
        document_category: Optional document category
        company_name: Optional company name
        financial_year: Optional financial year
        db: Database session
        
    Returns:
        Dict: Upload status and document info
    """
    try:
        # Validate file type
        if file.content_type not in [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ]:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Only PDF, DOCX, and XLSX files are allowed."
            )
        
        # Validate file size (100MB default)
        max_size = 100 * 1024 * 1024  # 100MB in bytes
        file_content = await file.read()
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {max_size // (1024*1024)}MB."
            )
        
        # Generate unique filename and checksum
        file_checksum = hashlib.sha256(file_content).hexdigest()
        file_extension = os.path.splitext(file.filename)[1].lower()
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(settings.upload_dir, unique_filename)
        
        # Ensure upload directory exists
        os.makedirs(settings.upload_dir, exist_ok=True)
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Check if document already exists
        existing_doc = db.query(Document).filter(Document.checksum == file_checksum).first()
        if existing_doc:
            # Remove uploaded file since it's a duplicate
            os.remove(file_path)
            return {
                "status": "duplicate",
                "message": "Document already exists",
                "document_id": existing_doc.id,
                "filename": existing_doc.original_filename
            }
        
        # Determine document type
        if file_extension == ".pdf":
            doc_type = DocumentType.PDF
        elif file_extension == ".docx":
            doc_type = DocumentType.DOCX
        elif file_extension == ".xlsx":
            doc_type = DocumentType.XLSX
        else:
            doc_type = DocumentType.OTHER
        
        # Create document record
        document = Document(
            filename=unique_filename,
            original_filename=file.filename,
            file_path=file_path,
            file_size=len(file_content),
            file_type=doc_type,
            mime_type=file.content_type,
            checksum=file_checksum,
            processing_status=ProcessingStatus.PENDING,
            document_category=document_category,
            company_name=company_name,
            financial_year=financial_year
        )
        
        db.add(document)
        db.commit()
        db.refresh(document)
        
        # TODO: Start background processing job
        # This will be implemented in Phase 2
        
        logger.info(f"Document uploaded successfully: {file.filename} -> {document.id}")
        
        return {
            "status": "success",
            "message": "Document uploaded successfully",
            "document_id": document.id,
            "filename": file.filename,
            "file_size": len(file_content),
            "file_type": doc_type.value,
            "processing_status": document.processing_status.value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        # Clean up uploaded file if it exists
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail="Failed to upload document")


@router.get("/")
async def get_documents(
    skip: int = 0,
    limit: int = 20,
    category: Optional[str] = None,
    company: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get list of documents with filtering and pagination.
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        category: Filter by document category
        company: Filter by company name
        status: Filter by processing status
        db: Database session
        
    Returns:
        Dict: Documents list with pagination info
    """
    try:
        query = db.query(Document)
        
        if category:
            query = query.filter(Document.document_category == category)
        if company:
            query = query.filter(Document.company_name.ilike(f"%{company}%"))
        if status:
            query = query.filter(Document.processing_status == status)
        
        total = query.count()
        documents = query.offset(skip).limit(limit).order_by(Document.uploaded_at.desc()).all()
        
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "documents": [
                {
                    "id": doc.id,
                    "filename": doc.original_filename,
                    "file_type": doc.file_type.value,
                    "file_size": doc.file_size,
                    "processing_status": doc.processing_status.value,
                    "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
                    "processing_completed_at": doc.processing_completed_at.isoformat() if doc.processing_completed_at else None,
                    "document_category": doc.document_category,
                    "company_name": doc.company_name,
                    "financial_year": doc.financial_year,
                    "page_count": doc.page_count,
                    "word_count": doc.word_count
                }
                for doc in documents
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to get documents")


@router.get("/{document_id}")
async def get_document(
    document_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get detailed information about a specific document.
    
    Args:
        document_id: ID of the document
        db: Database session
        
    Returns:
        Dict: Document details
    """
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get chunks count
        chunks_count = len(document.chunks) if document.chunks else 0
        
        return {
            "id": document.id,
            "filename": document.original_filename,
            "file_type": document.file_type.value,
            "file_size": document.file_size,
            "mime_type": document.mime_type,
            "processing_status": document.processing_status.value,
            "processing_started_at": document.processing_started_at.isoformat() if document.processing_started_at else None,
            "processing_completed_at": document.processing_completed_at.isoformat() if document.processing_completed_at else None,
            "processing_error": document.processing_error,
            "retry_count": document.retry_count,
            "title": document.title,
            "author": document.author,
            "created_date": document.created_date.isoformat() if document.created_date else None,
            "modified_date": document.modified_date.isoformat() if document.modified_date else None,
            "page_count": document.page_count,
            "word_count": document.word_count,
            "document_category": document.document_category,
            "company_name": document.company_name,
            "financial_year": document.financial_year,
            "document_date": document.document_date.isoformat() if document.document_date else None,
            "uploaded_at": document.uploaded_at.isoformat() if document.uploaded_at else None,
            "updated_at": document.updated_at.isoformat() if document.updated_at else None,
            "chunks_count": chunks_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get document")


@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: int,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get chunks for a specific document.
    
    Args:
        document_id: ID of the document
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session
        
    Returns:
        Dict: Document chunks with pagination info
    """
    try:
        # Check if document exists
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get chunks
        query = db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id)
        total = query.count()
        chunks = query.offset(skip).limit(limit).order_by(DocumentChunk.chunk_index).all()
        
        return {
            "document_id": document_id,
            "total": total,
            "skip": skip,
            "limit": limit,
            "chunks": [
                {
                    "id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    "content_length": chunk.content_length,
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "chunk_type": chunk.chunk_type,
                    "contains_financial_data": chunk.contains_financial_data,
                    "created_at": chunk.created_at.isoformat() if chunk.created_at else None
                }
                for chunk in chunks
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chunks for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get document chunks")


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Delete a document and all its associated data.
    
    Args:
        document_id: ID of the document to delete
        db: Database session
        
    Returns:
        Dict: Deletion status
    """
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from Milvus
        try:
            collection_name = "document_chunks"
            milvus_service.delete_by_document_id(collection_name, document_id)
        except Exception as e:
            logger.warning(f"Failed to delete from Milvus: {e}")
        
        # Delete file from filesystem
        try:
            if os.path.exists(document.file_path):
                os.remove(document.file_path)
        except Exception as e:
            logger.warning(f"Failed to delete file: {e}")
        
        # Delete from database (cascades to chunks)
        db.delete(document)
        db.commit()
        
        logger.info(f"Document {document_id} deleted successfully")
        
        return {"status": "success", "message": f"Document {document_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete document")


@router.post("/{document_id}/reprocess")
async def reprocess_document(
    document_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Re-process a document (reset status and restart processing).
    
    Args:
        document_id: ID of the document to reprocess
        db: Database session
        
    Returns:
        Dict: Reprocessing status
    """
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Reset processing status
        document.processing_status = ProcessingStatus.PENDING
        document.processing_started_at = None
        document.processing_completed_at = None
        document.processing_error = None
        document.retry_count += 1
        
        db.commit()
        
        # TODO: Restart background processing job
        # This will be implemented in Phase 2
        
        logger.info(f"Document {document_id} queued for reprocessing")
        
        return {
            "status": "success",
            "message": f"Document {document_id} queued for reprocessing",
            "retry_count": document.retry_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing document {document_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to reprocess document")


@router.get("/categories/list")
async def get_document_categories(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get list of all document categories.
    
    Args:
        db: Database session
        
    Returns:
        Dict: Document categories
    """
    try:
        categories = db.query(Document.document_category).filter(
            Document.document_category.isnot(None)
        ).distinct().all()
        
        category_list = [cat[0] for cat in categories if cat[0]]
        
        return {
            "categories": sorted(category_list),
            "count": len(category_list)
        }
        
    except Exception as e:
        logger.error(f"Error getting document categories: {e}")
        raise HTTPException(status_code=500, detail="Failed to get document categories")


@router.get("/companies/list")
async def get_companies(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get list of all company names.
    
    Args:
        db: Database session
        
    Returns:
        Dict: Company names
    """
    try:
        companies = db.query(Document.company_name).filter(
            Document.company_name.isnot(None)
        ).distinct().all()
        
        company_list = [company[0] for company in companies if company[0]]
        
        return {
            "companies": sorted(company_list),
            "count": len(company_list)
        }
        
    except Exception as e:
        logger.error(f"Error getting companies: {e}")
        raise HTTPException(status_code=500, detail="Failed to get companies")
