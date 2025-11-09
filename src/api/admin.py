"""
Admin API endpoints for FATRAG.
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse
from typing import Dict, Any, List, Optional
import logging

from src.core.database import get_db, Session
from src.models.database import Document, Query, Job, SystemConfig, User
from src.services.vector_db import milvus_service
from src.services.ollama_service import ollama_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def admin_dashboard():
    """
    Admin dashboard endpoint.
    
    Returns:
        HTMLResponse: Admin dashboard HTML
    """
    # This will be handled by the main app's template rendering
    # The endpoint is here for routing completeness
    return {"message": "Admin dashboard - see templates/admin.html"}


@router.get("/stats")
async def get_system_stats(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get system statistics.
    
    Args:
        db: Database session
        
    Returns:
        Dict: System statistics
    """
    try:
        # Document statistics
        total_documents = db.query(Document).count()
        completed_documents = db.query(Document).filter(
            Document.processing_status == "completed"
        ).count()
        failed_documents = db.query(Document).filter(
            Document.processing_status == "failed"
        ).count()
        
        # Query statistics
        total_queries = db.query(Query).count()
        
        # Job statistics
        total_jobs = db.query(Job).count()
        running_jobs = db.query(Job).filter(Job.status == "running").count()
        
        # User statistics
        total_users = db.query(User).count()
        
        # Vector database statistics
        milvus_stats = milvus_service.health_check()
        
        # Ollama statistics
        ollama_stats = await ollama_service.health_check()
        
        return {
            "documents": {
                "total": total_documents,
                "completed": completed_documents,
                "failed": failed_documents,
                "pending": total_documents - completed_documents - failed_documents
            },
            "queries": {
                "total": total_queries
            },
            "jobs": {
                "total": total_jobs,
                "running": running_jobs
            },
            "users": {
                "total": total_users
            },
            "services": {
                "milvus": milvus_stats,
                "ollama": ollama_stats
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system statistics")


@router.get("/documents")
async def get_documents(
    skip: int = 0,
    limit: int = 50,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get list of documents with pagination.
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        status: Filter by processing status
        db: Database session
        
    Returns:
        Dict: Documents list with pagination info
    """
    try:
        query = db.query(Document)
        
        if status:
            query = query.filter(Document.processing_status == status)
        
        total = query.count()
        documents = query.offset(skip).limit(limit).all()
        
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
                    "company_name": doc.company_name
                }
                for doc in documents
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to get documents")


@router.get("/queries")
async def get_queries(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get list of queries with pagination.
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session
        
    Returns:
        Dict: Queries list with pagination info
    """
    try:
        total = db.query(Query).count()
        queries = db.query(Query).offset(skip).limit(limit).order_by(Query.created_at.desc()).all()
        
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "queries": [
                {
                    "id": query.id,
                    "session_id": query.session_id,
                    "query_text": query.query_text[:100] + "..." if len(query.query_text) > 100 else query.query_text,
                    "query_type": query.query_type,
                    "response_time_ms": query.response_time_ms,
                    "confidence_score": query.confidence_score,
                    "user_rating": query.user_rating,
                    "created_at": query.created_at.isoformat() if query.created_at else None
                }
                for query in queries
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting queries: {e}")
        raise HTTPException(status_code=500, detail="Failed to get queries")


@router.get("/jobs")
async def get_jobs(
    skip: int = 0,
    limit: int = 50,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get list of background jobs with pagination.
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        status: Filter by job status
        db: Database session
        
    Returns:
        Dict: Jobs list with pagination info
    """
    try:
        query = db.query(Job)
        
        if status:
            query = query.filter(Job.status == status)
        
        total = query.count()
        jobs = query.offset(skip).limit(limit).order_by(Job.created_at.desc()).all()
        
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "jobs": [
                {
                    "id": job.id,
                    "job_id": job.job_id,
                    "job_type": job.job_type,
                    "status": job.status,
                    "progress": job.progress,
                    "total_items": job.total_items,
                    "processed_items": job.processed_items,
                    "created_at": job.created_at.isoformat() if job.created_at else None,
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "error_message": job.error_message
                }
                for job in jobs
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting jobs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get jobs")


@router.get("/config")
async def get_system_config(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get system configuration.
    
    Args:
        db: Database session
        
    Returns:
        Dict: System configuration
    """
    try:
        configs = db.query(SystemConfig).all()
        
        return {
            "config": [
                {
                    "key": config.key,
                    "value": config.value,
                    "description": config.description,
                    "updated_at": config.updated_at.isoformat() if config.updated_at else None
                }
                for config in configs
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting system config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system configuration")


@router.post("/config")
async def update_system_config(
    config_updates: Dict[str, Any],
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Update system configuration.
    
    Args:
        config_updates: Configuration updates
        db: Database session
        
    Returns:
        Dict: Update status
    """
    try:
        for key, value in config_updates.items():
            config = db.query(SystemConfig).filter(SystemConfig.key == key).first()
            if config:
                config.value = value
                config.updated_at = "2025-01-09T11:07:10Z"  # Current time
            else:
                new_config = SystemConfig(
                    key=key,
                    value=value,
                    description=f"Updated via API"
                )
                db.add(new_config)
        
        db.commit()
        
        return {"status": "success", "message": "Configuration updated"}
        
    except Exception as e:
        logger.error(f"Error updating system config: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update system configuration")


@router.get("/models")
async def get_available_models() -> Dict[str, Any]:
    """
    Get available models from Ollama.
    
    Returns:
        Dict: Available models
    """
    try:
        models = await ollama_service.list_models()
        
        return {
            "models": models,
            "count": len(models)
        }
        
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        raise HTTPException(status_code=500, detail="Failed to get available models")


@router.post("/models/pull")
async def pull_model(model_name: str) -> Dict[str, str]:
    """
    Pull a model from Ollama registry.
    
    Args:
        model_name: Name of the model to pull
        
    Returns:
        Dict: Pull status
    """
    try:
        success = await ollama_service.pull_model(model_name)
        
        if success:
            return {"status": "success", "message": f"Model {model_name} pulled successfully"}
        else:
            return {"status": "error", "message": f"Failed to pull model {model_name}"}
        
    except Exception as e:
        logger.error(f"Error pulling model {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to pull model")


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Delete a document and its associated data.
    
    Args:
        document_id: ID of the document to delete
        db: Database session
        
    Returns:
        Dict: Deletion status
    """
    try:
        # Get document
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from Milvus
        collection_name = "document_chunks"  # Default collection name
        milvus_service.delete_by_document_id(collection_name, document_id)
        
        # Delete from database (cascades to chunks)
        db.delete(document)
        db.commit()
        
        return {"status": "success", "message": f"Document {document_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete document")
