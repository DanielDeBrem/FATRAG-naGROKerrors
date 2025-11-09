"""
Health check API endpoints for FATRAG.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

from src.core.database import db_manager
from src.services.vector_db import milvus_service
from src.services.ollama_service import ollama_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check endpoint.
    
    Returns:
        Dict: Health status of all system components
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": "2025-01-09T11:06:50Z",  # Using current time from context
            "version": "1.0.0",
            "components": {}
        }
        
        # Check database health
        try:
            db_health = db_manager.health_check()
            health_status["components"]["database"] = db_health
        except Exception as e:
            health_status["components"]["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check Milvus health
        try:
            milvus_health = milvus_service.health_check()
            health_status["components"]["milvus"] = milvus_health
            if milvus_health.get("status") != "healthy":
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["components"]["milvus"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check Ollama health
        try:
            ollama_health = await ollama_service.health_check()
            health_status["components"]["ollama"] = ollama_health
            if ollama_health.get("status") != "healthy":
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["components"]["ollama"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/database")
async def database_health() -> Dict[str, Any]:
    """
    Database health check endpoint.
    
    Returns:
        Dict: Database health status
    """
    try:
        return db_manager.health_check()
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        raise HTTPException(status_code=500, detail="Database health check failed")


@router.get("/milvus")
async def milvus_health() -> Dict[str, Any]:
    """
    Milvus health check endpoint.
    
    Returns:
        Dict: Milvus health status
    """
    try:
        return milvus_service.health_check()
    except Exception as e:
        logger.error(f"Milvus health check failed: {e}")
        raise HTTPException(status_code=500, detail="Milvus health check failed")


@router.get("/ollama")
async def ollama_health() -> Dict[str, Any]:
    """
    Ollama health check endpoint.
    
    Returns:
        Dict: Ollama health status
    """
    try:
        return await ollama_service.health_check()
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        raise HTTPException(status_code=500, detail="Ollama health check failed")


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check endpoint (for Kubernetes/liveness probes).
    
    Returns:
        Dict: Readiness status
    """
    try:
        # Check if critical components are ready
        db_healthy = db_manager.health_check().get("status") == "healthy"
        
        # For readiness, we require database to be healthy
        # Other services can be optional or degraded
        ready = db_healthy
        
        return {
            "ready": ready,
            "database_healthy": db_healthy
        }
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {
            "ready": False,
            "error": str(e)
        }


@router.get("/live")
async def liveness_check() -> Dict[str, str]:
    """
    Liveness check endpoint (for Kubernetes/liveness probes).
    
    Returns:
        Dict: Liveness status
    """
    return {
        "alive": "true",
        "message": "FATRAG is alive"
    }
