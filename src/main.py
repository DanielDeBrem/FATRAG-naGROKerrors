"""
Main FastAPI application for FATRAG.
"""
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import logging
import time
import uvicorn

from src.core.config import settings
from src.core.database import init_database, db_manager
from src.services.vector_db import milvus_service
from src.services.ollama_service import ollama_service
from src.api import admin, documents, queries, health

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting FATRAG application...")
    
    try:
        # Initialize database
        logger.info("Initializing database...")
        init_database()
        
        # Connect to Milvus
        logger.info("Connecting to Milvus...")
        milvus_connected = milvus_service.connect()
        if not milvus_connected:
            logger.warning("Failed to connect to Milvus, but continuing...")
        
        # Check Ollama connection
        logger.info("Checking Ollama connection...")
        ollama_connected = await ollama_service.check_connection()
        if not ollama_connected:
            logger.warning("Failed to connect to Ollama, but continuing...")
        
        logger.info("FATRAG application started successfully!")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down FATRAG application...")
    
    try:
        # Disconnect from Milvus
        milvus_service.disconnect()
        logger.info("Disconnected from Milvus")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Financial Advisory Tool using RAG architecture",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")


# Include API routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(admin.router, prefix="/admin", tags=["Admin"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(queries.router, prefix="/api/queries", tags=["Queries"])


# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Root endpoint serving the main interface."""
    return templates.TemplateResponse("index.html", {"request": request})


# Admin interface endpoint
@app.get("/admin", response_class=HTMLResponse)
async def admin_interface(request: Request):
    """Admin interface endpoint."""
    return templates.TemplateResponse("admin.html", {"request": request})


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return HTTPException(
        status_code=500,
        detail="Internal server error"
    )


# Health check endpoint (alternative to /health)
@app.get("/ping")
async def ping():
    """Simple ping endpoint for basic health check."""
    return {"status": "ok", "message": "FATRAG is running"}


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
        log_level=settings.log_level.lower()
    )
