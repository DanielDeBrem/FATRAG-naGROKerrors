"""
Database connection and session management for FATRAG.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator
import logging

from src.core.config import settings
from src.models.database import Base

logger = logging.getLogger(__name__)

# Create database engine
engine = create_engine(
    settings.database_url,
    poolclass=StaticPool,
    pool_pre_ping=True,
    echo=settings.app_debug,
    connect_args={
        "charset": "utf8mb4",
        "use_unicode": True,
    } if "mysql" in settings.database_url else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """Create all database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


def get_db() -> Generator[Session, None, None]:
    """
    Get database session.
    
    Yields:
        Session: Database session
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def init_database():
    """Initialize database with required data."""
    try:
        # Create tables
        create_tables()
        
        # Add initial system configuration if needed
        from src.models.database import SystemConfig
        
        db = SessionLocal()
        try:
            # Check if system is already initialized
            existing_config = db.query(SystemConfig).filter(
                SystemConfig.key == "system_initialized"
            ).first()
            
            if not existing_config:
                # Add initial system configuration
                initial_configs = [
                    SystemConfig(
                        key="system_initialized",
                        value=True,
                        description="Flag indicating system initialization status"
                    ),
                    SystemConfig(
                        key="default_embedding_model",
                        value=settings.ollama_embedding_model,
                        description="Default model for text embeddings"
                    ),
                    SystemConfig(
                        key="default_generation_model",
                        value=settings.ollama_generation_model,
                        description="Default model for text generation"
                    ),
                    SystemConfig(
                        key="max_file_size_mb",
                        value=100,
                        description="Maximum file size for uploads in MB"
                    ),
                    SystemConfig(
                        key="chunk_size",
                        value=1000,
                        description="Default chunk size for document processing"
                    ),
                    SystemConfig(
                        key="chunk_overlap",
                        value=200,
                        description="Default overlap between chunks"
                    )
                ]
                
                for config in initial_configs:
                    db.add(config)
                
                db.commit()
                logger.info("Database initialized with default configuration")
            else:
                logger.info("Database already initialized")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            db.rollback()
            raise
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def check_database_connection() -> bool:
    """
    Check database connection health.
    
    Returns:
        bool: True if connection is healthy
    """
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


class DatabaseManager:
    """Database management utilities."""
    
    @staticmethod
    def get_session() -> Session:
        """Get a new database session."""
        return SessionLocal()
    
    @staticmethod
    def close_session(session: Session):
        """Close database session."""
        try:
            session.close()
        except Exception as e:
            logger.error(f"Error closing database session: {e}")
    
    @staticmethod
    def health_check() -> dict:
        """
        Perform database health check.
        
        Returns:
            dict: Health check results
        """
        try:
            db = SessionLocal()
            
            # Test basic connectivity
            result = db.execute("SELECT 1 as health_check").fetchone()
            connection_ok = result[0] == 1 if result else False
            
            # Test table access
            table_count = len(Base.metadata.tables)
            
            db.close()
            
            return {
                "status": "healthy" if connection_ok else "unhealthy",
                "connection": connection_ok,
                "tables": table_count,
                "database_url": settings.database_url.split("@")[-1] if "@" in settings.database_url else "unknown"
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "connection": False,
                "error": str(e),
                "database_url": settings.database_url.split("@")[-1] if "@" in settings.database_url else "unknown"
            }


# Global database manager instance
db_manager = DatabaseManager()
