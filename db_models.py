"""
SQLAlchemy database models for FATRAG
Maps to MySQL schema defined in schema.sql
"""
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, Boolean, JSON, TIMESTAMP, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
import os

Base = declarative_base()

# Database connection configuration
DB_USER = os.getenv("DB_USER", "fatrag")
DB_PASSWORD = os.getenv("DB_PASSWORD", "fatrag_pw")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "fatrag")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create engine and session
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency for FastAPI to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String(128), unique=True, nullable=False, index=True)
    filename = Column(String(512), nullable=False)
    source_type = Column(String(64), nullable=False, index=True)
    file_path = Column(String(1024))
    file_size = Column(Integer, default=0)
    checksum = Column(String(128))
    chunk_count = Column(Integer, default=0)
    status = Column(String(32), default="pending", index=True)
    error_message = Column(Text)
    meta_data = Column("metadata", JSON)
    uploaded_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    indexed_at = Column(TIMESTAMP, nullable=True)
    deleted_at = Column(TIMESTAMP, nullable=True, index=True)


class Query(Base):
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(String(128), unique=True, nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text)
    language = Column(String(10), default="nl", index=True)
    retriever_k = Column(Integer, default=5)
    temperature = Column(Float, default=0.7)
    model_used = Column(String(128))
    response_time_ms = Column(Integer)
    chunks_retrieved = Column(JSON)
    confidence_score = Column(Float)
    status = Column(String(32), default="completed", index=True)
    error_message = Column(Text)
    user_role = Column(String(64))
    session_id = Column(String(128), index=True)
    meta_data = Column("metadata", JSON)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp(), index=True)

    # Relationship to feedback
    feedback_items = relationship("Feedback", back_populates="query")


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    feedback_id = Column(String(128), unique=True, nullable=False, index=True)
    query_id = Column(String(128), ForeignKey("queries.query_id", ondelete="SET NULL"), index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    rating = Column(String(10), index=True)
    corrected_answer = Column(Text)
    tags = Column(JSON)
    user_role = Column(String(64))
    status = Column(String(32), default="pending", index=True)
    moderator_notes = Column(Text)
    meta_data = Column("metadata", JSON)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp(), index=True)
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

    # Relationship to query
    query = relationship("Query", back_populates="feedback_items")


class Config(Base):
    __tablename__ = "config"

    id = Column(Integer, primary_key=True, autoincrement=True)
    config_key = Column(String(128), unique=True, nullable=False, index=True)
    config_value = Column(Text, nullable=False)
    value_type = Column(String(32), default="string")
    description = Column(Text)
    is_secret = Column(Boolean, default=False, index=True)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(128), unique=True, nullable=False, index=True)
    job_type = Column(String(64), nullable=False, index=True)
    status = Column(String(32), default="pending", index=True)
    priority = Column(Integer, default=5, index=True)
    params = Column(JSON)
    result = Column(JSON)
    error_message = Column(Text)
    progress = Column(Integer, default=0)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    started_at = Column(TIMESTAMP, nullable=True)
    completed_at = Column(TIMESTAMP, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp(), index=True)
    updated_at = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event = Column(String(128), nullable=False, index=True)
    event_type = Column(String(32), default="info", index=True)
    user_id = Column(String(128))
    payload = Column(JSON)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp(), index=True)


# Helper functions
def init_db():
    """Initialize database tables (only if they don't exist)"""
    Base.metadata.create_all(bind=engine)


def test_connection():
    """Test database connectivity"""
    try:
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False
