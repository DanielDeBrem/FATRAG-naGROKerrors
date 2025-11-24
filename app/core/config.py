"""Application configuration and settings."""

import os
from typing import List, Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8020, env="PORT")

    # File upload settings
    max_upload_size: int = Field(default=50 * 1024 * 1024, description="Max file size in bytes (50MB)")
    allowed_file_types: List[str] = Field(
        default=["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
        description="Allowed MIME types for uploads"
    )
    upload_directory: str = Field(default="fatrag_data/uploads", env="UPLOAD_DIRECTORY")

    # Database settings
    mysql_host: str = Field(default="localhost", env="MYSQL_HOST")
    mysql_port: int = Field(default=3306, env="MYSQL_PORT")
    mysql_user: str = Field(default="root", env="MYSQL_USER")
    mysql_password: str = Field(default="", env="MYSQL_PASSWORD")
    mysql_database: str = Field(default="fatrag", env="MYSQL_DATABASE")

    # Vector database settings
    chroma_directory: str = Field(default="./fatrag_chroma_db", env="CHROMA_DIR")
    chroma_collection: str = Field(default="fatrag", env="CHROMA_COLLECTION")

    # LLM settings
    llm_model: str = Field(default="llama3.1:8b", env="OLLAMA_LLM_MODEL")
    embed_model: str = Field(default="gemma2:2b", env="OLLAMA_EMBED_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")

    # Worker routing (for multi-GPU)
    _ollama_worker_ports: List[int] = []

    @property
    def ollama_worker_ports(self) -> List[int]:
        """Get list of Ollama worker ports."""
        return self._ollama_worker_ports or [11434]

    # Processing settings
    chunk_size: int = Field(default=500, description="Text chunk size for embeddings")
    chunk_overlap: int = Field(default=100, description="Overlap between chunks")
    max_chunks: int = Field(default=25, description="Maximum chunks per document")

    # Feature flags
    feedback_enabled: bool = Field(default=True, env="FEEDBACK_ENABLED")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

    @property
    def mysql_connection_string(self) -> str:
        """Get MySQL connection string."""
        return f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"

    @property
    def ollama_worker_urls(self) -> List[str]:
        """Get list of Ollama worker URLs."""
        return [f"http://127.0.0.1:{port}" for port in self.ollama_worker_ports]


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
