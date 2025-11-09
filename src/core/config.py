"""
Application configuration settings for FATRAG.
"""
from functools import lru_cache
from typing import Optional, List
from pydantic import BaseSettings, validator
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # Application Configuration
    app_name: str = "FATRAG"
    app_version: str = "1.0.0"
    app_host: str = "0.0.0.0"
    app_port: int = 8050
    app_debug: bool = False
    app_reload: bool = False
    
    # Database Configuration
    database_url: str
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "fatrag"
    mysql_password: str
    mysql_database: str = "fatrag"
    
    # Vector Database (Milvus)
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_user: Optional[str] = None
    milvus_password: Optional[str] = None
    
    # Ollama Configuration
    ollama_host: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_generation_model: str = "llama2"
    ollama_timeout: int = 30
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # File Storage
    upload_dir: str = "./uploads"
    max_file_size: str = "100MB"
    allowed_extensions: str = "pdf,docx,xlsx"
    
    @validator("allowed_extensions")
    def parse_extensions(cls, v):
        return [ext.strip() for ext in v.split(",")]
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Monitoring
    prometheus_port: int = 9090
    
    # Financial Data APIs
    yahoo_finance_enabled: bool = True
    alpha_vantage_api_key: Optional[str] = None
    
    # Dutch Language Settings
    default_language: str = "nl"
    summary_max_length: int = 500
    pii_detection_enabled: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
