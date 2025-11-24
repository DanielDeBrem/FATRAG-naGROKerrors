"""Data Transfer Objects for upload operations."""

from pydantic import BaseModel
from typing import Optional


class UploadResponse(BaseModel):
    """Response model for document upload."""

    document_id: str
    filename: str
    mime_type: str
    file_size: int
    status: str
    message: Optional[str] = None

    class Config:
        from_attributes = True


class UploadValidationError(BaseModel):
    """Error model for upload validation failures."""

    field: str
    error: str
    value: Optional[str] = None
