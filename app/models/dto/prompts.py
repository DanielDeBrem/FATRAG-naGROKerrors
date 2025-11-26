from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class PromptDTO(BaseModel):
    """DTO voor één promptconfiguratie (voor admin API en UI)."""

    doc_type: str = Field(..., description="Documenttype, bijv. 'jaarrekening', 'factuur'")
    prompt_kind: str = Field(..., description="Soort prompt: 'extract', 'summary' of 'risk'")
    template: str = Field(..., description="De volledige prompttekst")
    max_context_tokens: Optional[int] = Field(
        None,
        description="Optionele limiet op het aantal context-tokens voor deze prompt",
    )
    active: bool = Field(True, description="Of deze prompt actief is voor dit type")


class DocTypePromptsDTO(BaseModel):
    """DTO voor alle prompts behorend bij één doc_type."""

    doc_type: str
    prompts: Dict[str, PromptDTO]  # keys: 'extract', 'summary', 'risk'


class PromptDocTypeOverviewDTO(BaseModel):
    """DTO voor lijstweergave van bekende doc_types + statistieken."""

    doc_type: str
    prompt_count: int
    active_prompts: int
