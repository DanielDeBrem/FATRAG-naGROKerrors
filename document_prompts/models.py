from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel


class PromptVariant(BaseModel):
    """Eén concrete prompt-variant (bijv. extract/summary/risk) voor een doc_type.

    - template: de prompttekst, meestal met placeholders zoals {text}
    - max_context_tokens: optionele limiet voor de context die we meesturen
    """

    template: str
    max_context_tokens: Optional[int] = None


class DocTypePrompts(BaseModel):
    """Alle prompts voor één documenttype (bijv. jaarrekening, factuur, etc.)."""

    doc_type: str
    prompts: Dict[str, PromptVariant]  # keys: "extract", "summary", "risk", ...
