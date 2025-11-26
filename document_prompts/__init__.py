from __future__ import annotations

from pathlib import Path
from typing import Optional

from .models import DocTypePrompts, PromptVariant
from .store import PromptStore, get_prompt as _get_prompt, load_prompts as _load_prompts
from .context import apply_context_limit as _apply_context_limit

__all__ = [
    "PromptVariant",
    "DocTypePrompts",
    "PromptStore",
    "get_prompt",
    "load_prompts",
    "apply_context_limit",
]


def get_prompt(doc_type: Optional[str], prompt_kind: str) -> PromptVariant:
    """Shortcut naar de globale PromptStore.

    Voorbeeld:
        from document_prompts import get_prompt, apply_context_limit

        pv = get_prompt(document.document_type, "summary")
        limited_text = apply_context_limit(document.raw_text or "", pv.max_context_tokens)
        prompt = pv.template.format(text=limited_text)
    """
    return _get_prompt(doc_type, prompt_kind)


def load_prompts(force: bool = False) -> None:
    """Laad of herlaad alle prompts in de globale store."""
    _load_prompts(force=force)


def apply_context_limit(text: str, max_context_tokens: Optional[int] = None) -> str:
    """Pas een ruwe contextlimiet toe vóór het aanroepen van de LLM."""
    return _apply_context_limit(text, max_context_tokens=max_context_tokens)
