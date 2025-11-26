from __future__ import annotations

from typing import Callable, Dict, Optional

from .models import DocTypePrompts, PromptVariant
from app.services.document_prompts_service import DocumentPromptsService


class PromptStore:
    """Beheert prompts per documenttype op basis van de database.

    - Geen YAML/JSON-bestanden meer: MySQL is de enige bron.
    - Biedt een eenvoudige interface:
        get_prompt(doc_type, kind) -> PromptVariant
    - Bevat een expliciete default-doc_type (bijv. 'default') voor unknown/lege types.
    """

    def __init__(
        self,
        loader: Callable[[], Dict[str, DocTypePrompts]],
        default_key: str = "default",
    ) -> None:
        """
        Args:
            loader: Callable die alle DocTypePrompts uit de bron laadt (DB).
            default_key: doc_type die wordt gebruikt als fallback wanneer
                         doc_type leeg/None is.
        """
        self.loader = loader
        self.default_key = default_key.lower()
        self._cache: Dict[str, DocTypePrompts] = {}
        self._loaded: bool = False

    def load_all(self, force: bool = False) -> None:
        """Laad of herlaad alle promptconfiguraties vanuit de bron (DB)."""
        if self._loaded and not force:
            return

        cache = self.loader() or {}
        # Normaliseer sleutels naar lowercase
        norm_cache: Dict[str, DocTypePrompts] = {}
        for key, cfg in cache.items():
            norm_cache[(key or "").lower()] = cfg

        self._cache = norm_cache
        self._loaded = True

    def get_prompt(self, doc_type: Optional[str], kind: str) -> PromptVariant:
        """Haal de juiste prompt op o.b.v. doc_type en soort (extract/summary/risk).

        Gedrag:
        - doc_type wordt naar lowercase genormaliseerd.
        - Lege/None doc_type valt terug op default_key (bijv. 'default').
        - Als er geen entry bestaat voor het gevraagde type:
          - proberen we NIET andere bronnen (geen YAML).
          - als er wel een default bestaat, kun je er bewust voor kiezen
            doc_type in de aanroep al te mappen naar 'default'.
        """
        if not self._loaded:
            self.load_all()

        key = (doc_type or "").lower().strip() or self.default_key

        cfg = self._cache.get(key)
        if cfg is None:
            # Er is geen configuratie voor dit doc_type geladen
            raise RuntimeError(
                f"Geen promptconfig gevonden voor doc_type={doc_type!r}. "
                f"Zorg dat dit type is geconfigureerd in de document_prompts-tabel."
            )

        prompt = cfg.prompts.get(kind)
        if prompt is None:
            raise RuntimeError(
                f"Geen prompt gevonden voor doc_type={doc_type!r}, kind={kind!r}. "
                f"Controleer de document_prompts-tabel voor dit type."
            )

        return prompt


# ---------- Globale helper ----------

# DocumentPromptsService laadt alle DocTypePrompts uit MySQL
_service = DocumentPromptsService()
_prompt_store = PromptStore(loader=_service.load_all, default_key="default")


def load_prompts(force: bool = False) -> None:
    """Laad of herlaad alle prompts in de globale store vanuit de database."""
    _prompt_store.load_all(force=force)


def get_prompt(doc_type: Optional[str], prompt_kind: str) -> PromptVariant:
    """Publieke helper voor gebruik elders in de codebase.

    Voorbeeld:
        from document_prompts.store import get_prompt, load_prompts

        load_prompts()  # bij startup of na admin-wijziging
        pv = get_prompt(document.document_type, "summary")
        prompt_text = pv.template.format(text=...)
    """
    return _prompt_store.get_prompt(doc_type, prompt_kind)
