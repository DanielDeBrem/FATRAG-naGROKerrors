"""Service layer to load document prompts from MySQL into DocTypePrompts."""

from __future__ import annotations

from typing import Dict, Optional

from document_prompts.models import DocTypePrompts, PromptVariant
from app.repositories.prompts import PromptsRepository, PromptRecord


class DocumentPromptsService:
    """Loads promptconfiguraties uit de document_prompts-tabel.

    Dit vormt de brug tussen de MySQL-opslag en de document_prompts.PromptStore.
    """

    def __init__(self, repository: Optional[PromptsRepository] = None) -> None:
        self.repository = repository or PromptsRepository()

    def load_all(self) -> Dict[str, DocTypePrompts]:
        """Laad alle prompts uit de database, gegroepeerd per doc_type.

        Returns:
            Dict[str, DocTypePrompts], waarbij de sleutel de lowercase doc_type is.
        """
        result: Dict[str, DocTypePrompts] = {}

        # Eerst alle bekende doc_types ophalen (met stats)
        doc_type_rows = self.repository.list_doc_types()
        doc_types = [row["doc_type"] for row in doc_type_rows]

        for doc_type in doc_types:
            records = self.repository.list_prompts_for_type(doc_type)
            if not records:
                continue

            prompts: Dict[str, PromptVariant] = {}
            for rec in records:
                if not rec.get("active", True):
                    continue
                kind = str(rec["prompt_kind"])
                template = str(rec["template"])
                max_ctx = rec.get("max_context_tokens")
                prompts[kind] = PromptVariant(
                    template=template,
                    max_context_tokens=int(max_ctx) if max_ctx is not None else None,
                )

            if not prompts:
                continue

            key = str(doc_type).lower()
            result[key] = DocTypePrompts(doc_type=doc_type, prompts=prompts)

        return result
