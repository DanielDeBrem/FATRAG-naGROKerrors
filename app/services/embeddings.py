"""Central embeddings & vectorstore service (Fase D).

Doel:
- Eén centrale plek voor:
  - Chroma-instantie (vectorstore)
  - Toevoegen van document-chunks met rijke metadata
  - Ophalen van retrievers gefilterd op project_id / client_id / document_type

LET OP:
- Dit is een eerste implementatie. In main.py wordt nog een eigen `vectorstore`
  aangemaakt; in een volgende stap gaan we main.py ombouwen om deze service te
  gebruiken in plaats van de globale `vectorstore`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


# Basisconfig uit env (later kunnen we dit ook uit app.state.config halen)
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", os.getenv("EMBED_MODEL", "gemma2:2b"))
CHROMA_DIR = os.getenv("CHROMA_DIR", "./fatrag_chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "fatrag")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# Module-brede singleton voor embeddings en vectorstore
_embed_model = OllamaEmbeddings(
    model=EMBED_MODEL,
    base_url=OLLAMA_BASE_URL,
)

_vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=_embed_model,
    collection_name=CHROMA_COLLECTION,
)


class EmbeddingsService:
    """Centrale adapter rond Chroma, voor indexeren en ophalen.

    - add_document_chunks: voegt chunks toe met consistente metadata.
    - get_retriever: levert een retriever met optionele filters.
    """

    @staticmethod
    def add_document_chunks(
        *,
        doc_id: str,
        project_id: Optional[str],
        client_id: Optional[str],
        document_type: Optional[str],
        source_type: str,
        chunks: List[str],
        extra_metadata: Optional[Dict[str, Any]] = None,
        persist: bool = True,
    ) -> int:
        """Voeg chunks toe aan de vectorstore met consistente metadata.

        Metadata bevat minimaal:
        - doc_id
        - project_id
        - client_id
        - document_type
        - source_type
        plus alles uit extra_metadata (bijv. filename).
        """
        if not chunks:
            return 0

        base_meta: Dict[str, Any] = {
            "doc_id": doc_id,
            "source_type": source_type,
        }
        if project_id:
            base_meta["project_id"] = project_id
        if client_id:
            base_meta["client_id"] = client_id
        if document_type:
            base_meta["document_type"] = document_type

        if extra_metadata:
            base_meta.update(extra_metadata)

        # Zelfde metadata voor alle chunks, eventueel later uitbreiden met chunk_index
        metas = [base_meta for _ in chunks]

        _vectorstore.add_texts(texts=chunks, metadatas=metas)
        if persist:
            try:
                _vectorstore.persist()
            except Exception:
                # Persist-fouten mogen ingest niet laten crashen
                pass
        return len(chunks)

    @staticmethod
    def get_retriever(
        *,
        k: int = 5,
        project_id: Optional[str] = None,
        client_id: Optional[str] = None,
        document_type: Optional[str] = None,
        extra_filter: Optional[Dict[str, Any]] = None,
    ):
        """Retourneer een retriever met standaardfilters.

        Filters:
        - project_id
        - client_id
        - document_type
        - extra_filter (wordt gemerged)
        """
        search_kwargs: Dict[str, Any] = {"k": k}
        filt: Dict[str, Any] = {}

        if project_id:
            filt["project_id"] = project_id
        if client_id:
            filt["client_id"] = client_id
        if document_type:
            filt["document_type"] = document_type
        if extra_filter:
            # extra_filter heeft voorrang bij conflicterende keys
            filt.update(extra_filter)

        if filt:
            search_kwargs["filter"] = filt

        return _vectorstore.as_retriever(search_kwargs=search_kwargs)

    @staticmethod
    def raw_vectorstore() -> Chroma:
        """Geef directe toegang tot de onderliggende vectorstore (voor scripts/tools).

        Let op: gebruik bij voorkeur add_document_chunks/get_retriever voor nieuwe code.
        """
        return _vectorstore
