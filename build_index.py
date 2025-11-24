import os
from typing import List

# Disable ChromaDB telemetry to prevent PostHog errors
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY'] = 'False'

from prepare_data import build_chunks
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


def main() -> None:
    # Config via env (keeps parity with main.py defaults)
    EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "gemma2:2b")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    CHROMA_DIR = os.getenv("CHROMA_DIR", "./fatrag_chroma_db")
    COLLECTION = os.getenv("CHROMA_COLLECTION", "fatrag")

    # 1) Load/construct documents
    chunks: List[str] = build_chunks()
    print(f"Loaded {len(chunks)} chunks; building vector index in '{CHROMA_DIR}' (collection='{COLLECTION}')...")

    # 2) Build embeddings
    embed_model = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    # 3) Create/update Chroma persistence
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embed_model,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )
    vectorstore.persist()
    print("Index built and persisted successfully.")


if __name__ == "__main__":
    main()
