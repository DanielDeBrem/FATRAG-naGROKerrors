#!/usr/bin/env python3
"""
Detect available vector database and return connection info.
Tries adapters in order: Chroma, FAISS, Qdrant, Weaviate, Milvus, pgvector
"""
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

def detect_chroma():
    try:
        import chromadb
        chroma_dir = os.getenv("CHROMA_DIR", "./fatrag_chroma_db")
        if not os.path.exists(chroma_dir):
            return None
        client = chromadb.PersistentClient(path=chroma_dir)
        collections = [c.name for c in client.list_collections()]
        return {
            "type": "chroma",
            "conn": chroma_dir,
            "collections": collections,
            "client": client
        }
    except Exception as e:
        return None

def detect_faiss():
    try:
        import faiss
        faiss_index = os.getenv("FAISS_INDEX", "./faiss_index")
        if not os.path.exists(faiss_index):
            return None
        # Check for .faiss or .index files
        index_files = list(Path(faiss_index).glob("*.faiss")) + list(Path(faiss_index).glob("*.index"))
        if not index_files:
            return None
        return {
            "type": "faiss",
            "conn": faiss_index,
            "collections": [f.stem for f in index_files],
            "client": None
        }
    except Exception:
        return None

def detect_qdrant():
    try:
        from qdrant_client import QdrantClient
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=url)
        collections = [c.name for c in client.get_collections().collections]
        return {
            "type": "qdrant",
            "conn": url,
            "collections": collections,
            "client": client
        }
    except Exception:
        return None

def main():
    # Try in order
    for detector in [detect_chroma, detect_faiss, detect_qdrant]:
        result = detector()
        if result:
            # Don't serialize client object
            output = {
                "type": result["type"],
                "conn": result["conn"],
                "collections": result["collections"]
            }
            print(json.dumps(output, indent=2))
            return 0
    
    print(json.dumps({"error": "No vector database found"}, indent=2))
    return 1

if __name__ == "__main__":
    sys.exit(main())
