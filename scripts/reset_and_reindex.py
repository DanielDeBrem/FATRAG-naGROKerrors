#!/usr/bin/env python3
"""
Reset and Reindex Script for FATRAG

Clears the existing Chroma vector database collection and re-embeds all documents.
This ensures a fresh start for the FATRAG analysis pipeline.
"""

import os
import sys
import shutil
from typing import List, Dict, Any
import glob

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import chromadb
    from langchain_ollama import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma
    from prepare_data import build_chunks
except ImportError as e:
    print(f"ERROR: Missing required dependency: {e}")
    print("Please ensure langchain_ollama, chromadb, and prepare_data are available")
    sys.exit(1)


def log(msg: str) -> None:
    print(f"[RESET] {msg}", flush=True)


def clear_chroma_collection(chroma_dir: str, collection: str) -> None:
    """
    Clear the specified Chroma collection by deleting its directory.
    """
    log(f"Clearing Chroma collection '{collection}' at {chroma_dir}")
    
    if not os.path.isdir(chroma_dir):
        log(f"Directory {chroma_dir} does not exist - will be created fresh")
        return
    
    try:
        # Initialize client to properly delete collection
        client = chromadb.PersistentClient(path=chroma_dir)
        collections = client.list_collections()
        col_names = [c.name for c in collections]
        
        if collection in col_names:
            log(f"Deleting existing collection '{collection}'")
            client.delete_collection(name=collection)
            log(f"✓ Collection '{collection}' deleted")
        else:
            log(f"Collection '{collection}' does not exist - nothing to delete")
            
    except Exception as e:
        log(f"WARNING: Could not delete collection via client: {e}")
        log("Attempting to clear by removing directory...")
        try:
            shutil.rmtree(chroma_dir)
            log(f"✓ Directory {chroma_dir} removed")
        except Exception as e2:
            log(f"WARNING: Could not remove directory: {e2}")


def collect_documents() -> List[str]:
    """
    Collect all documents from fatrag_data/ and fatrag_data/uploads/
    Returns list of file paths.
    """
    log("Collecting documents for indexing...")
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fatrag_data")
    uploads_dir = os.path.join(data_dir, "uploads")
    
    documents = []
    
    # Collect from main data directory
    if os.path.isdir(data_dir):
        for ext in ["*.txt", "*.md", "*.pdf"]:
            pattern = os.path.join(data_dir, ext)
            documents.extend(glob.glob(pattern))
    
    # Collect from uploads directory
    if os.path.isdir(uploads_dir):
        for ext in ["*.txt", "*.md", "*.pdf"]:
            pattern = os.path.join(uploads_dir, ext)
            documents.extend(glob.glob(pattern))
    
    # Filter out any that are in subdirectories we want to skip
    documents = [d for d in documents if "/uploads/" in d or os.path.dirname(d) == data_dir]
    
    log(f"Found {len(documents)} documents to index")
    return documents


def reindex_documents(chroma_dir: str, collection: str, embed_model: str, ollama_base_url: str) -> None:
    """
    Re-embed and re-index all documents into the Chroma collection.
    """
    log("Starting re-indexing process...")
    
    # Build chunks using existing prepare_data logic
    try:
        chunks = build_chunks()
        log(f"Built {len(chunks)} chunks from documents")
    except Exception as e:
        log(f"ERROR: Failed to build chunks: {e}")
        sys.exit(1)
    
    if not chunks:
        log("WARNING: No chunks to index - vector DB will be empty")
        return
    
    # Create embeddings model
    log(f"Initializing embeddings with model: {embed_model}")
    try:
        embeddings = OllamaEmbeddings(
            model=embed_model,
            base_url=ollama_base_url
        )
    except Exception as e:
        log(f"ERROR: Failed to initialize embeddings: {e}")
        sys.exit(1)
    
    # Create new vector store with all chunks
    log(f"Creating Chroma vector store at {chroma_dir} (collection={collection})")
    try:
        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            collection_name=collection,
            persist_directory=chroma_dir,
        )
        vectorstore.persist()
        log(f"✓ Successfully indexed {len(chunks)} chunks into collection '{collection}'")
    except Exception as e:
        log(f"ERROR: Failed to create vector store: {e}")
        sys.exit(1)


def main():
    # Load configuration from environment
    CHROMA_DIR = os.getenv("CHROMA_DIR", "./fatrag_chroma_db")
    CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "fatrag")
    EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "gemma2:2b")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    log("=" * 80)
    log("FATRAG Vector Database Reset and Reindex")
    log("=" * 80)
    log(f"Chroma directory: {CHROMA_DIR}")
    log(f"Collection: {CHROMA_COLLECTION}")
    log(f"Embed model: {EMBED_MODEL}")
    log(f"Ollama base URL: {OLLAMA_BASE_URL}")
    log("")
    
    # Step 1: Clear existing collection
    clear_chroma_collection(CHROMA_DIR, CHROMA_COLLECTION)
    log("")
    
    # Step 2: Re-index all documents
    reindex_documents(CHROMA_DIR, CHROMA_COLLECTION, EMBED_MODEL, OLLAMA_BASE_URL)
    log("")
    
    log("=" * 80)
    log("✅ Reset and reindex complete!")
    log("=" * 80)


if __name__ == "__main__":
    main()
