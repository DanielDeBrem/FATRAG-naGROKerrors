import os
from typing import Iterable, List, Dict, Any, Optional, Tuple

# Disable ChromaDB telemetry to prevent PostHog errors
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY'] = 'False'

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "fatrag_data", "uploads")


def ensure_dirs() -> None:
    os.makedirs(UPLOADS_DIR, exist_ok=True)


# ---------- Text Cleaning & Evidence Utilities ----------

import re
from collections import Counter

def clean_extracted_text(text: str) -> str:
    """
    Clean raw extracted text:
    - normalize newlines
    - fix hyphenation across line breaks
    - drop common headers/footers (page numbers, repeated short lines)
    - collapse excessive blank lines
    """
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix hyphenation at EOL: "waarde-\nring" -> "waardering"
    t = re.sub(r'(\w)-\n(\w)', r'\1\2', t)

    lines = t.split("\n")

    # Remove explicit page/header/footer patterns
    page_footer_patterns = [
        r'^\s*(Page|Pagina|Bladzijde)\s+\d+(\s*(of|van|/)\s*\d+)?\s*$',
        r'^\s*Confidential.*$',
        r'^\s*All rights reserved.*$',
        r'^\s*This document.*$',
        r'^\s*Deze pagina is.*$',
        r'^\s*Footer.*$',
    ]

    # Count short line frequencies to identify repeated headers/footers
    short = [ln.strip() for ln in lines if ln.strip() and len(ln.strip()) <= 120]
    freq = Counter(short)

    cleaned_lines: list[str] = []
    for ln in lines:
        s = ln.strip()
        # drop typical footer/header patterns
        if any(re.match(p, s, flags=re.IGNORECASE) for p in page_footer_patterns):
            continue
        # drop very frequent short lines (likely headers/footers), seen 3+ times
        if s and len(s) <= 120 and freq.get(s, 0) >= 3:
            continue
        cleaned_lines.append(ln)

    t = "\n".join(cleaned_lines)

    # Collapse 3+ blank lines into max 2
    t = re.sub(r'\n{3,}', '\n\n', t)

    return t


def extract_financial_evidence(text: str) -> Dict[str, List[str]]:
    """
    Best-effort regex extraction of financial signals to guide LLM prompts.
    Returns unique lists per category.
    """
    if not text:
        return {"amounts": [], "percentages": [], "rates": [], "dates": [], "entities": []}

    amounts = re.findall(r'(?:(?:EUR|€)\s?[\d\.\,]+(?:\s?(?:mln|miljoen|k))?)', text, flags=re.IGNORECASE)
    percentages = re.findall(r'\b\d{1,3}(?:[\.,]\d+)?\s?%', text)
    rates = re.findall(r'\b(?:rente|interest|rate)\s*[:=]?\s*\d{1,2}(?:[\.,]\d+)?\s?%', text, flags=re.IGNORECASE)
    dates = re.findall(r'\b(?:20\d{2}|19\d{2})\b', text)
    entities = re.findall(r'\b(?:BV|N\.V\.|V\.O\.F\.|Stichting|Coöperatie|Holding)\b.*', text)

    # deduplicate preserving order
    def uniq(seq: List[str]) -> List[str]:
        return list(dict.fromkeys(seq))

    return {
        "amounts": uniq(amounts),
        "percentages": uniq(percentages),
        "rates": uniq(rates),
        "dates": uniq(dates),
        "entities": uniq(entities),
    }


def read_text_file(path: str, encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        raw = f.read()
    return clean_extracted_text(raw)


def read_pdf_file(path: str) -> str:
    """Extract text from PDF file with OCR fallback"""
    try:
        from analysis import extract_pdf_text
        raw = extract_pdf_text(path)
        cleaned = clean_extracted_text(raw)
        
        # If extraction yielded very little text, try OCR
        if len(cleaned) < 100:
            print(f"    Regular extraction yielded only {len(cleaned)} chars, trying OCR...")
            ocr_text = read_pdf_with_ocr(path)
            if len(ocr_text) > len(cleaned):
                print(f"    OCR succeeded: {len(ocr_text)} chars extracted")
                return ocr_text
        
        return cleaned
    except Exception as e:
        # Try OCR as last resort
        try:
            print(f"    PDF extraction failed ({e}), trying OCR...")
            return read_pdf_with_ocr(path)
        except:
            return f"[PDF extraction error: {str(e)}]"


def read_pdf_with_ocr(path: str) -> str:
    """Extract text from PDF using OCR (for scanned documents)"""
    try:
        from pdf2image import convert_from_path
        import pytesseract
        
        print(f"    Converting PDF to images...")
        # Convert PDF to images
        images = convert_from_path(path, dpi=300)
        
        print(f"    Running OCR on {len(images)} pages...")
        # Extract text from each page
        text_parts = []
        for i, image in enumerate(images):
            # Use Dutch language if available, otherwise default
            try:
                page_text = pytesseract.image_to_string(image, lang='nld')
            except:
                page_text = pytesseract.image_to_string(image)
            
            if page_text.strip():
                text_parts.append(f"[Page {i+1}]\n{page_text}")
        
        full_text = "\n\n".join(text_parts)
        return clean_extracted_text(full_text)
        
    except Exception as e:
        return f"[OCR extraction error: {str(e)}]"


def read_excel_file(path: str) -> str:
    """Extract text from Excel file"""
    try:
        import pandas as pd
        xl = pd.read_excel(path, sheet_name=None)
        text_parts = []
        for sheet_name, df in xl.items():
            text_parts.append(f"=== Sheet: {sheet_name} ===\n")
            text_parts.append(df.to_string())
            text_parts.append("\n\n")
        return clean_extracted_text("".join(text_parts))
    except Exception as e:
        return f"[Excel extraction error: {str(e)}]"


def chunk_texts(
    texts: Iterable[str], chunk_size: int = 500, chunk_overlap: int = 100
) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: List[str] = []
    # Support both modern split_texts API and older split_text per-text
    if hasattr(splitter, "split_texts"):
        chunks = splitter.split_texts(list(texts))
    else:
        for t in texts:
            chunks.extend(splitter.split_text(t))
    return chunks


def ingest_texts(
    vectorstore: Chroma,
    texts: List[str],
    base_metadata: Optional[Dict[str, Any]] = None,
    persist: bool = True,
) -> int:
    """
    Add chunks to an existing Chroma vectorstore with metadata.
    """
    base_metadata = base_metadata or {}
    if not texts:
        return 0
    # Use add_texts with duplicate filter handled by Chroma config (if any)
    vectorstore.add_texts(texts=texts, metadatas=[base_metadata] * len(texts))
    if persist:
        vectorstore.persist()
    return len(texts)


def ingest_files(
    vectorstore: Chroma,
    file_paths: List[str],
    user: Optional[str] = None,
    kind: str = "upload",
    extra_metadata: Optional[Dict[str, Any]] = None,
    persist: bool = True,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> Dict[str, Any]:
    """
    Read text, PDF, and Excel files, chunk, and ingest to vectorstore. Returns a report.
    Supported: .txt, .md, .pdf, .xlsx, .xls
    """
    ensure_dirs()
    supported_ext = {".txt", ".md", ".pdf", ".xlsx", ".xls"}
    total_chunks = 0
    docs_info: List[Dict[str, Any]] = []

    for p in file_paths:
        name = os.path.basename(p)
        ext = os.path.splitext(name)[1].lower()
        if ext not in supported_ext:
            docs_info.append({"filename": name, "status": "skipped", "reason": f"unsupported extension {ext}"})
            continue

        try:
            # Read file based on type
            if ext == ".pdf":
                text = read_pdf_file(p)
            elif ext in [".xlsx", ".xls"]:
                text = read_excel_file(p)
            else:
                text = read_text_file(p)
            
            chunks = chunk_texts([text], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            meta = {
                "source": name,
                "doc_id": name,
                "kind": kind,
                "uploaded_by": user or "admin",
                "path": p,
                "file_type": ext[1:],  # Remove dot
            }
            if extra_metadata:
                meta.update(extra_metadata)
            n = ingest_texts(vectorstore, chunks, meta, persist=False)
            total_chunks += n
            docs_info.append({"filename": name, "status": "ingested", "chunks": n, "file_type": ext[1:]})
        except Exception as e:
            docs_info.append({"filename": name, "status": "error", "error": str(e)})

    if persist:
        try:
            vectorstore.persist()
        except Exception:
            # ignore persist errors
            pass

    return {"ingested_chunks": total_chunks, "documents": docs_info}


def list_uploaded_files() -> List[Dict[str, Any]]:
    ensure_dirs()
    items: List[Dict[str, Any]] = []
    for fname in sorted(os.listdir(UPLOADS_DIR)):
        fpath = os.path.join(UPLOADS_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        st = os.stat(fpath)
        items.append(
            {
                "filename": fname,
                "size": st.st_size,
                "mtime": int(st.st_mtime),
                "path": fpath,
            }
        )
    return items


def delete_by_source(vectorstore: Chroma, source_name: str, persist: bool = True) -> Tuple[int, bool]:
    """
    Attempt to delete all chunks with metadata source == source_name from Chroma.
    Returns (deleted_count_unknown_or_0, persisted_flag).
    Note: LangChain Chroma supports delete(where=...) on underlying collection.
    """
    try:
        # Best-effort: where filter on metadata
        vectorstore.delete(where={"source": source_name})
        if persist:
            vectorstore.persist()
        return (0, True)
    except Exception:
        return (0, False)
