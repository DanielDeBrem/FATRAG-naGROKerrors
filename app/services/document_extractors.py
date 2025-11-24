"""Document extraction & normalization service.

This wraps the existing helpers in `ingestion.py` into a small, focused
service that can be used from ingest flows and later from classifiers.

Fase B van het plan:
- Eén plek waar we per bestandstype naar genormaliseerde tekst gaan.
- Bestaande implementatie uit `ingestion.py` wordt hergebruikt.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

from ingestion import (
    read_text_file,
    read_pdf_file,
    read_excel_file,
    clean_extracted_text,
)


SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXCEL_EXTENSIONS = {".xlsx", ".xls"}


def detect_extension(path: str) -> str:
    """Return lowercase extension including dot, e.g. '.pdf'."""
    return os.path.splitext(path)[1].lower()


def extract_normalized_text(path: str) -> str:
    """Extract and normalize text from a single file.

    - PDF: gebruikt `read_pdf_file` (met OCR fallback).
    - XLSX/XLS: gebruikt `read_excel_file`.
    - TXT/MD: gebruikt `read_text_file`.
    - Onbekend type: levert een korte foutstring terug i.p.v. exception.

    Deze functie gooit alleen bij echt onbruikbare input een exception; in
    de meeste gevallen wordt een tekst (eventueel met foutlabel) geretourneerd.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = detect_extension(path)

    try:
        if ext in SUPPORTED_PDF_EXTENSIONS:
            return read_pdf_file(path)
        if ext in SUPPORTED_EXCEL_EXTENSIONS:
            return read_excel_file(path)
        if ext in SUPPORTED_TEXT_EXTENSIONS:
            return read_text_file(path)
        # Fallback: probeer als tekst te lezen en schoon te maken
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        return clean_extracted_text(raw)
    except Exception as e:
        # Voor nu: encodeer de fout in de tekst, zodat bovenliggende lagen
        # dit kunnen herkennen en netjes afhandelen.
        return f"[Text extraction error: {str(e)}]"


def extract_for_db(path: str) -> Tuple[str, int]:
    """Convenience helper voor ingest:

    Returns:
        (normalized_text, file_size_bytes)

    Deze helper combineert:
    - bestandsgrootte ophalen
    - normalisatie naar tekst
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    st = os.stat(path)
    size = st.st_size
    text = extract_normalized_text(path)
    return text, size


def is_supported_for_normalization(path: str) -> bool:
    """Return True als het bestand in principe normaliseerbaar is.

    Dit is een zachte check; onbekende extensies kunnen nog steeds via
    fallback naar tekst worden verwerkt.
    """
    ext = detect_extension(path)
    return ext in SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS | SUPPORTED_EXCEL_EXTENSIONS
