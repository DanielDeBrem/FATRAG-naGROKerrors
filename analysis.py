"""
Document Analysis Module
Handles PDF/Excel extraction, document type detection, and financial data extraction
"""

import os
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import PyPDF2
import pdfplumber
import pandas as pd
from tabulate import tabulate
import openpyxl


# ============= PDF PROCESSING =============

def extract_pdf_text(file_path: str) -> str:
    """Extract plain text from PDF using PyPDF2"""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
            return text
    except Exception as e:
        raise Exception(f"PDF text extraction failed: {str(e)}")


def extract_pdf_tables(file_path: str) -> List[Dict[str, Any]]:
    """Extract tables from PDF using pdfplumber"""
    tables_data = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                for table_num, table in enumerate(tables, 1):
                    if table and len(table) > 0:
                        # Convert to pandas DataFrame for better handling
                        df = pd.DataFrame(table[1:], columns=table[0] if table[0] else None)
                        tables_data.append({
                            "page": page_num,
                            "table_num": table_num,
                            "headers": table[0] if table[0] else [],
                            "rows": len(df),
                            "data": df.to_dict('records')[:5],  # First 5 rows as sample
                        })
        return tables_data
    except Exception as e:
        return [{"error": f"Table extraction failed: {str(e)}"}]


def analyze_pdf(file_path: str) -> Dict[str, Any]:
    """Complete PDF analysis with metadata, text, and tables"""
    result = {
        "file_path": file_path,
        "file_size": os.path.getsize(file_path),
        "analyzed_at": datetime.now().isoformat(),
    }
    
    try:
        # Basic metadata
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            result["page_count"] = len(reader.pages)
            result["metadata"] = dict(reader.metadata) if reader.metadata else {}
        
        # Extract text
        text = extract_pdf_text(file_path)
        result["text"] = text
        result["text_length"] = len(text)
        
        # Extract tables
        tables = extract_pdf_tables(file_path)
        result["tables"] = tables
        result["table_count"] = len(tables)
        
        # Detect document type based on content
        result["detected_type"] = detect_document_type(text)
        
        # Extract financial data if applicable
        if result["detected_type"] in ["taxatie", "jaarrekening", "balans"]:
            result["financial_data"] = extract_financial_data(text, result["detected_type"])
        
        # Extract parties/persons mentioned
        result["parties"] = extract_parties(text)
        
        result["status"] = "success"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


# ============= EXCEL PROCESSING =============

def analyze_excel(file_path: str) -> Dict[str, Any]:
    """Analyze Excel file (.xlsx or .xls)"""
    result = {
        "file_path": file_path,
        "file_size": os.path.getsize(file_path),
        "analyzed_at": datetime.now().isoformat(),
    }
    
    try:
        # Read all sheets
        xl = pd.read_excel(file_path, sheet_name=None, nrows=100)  # Limit to first 100 rows per sheet
        
        result["sheet_count"] = len(xl)
        result["sheets"] = {}
        
        for sheet_name, df in xl.items():
            result["sheets"][sheet_name] = {
                "rows": len(df),
                "columns": list(df.columns),
                "column_count": len(df.columns),
                "sample_data": df.head(10).to_dict('records'),  # First 10 rows
                "summary": {
                    "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
                    "text_columns": df.select_dtypes(include=['object']).columns.tolist(),
                }
            }
        
        # Try to detect if it's a financial statement
        all_text = " ".join([str(df.to_string()) for df in xl.values()])
        result["detected_type"] = detect_document_type(all_text)
        
        result["status"] = "success"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


# ============= DOCUMENT TYPE DETECTION =============

DOC_TYPE_PATTERNS = {
    "taxatie": [
        r"taxatie(verslag|rapport)?",
        r"waardebepaling",
        r"getaxeerde waarde",
        r"marktwaarde",
        r"woz[\-\s]waarde",
    ],
    "jaarrekening": [
        r"jaarrekening",
        r"jaarverslag",
        r"financieel verslag",
        r"balans per",
        r"winst[- ]en verliesrekening",
    ],
    "balans": [
        r"balans per \d{1,2}[/-]\d{1,2}[/-]\d{4}",
        r"activa.*passiva",
        r"vaste activa",
        r"vlottende activa",
    ],
    "akte": [
        r"akte van oprichting",
        r"notariële akte",
        r"voor mij,.*notaris",
        r"repertorium.*nummer",
    ],
    "testament": [
        r"testament",
        r"uiterste wil",
        r"erfgenamen",
        r"legaat",
        r"executeur",
    ],
    "aangifte": [
        r"aangiftebiljet",
        r"inkomstenbelasting",
        r"vennootschapsbelasting",
        r"btw[\-\s]aangifte",
    ],
    "huwelijkse_voorwaarden": [
        r"huwelijkse voorwaarden",
        r"huwelijksvoorwaarden",
        r"gemeenschap van goederen",
        r"koude uitsluiting",
    ],
}


def detect_document_type(text: str) -> str:
    """Detect document type based on content patterns"""
    text_lower = text.lower()
    
    for doc_type, patterns in DOC_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return doc_type
    
    return "algemeen"


# ============= FINANCIAL DATA EXTRACTION =============

def extract_financial_data(text: str, doc_type: str) -> Dict[str, Any]:
    """Extract financial figures based on document type"""
    data = {
        "doc_type": doc_type,
        "amounts": [],
        "dates": [],
        "percentages": [],
    }
    
    # Extract amounts (EUR)
    amount_pattern = r"€\s?(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)|(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)\s?euro"
    amounts = re.findall(amount_pattern, text)
    data["amounts"] = [a[0] or a[1] for a in amounts if a[0] or a[1]]
    
    # Extract dates
    date_pattern = r"\d{1,2}[/-]\d{1,2}[/-]\d{4}"
    data["dates"] = re.findall(date_pattern, text)
    
    # Extract percentages
    pct_pattern = r"(\d{1,3}(?:,\d{1,2})?)\s?%"
    data["percentages"] = re.findall(pct_pattern, text)
    
    # Type-specific extraction
    if doc_type == "taxatie":
        data["valuation"] = extract_valuation_data(text)
    elif doc_type in ["jaarrekening", "balans"]:
        data["financial_statement"] = extract_balance_data(text)
    elif doc_type == "akte":
        data["deed_info"] = extract_deed_data(text)
    
    return data


def extract_valuation_data(text: str) -> Dict[str, Any]:
    """Extract valuation-specific data"""
    data = {}
    
    # Marktwaarde pattern
    marktwaarde_pattern = r"marktwaarde[:\s]+€?\s?(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)"
    match = re.search(marktwaarde_pattern, text, re.IGNORECASE)
    if match:
        data["marktwaarde"] = match.group(1)
    
    # WOZ-waarde pattern  
    woz_pattern = r"woz[\-\s]waarde[:\s]+€?\s?(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)"
    match = re.search(woz_pattern, text, re.IGNORECASE)
    if match:
        data["woz_waarde"] = match.group(1)
    
    return data


def extract_balance_data(text: str) -> Dict[str, Any]:
    """Extract balance sheet data"""
    data = {}
    
    # Totaal activa
    activa_pattern = r"totaal activa[:\s]+€?\s?(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)"
    match = re.search(activa_pattern, text, re.IGNORECASE)
    if match:
        data["totaal_activa"] = match.group(1)
    
    # Eigen vermogen
    ev_pattern = r"eigen vermogen[:\s]+€?\s?(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)"
    match = re.search(ev_pattern, text, re.IGNORECASE)
    if match:
        data["eigen_vermogen"] = match.group(1)
    
    return data


def extract_deed_data(text: str) -> Dict[str, Any]:
    """Extract notarial deed data"""
    data = {}
    
    # Notaris naam
    notaris_pattern = r"voor mij,\s+([^,]+),\s+notaris"
    match = re.search(notaris_pattern, text, re.IGNORECASE)
    if match:
        data["notaris"] = match.group(1).strip()
    
    # Repertorium nummer
    rep_pattern = r"repertorium[\-\s]?nummer[:\s]+(\d+)"
    match = re.search(rep_pattern, text, re.IGNORECASE)
    if match:
        data["repertorium_nummer"] = match.group(1)
    
    return data


# ============= PARTY/PERSON EXTRACTION =============

def extract_parties(text: str) -> List[Dict[str, str]]:
    """Extract mentioned parties/persons from document"""
    parties = []
    
    # Pattern for "naam + BSN" or "naam + KvK"
    bsn_pattern = r"([A-Z][a-zA-Z\s\-]+)\s+(?:BSN|bsn)[:\s]+(\d{9})"
    kvk_pattern = r"([A-Z][a-zA-Z\s\-]+)\s+(?:KvK|kvk)[:\s]+(\d{8})"
    
    for match in re.finditer(bsn_pattern, text):
        parties.append({"name": match.group(1).strip(), "bsn": match.group(2), "type": "individual"})
    
    for match in re.finditer(kvk_pattern, text):
        parties.append({"name": match.group(1).strip(), "kvk": match.group(2), "type": "business"})
    
    # Pattern for BV/NV names
    company_pattern = r"([A-Z][a-zA-Z\s\-]+ (?:B\.?V\.?|N\.?V\.?))"
    for match in re.finditer(company_pattern, text):
        name = match.group(1).strip()
        if not any(p["name"] == name for p in parties):
            parties.append({"name": name, "type": "company"})
    
    return parties[:10]  # Limit to first 10 parties


# ============= ORGANOGRAM DATA GENERATION =============

def generate_organogram_data(documents: List[str], project_id: str) -> Dict[str, Any]:
    """
    Generate organogram structure from document content
    Returns vis.js compatible format: {nodes: [...], edges: [...]}
    """
    nodes = []
    edges = []
    node_id = 1
    
    # Placeholder: in real implementation, parse documents to find entities and relationships
    # For now, create a sample structure
    
    nodes.append({
        "id": node_id,
        "label": "Holding BV",
        "shape": "box",
        "color": "#003d7a",
        "title": "Moeder entiteit",
    })
    root_id = node_id
    node_id += 1
    
    # This would be populated by analyzing the documents
    # Look for patterns like "X houdt Y% van de aandelen in Z"
    
    return {
        "nodes": nodes,
        "edges": edges,
        "generated_at": datetime.now().isoformat(),
        "source": "auto_generated",
        "project_id": project_id,
    }
