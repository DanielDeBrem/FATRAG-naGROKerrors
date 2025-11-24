"""
Organogram Service Module
Handles CRUD operations for interactive organizational charts (vis.js format)
"""

import os
import uuid
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import pymysql
from pymysql.cursors import DictCursor


def get_db_connection():
    """Create MySQL connection using environment variables"""
    return pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER", "fatrag"),
        password=os.getenv("DB_PASSWORD", "fatrag_pw"),
        database=os.getenv("DB_NAME", "fatrag"),
        charset="utf8mb4",
        cursorclass=DictCursor,
    )


def generate_id(prefix: str = "organogram-") -> str:
    """Generate unique ID"""
    unique = str(uuid.uuid4())[:8]
    return f"{prefix}{unique}"


# ============= ORGANOGRAM OPERATIONS =============

def create_organogram(
    project_id: str,
    name: str,
    structure_data: Dict[str, Any],
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new organogram for a project
    structure_data should be vis.js format: {"nodes": [...], "edges": [...]}
    """
    organogram_id = generate_id()
    
    # Validate structure_data format
    if not isinstance(structure_data, dict):
        raise ValueError("structure_data must be a dictionary")
    if "nodes" not in structure_data or "edges" not in structure_data:
        raise ValueError("structure_data must contain 'nodes' and 'edges' keys")
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO organograms (organogram_id, project_id, name, structure_data, notes)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(
                sql,
                (
                    organogram_id,
                    project_id,
                    name,
                    json.dumps(structure_data),
                    notes,
                ),
            )
        conn.commit()
        return get_organogram(organogram_id)
    finally:
        conn.close()


def list_organograms(project_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """List organograms, optionally filtered by project"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            if project_id:
                sql = "SELECT * FROM organograms WHERE project_id = %s ORDER BY version DESC, created_at DESC"
                cursor.execute(sql, (project_id,))
            else:
                sql = "SELECT * FROM organograms ORDER BY created_at DESC"
                cursor.execute(sql)
            
            results = cursor.fetchall()
            # Parse JSON structure_data
            for result in results:
                if result.get("structure_data"):
                    result["structure_data"] = json.loads(result["structure_data"])
            return results
    finally:
        conn.close()


def get_organogram(organogram_id: str) -> Optional[Dict[str, Any]]:
    """Get organogram by ID"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM organograms WHERE organogram_id = %s"
            cursor.execute(sql, (organogram_id,))
            result = cursor.fetchone()
            if result and result.get("structure_data"):
                # Parse JSON structure_data
                result["structure_data"] = json.loads(result["structure_data"])
            return result
    finally:
        conn.close()


def update_organogram(
    organogram_id: str,
    name: Optional[str] = None,
    structure_data: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None,
    increment_version: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Update organogram
    If increment_version=True, creates a new version
    """
    conn = get_db_connection()
    try:
        # Get current version if incrementing
        if increment_version:
            current = get_organogram(organogram_id)
            if current:
                current_version = current.get("version", 1)
        
        with conn.cursor() as cursor:
            updates = []
            params = []
            
            if name is not None:
                updates.append("name = %s")
                params.append(name)
            
            if structure_data is not None:
                if not isinstance(structure_data, dict):
                    raise ValueError("structure_data must be a dictionary")
                if "nodes" not in structure_data or "edges" not in structure_data:
                    raise ValueError("structure_data must contain 'nodes' and 'edges' keys")
                updates.append("structure_data = %s")
                params.append(json.dumps(structure_data))
            
            if notes is not None:
                updates.append("notes = %s")
                params.append(notes)
            
            if increment_version:
                updates.append("version = version + 1")
            
            if not updates:
                return get_organogram(organogram_id)
            
            params.append(organogram_id)
            sql = f"UPDATE organograms SET {', '.join(updates)} WHERE organogram_id = %s"
            cursor.execute(sql, params)
        
        conn.commit()
        return get_organogram(organogram_id)
    finally:
        conn.close()


def delete_organogram(organogram_id: str) -> bool:
    """Delete organogram permanently"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = "DELETE FROM organograms WHERE organogram_id = %s"
            cursor.execute(sql, (organogram_id,))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


# ============= AUTO-GENERATION HELPERS =============

def auto_generate_organogram(
    project_id: str,
    document_texts: List[str],
    name: str = "Auto-generated Organogram",
) -> Dict[str, Any]:
    """
    Auto-generate organogram from document content
    This is a simplified version - real implementation would use NLP/pattern matching
    """
    from analysis import extract_parties
    
    nodes = []
    edges = []
    node_id_counter = 1
    party_to_node = {}
    
    # Extract all parties from documents
    all_parties = []
    for text in document_texts:
        parties = extract_parties(text)
        all_parties.extend(parties)
    
    # Deduplicate parties by name
    unique_parties = {}
    for party in all_parties:
        name = party.get("name", "")
        if name and name not in unique_parties:
            unique_parties[name] = party
    
    # Create nodes for each party
    for party_name, party_data in unique_parties.items():
        node = {
            "id": node_id_counter,
            "label": party_name,
            "shape": "box" if party_data.get("type") == "company" else "ellipse",
            "color": {
                "background": "#003d7a" if party_data.get("type") == "company" else "#4a90e2",
                "border": "#002855",
            },
            "font": {"color": "#ffffff"},
        }
        
        if party_data.get("kvk"):
            node["title"] = f"KvK: {party_data['kvk']}"
        elif party_data.get("bsn"):
            node["title"] = f"BSN: {party_data['bsn']}"
        
        nodes.append(node)
        party_to_node[party_name] = node_id_counter
        node_id_counter += 1
    
    # Try to detect relationships from text patterns
    # Simple pattern: "X houdt Y% aandelen in Z" or "X is aandeelhouder van Z"
    for text in document_texts:
        # Pattern: "X houdt/bezit N% (van de aandelen) (in) Y"
        import re
        pattern = r"([A-Z][a-zA-Z\s\-]+ (?:B\.?V\.?|N\.?V\.?)?)\s+(?:houdt|bezit)\s+(\d+)%\s+(?:van de aandelen in|in|van)\s+([A-Z][a-zA-Z\s\-]+ (?:B\.?V\.?|N\.?V\.?))"
        
        for match in re.finditer(pattern, text):
            from_party = match.group(1).strip()
            percentage = match.group(2)
            to_party = match.group(3).strip()
            
            if from_party in party_to_node and to_party in party_to_node:
                edges.append({
                    "from": party_to_node[from_party],
                    "to": party_to_node[to_party],
                    "label": f"{percentage}%",
                    "arrows": "to",
                    "color": {"color": "#d4af37"},
                })
    
    structure_data = {
        "nodes": nodes,
        "edges": edges,
        "options": {
            "layout": {
                "hierarchical": {
                    "enabled": True,
                    "direction": "UD",
                    "sortMethod": "directed",
                }
            },
            "physics": {
                "enabled": False,
            },
        },
    }
    
    # Create and save organogram
    return create_organogram(
        project_id=project_id,
        name=name,
        structure_data=structure_data,
        notes="Automatically generated from document content",
    )


# ============= TEMPLATE HELPERS =============

def create_empty_organogram_template() -> Dict[str, Any]:
    """Create an empty organogram template"""
    return {
        "nodes": [],
        "edges": [],
        "options": {
            "layout": {
                "hierarchical": {
                    "enabled": True,
                    "direction": "UD",
                    "sortMethod": "directed",
                }
            },
            "interaction": {
                "dragNodes": True,
                "dragView": True,
                "zoomView": True,
            },
            "manipulation": {
                "enabled": True,
            },
        },
    }


def create_simple_holding_template(holding_name: str = "Holding BV") -> Dict[str, Any]:
    """Create a simple holding structure template"""
    return {
        "nodes": [
            {
                "id": 1,
                "label": holding_name,
                "shape": "box",
                "color": {
                    "background": "#003d7a",
                    "border": "#002855",
                },
                "font": {"color": "#ffffff", "size": 16},
            },
            {
                "id": 2,
                "label": "Werk BV",
                "shape": "box",
                "color": {
                    "background": "#4a90e2",
                    "border": "#003d7a",
                },
                "font": {"color": "#ffffff"},
            },
            {
                "id": 3,
                "label": "Vastgoed BV",
                "shape": "box",
                "color": {
                    "background": "#4a90e2",
                    "border": "#003d7a",
                },
                "font": {"color": "#ffffff"},
            },
        ],
        "edges": [
            {"from": 1, "to": 2, "label": "100%", "arrows": "to", "color": {"color": "#d4af37"}},
            {"from": 1, "to": 3, "label": "100%", "arrows": "to", "color": {"color": "#d4af37"}},
        ],
        "options": {
            "layout": {
                "hierarchical": {
                    "enabled": True,
                    "direction": "UD",
                    "sortMethod": "directed",
                    "nodeSpacing": 150,
                    "levelSeparation": 200,
                }
            },
            "physics": {
                "enabled": False,
            },
            "interaction": {
                "dragNodes": True,
                "dragView": True,
                "zoomView": True,
            },
        },
    }
