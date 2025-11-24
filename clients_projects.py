"""
Client and Project Management Module
Handles CRUD operations for clients and projects with MySQL backend
"""

import os
import uuid
import json
from typing import List, Dict, Any, Optional
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


def generate_id(prefix: str = "") -> str:
    """Generate unique ID with optional prefix"""
    unique = str(uuid.uuid4())[:8]
    return f"{prefix}{unique}" if prefix else unique


# ============= CLIENT OPERATIONS =============

def create_client(
    name: str,
    type: str = "individual",
    tax_id: Optional[str] = None,
    contact_info: Optional[Dict] = None,
    notes: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Create a new client"""
    client_id = generate_id("client-")
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO clients (client_id, name, type, tax_id, contact_info, notes, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(
                sql,
                (
                    client_id,
                    name,
                    type,
                    tax_id,
                    json.dumps(contact_info) if contact_info else None,
                    notes,
                    json.dumps(metadata) if metadata else None,
                ),
            )
        conn.commit()
        return get_client(client_id)
    finally:
        conn.close()


def list_clients(archived: bool = False) -> List[Dict[str, Any]]:
    """List all clients, optionally including archived"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            if archived:
                sql = "SELECT * FROM clients ORDER BY created_at DESC"
            else:
                sql = "SELECT * FROM clients WHERE archived_at IS NULL ORDER BY created_at DESC"
            cursor.execute(sql)
            return cursor.fetchall()
    finally:
        conn.close()


def get_client(client_id: str) -> Optional[Dict[str, Any]]:
    """Get client by ID"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM clients WHERE client_id = %s"
            cursor.execute(sql, (client_id,))
            return cursor.fetchone()
    finally:
        conn.close()


def update_client(
    client_id: str,
    name: Optional[str] = None,
    type: Optional[str] = None,
    tax_id: Optional[str] = None,
    contact_info: Optional[Dict] = None,
    notes: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> Optional[Dict[str, Any]]:
    """Update client information"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            updates = []
            params = []
            
            if name is not None:
                updates.append("name = %s")
                params.append(name)
            if type is not None:
                updates.append("type = %s")
                params.append(type)
            if tax_id is not None:
                updates.append("tax_id = %s")
                params.append(tax_id)
            if contact_info is not None:
                updates.append("contact_info = %s")
                params.append(json.dumps(contact_info))
            if notes is not None:
                updates.append("notes = %s")
                params.append(notes)
            if metadata is not None:
                updates.append("metadata = %s")
                params.append(json.dumps(metadata))
            
            if not updates:
                return get_client(client_id)
            
            params.append(client_id)
            sql = f"UPDATE clients SET {', '.join(updates)} WHERE client_id = %s"
            cursor.execute(sql, params)
        conn.commit()
        return get_client(client_id)
    finally:
        conn.close()


def archive_client(client_id: str) -> Optional[Dict[str, Any]]:
    """Archive a client (soft delete)"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = "UPDATE clients SET archived_at = NOW() WHERE client_id = %s"
            cursor.execute(sql, (client_id,))
        conn.commit()
        return get_client(client_id)
    finally:
        conn.close()


# ============= PROJECT OPERATIONS =============

def create_project(
    client_id: str,
    name: str,
    type: str = "general",
    description: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Create a new project for a client"""
    project_id = generate_id("project-")
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO projects (project_id, client_id, name, type, description, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(
                sql,
                (
                    project_id,
                    client_id,
                    name,
                    type,
                    description,
                    json.dumps(metadata) if metadata else None,
                ),
            )
        conn.commit()
        return get_project(project_id)
    finally:
        conn.close()


def list_projects(
    client_id: Optional[str] = None,
    status: Optional[str] = None,
    archived: bool = False,
) -> List[Dict[str, Any]]:
    """List projects, optionally filtered by client or status"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            conditions = []
            params = []
            
            if not archived:
                conditions.append("archived_at IS NULL")
            if client_id:
                conditions.append("client_id = %s")
                params.append(client_id)
            if status:
                conditions.append("status = %s")
                params.append(status)
            
            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            sql = f"SELECT * FROM projects {where_clause} ORDER BY created_at DESC"
            cursor.execute(sql, params)
            return cursor.fetchall()
    finally:
        conn.close()


def get_project(project_id: str) -> Optional[Dict[str, Any]]:
    """Get project by ID"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM projects WHERE project_id = %s"
            cursor.execute(sql, (project_id,))
            return cursor.fetchone()
    finally:
        conn.close()


def get_project_with_documents(project_id: str) -> Optional[Dict[str, Any]]:
    """Get project with all associated documents"""
    project = get_project(project_id)
    if not project:
        return None
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
                SELECT * FROM documents 
                WHERE project_id = %s AND deleted_at IS NULL
                ORDER BY uploaded_at DESC
            """
            cursor.execute(sql, (project_id,))
            project["documents"] = cursor.fetchall()
        return project
    finally:
        conn.close()


def update_project(
    project_id: str,
    name: Optional[str] = None,
    type: Optional[str] = None,
    status: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> Optional[Dict[str, Any]]:
    """Update project information"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            updates = []
            params = []
            
            if name is not None:
                updates.append("name = %s")
                params.append(name)
            if type is not None:
                updates.append("type = %s")
                params.append(type)
            if status is not None:
                updates.append("status = %s")
                params.append(status)
            if description is not None:
                updates.append("description = %s")
                params.append(description)
            if metadata is not None:
                updates.append("metadata = %s")
                params.append(json.dumps(metadata))
            
            if not updates:
                return get_project(project_id)
            
            params.append(project_id)
            sql = f"UPDATE projects SET {', '.join(updates)} WHERE project_id = %s"
            cursor.execute(sql, params)
        conn.commit()
        return get_project(project_id)
    finally:
        conn.close()


def archive_project(project_id: str) -> Optional[Dict[str, Any]]:
    """Archive a project (soft delete)"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = "UPDATE projects SET archived_at = NOW(), status = 'archived' WHERE project_id = %s"
            cursor.execute(sql, (project_id,))
        conn.commit()
        return get_project(project_id)
    finally:
        conn.close()


# ============= DOCUMENT-PROJECT LINKING =============

def link_document_to_project(doc_id: str, project_id: str, client_id: Optional[str] = None) -> bool:
    """Link an existing document to a project"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = "UPDATE documents SET project_id = %s, client_id = %s WHERE doc_id = %s"
            cursor.execute(sql, (project_id, client_id, doc_id))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def get_project_documents(project_id: str) -> List[Dict[str, Any]]:
    """Get all documents for a project"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
                SELECT * FROM documents 
                WHERE project_id = %s AND deleted_at IS NULL
                ORDER BY uploaded_at DESC
            """
            cursor.execute(sql, (project_id,))
            return cursor.fetchall()
    finally:
        conn.close()
