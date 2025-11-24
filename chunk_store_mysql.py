"""
Chunk Analysis Store - MySQL persistence for chunked progressive analysis
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
import pymysql
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "fatrag")
DB_PASS = os.getenv("DB_PASSWORD", "fatrag_pw")
DB_NAME = os.getenv("DB_NAME", "fatrag")


def get_connection():
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False
    )


def create_chunk_job(
    job_id: str,
    project_id: str,
    doc_path: str,
    chunk_size: int,
    chunk_overlap: int,
    total_chunks: int,
    model_name: str,
    worker_count: int,
    output_dir: str
) -> Dict[str, Any]:
    """Create a new chunk analysis job"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chunk_job_metadata 
                (job_id, project_id, doc_path, chunk_size, chunk_overlap, 
                 total_chunks, model_name, worker_count, output_dir, status, started_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'chunking', NOW())
            """, (job_id, project_id, doc_path, chunk_size, chunk_overlap,
                  total_chunks, model_name, worker_count, output_dir))
            conn.commit()
        return get_chunk_job(job_id)
    finally:
        conn.close()


def update_chunk_job(
    job_id: str,
    status: Optional[str] = None,
    chunks_completed: Optional[int] = None,
    chunks_failed: Optional[int] = None,
    eta_minutes: Optional[float] = None,
    throughput: Optional[float] = None,
    error_message: Optional[str] = None
) -> Dict[str, Any]:
    """Update chunk job metadata"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            updates = []
            params = []
            
            if status:
                updates.append("status = %s")
                params.append(status)
                if status in ('completed', 'failed'):
                    updates.append("completed_at = NOW()")
            
            if chunks_completed is not None:
                updates.append("chunks_completed = %s")
                params.append(chunks_completed)
            
            if chunks_failed is not None:
                updates.append("chunks_failed = %s")
                params.append(chunks_failed)
            
            if eta_minutes is not None:
                updates.append("eta_minutes = %s")
                params.append(eta_minutes)
            
            if throughput is not None:
                updates.append("throughput_chunks_per_min = %s")
                params.append(throughput)
            
            if error_message is not None:
                updates.append("error_message = %s")
                params.append(error_message)
            
            if updates:
                params.append(job_id)
                sql = f"UPDATE chunk_job_metadata SET {', '.join(updates)} WHERE job_id = %s"
                cur.execute(sql, params)
                conn.commit()
        
        return get_chunk_job(job_id)
    finally:
        conn.close()


def get_chunk_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get chunk job metadata"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM chunk_job_metadata WHERE job_id = %s
            """, (job_id,))
            return cur.fetchone()
    finally:
        conn.close()


def create_chunk(
    chunk_id: str,
    job_id: str,
    doc_name: str,
    chunk_index: int,
    total_chunks: int,
    chunk_text: str
) -> Dict[str, Any]:
    """Create a chunk record"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chunk_analysis 
                (chunk_id, job_id, doc_name, chunk_index, total_chunks, chunk_text, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'pending')
            """, (chunk_id, job_id, doc_name, chunk_index, total_chunks, chunk_text))
            conn.commit()
        return get_chunk(chunk_id)
    finally:
        conn.close()


def update_chunk(
    chunk_id: str,
    status: Optional[str] = None,
    result_json: Optional[str] = None,
    error_message: Optional[str] = None,
    processing_time_sec: Optional[float] = None
) -> Dict[str, Any]:
    """Update chunk status and result"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            updates = []
            params = []
            
            if status == 'processing':
                updates.append("status = %s")
                updates.append("started_at = NOW()")
                params.append(status)
            elif status in ('completed', 'failed'):
                updates.append("status = %s")
                updates.append("completed_at = NOW()")
                params.append(status)
                if status == 'failed':
                    updates.append("retry_count = retry_count + 1")
            
            if result_json is not None:
                updates.append("result_json = %s")
                params.append(result_json)
            
            if error_message is not None:
                updates.append("error_message = %s")
                params.append(error_message)
            
            if processing_time_sec is not None:
                updates.append("processing_time_sec = %s")
                params.append(processing_time_sec)
            
            if updates:
                params.append(chunk_id)
                sql = f"UPDATE chunk_analysis SET {', '.join(updates)} WHERE chunk_id = %s"
                cur.execute(sql, params)
                conn.commit()
        
        return get_chunk(chunk_id)
    finally:
        conn.close()


def get_chunk(chunk_id: str) -> Optional[Dict[str, Any]]:
    """Get chunk record"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM chunk_analysis WHERE chunk_id = %s", (chunk_id,))
            return cur.fetchone()
    finally:
        conn.close()


def get_pending_chunks(job_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Get pending chunks for processing"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM chunk_analysis 
                WHERE job_id = %s AND status IN ('pending', 'failed')
                AND retry_count < 3
                ORDER BY chunk_index
                LIMIT %s
            """, (job_id, limit))
            return cur.fetchall()
    finally:
        conn.close()


def get_job_chunks(job_id: str) -> List[Dict[str, Any]]:
    """Get all chunks for a job"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM chunk_analysis 
                WHERE job_id = %s 
                ORDER BY chunk_index
            """, (job_id,))
            return cur.fetchall()
    finally:
        conn.close()


def get_job_progress(job_id: str) -> Dict[str, Any]:
    """Get job progress summary"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Get job metadata
            cur.execute("SELECT * FROM chunk_job_metadata WHERE job_id = %s", (job_id,))
            job = cur.fetchone()
            
            if not job:
                return None
            
            # Get chunk stats
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) as processing,
                    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                    AVG(CASE WHEN status = 'completed' THEN processing_time_sec ELSE NULL END) as avg_time
                FROM chunk_analysis WHERE job_id = %s
            """, (job_id,))
            stats = cur.fetchone()
            
            return {
                'job_id': job_id,
                'status': job['status'],
                'total_chunks': job['total_chunks'],
                'completed': stats['completed'] or 0,
                'failed': stats['failed'] or 0,
                'processing': stats['processing'] or 0,
                'pending': stats['pending'] or 0,
                'progress_pct': round((stats['completed'] or 0) / job['total_chunks'] * 100, 1) if job['total_chunks'] > 0 else 0,
                'avg_processing_time': round(stats['avg_time'], 2) if stats['avg_time'] else None,
                'eta_minutes': job['eta_minutes'],
                'throughput': job['throughput_chunks_per_min'],
                'created_at': job['created_at'],
                'started_at': job['started_at'],
                'completed_at': job['completed_at']
            }
    finally:
        conn.close()


def generate_chunk_id(job_id: str, chunk_index: int) -> str:
    """Generate unique chunk ID"""
    s = f"{job_id}:{chunk_index}"
    return hashlib.sha256(s.encode()).hexdigest()[:16]
