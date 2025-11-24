"""
Upload Progress Tracking Store
Manages progress tracking for file uploads through the ingestion pipeline
"""

import os
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import pymysql.cursors


def get_db_connection():
    """Get MySQL connection with project configuration"""
    return pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "fatrag_user"),
        password=os.getenv("DB_PASSWORD", "fatrag_password"),
        database=os.getenv("DB_NAME", "fatrag_db"),
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


def generate_id(prefix: str = "upload-") -> str:
    """Generate unique ID with timestamp"""
    return f"{prefix}{int(time.time() * 1000)}"


class UploadProgressTracker:
    """Track upload progress through all pipeline stages"""
    
    def __init__(self):
        self.conn = None
    
    def _get_conn(self):
        """Lazy connection getter"""
        if self.conn is None:
            self.conn = get_db_connection()
        return self.conn
    
    def create_upload(
        self,
        upload_id: str,
        filename: str,
        project_id: Optional[str] = None,
        client_id: Optional[str] = None,
        file_size: int = 0,
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new upload progress record"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO upload_progress 
                    (upload_id, filename, project_id, client_id, file_size, file_path, 
                     status, progress_percent, current_stage, started_at)
                    VALUES (%s, %s, %s, %s, %s, %s, 'queued', 0, 'queued', NOW())
                """
                cursor.execute(sql, (upload_id, filename, project_id, client_id, file_size, file_path))
            conn.commit()
            return self.get_upload(upload_id)
        except Exception as e:
            conn.rollback()
            raise Exception(f"Failed to create upload progress: {e}")
    
    def update_upload(
        self,
        upload_id: str,
        status: Optional[str] = None,
        progress_percent: Optional[int] = None,
        current_stage: Optional[str] = None,
        total_chunks: Optional[int] = None,
        total_tokens: Optional[int] = None,
        embedding_dimensions: Optional[int] = None,
        doc_id: Optional[str] = None,
        extraction_time_ms: Optional[int] = None,
        tokenization_time_ms: Optional[int] = None,
        embedding_time_ms: Optional[int] = None,
        indexing_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        error_stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update upload progress"""
        conn = self._get_conn()
        
        updates = []
        values = []
        
        if status is not None:
            updates.append("status = %s")
            values.append(status)
            if status == "completed":
                updates.append("completed_at = NOW()")
        
        if progress_percent is not None:
            updates.append("progress_percent = %s")
            values.append(progress_percent)
        
        if current_stage is not None:
            updates.append("current_stage = %s")
            values.append(current_stage)
        
        if total_chunks is not None:
            updates.append("total_chunks = %s")
            values.append(total_chunks)
        
        if total_tokens is not None:
            updates.append("total_tokens = %s")
            values.append(total_tokens)
        
        if embedding_dimensions is not None:
            updates.append("embedding_dimensions = %s")
            values.append(embedding_dimensions)
        
        if doc_id is not None:
            updates.append("doc_id = %s")
            values.append(doc_id)
        
        if extraction_time_ms is not None:
            updates.append("extraction_time_ms = %s")
            values.append(extraction_time_ms)
        
        if tokenization_time_ms is not None:
            updates.append("tokenization_time_ms = %s")
            values.append(tokenization_time_ms)
        
        if embedding_time_ms is not None:
            updates.append("embedding_time_ms = %s")
            values.append(embedding_time_ms)
        
        if indexing_time_ms is not None:
            updates.append("indexing_time_ms = %s")
            values.append(indexing_time_ms)
        
        if error_message is not None:
            updates.append("error_message = %s")
            values.append(error_message)
        
        if error_stage is not None:
            updates.append("error_stage = %s")
            values.append(error_stage)
        
        if not updates:
            return self.get_upload(upload_id)
        
        values.append(upload_id)
        sql = f"UPDATE upload_progress SET {', '.join(updates)} WHERE upload_id = %s"
        
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, tuple(values))
            conn.commit()
            return self.get_upload(upload_id)
        except Exception as e:
            conn.rollback()
            raise Exception(f"Failed to update upload progress: {e}")
    
    def get_upload(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Get upload progress by ID"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                sql = "SELECT * FROM upload_progress WHERE upload_id = %s"
                cursor.execute(sql, (upload_id,))
                return cursor.fetchone()
        except Exception:
            return None
    
    def list_uploads(
        self,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List uploads with optional filters"""
        conn = self._get_conn()
        try:
            conditions = []
            values = []
            
            if project_id:
                conditions.append("project_id = %s")
                values.append(project_id)
            
            if status:
                conditions.append("status = %s")
                values.append(status)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            sql = f"""
                SELECT * FROM upload_progress 
                WHERE {where_clause}
                ORDER BY created_at DESC 
                LIMIT %s
            """
            values.append(limit)
            
            with conn.cursor() as cursor:
                cursor.execute(sql, tuple(values))
                return cursor.fetchall()
        except Exception as e:
            print(f"Error listing uploads: {e}")
            return []
    
    def create_batch(self, batch_id: str, project_id: str, total_files: int) -> Dict[str, Any]:
        """Create a batch upload record"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO upload_batches 
                    (batch_id, project_id, total_files, status, started_at)
                    VALUES (%s, %s, %s, 'queued', NOW())
                """
                cursor.execute(sql, (batch_id, project_id, total_files))
            conn.commit()
            return self.get_batch(batch_id)
        except Exception as e:
            conn.rollback()
            raise Exception(f"Failed to create upload batch: {e}")
    
    def update_batch(
        self,
        batch_id: str,
        status: Optional[str] = None,
        completed_files: Optional[int] = None,
        failed_files: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update batch upload progress"""
        conn = self._get_conn()
        
        updates = []
        values = []
        
        if status is not None:
            updates.append("status = %s")
            values.append(status)
            if status in ["completed", "partial_failure", "failed"]:
                updates.append("completed_at = NOW()")
        
        if completed_files is not None:
            updates.append("completed_files = %s")
            values.append(completed_files)
        
        if failed_files is not None:
            updates.append("failed_files = %s")
            values.append(failed_files)
        
        if not updates:
            return self.get_batch(batch_id)
        
        values.append(batch_id)
        sql = f"UPDATE upload_batches SET {', '.join(updates)} WHERE batch_id = %s"
        
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, tuple(values))
            conn.commit()
            return self.get_batch(batch_id)
        except Exception as e:
            conn.rollback()
            raise Exception(f"Failed to update upload batch: {e}")
    
    def get_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get batch upload by ID"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                sql = "SELECT * FROM upload_batches WHERE batch_id = %s"
                cursor.execute(sql, (batch_id,))
                return cursor.fetchone()
        except Exception:
            return None
    
    def get_batch_uploads(self, batch_id: str) -> List[Dict[str, Any]]:
        """Get all uploads in a batch - matched by timing and project"""
        batch = self.get_batch(batch_id)
        if not batch:
            return []
        
        # Find uploads that were created around the same time as the batch
        # This is a heuristic since we don't have explicit batch linkage yet
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                sql = """
                    SELECT * FROM upload_progress 
                    WHERE project_id = %s 
                    AND ABS(TIMESTAMPDIFF(SECOND, created_at, %s)) < 5
                    ORDER BY created_at ASC
                """
                cursor.execute(sql, (batch["project_id"], batch["created_at"]))
                return cursor.fetchall()
        except Exception:
            return []
    
    def __del__(self):
        """Close connection on cleanup"""
        if self.conn:
            try:
                self.conn.close()
            except:
                pass


# Create progress callback function factory
def create_progress_callback(tracker: UploadProgressTracker, upload_id: str) -> Callable:
    """
    Create a callback function that can be used to report progress during ingestion.
    
    Returns a function with signature: callback(stage, progress_percent, **kwargs)
    """
    def callback(stage: str, progress_percent: int, **kwargs):
        """Progress callback to update upload status"""
        try:
            # Map stage to status
            status_map = {
                "uploading": "uploading",
                "uploaded": "uploaded",
                "extracting": "extracting",
                "tokenizing": "tokenizing",
                "embedding": "embedding",
                "indexing": "indexing",
                "completed": "completed",
                "failed": "failed",
            }
            
            status = status_map.get(stage, "queued")
            
            # Extract timing and metrics from kwargs
            update_args = {
                "status": status,
                "progress_percent": min(100, max(0, progress_percent)),
                "current_stage": stage,
            }
            
            # Add optional metrics
            if "total_chunks" in kwargs:
                update_args["total_chunks"] = kwargs["total_chunks"]
            if "total_tokens" in kwargs:
                update_args["total_tokens"] = kwargs["total_tokens"]
            if "embedding_dimensions" in kwargs:
                update_args["embedding_dimensions"] = kwargs["embedding_dimensions"]
            if "doc_id" in kwargs:
                update_args["doc_id"] = kwargs["doc_id"]
            if "extraction_time_ms" in kwargs:
                update_args["extraction_time_ms"] = kwargs["extraction_time_ms"]
            if "tokenization_time_ms" in kwargs:
                update_args["tokenization_time_ms"] = kwargs["tokenization_time_ms"]
            if "embedding_time_ms" in kwargs:
                update_args["embedding_time_ms"] = kwargs["embedding_time_ms"]
            if "indexing_time_ms" in kwargs:
                update_args["indexing_time_ms"] = kwargs["indexing_time_ms"]
            if "error_message" in kwargs:
                update_args["error_message"] = kwargs["error_message"]
            if "error_stage" in kwargs:
                update_args["error_stage"] = kwargs["error_stage"]
            
            tracker.update_upload(upload_id, **update_args)
        except Exception as e:
            print(f"Progress callback error: {e}")
    
    return callback
