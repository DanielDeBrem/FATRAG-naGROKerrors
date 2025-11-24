import os
from typing import Optional, Dict, Any, List
import pymysql
from pymysql.cursors import DictCursor
from datetime import datetime
import json


def get_db_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER", "fatrag"),
        password=os.getenv("DB_PASSWORD", "fatrag_pw"),
        database=os.getenv("DB_NAME", "fatrag"),
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=False,
    )


def ensure_jobs_table(conn) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                job_id VARCHAR(128) NOT NULL UNIQUE,
                job_type VARCHAR(64) NOT NULL,
                project_id VARCHAR(128) NULL,
                status VARCHAR(32) NOT NULL DEFAULT 'queued',
                progress INT NOT NULL DEFAULT 0,
                result_filename VARCHAR(512) NULL,
                error_message TEXT NULL,
                metadata JSON NULL,
                created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX (project_id),
                INDEX (status),
                INDEX (job_type)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )

        # Backward-compatible schema upgrades: add any missing columns/indexes if table existed before.
        def _column_missing(col_name):
            cursor.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = 'jobs'
                  AND COLUMN_NAME = %s
                """,
                (col_name,),
            )
            row = cursor.fetchone() or {}
            return int(row.get("cnt", 0)) == 0

        alters = []
        if _column_missing("project_id"):
            alters.append("ADD COLUMN project_id VARCHAR(128) NULL")
            alters.append("ADD INDEX idx_jobs_project_id (project_id)")
        if _column_missing("progress"):
            alters.append("ADD COLUMN progress INT NOT NULL DEFAULT 0")
        if _column_missing("result_filename"):
            alters.append("ADD COLUMN result_filename VARCHAR(512) NULL")
        if _column_missing("error_message"):
            alters.append("ADD COLUMN error_message TEXT NULL")
        if _column_missing("metadata"):
            alters.append("ADD COLUMN metadata JSON NULL")
        if _column_missing("created_at"):
            alters.append("ADD COLUMN created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP")
        if _column_missing("updated_at"):
            alters.append("ADD COLUMN updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP")

        if alters:
            cursor.execute(f"ALTER TABLE jobs {', '.join(alters)}")
            conn.commit()


def create_job(job_id: str, job_type: str, project_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    conn = get_db_connection()
    try:
        ensure_jobs_table(conn)
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO jobs (job_id, job_type, project_id, status, progress, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(
                sql,
                (job_id, job_type, project_id, "queued", 0, (json.dumps(metadata, ensure_ascii=False) if metadata else None)),
            )
        conn.commit()
        return get_job(job_id) or {}
    finally:
        conn.close()


def update_job(
    job_id: str,
    *,
    status: Optional[str] = None,
    progress: Optional[int] = None,
    result_filename: Optional[str] = None,
    error_message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        ensure_jobs_table(conn)
        sets = []
        params: List[Any] = []
        if status is not None:
            sets.append("status = %s")
            params.append(status)
        if progress is not None:
            sets.append("progress = %s")
            params.append(max(0, min(100, progress)))
        if result_filename is not None:
            sets.append("result_filename = %s")
            params.append(result_filename)
        if error_message is not None:
            sets.append("error_message = %s")
            params.append(error_message)
        if metadata is not None:
            sets.append("metadata = %s")
            params.append(json.dumps(metadata, ensure_ascii=False))

        if not sets:
            return get_job(job_id)

        params.append(job_id)
        with conn.cursor() as cursor:
            cursor.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = %s", params)
        conn.commit()
        return get_job(job_id)
    finally:
        conn.close()


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        ensure_jobs_table(conn)
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM jobs WHERE job_id = %s", (job_id,))
            return cursor.fetchone()
    finally:
        conn.close()


def list_jobs(
    *,
    status: Optional[str] = None,
    job_type: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        ensure_jobs_table(conn)
        where = []
        params: List[Any] = []
        if status:
            where.append("status = %s")
            params.append(status)
        if job_type:
            where.append("job_type = %s")
            params.append(job_type)
        if project_id:
            where.append("project_id = %s")
            params.append(project_id)
        where_clause = f"WHERE {' AND '.join(where)}" if where else ""
        sql = f"SELECT * FROM jobs {where_clause} ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()
    finally:
        conn.close()
