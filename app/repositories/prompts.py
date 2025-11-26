"""Repository for document prompt configuration in MySQL."""

from __future__ import annotations

from typing import Dict, List, Optional

import clients_projects as cp
from app.core.config import get_settings


class PromptRecord(Dict[str, object]):
    """Simple dict-based record for a document prompt row.

    Keys:
      - id: int
      - doc_type: str
      - prompt_kind: str
      - template: str
      - max_context_tokens: Optional[int]
      - active: bool
    """

    # Typing helper only; behaves as a plain dict at runtime.
    pass


class PromptsRepository:
    """Repository for CRUD operations on document_prompts."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def _get_connection(self):
        """Reuse the shared DB connection factory (same as other repositories)."""
        return cp.get_db_connection()

    def list_doc_types(self) -> List[Dict[str, object]]:
        """Return distinct doc_types with simple stats.

        Result items:
          {
            "doc_type": str,
            "prompt_count": int,
            "active_prompts": int,
          }
        """
        sql = """
            SELECT
              doc_type,
              COUNT(*) AS prompt_count,
              SUM(CASE WHEN active = 1 THEN 1 ELSE 0 END) AS active_prompts
            FROM document_prompts
            GROUP BY doc_type
            ORDER BY doc_type
        """
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall() or []

        return [
            {
                "doc_type": row["doc_type"],
                "prompt_count": int(row.get("prompt_count", 0)),
                "active_prompts": int(row.get("active_prompts", 0)),
            }
            for row in rows
        ]

    def list_prompts_for_type(self, doc_type: str) -> List[PromptRecord]:
        """List all prompts for a given doc_type."""
        sql = """
            SELECT
              id,
              doc_type,
              prompt_kind,
              template,
              max_context_tokens,
              active
            FROM document_prompts
            WHERE doc_type = %s
            ORDER BY prompt_kind
        """
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (doc_type,))
                rows = cursor.fetchall() or []

        results: List[PromptRecord] = []
        for row in rows:
            rec: PromptRecord = PromptRecord(
                id=row["id"],
                doc_type=row["doc_type"],
                prompt_kind=row["prompt_kind"],
                template=row["template"],
                max_context_tokens=row.get("max_context_tokens"),
                active=bool(row.get("active", 1)),
            )
            results.append(rec)
        return results

    def get_prompt(self, doc_type: str, prompt_kind: str) -> Optional[PromptRecord]:
        """Get a single prompt record by (doc_type, prompt_kind)."""
        sql = """
            SELECT
              id,
              doc_type,
              prompt_kind,
              template,
              max_context_tokens,
              active
            FROM document_prompts
            WHERE doc_type = %s AND prompt_kind = %s
            LIMIT 1
        """
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (doc_type, prompt_kind))
                row = cursor.fetchone()
                if not row:
                    return None

        rec: PromptRecord = PromptRecord(
            id=row["id"],
            doc_type=row["doc_type"],
            prompt_kind=row["prompt_kind"],
            template=row["template"],
            max_context_tokens=row.get("max_context_tokens"),
            active=bool(row.get("active", 1)),
        )
        return rec

    def upsert_prompt(
        self,
        doc_type: str,
        prompt_kind: str,
        template: str,
        max_context_tokens: Optional[int] = None,
        active: bool = True,
    ) -> PromptRecord:
        """Insert or update a prompt for (doc_type, prompt_kind).

        If a record already exists, it is updated in-place.
        Returns the final stored record.
        """
        sql_insert = """
            INSERT INTO document_prompts
              (doc_type, prompt_kind, template, max_context_tokens, active)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
              template = VALUES(template),
              max_context_tokens = VALUES(max_context_tokens),
              active = VALUES(active),
              updated_at = CURRENT_TIMESTAMP
        """
        params = (
            doc_type,
            prompt_kind,
            template,
            max_context_tokens,
            1 if active else 0,
        )
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql_insert, params)
                conn.commit()

        # Re-read to return canonical row (with id, timestamps, etc.)
        rec = self.get_prompt(doc_type, prompt_kind)
        if not rec:
            # This should not normally happen; raise a clear error.
            raise RuntimeError(
                f"Failed to upsert prompt for doc_type={doc_type!r}, kind={prompt_kind!r}"
            )
        return rec

    def delete_prompt(self, doc_type: str, prompt_kind: str) -> bool:
        """Hard-delete a prompt by (doc_type, prompt_kind).

        For now we use a hard delete; if we need history later we can add a
        'deleted_at' column instead and perform a soft delete.
        """
        sql = """
            DELETE FROM document_prompts
            WHERE doc_type = %s AND prompt_kind = %s
        """
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (doc_type, prompt_kind))
                conn.commit()
                return cursor.rowcount > 0
