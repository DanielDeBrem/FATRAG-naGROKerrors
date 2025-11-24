import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

FEEDBACK_DIR = os.path.join(os.path.dirname(__file__), "logs")
FEEDBACK_PATH = os.path.join(FEEDBACK_DIR, "feedback.jsonl")

_LOCK = threading.Lock()


def _ensure_dirs() -> None:
    os.makedirs(FEEDBACK_DIR, exist_ok=True)


def _now() -> float:
    return time.time()


def submit_feedback(
    question: str,
    answer: str,
    rating: Optional[str] = None,  # "up" | "down" | None
    corrected_answer: Optional[str] = None,
    tags: Optional[List[str]] = None,
    user_role: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Append a feedback entry with status='pending'. Returns the stored record.
    """
    _ensure_dirs()
    rec = {
        "id": f"{int(_now()*1000)}-{uuid.uuid4().hex[:8]}",
        "ts": _now(),
        "status": "pending",
        "question": question,
        "answer": answer,
        "rating": rating,
        "corrected_answer": corrected_answer,
        "tags": tags or [],
        "user_role": user_role,
        "meta": meta or {},
    }
    line = json.dumps(rec, ensure_ascii=False)
    with _LOCK:
        with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return rec


def _read_all() -> List[Dict[str, Any]]:
    if not os.path.isfile(FEEDBACK_PATH):
        return []
    items: List[Dict[str, Any]] = []
    with _LOCK:
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    # skip corrupt line
                    continue
    return items


def list_feedback(status: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    """
    Return most recent first, optionally filtered by status.
    """
    items = _read_all()
    if status:
        items = [x for x in items if x.get("status") == status]
    items.sort(key=lambda x: x.get("ts", 0), reverse=True)
    return items[: max(1, min(limit, 2000))]


def get_feedback(feedback_id: str) -> Optional[Dict[str, Any]]:
    for x in _read_all():
        if x.get("id") == feedback_id:
            return x
    return None


def _rewrite_all(items: List[Dict[str, Any]]) -> None:
    _ensure_dirs()
    with _LOCK:
        with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
            for x in items:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")


def update_status(
    feedback_id: str,
    status: str,
    corrected_answer: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Update status to 'approved' or 'rejected'. Optionally set corrected_answer.
    """
    assert status in {"approved", "rejected"}
    items = _read_all()
    found = None
    for x in items:
        if x.get("id") == feedback_id:
            x["status"] = status
            if corrected_answer is not None:
                x["corrected_answer"] = corrected_answer
            if extra_meta:
                m = x.get("meta") or {}
                m.update(extra_meta)
                x["meta"] = m
            x["updated_ts"] = _now()
            found = x
            break
    if found is None:
        return None
    _rewrite_all(items)
    return found
