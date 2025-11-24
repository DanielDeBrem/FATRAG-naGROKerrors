"""
MySQL-based feedback store for FATRAG
Replaces file-based feedback_store.py
"""
import time
import uuid
from typing import Any, Dict, List, Optional
from db_models import SessionLocal, Feedback, Query
from datetime import datetime


def _now() -> float:
    """Return current Unix timestamp"""
    return time.time()


def submit_feedback(
    question: str,
    answer: str,
    rating: Optional[str] = None,  # "up" | "down" | None
    corrected_answer: Optional[str] = None,
    tags: Optional[List[str]] = None,
    user_role: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    query_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Submit user feedback and store in MySQL.
    Returns the stored record as a dictionary.
    """
    db = SessionLocal()
    try:
        # Generate unique feedback ID
        feedback_id = f"{int(_now()*1000)}-{uuid.uuid4().hex[:8]}"
        
        # Create feedback record
        feedback = Feedback(
            feedback_id=feedback_id,
            query_id=query_id,
            question=question,
            answer=answer,
            rating=rating,
            corrected_answer=corrected_answer,
            tags=tags or [],
            user_role=user_role,
            status="pending",
            meta_data=meta or {}
        )
        
        db.add(feedback)
        db.commit()
        db.refresh(feedback)
        
        # Convert to dict for return
        return {
            "id": feedback.feedback_id,
            "ts": feedback.created_at.timestamp() if feedback.created_at else _now(),
            "status": feedback.status,
            "question": feedback.question,
            "answer": feedback.answer,
            "rating": feedback.rating,
            "corrected_answer": feedback.corrected_answer,
            "tags": feedback.tags or [],
            "user_role": feedback.user_role,
            "meta": feedback.meta_data or {},
        }
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def list_feedback(status: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    """
    List feedback entries, optionally filtered by status.
    Returns most recent first.
    """
    db = SessionLocal()
    try:
        query = db.query(Feedback)
        
        # Filter by status if provided
        if status:
            query = query.filter(Feedback.status == status)
        
        # Order by created_at descending (newest first)
        query = query.order_by(Feedback.created_at.desc())
        
        # Apply limit
        query = query.limit(min(limit, 2000))
        
        # Execute and convert to list of dicts
        items = []
        for feedback in query.all():
            items.append({
                "id": feedback.feedback_id,
                "ts": feedback.created_at.timestamp() if feedback.created_at else 0,
                "status": feedback.status,
                "question": feedback.question,
                "answer": feedback.answer,
                "rating": feedback.rating,
                "corrected_answer": feedback.corrected_answer,
                "tags": feedback.tags or [],
                "user_role": feedback.user_role,
                "meta": feedback.meta_data or {},
                "query_id": feedback.query_id,
            })
        
        return items
    except Exception as e:
        print(f"Error listing feedback: {e}")
        return []
    finally:
        db.close()


def get_feedback(feedback_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific feedback entry by ID"""
    db = SessionLocal()
    try:
        feedback = db.query(Feedback).filter(
            Feedback.feedback_id == feedback_id
        ).first()
        
        if not feedback:
            return None
        
        return {
            "id": feedback.feedback_id,
            "ts": feedback.created_at.timestamp() if feedback.created_at else 0,
            "status": feedback.status,
            "question": feedback.question,
            "answer": feedback.answer,
            "rating": feedback.rating,
            "corrected_answer": feedback.corrected_answer,
            "tags": feedback.tags or [],
            "user_role": feedback.user_role,
            "meta": feedback.meta_data or {},
            "query_id": feedback.query_id,
        }
    except Exception as e:
        print(f"Error getting feedback: {e}")
        return None
    finally:
        db.close()


def update_status(
    feedback_id: str,
    status: str,
    corrected_answer: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Update feedback status to 'approved' or 'rejected'.
    Optionally set corrected_answer and additional metadata.
    """
    assert status in {"approved", "rejected"}, "Status must be 'approved' or 'rejected'"
    
    db = SessionLocal()
    try:
        feedback = db.query(Feedback).filter(
            Feedback.feedback_id == feedback_id
        ).first()
        
        if not feedback:
            return None
        
        # Update fields
        feedback.status = status
        
        if corrected_answer is not None:
            feedback.corrected_answer = corrected_answer
        
        if extra_meta:
            current_meta = feedback.meta_data or {}
            current_meta.update(extra_meta)
            feedback.meta_data = current_meta
        
        db.commit()
        db.refresh(feedback)
        
        return {
            "id": feedback.feedback_id,
            "ts": feedback.created_at.timestamp() if feedback.created_at else 0,
            "status": feedback.status,
            "question": feedback.question,
            "answer": feedback.answer,
            "rating": feedback.rating,
            "corrected_answer": feedback.corrected_answer,
            "tags": feedback.tags or [],
            "user_role": feedback.user_role,
            "meta": feedback.meta_data or {},
            "query_id": feedback.query_id,
            "updated_ts": feedback.updated_at.timestamp() if feedback.updated_at else _now(),
        }
    except Exception as e:
        db.rollback()
        print(f"Error updating feedback status: {e}")
        return None
    finally:
        db.close()
