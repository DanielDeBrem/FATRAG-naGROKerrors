"""
Query and RAG API endpoints for FATRAG.
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
import time
import uuid

from src.core.database import get_db, Session
from src.models.database import Query, Document
from src.services.vector_db import milvus_service
from src.services.ollama_service import ollama_service

logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    """Query request model."""
    query_text: str
    query_type: Optional[str] = "general"
    session_id: Optional[str] = None
    max_results: Optional[int] = 10
    document_filter: Optional[Dict[str, Any]] = None
    language: Optional[str] = "nl"


class QueryResponse(BaseModel):
    """Query response model."""
    query_id: int
    response_text: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    response_time_ms: int
    session_id: str


class FeedbackRequest(BaseModel):
    """Feedback request model."""
    query_id: int
    rating: int  # 1-5 rating
    feedback: Optional[str] = None


@router.post("/ask", response_model=QueryResponse)
async def ask_query(
    query_request: QueryRequest,
    db: Session = Depends(get_db)
) -> QueryResponse:
    """
    Process a natural language query using RAG.
    
    Args:
        query_request: Query request data
        db: Database session
        
    Returns:
        QueryResponse: Query response with sources
    """
    start_time = time.time()
    
    try:
        # Generate session ID if not provided
        session_id = query_request.session_id or str(uuid.uuid4())
        
        # Generate query embedding
        query_embedding = await ollama_service.generate_embedding(query_request.query_text)
        
        # Build filter expression for Milvus if document filter is provided
        expr_filter = None
        if query_request.document_filter:
            filters = []
            if "company_name" in query_request.document_filter:
                filters.append(f'company_name == "{query_request.document_filter["company_name"]}"')
            if "financial_year" in query_request.document_filter:
                filters.append(f'financial_year == {query_request.document_filter["financial_year"]}')
            if "document_category" in query_request.document_filter:
                filters.append(f'document_category == "{query_request.document_filter["document_category"]}"')
            if filters:
                expr_filter = " and ".join(filters)
        
        # Search for relevant chunks in Milvus
        search_results = milvus_service.search_similar(
            collection_name="document_chunks",
            query_embedding=query_embedding,
            limit=query_request.max_results,
            expr_filter=expr_filter
        )
        
        if not search_results:
            # No results found
            response_text = "onvoldoende data"
            confidence_score = 0.0
            sources = []
        else:
            # Build context from search results
            context_chunks = []
            sources = []
            
            for result in search_results:
                # Get document information
                document = db.query(Document).filter(
                    Document.id == result["document_id"]
                ).first()
                
                if document:
                    source_info = {
                        "document_id": result["document_id"],
                        "document_title": document.original_filename,
                        "document_company": document.company_name,
                        "document_category": document.document_category,
                        "chunk_id": result["id"],
                        "chunk_index": result["chunk_index"],
                        "page_number": result["page_number"],
                        "section_title": result["section_title"],
                        "score": result["score"]
                    }
                    sources.append(source_info)
                    
                    # Add chunk content to context
                    context_chunks.append({
                        "content": result["content"],
                        "source": f"Document: {document.original_filename}, Page {result['page_number']}"
                    })
            
            # Build context for LLM
            context_text = "\n\n".join([
                f"Context {i+1}:\n{chunk['content']}\nSource: {chunk['source']}"
                for i, chunk in enumerate(context_chunks)
            ])
            
            # Generate response using Ollama
            system_prompt = """Je bent een financiële assistent die vragen beantwoordt op basis van de verstrekte documenten.
            Geef antwoorden in het Nederlands.
            Wees beknopt en professioneel.
            Als er onvoldoende informatie is, zeg "onvoldoende data".
            Vermijd het verzinnen van informatie die niet in de documenten staat."""
            
            user_prompt = f"""Vraag: {query_request.query_text}

Context informatie:
{context_text}

Geef een antwoord op basis van de bovenstaande context informatie."""
            
            response_text = await ollama_service.generate_response(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=500
            )
            
            # Calculate confidence score based on search results
            if search_results:
                avg_score = sum(result["score"] for result in search_results) / len(search_results)
                confidence_score = min(avg_score * 2, 1.0)  # Normalize to 0-1 range
            else:
                confidence_score = 0.0
        
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Save query to database
        query_record = Query(
            session_id=session_id,
            query_text=query_request.query_text,
            query_language=query_request.language,
            query_type=query_request.query_type,
            response_text=response_text,
            response_sources=sources,
            confidence_score=confidence_score,
            response_time_ms=response_time_ms,
            embedding_model=ollama_service.embedding_model,
            generation_model=ollama_service.generation_model,
            context_chunks_count=len(search_results),
            context_document_ids=list(set(result["document_id"] for result in search_results))
        )
        
        db.add(query_record)
        db.commit()
        db.refresh(query_record)
        
        return QueryResponse(
            query_id=query_record.id,
            response_text=response_text,
            sources=sources,
            confidence_score=confidence_score,
            response_time_ms=response_time_ms,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Failed to process query")


@router.get("/")
async def get_queries(
    skip: int = 0,
    limit: int = 20,
    session_id: Optional[str] = None,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get list of queries with pagination.
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        session_id: Filter by session ID
        db: Database session
        
    Returns:
        Dict: Queries list with pagination info
    """
    try:
        query = db.query(Query)
        
        if session_id:
            query = query.filter(Query.session_id == session_id)
        
        total = query.count()
        queries = query.offset(skip).limit(limit).order_by(Query.created_at.desc()).all()
        
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "queries": [
                {
                    "id": q.id,
                    "session_id": q.session_id,
                    "query_text": q.query_text,
                    "query_type": q.query_type,
                    "query_language": q.query_language,
                    "response_text": q.response_text[:200] + "..." if q.response_text and len(q.response_text) > 200 else q.response_text,
                    "confidence_score": q.confidence_score,
                    "response_time_ms": q.response_time_ms,
                    "context_chunks_count": q.context_chunks_count,
                    "user_rating": q.user_rating,
                    "created_at": q.created_at.isoformat() if q.created_at else None
                }
                for q in queries
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting queries: {e}")
        raise HTTPException(status_code=500, detail="Failed to get queries")


@router.get("/{query_id}")
async def get_query(
    query_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get detailed information about a specific query.
    
    Args:
        query_id: ID of the query
        db: Database session
        
    Returns:
        Dict: Query details
    """
    try:
        query = db.query(Query).filter(Query.id == query_id).first()
        if not query:
            raise HTTPException(status_code=404, detail="Query not found")
        
        return {
            "id": query.id,
            "session_id": query.session_id,
            "query_text": query.query_text,
            "query_type": query.query_type,
            "query_language": query.query_language,
            "response_text": query.response_text,
            "response_sources": query.response_sources,
            "confidence_score": query.confidence_score,
            "response_time_ms": query.response_time_ms,
            "embedding_model": query.embedding_model,
            "generation_model": query.generation_model,
            "context_chunks_count": query.context_chunks_count,
            "context_document_ids": query.context_document_ids,
            "user_rating": query.user_rating,
            "user_feedback": query.user_feedback,
            "created_at": query.created_at.isoformat() if query.created_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting query {query_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get query")


@router.post("/{query_id}/feedback")
async def submit_feedback(
    query_id: int,
    feedback: FeedbackRequest,
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Submit feedback for a query response.
    
    Args:
        query_id: ID of the query
        feedback: Feedback data
        db: Database session
        
    Returns:
        Dict: Feedback submission status
    """
    try:
        query = db.query(Query).filter(Query.id == query_id).first()
        if not query:
            raise HTTPException(status_code=404, detail="Query not found")
        
        # Validate rating
        if feedback.rating < 1 or feedback.rating > 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        
        # Update query with feedback
        query.user_rating = feedback.rating
        query.user_feedback = feedback.feedback
        
        db.commit()
        
        logger.info(f"Feedback submitted for query {query_id}: rating={feedback.rating}")
        
        return {"status": "success", "message": "Feedback submitted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback for query {query_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to submit feedback")


@router.get("/sessions/{session_id}")
async def get_session_queries(
    session_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get all queries for a specific session.
    
    Args:
        session_id: Session ID
        db: Database session
        
    Returns:
        Dict: Session queries
    """
    try:
        queries = db.query(Query).filter(
            Query.session_id == session_id
        ).order_by(Query.created_at.asc()).all()
        
        return {
            "session_id": session_id,
            "query_count": len(queries),
            "queries": [
                {
                    "id": q.id,
                    "query_text": q.query_text,
                    "query_type": q.query_type,
                    "response_text": q.response_text[:200] + "..." if q.response_text and len(q.response_text) > 200 else q.response_text,
                    "confidence_score": q.confidence_score,
                    "response_time_ms": q.response_time_ms,
                    "user_rating": q.user_rating,
                    "created_at": q.created_at.isoformat() if q.created_at else None
                }
                for q in queries
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting session queries for {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session queries")


@router.delete("/{query_id}")
async def delete_query(
    query_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Delete a query record.
    
    Args:
        query_id: ID of the query to delete
        db: Database session
        
    Returns:
        Dict: Deletion status
    """
    try:
        query = db.query(Query).filter(Query.id == query_id).first()
        if not query:
            raise HTTPException(status_code=404, detail="Query not found")
        
        db.delete(query)
        db.commit()
        
        logger.info(f"Query {query_id} deleted successfully")
        
        return {"status": "success", "message": f"Query {query_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting query {query_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete query")
