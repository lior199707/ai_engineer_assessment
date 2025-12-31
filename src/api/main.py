"""FastAPI Backend Implementation for the SoluGen RAG Assignment.

This module serves as the primary API layer, providing endpoints for 
semantic document retrieval and serving frontend assets.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from src.retrieval.vector_db import get_vector_db
from src.utils import setup_logger

# Initialize Logger
logger = setup_logger(__name__)

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(
    title="SoluGen AI RAG Assignment",
    description="Professional Vector Search API for HR and Recruitment Data",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class SearchRequest(BaseModel):
    """Schema for incoming semantic search requests."""
    query: str = Field(..., json_schema_extra={"example": "Python developer..."})
    k: int = Field(default=5, ge=1, le=20, description="Number of results to return")

class SearchResponse(BaseModel):
    """Schema for structured retrieval results."""
    results: List[Dict[str, Any]]

@app.get("/", tags=["UI"])
async def read_root() -> FileResponse:
    """
    Serves the main SPA (Single Page Application) frontend.

    Returns:
        FileResponse: The index.html file.
    """
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend assets missing")
    return FileResponse(index_path)

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest) -> Dict[str, Any]:
    """
    Performs a semantic search using cosine relevance scores.

    This implementation accesses the underlying vector store property to 
    provide similarity scores and chunk-level transparency.

    Args:
        request (SearchRequest): Search parameters (query and Top-K).

    Returns:
        Dict[str, Any]: Formatted results meeting the 0.3 similarity threshold.

    Raises:
        HTTPException: 500 if the search engine fails or DB is missing.
    """
    try:
        if not request.query.strip():
            return {"results": []}

        # 1. Retrieve the database wrapper and the internal store instance
        vector_db_wrapper = get_vector_db()
        store = vector_db_wrapper.store 
        
        # 2. Execute Similarity Search with Raw Relevance Scores
        # Result: List[Tuple[Document, float]]
        docs_with_scores = store.similarity_search_with_relevance_scores(
            request.query, 
            k=request.k
        )

        # 3. Filtering and Formatting
        SIMILARITY_THRESHOLD = 0.3  
        formatted_results = []
        
        for doc, score in docs_with_scores:
            if score >= SIMILARITY_THRESHOLD:
                formatted_results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "job_title": doc.metadata.get("source", "N/A"),
                    "score": round(float(score), 4),
                    "id": doc.metadata.get("row", "N/A")
                })

        logger.info(f"Retrieved {len(formatted_results)} results for: '{request.query}'")
        return {"results": formatted_results}

    except FileNotFoundError as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=503, detail="Search database not initialized. Run ingestion.")
    except Exception as e:
        logger.error(f"Critical search error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Search Engine Error")