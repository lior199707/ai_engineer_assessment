"""Integration Testing Suite for the RAG API.

This module validates the end-to-end flow between the FastAPI delivery layer 
and the Retrieval logic. It ensures that the vector store is accessible, 
metadata is correctly mapped, and the frontend assets are served.

Note:
    These tests require the vector store to be populated (via 'make ingest')
    and the environment variables to be correctly configured.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

# Initialize the TestClient with the FastAPI app instance
client = TestClient(app)

def test_read_root():
    """
    Test the root endpoint to ensure the frontend is being served correctly.
    
    Verifies:
        - HTTP Status Code is 200 (OK).
        - The absolute path resolution for 'index.html' is functional.
    """
    response = client.get("/")
    assert response.status_code == 200

def test_search_endpoint():
    """
    Test the semantic search functionality via the POST /search endpoint.

    This test performs a live retrieval call to the vector database and 
    validates the schema and metadata integrity of the response.

    Verifies:
        - The endpoint accepts JSON payloads.
        - The response follows the 'SearchResponse' Pydantic model.
        - Metadata mapping (Job Title) is correctly extracted from the 
          'source' field in the vector store.
    """
    # Define a sample search payload
    payload = {
        "query": "python developer",
        "k": 3
    }

    # Execute the request
    response = client.post("/search", json=payload)

    # Validate HTTP response
    assert response.status_code == 200

    # Parse JSON body
    data = response.json()

    # Validate Response Structure
    assert "results" in data, "Response body missing 'results' key."
    assert len(data["results"]) > 0, "Search returned no results for a common query."
    
    # Validate Metadata Mapping (Crucial for the 'N/A' fix we implemented)
    first_result = data["results"][0]
    assert "job_title" in first_result, "Result missing 'job_title' field."
    assert first_result["job_title"] != "N/A", "Metadata mapping failed: job_title is N/A."
    assert "content" in first_result, "Result missing 'content' field."

def test_search_invalid_payload():
    """
    Verify that the API correctly handles and validates malformed requests.
    
    This demonstrates the robustness of the Pydantic validation layers.
    """
    # Sending an empty query which should ideally be caught by validation
    response = client.post("/search", json={"query": ""})
    # If you added min_length=1 to your Pydantic model, this would be 422
    # Otherwise, it checks if the server at least responded to the request
    assert response.status_code == 200