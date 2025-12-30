"""Unit tests for the CLI Entry Point (main.py).

This module verifies the orchestration logic of the application.
It mocks the Factory functions and Manager classes to ensure that `main.py`
is decoupled from concrete implementations and simply wires components together.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.main import ingest, query

@pytest.mark.unit
class TestMainCLI:
    """Tests for the command-line interface logic."""

    # --- Ingestion Command Tests ---

    @patch("src.main.get_vector_db")  # <--- Mock the Factory
    @patch("src.main.IngestionManager")
    def test_ingest_success(self, mock_manager_cls, mock_get_db):
        """
        Verify the 'ingest' command flow using the Factory Pattern.
        
        Expected Flow:
        1. Initialize IngestionManager.
        2. Load documents -> Chunk documents.
        3. Get Vector DB from Factory.
        4. Persist chunks to Vector DB.
        """
        # Setup Mocks
        mock_manager = mock_manager_cls.return_value
        mock_db_instance = mock_get_db.return_value  # The object returned by factory
        
        # Simulate returning dummy data
        mock_manager.load.return_value = ["doc1"]
        mock_manager.chunk.return_value = ["chunk1", "chunk2"]

        # Execute
        ingest("dummy/dir")

        # Assertions
        mock_manager.load.assert_called_once_with("dummy/dir")
        mock_get_db.assert_called_once() # Was the factory called?
        mock_db_instance.create_vector_store.assert_called_once_with(["chunk1", "chunk2"]) # Was data persisted?

    # --- Query Command Tests ---

    @patch("src.main.RAGGenerator")
    @patch("src.main.get_vector_db") # <--- Mock the Factory
    def test_query_success(self, mock_get_db, mock_rag_cls):
        """
        Verify the 'query' command flow using the Factory Pattern.
        
        Expected Flow:
        1. Get Vector DB from Factory.
        2. Initialize Retriever.
        3. Initialize RAGGenerator and build Chain.
        4. Invoke Chain with the question.
        """
        # Setup Mocks
        mock_db_instance = mock_get_db.return_value
        mock_rag = mock_rag_cls.return_value
        mock_chain = mock_rag.get_chain.return_value
        
        # Simulate successful retrieval
        mock_retriever = MagicMock()
        mock_db_instance.as_retriever.return_value = mock_retriever
        
        # Execute
        query("What is AI?")

        # Assertions
        mock_get_db.assert_called_once()
        mock_db_instance.as_retriever.assert_called_once()
        mock_rag.get_chain.assert_called_once_with(mock_retriever)
        mock_chain.invoke.assert_called_once_with("What is AI?")