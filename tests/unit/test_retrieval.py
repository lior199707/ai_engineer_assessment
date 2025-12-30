"""Unit tests for the Retrieval Layer (Vector Database).

This module validates the Retrieval architecture, specifically:
1. The Factory Pattern (`get_vector_db`): Ensuring configuration correctly swaps implementations.
2. The Concrete Implementation (`ChromaVectorDB`): Ensuring the database wrapper works as expected.

We extensively use mocking here to prevent:
- Writing actual files to disk during testing.
- Initializing heavy embedding models.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.retrieval.vector_db import ChromaVectorDB, get_vector_db
from src.config import LLMProvider, VectorDBType

@pytest.mark.unit
class TestRetrievalFactory:
    """Tests for the 'get_vector_db' factory function."""

    def test_get_vector_db_returns_chroma(self):
        """
        Verify the factory returns the correct class (ChromaVectorDB) when configured.
        
        This proves that the Factory Pattern is correctly reading the 
        'vector_db_type' setting from the environment/config.
        """
        # Patch the setting to force Chroma
        with patch("src.config.settings.vector_db_type", VectorDBType.CHROMA):
            db_instance = get_vector_db()
            assert isinstance(db_instance, ChromaVectorDB)

    def test_get_vector_db_invalid(self):
        """
        Verify the factory raises a helpful error for unsupported types.
        
        This ensures fail-fast behavior if a user misconfigures the .env file.
        """
        with patch("src.config.settings.vector_db_type", "unsupported_db"):
            with pytest.raises(ValueError):
                get_vector_db()

@pytest.mark.unit
class TestChromaVectorDB:
    """Tests for the concrete ChromaDB implementation logic."""

    @pytest.fixture
    def mock_settings(self):
        """
        Fixture to patch global settings for controlled testing.
        
        This isolates the test from the actual .env file, ensuring consistent
        behavior regardless of the developer's local setup.
        """
        with patch("src.retrieval.vector_db.settings") as mock_settings:
            mock_settings.vector_db_path = "fake/db/path"
            mock_settings.llm_provider = LLMProvider.OPENAI
            mock_settings.openai_embedding_model = "fake-model"
            mock_settings.openai_api_key = "fake-key"
            yield mock_settings

    @patch("src.retrieval.vector_db.Chroma")
    @patch("src.retrieval.vector_db.OpenAIEmbeddings")
    def test_create_vector_store(self, mock_embeddings, mock_chroma, mock_settings):
        """
        Test that document chunks are passed to ChromaDB correctly.
        
        This verifies that our wrapper class delegates the 'persist' command
        to the underlying LangChain Chroma implementation with the right arguments.
        """
        db = ChromaVectorDB()
        docs = [Document(page_content="test")]

        db.create_vector_store(docs)

        mock_chroma.from_documents.assert_called_once()
        call_args = mock_chroma.from_documents.call_args
        # Ensure data integrity: the exact documents passed in must reach Chroma
        assert call_args.kwargs['documents'] == docs

    @patch("src.retrieval.vector_db.Chroma")
    @patch("src.retrieval.vector_db.OpenAIEmbeddings")
    @patch("os.path.exists", return_value=True) # Mock file system check
    def test_as_retriever_success(self, mock_exists, mock_embeddings, mock_chroma, mock_settings):
        """
        Test successful retriever initialization.
        
        We mock 'os.path.exists' to simulate an existing database, verifying
        that the class correctly loads the persistence directory.
        """
        db = ChromaVectorDB()
        retriever = db.as_retriever()

        mock_chroma.assert_called()
        # Verify we requested a retriever with the standard k=5 search kwargs
        mock_chroma.return_value.as_retriever.assert_called_with(search_kwargs={"k": 5})