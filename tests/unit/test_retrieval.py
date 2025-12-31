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
from src.config import EmbeddingProvider, VectorDBType

@pytest.mark.unit
class TestRetrievalFactory:
    """Tests for the 'get_vector_db' factory function."""

    @patch("src.retrieval.vector_db.settings")
    def test_get_vector_db_returns_chroma(self, mock_settings):
        """
        Verify the factory returns the correct class (ChromaVectorDB) when configured.
        """
        # FIX 1: Set Enum for logic AND Strings for Pydantic validation
        mock_settings.vector_db_type = VectorDBType.CHROMA
        mock_settings.embedding_provider = EmbeddingProvider.OPENAI
        mock_settings.openai_embedding_model = "text-embedding-test"
        mock_settings.openai_api_key = "sk-test-key"
        
        # Call the function directly
        db_instance = get_vector_db()
        assert isinstance(db_instance, ChromaVectorDB)

    @patch("src.retrieval.vector_db.settings")
    def test_get_vector_db_invalid(self, mock_settings):
        """
        Verify the factory raises a helpful error for unsupported types.
        """
        mock_settings.vector_db_type = "unsupported_db"
        
        with pytest.raises(ValueError):
            get_vector_db()

@pytest.mark.unit
class TestChromaVectorDB:
    """Tests for the concrete ChromaDB implementation logic."""

    @patch("src.retrieval.vector_db.settings")
    @patch("src.retrieval.vector_db.Chroma")
    @patch("src.retrieval.vector_db.OpenAIEmbeddings")
    @patch("src.retrieval.vector_db.os.path.exists", return_value=False) 
    def test_create_vector_store(self, mock_exists, mock_embeddings, mock_chroma, mock_settings):
        """
        Test that document chunks are passed to ChromaDB correctly.
        """
        # FIX 1: Set settings to valid strings/Enums
        mock_settings.embedding_provider = EmbeddingProvider.OPENAI
        mock_settings.openai_embedding_model = "text-embedding-test"
        mock_settings.openai_api_key = "sk-test-key"
        
        db = ChromaVectorDB()
        docs = [Document(page_content="test")]

        db.create_vector_store(docs)

        mock_chroma.from_documents.assert_called_once()
        
        # FIX 2: Check kwargs instead of positional args
        # Your code likely uses: Chroma.from_documents(documents=docs, ...)
        call_args = mock_chroma.from_documents.call_args
        assert call_args.kwargs['documents'] == docs 

    @patch("src.retrieval.vector_db.settings")
    @patch("src.retrieval.vector_db.Chroma")
    @patch("src.retrieval.vector_db.OpenAIEmbeddings")
    @patch("src.retrieval.vector_db.os.path.exists", return_value=True) 
    def test_as_retriever_success(self, mock_exists, mock_embeddings, mock_chroma, mock_settings):
        """
        Test successful retriever initialization.
        """
        # Set settings
        mock_settings.embedding_provider = EmbeddingProvider.OPENAI
        mock_settings.openai_embedding_model = "text-embedding-test"
        mock_settings.openai_api_key = "sk-test-key"
        
        db = ChromaVectorDB()
        retriever = db.as_retriever()

        mock_chroma.assert_called()
        # Verify we requested a retriever with the standard k=5 search kwargs
        mock_chroma.return_value.as_retriever.assert_called_with(search_kwargs={"k": 5})