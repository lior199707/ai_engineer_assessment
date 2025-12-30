"""Unit tests for the Ingestion Layer.

This module validates the behavior of the three ingestion components:
1. Loader (Functional): Loading files from disk.
2. Splitter (Functional): Breaking text into chunks.
3. Manager (Orchestrator): Coordinating the workflow and handling kwargs.

Design Note:
    We test the Manager using Mocks to ensure it delegates correctly.
    We test the Loader and Splitter closer to implementation to ensure logic is sound.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader  # <--- Added Import
from src.ingestion.loader import load_documents
from src.ingestion.splitter import split_documents
from src.ingestion.manager import IngestionManager

# --- Shared Fixtures ---
@pytest.fixture
def sample_docs():
    """Provides a list of dummy Document objects for testing."""
    return [
        Document(page_content="Hello World", metadata={"source": "doc1.pdf"}),
        Document(page_content="AI is great", metadata={"source": "doc2.pdf"}),
    ]

# --- 1. Tests for the Loader Module ---
@pytest.mark.unit
class TestLoader:
    """Tests for the functional loader logic in src.ingestion.loader."""
    
    @patch("src.ingestion.loader.os.path.exists", return_value=True)  # FIX 1: Mock file system check
    @patch("src.ingestion.loader.DirectoryLoader")
    def test_load_documents_defaults(self, mock_loader_cls, mock_exists, sample_docs):
        """
        Test loading with default settings (PDF).
        
        Verifies that if no arguments are passed, the loader looks for *.pdf
        and uses the PyPDFLoader class.
        """
        mock_loader_instance = mock_loader_cls.return_value
        mock_loader_instance.load.return_value = sample_docs

        load_documents("dummy/path")

        # Verify defaults: glob="*.pdf" and loader_cls is PyPDFLoader
        mock_loader_cls.assert_called_once_with(
            "dummy/path", 
            glob="*.pdf", 
            loader_cls=PyPDFLoader  # <--- FIX: Assert against the real class
        )

    @patch("src.ingestion.loader.os.path.exists", return_value=True) # FIX 1: Mock file system check
    @patch("src.ingestion.loader.DirectoryLoader")
    def test_load_documents_custom_pattern(self, mock_loader_cls, mock_exists, sample_docs):
        """
        Test loading with a custom file pattern (e.g., .txt).
        
        This ensures our loader is flexible enough to handle different file types
        passed via the Manager's kwargs.
        """
        mock_loader_instance = mock_loader_cls.return_value
        mock_loader_instance.load.return_value = sample_docs

        # Pass a custom pattern
        load_documents("dummy/path", glob_pattern="*.txt")

        # Verify custom glob was used, but loader_cls remains PyPDFLoader (default behavior)
        mock_loader_cls.assert_called_once_with(
            "dummy/path", 
            glob="*.txt", 
            loader_cls=PyPDFLoader  # <--- FIX: Assert against the real class
        )

# --- 2. Tests for the Splitter Module ---
@pytest.mark.unit
class TestSplitter:
    """Tests for the functional splitter logic in src.ingestion.splitter."""

    def test_split_documents_logic(self):
        """
        Test that the splitter actually splits text based on size.
        
        We force a small chunk_size to ensure the splitting logic triggers.
        We also patch 'settings' to ensure overlap isn't larger than the chunk size,
        which would cause a ValueError.
        """
        long_text = "word " * 500
        doc = Document(page_content=long_text, metadata={"id": 1})
        
        # We must reduce overlap to be smaller than the chunk_size (50)
        with patch("src.ingestion.splitter.settings") as mock_settings:
            mock_settings.chunk_size = 1000  # Default in config
            mock_settings.chunk_overlap = 10 # Small overlap for this test
            
            # Force a small chunk size to guarantee multiple chunks
            chunks = split_documents([doc], chunk_size=50)
            
        assert len(chunks) > 1
        assert isinstance(chunks[0], Document)

# --- 3. Tests for the Manager Class (The Conductor) ---
@pytest.mark.unit
class TestIngestionManager:
    """Tests for the IngestionManager class in src.ingestion.manager."""

    @patch("src.ingestion.manager.load_documents")
    def test_manager_passes_kwargs_to_loader(self, mock_load_fn):
        """
        Verify that kwargs passed to Manager.load are forwarded to the loader.
        
        This test ensures the Manager is not a bottleneck and respects the 
        generic interface contract by passing 'glob_pattern' correctly.
        """
        manager = IngestionManager()
        
        # Act: Pass a custom glob pattern via kwargs
        manager.load("data/raw", glob_pattern="*.txt")
        
        # Assert: The functional loader received the custom pattern
        mock_load_fn.assert_called_once_with("data/raw", glob_pattern="*.txt")

    @patch("src.ingestion.manager.load_documents")
    def test_manager_defaults_to_pdf(self, mock_load_fn):
        """
        Verify backward compatibility (defaults to PDF).
        
        If no kwargs are provided, the Manager should intelligently default
        to the project standard (*.pdf).
        """
        manager = IngestionManager()
        manager.load("data/raw")
        
        mock_load_fn.assert_called_once_with("data/raw", glob_pattern="*.pdf")

    @patch("src.ingestion.manager.split_documents")
    def test_manager_delegates_chunking(self, mock_split_fn, sample_docs):
        """
        Verify the Manager calls the splitter module correctly.
        
        Ensures that arguments (like chunk_size) are passed through 
        from the Manager to the 'splitter.py' module.
        """
        manager = IngestionManager()
        manager.chunk(sample_docs, chunk_size=500)
        
        # The manager should pass the arguments through to the functional splitter
        mock_split_fn.assert_called_once_with(sample_docs, chunk_size=500)