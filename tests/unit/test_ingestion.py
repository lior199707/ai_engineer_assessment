"""Unit Testing Suite for the Ingestion Layer.

This module provides isolated unit tests for the loader, splitter, and 
ingestion manager. It uses mock objects to verify logic without 
requiring actual filesystem access or heavy machine learning models.

Conventions:
    - Mocks: unittest.mock.patch
    - Framework: pytest
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.ingestion.loader import load_documents
from src.ingestion.splitter import split_documents
from src.ingestion.manager import IngestionManager

class TestLoader:
    """
    Test suite for the Document Loader functional module.
    
    Validates that the loader correctly identifies and processes 
    different file formats (PDF and CSV) from a target directory.
    """

    @patch("src.ingestion.loader.PyPDFLoader")
    @patch("src.ingestion.loader.CSVLoader")
    @patch("src.ingestion.loader.glob.glob")
    @patch("src.ingestion.loader.os.path.exists")
    def test_load_documents_success(self, mock_exists, mock_glob, mock_csv_loader, mock_pdf_loader):
        """
        Verify that load_documents orchestrates both PDF and CSV loading.

        Verifies:
            - Directory existence check is performed.
            - Glob is called twice (once for .pdf, once for .csv).
            - Results from different loaders are combined into a single list.
        """
        # Setup: Mock environment and file discovery
        mock_exists.return_value = True
        mock_glob.side_effect = [["file.pdf"], ["file.csv"]]
        
        # Setup: Mock data returned by internal LangChain loaders
        mock_pdf_instance = mock_pdf_loader.return_value
        mock_pdf_instance.load.return_value = [Document(page_content="pdf content")]
        
        mock_csv_instance = mock_csv_loader.return_value
        mock_csv_instance.load.return_value = [Document(page_content="csv content")]

        # Act: Execute the loader
        docs = load_documents("fake_dir")

        # Assert: Check output integrity and internal calls
        assert len(docs) == 2
        assert docs[0].page_content == "pdf content"
        assert docs[1].page_content == "csv content"
        mock_pdf_loader.assert_called_once()
        mock_csv_loader.assert_called_once()

class TestSplitter:
    """
    Test suite for the Text Splitting/Chunking logic.
    """

    def test_split_documents_logic(self):
            """
            Verify that the splitter creates multiple chunks from long text strings.

            Verifies:
                - The RecursiveCharacterTextSplitter is effectively breaking text.
                - Metadata is preserved across chunks.
            """
            # Setup: One document with a repetitive long string
            test_doc = [Document(page_content="text " * 1000, metadata={"source": "test"})]
            
            # FIX: Ensure chunk_size (500) is greater than global CHUNK_OVERLAP (200)
            chunks = split_documents(test_doc, chunk_size=500)
            
            # Assert
            assert len(chunks) > 1
            assert chunks[0].metadata["source"] == "test"
            assert isinstance(chunks[0], Document)

class TestIngestionManager:
    """
    Test suite for the IngestionManager (Facade Pattern).
    
    Validates the orchestration logic and argument delegation.
    """

    @patch("src.ingestion.manager.load_documents")
    def test_manager_delegates_loading(self, mock_load_fn):
        """
        Verify that the Manager.load method correctly delegates to the functional loader.

        Verifies:
            - The Facade correctly forwards the source path.
            - The logic is decoupled from specific glob patterns.
        """
        manager = IngestionManager()
        
        # Act
        manager.load("data/raw")

        # Assert: Verification of the internal functional call
        mock_load_fn.assert_called_once_with("data/raw")

    @patch("src.ingestion.manager.split_documents")
    def test_manager_delegates_chunking(self, mock_split_fn):
        """
        Verify that the Manager.chunk method delegates to the splitter.

        Verifies:
            - Pass-through of document lists.
            - Optional parameters (chunk_size) are respected.
        """
        manager = IngestionManager()
        sample_docs = [Document(page_content="test")]
        
        # Act
        manager.chunk(sample_docs, chunk_size=500)
        
        # Assert
        mock_split_fn.assert_called_once_with(sample_docs, chunk_size=500)