"""Ingestion Manager Implementation.

This module serves as the orchestrator for the Ingestion Layer. 
It implements the `BaseIngestion` interface but delegates the actual heavy lifting 
to specialized functional modules (`loader.py` and `splitter.py`).
"""

from typing import List, Optional, Any
from langchain_core.documents import Document
from src.ingestion.base import BaseIngestion
from src.ingestion.loader import load_documents
from src.ingestion.splitter import split_documents
from src.utils import setup_logger

logger = setup_logger(__name__)

class IngestionManager(BaseIngestion):
    """
    Concrete implementation of the Ingestion Layer.
    
    This class ties together the loading and chunking logic. It uses the 
    Facade pattern to provide a simple API while managing the complexity 
    of underlying worker modules.
    """

    def load(self, source: str, **kwargs: Any) -> List[Document]:
        """
        Loads documents from the file system.
        
        Delegates the loading process to `loader.load_documents`, which 
        automatically handles supported file types (CSV, PDF) found in the source directory.

        Args:
            source (str): The directory path to load from.
            **kwargs: Reserved for future extensibility (e.g., recursive loading).

        Returns:
            List[Document]: A list of loaded documents.
        """
        logger.info(f"Manager delegating load to loader module for source: {source}")
        
        # We no longer pass 'glob_pattern' because the loader now intelligently 
        # scans for all supported formats (CSV & PDF)
        return load_documents(source)

    def chunk(self, documents: List[Document], chunk_size: Optional[int] = None) -> List[Document]:
        """
        Splits documents into semantic chunks.

        Delegates to the functional splitter module.

        Args:
            documents (List[Document]): The raw documents to be split.
            chunk_size (int, optional): The target size for each text chunk. 
                                        If not provided, uses the global setting.

        Returns:
            List[Document]: A list of chunked Document objects.
        """
        return split_documents(documents, chunk_size=chunk_size)