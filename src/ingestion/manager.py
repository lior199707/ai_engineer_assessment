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
from src.config import settings
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
        
        It extracts specific configuration options from `kwargs` to pass
        to the underlying loader, maintaining flexibility.

        Args:
            source (str): The directory path to load from.
            **kwargs: specific options. Supported keys:
                      - glob_pattern (str): The file pattern to match. Defaults to "*.pdf".

        Returns:
            List[Document]: A list of loaded documents.
        """
        # Extract the pattern from kwargs, defaulting to PDF if not provided
        # This keeps the Manager robust: it handles missing args gracefully
        pattern = kwargs.get("glob_pattern", "*.pdf")
        
        logger.info(f"Manager delegating load to loader module with pattern='{pattern}'")
        return load_documents(source, glob_pattern=pattern)

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