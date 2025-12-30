"""Abstract Interface for the Data Ingestion Layer.

This module defines the contract that any ingestion implementation must follow.
It adheres to the Dependency Inversion Principle, ensuring high-level modules
depend on abstractions rather than concrete details.
"""

from abc import ABC, abstractmethod
from typing import List, Any
from langchain_core.documents import Document

class BaseIngestion(ABC):
    """
    Abstract Interface for the Data Ingestion Layer.
    """

    @abstractmethod
    def load(self, source: str, **kwargs: Any) -> List[Document]:
        """
        Loads raw data from a specified source.

        Args:
            source (str): The origin of the data (e.g., file path, URL, database connection).
            **kwargs: Additional arguments specific to the concrete implementation.
                      This allows flexibility without breaking the interface contract.
                      Examples:
                      - File System: `glob_pattern="*.txt"`
                      - Web Scraper: `depth=2`

        Returns:
            List[Document]: A list of raw LangChain Document objects.
        """
        pass

    @abstractmethod
    def chunk(self, documents: List[Document], chunk_size: int = 1000) -> List[Document]:
        """
        Splits raw documents into smaller semantic units suitable for embedding.
        
        Args:
            documents (List[Document]): The raw documents to split.
            chunk_size (int): Target size for each chunk.

        Returns:
            List[Document]: List of chunked documents.
        """
        pass