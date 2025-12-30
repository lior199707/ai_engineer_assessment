"""Abstract Interface for the Vector Database Layer.

This module defines the contract that any vector database implementation must follow.
It strictly adheres to the Dependency Inversion Principle (DIP), ensuring that the
rest of the application depends on this high-level abstraction rather than specific
implementations like ChromaDB, Pinecone, or FAISS.

Key Benefits:
    1. **Decoupling:** The application logic doesn't care which DB is used.
    2. **Testability:** We can easily swap in a 'MockVectorDB' for unit tests.
    3. **Flexibility:** Switching providers (e.g., to Pinecone) only requires 
       creating a new class that inherits from this interface.
"""

from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

class BaseVectorDB(ABC):
    """
    Abstract Base Class for Vector Database Management.
    
    All concrete vector store implementations must inherit from this class
    and implement the `create_vector_store` and `as_retriever` methods.
    """

    @abstractmethod
    def create_vector_store(self, chunks: List[Document]) -> None:
        """
        Ingests document chunks into the vector database and persists them.

        This method handles the embedding generation and storage process.
        It is designed to be idempotent where possible (depending on implementation).

        Args:
            chunks (List[Document]): A list of pre-chunked LangChain Document objects
                                     ready for embedding.
        
        Returns:
            None: The operation is expected to perform side effects (persistence).
        """
        pass

    @abstractmethod
    def as_retriever(self) -> VectorStoreRetriever:
        """
        Returns a retriever object capable of semantic similarity search.

        The retriever is the interface used by the LLM chain to fetch relevant context.
        It abstracts away the underlying vector search logic (k-NN, MMR, etc.).

        Returns:
            VectorStoreRetriever: A LangChain retriever initialized with the 
                                  correct embedding model and search parameters.
        """
        pass