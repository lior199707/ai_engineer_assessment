"""Vector Database Factory and Implementations.

This module uses the Factory Pattern to provide a unified interface for 
different vector database backends. It handles the initialization of 
embedding models (OpenAI, Google, HuggingFace) and provides a secure 
property to access the underlying vector store engine.
"""

import os
import shutil
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

# Imports for all supported embedding providers
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings, EmbeddingProvider, VectorDBType
from src.utils import setup_logger
from src.retrieval.base import BaseVectorDB

logger = setup_logger(__name__)

class ChromaVectorDB(BaseVectorDB):
    """
    ChromaDB implementation of the vector store wrapper.
    
    This class encapsulates the LangChain Chroma instance, providing 
    lifecycle management and a standardized interface for retrieval.

    Attributes:
        persist_directory (str): Local path where the vector DB is saved.
        embeddings (Embeddings): The initialized LangChain embedding model.
        _store (Optional[Chroma]): Internal cache of the LangChain Chroma instance.
    """

    def __init__(self):
        """Initializes ChromaDB wrapper with settings from the configuration."""
        self.persist_directory = settings.vector_db_path
        self.embeddings = self._get_embedding_model()
        self._store = None

    @property
    def store(self) -> Chroma:
        """
        Exposes the underlying LangChain Chroma instance.

        This property allows the API layer to perform advanced searches 
        (e.g., similarity_search_with_relevance_scores) that aren't 
        available through the standard high-level retriever.

        Returns:
            Chroma: The initialized LangChain Chroma object.

        Raises:
            FileNotFoundError: If the database hasn't been created on disk.
        """
        if self._store is None:
            if not os.path.exists(self.persist_directory):
                raise FileNotFoundError(f"No DB found at {self.persist_directory}")
            self._store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        return self._store

    def _get_embedding_model(self):
        """
        Selects and initializes the embedding model based on configuration.
        
        Returns:
            Embeddings: The initialized LangChain embedding model instance.
            
        Raises:
            ValueError: If an unsupported provider is configured.
        """
        provider = settings.embedding_provider

        if provider == EmbeddingProvider.OPENAI:
            logger.info(f"Using OpenAI Embeddings: {settings.openai_embedding_model}")
            return OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                api_key=settings.openai_api_key
            )
        
        elif provider == EmbeddingProvider.GOOGLE:
            logger.info(f"Using Google Embeddings: {settings.google_embedding_model}")
            return GoogleGenerativeAIEmbeddings(
                model=settings.google_embedding_model,
                google_api_key=settings.google_api_key
            )
            
        elif provider == EmbeddingProvider.HUGGINGFACE:
            logger.info(f"Using Local HuggingFace Embeddings: {settings.huggingface_embedding_model}")
            return HuggingFaceEmbeddings(model_name=settings.huggingface_embedding_model)
        
        else:
            raise ValueError(f"Unsupported Embedding Provider: {provider}")

    def create_vector_store(self, chunks: List[Document]) -> None:
        """
        Persists document chunks to disk using the Chroma engine.

        Args:
            chunks (List[Document]): Processed LangChain documents to be indexed.
        """
        if not chunks:
            logger.warning("No chunks provided for ingestion. Skipping.")
            return

        logger.info(f"Persisting {len(chunks)} chunks to {self.persist_directory}...")
        try:
            if os.path.exists(self.persist_directory):
                logger.warning("Existing DB found. Clearing to prevent dimension mismatch.")
                shutil.rmtree(self.persist_directory)

            self._store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            logger.info("Vector store successfully created and persisted.")
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise e

    def as_retriever(self) -> VectorStoreRetriever:
        """
        Returns the vector store as a standard LangChain retriever.

        Returns:
            VectorStoreRetriever: A hydrated retriever with k=5.
        """
        return self.store.as_retriever(search_kwargs={"k": 5})

def get_vector_db() -> BaseVectorDB:
    """
    Factory function to retrieve the configured Vector DB instance.
    
    Returns:
        BaseVectorDB: A concrete database instance (e.g., ChromaVectorDB).
    """
    if settings.vector_db_type == VectorDBType.CHROMA:
        return ChromaVectorDB()
    else:
        raise ValueError(f"Unsupported Vector DB Type: {settings.vector_db_type}")