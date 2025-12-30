"""Vector Database Factory and Implementations.

This module uses the Factory Pattern to provide a unified interface for 
different vector database backends (e.g., Chroma, Pinecone). 
It allows the application to switch databases just by changing configuration.
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import settings, LLMProvider, VectorDBType
from src.utils import setup_logger
from src.retrieval.base import BaseVectorDB

logger = setup_logger(__name__)

class ChromaVectorDB(BaseVectorDB):
    """
    ChromaDB implementation of the vector store.
    
    Attributes:
        persist_directory (str): Local path where the vector DB is saved.
        embeddings (Embeddings): The LangChain embedding model instance.
    """

    def __init__(self):
        """Initializes ChromaDB with settings from config."""
        self.persist_directory = settings.vector_db_path
        self.embeddings = self._get_embedding_model()

    def _get_embedding_model(self):
        """Selects the embedding model based on configuration."""
        if settings.llm_provider == LLMProvider.GOOGLE:
            logger.info(f"Using Google Embeddings: {settings.google_embedding_model}")
            return GoogleGenerativeAIEmbeddings(
                model=settings.google_embedding_model,
                google_api_key=settings.google_api_key
            )
        else:
            logger.info(f"Using OpenAI Embeddings: {settings.openai_embedding_model}")
            return OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                openai_api_key=settings.openai_api_key
            )

    def create_vector_store(self, chunks: List[Document]) -> None:
        """Persists document chunks to disk using Chroma."""
        if not chunks:
            logger.warning("No chunks provided. Skipping.")
            return

        logger.info(f"Persisting {len(chunks)} chunks to {self.persist_directory}...")
        try:
            Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            logger.info("Vector store successfully created and persisted.")
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise e

    def as_retriever(self) -> VectorStoreRetriever:
        """Hydrates the DB from disk and returns a retriever."""
        if not os.path.exists(self.persist_directory):
            error_msg = f"Vector store not found at {self.persist_directory}."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        logger.info(f"Loading vector store from {self.persist_directory}")
        vectorstore = Chroma(
            persist_directory=self.persist_directory, 
            embedding_function=self.embeddings
        )
        return vectorstore.as_retriever(search_kwargs={"k": 5})


def get_vector_db() -> BaseVectorDB:
    """
    Factory function to get the configured Vector DB instance.
    
    Returns:
        BaseVectorDB: A concrete instance (e.g., ChromaVectorDB) based on settings.
    
    Raises:
        ValueError: If the configured vector_db_type is unsupported.
    """
    if settings.vector_db_type == VectorDBType.CHROMA:
        return ChromaVectorDB()
    
    # Future extensibility:
    # elif settings.vector_db_type == VectorDBType.PINECONE:
    #     return PineconeVectorDB()
    
    else:
        raise ValueError(f"Unsupported Vector DB Type: {settings.vector_db_type}")