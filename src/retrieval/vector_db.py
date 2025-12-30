"""Vector Database management module.

This module handles the interaction with the vector store (ChromaDB), including
generating embeddings for text chunks, persisting data to disk, and loading
the database for retrieval.
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import settings, LLMProvider
from src.utils import setup_logger

logger = setup_logger(__name__)

class VectorStoreManager:
    """Manages the creation and retrieval of vector embeddings.

    Attributes:
        embeddings: The embedding model instance (OpenAI or Google).
        persist_directory (str): Path where the vector DB is saved.
    """

    def __init__(self):
        """Initializes the VectorStoreManager with settings from config."""
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
        else: # LLMProvider.OPENAI
            logger.info(f"Using OpenAI Embeddings: {settings.openai_embedding_model}")
            return OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                openai_api_key=settings.openai_api_key
            )

    def create_vector_store(self, chunks: List[Document]) -> Chroma:
        """Creates a new vector store from document chunks and saves it to disk.

        Args:
            chunks (List[Document]): List of document chunks to embed.

        Returns:
            Chroma: The initialized and persisted Chroma vector store.
        """
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        return vectorstore

    def load_vector_store(self) -> Chroma:
        """Loads an existing vector store from the persistence directory.

        Returns:
            Chroma: The loaded vector store object.

        Raises:
            FileNotFoundError: If the persistence directory does not exist,
                indicating that ingestion has not been run yet.
        """
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(
                f"Vector store not found at {self.persist_directory}. "
                "Please run ingestion first."
            )
            
        return Chroma(
            persist_directory=self.persist_directory, 
            embedding_function=self.embeddings
        )