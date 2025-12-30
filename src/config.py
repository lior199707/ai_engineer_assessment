"""Configuration settings for the RAG application.

This module manages environment variables and application constants using Pydantic.
It ensures type safety and provides default values for critical settings.
"""

from enum import Enum
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class VectorDBType(str, Enum):
    """Supported Vector Database providers."""
    CHROMA = "chroma"
    # PINECONE = "pinecone" (Future proofing placeholder)

class LLMProvider(str, Enum):
    """Supported Large Language Model providers."""
    OPENAI = "openai"
    GOOGLE = "google"

class Settings(BaseSettings):
    """Global application settings loaded from environment variables.

    Attributes:
        llm_provider (LLMProvider): The provider to use ('openai' or 'google').
        vector_db_type (VectorDBType): The vector database to use (default: 'chroma').
        
        vector_db_path (str): Local path to persist the vector database (Chroma-specific).
        
        # API Keys
        openai_api_key (str): API key for OpenAI services (Optional).
        google_api_key (str): API key for Google Gemini services (Optional).
        
        # Provider specific models
        openai_model_name (str): OpenAI model name (e.g., 'gpt-4o').
        google_model_name (str): Google model name (e.g., 'gemini-1.5-flash').
        
        openai_embedding_model (str): OpenAI embedding model.
        google_embedding_model (str): Google embedding model.
        
        # Ingestion params
        chunk_size (int): Number of characters per text chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.
        
        log_level (str): Logging verbosity (DEBUG, INFO, WARNING, ERROR).
    """
    
    # Core Application Logic
    llm_provider: LLMProvider = LLMProvider.GOOGLE
    vector_db_type: VectorDBType = VectorDBType.CHROMA  # <--- New setting
    
    # Ingestion & Persistence
    vector_db_path: str = "data/vector_store"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    log_level: str = "INFO"

    # OpenAI Settings (Optional)
    openai_api_key: Optional[str] = None
    openai_model_name: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # Google Settings (Optional)
    google_api_key: Optional[str] = None
    google_model_name: str = "gemini-1.5-flash"
    google_embedding_model: str = "models/embedding-001"

    # Pydantic configuration to load from .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# Singleton instance for import across the app
settings = Settings()