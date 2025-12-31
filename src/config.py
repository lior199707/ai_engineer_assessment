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

class LLMProvider(str, Enum):
    """Supported Large Language Model providers (for generation)."""
    OPENAI = "openai"
    GOOGLE = "google"

class EmbeddingProvider(str, Enum):
    """Supported Embedding Model providers (for retrieval)."""
    OPENAI = "openai"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"

class Settings(BaseSettings):
    """Global application settings loaded from environment variables."""
    
    # --- Core Application Logic ---
    # SoluGen Assignment Requirement: No LLM Generation, but we keep the structure.
    llm_provider: LLMProvider = LLMProvider.GOOGLE
    
    # SoluGen Assignment Requirement: Must use OpenAI Embeddings
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI 
    vector_db_type: VectorDBType = VectorDBType.CHROMA
    
    # --- Ingestion & Persistence ---
    vector_db_path: str = "data/vector_store"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    log_level: str = "INFO"

    # --- API Keys ---
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # --- Model Names ---
    openai_model_name: str = "gpt-4o"
    google_model_name: str = "gemini-2.5-flash"

    # Embedding Models
    # SoluGen Assignment Requirement: "text-embedding-3-small"
    openai_embedding_model: str = "text-embedding-3-small"
    google_embedding_model: str = "models/embedding-001"
    huggingface_embedding_model: str = "all-MiniLM-L6-v2"

    # --- Pydantic Config ---
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Singleton instance for import across the app
settings = Settings()