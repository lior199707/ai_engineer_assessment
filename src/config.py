"""Configuration settings for the RAG application.

This module manages environment variables and application constants using Pydantic.
It ensures type safety and provides default values for critical settings.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Global application settings loaded from environment variables.

    Attributes:
        openai_api_key (str): API key for OpenAI services.
        vector_db_path (str): Local path to persist the vector database.
        model_name (str): Name of the LLM model to use (e.g., 'gpt-4o').
        embedding_model (str): Name of the embedding model.
        chunk_size (int): Number of characters per text chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.
        log_level (str): Logging verbosity (DEBUG, INFO, WARNING, ERROR).
    """
    openai_api_key: str
    vector_db_path: str = "data/vector_store"
    model_name: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    log_level: str = "INFO"

    # Pydantic configuration to load from .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# Singleton instance for import across the app
settings = Settings()