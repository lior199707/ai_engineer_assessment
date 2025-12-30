"""Unit tests for the Configuration module.

This module verifies that the application settings (Pydantic models) are loaded correctly,
default values are respected, and Enum validations work as expected.
"""

import pytest
from src.config import Settings, LLMProvider, VectorDBType

@pytest.mark.unit
class TestConfig:
    """Tests for the Settings class and configuration logic."""

    def test_default_values(self):
        """
        Verify that settings load with safe defaults when environment variables are missing.
        
        This ensures that the application has a valid baseline configuration
        even if the user forgets to set specific non-critical variables.
        """
        # Clear env vars for isolation to ensure we test defaults, not actual envs
        with pytest.MonkeyPatch.context() as m:
            m.delenv("OPENAI_API_KEY", raising=False)
            m.delenv("GOOGLE_API_KEY", raising=False)
            
            # Re-instantiate settings (Pydantic reads env at creation)
            # We explicitly pass _env_file=None to ignore the .env file for this test
            settings = Settings(_env_file=None) # type: ignore
            
            assert settings.chunk_size == 1000
            assert settings.log_level == "INFO"
            assert settings.llm_provider == LLMProvider.GOOGLE  # Default provider
            assert settings.vector_db_type == VectorDBType.CHROMA # Default DB

    def test_provider_enum_handling(self):
        """
        Verify that string inputs are correctly mapped to the LLMProvider Enum.
        
        This tests the robustness of the configuration against string formatting,
        ensuring 'google' becomes LLMProvider.GOOGLE.
        """
        # Test Google mapping
        settings_google = Settings(llm_provider="google")
        assert settings_google.llm_provider == LLMProvider.GOOGLE
        assert settings_google.llm_provider == "google"  # Enum inherits from str

        # Test OpenAI mapping
        settings_openai = Settings(llm_provider="openai")
        assert settings_openai.llm_provider == LLMProvider.OPENAI

    def test_vector_db_enum_handling(self):
        """
        Verify that Vector DB strings are correctly mapped to Enums.
        
        This ensures that if a user puts 'chroma' in their .env file,
        it correctly resolves to the VectorDBType.CHROMA enum for the factory.
        """
        # Test default instantiation
        settings = Settings()
        assert settings.vector_db_type == VectorDBType.CHROMA
        
        # Test explicit string assignment
        settings_chroma = Settings(vector_db_type="chroma")
        assert settings_chroma.vector_db_type == VectorDBType.CHROMA
        assert settings_chroma.vector_db_type == "chroma"