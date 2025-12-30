"""Unit tests for the Generation Layer (LLM & RAG).

This module tests the RAG pipeline construction. It ensures that the
correct LLM provider is initialized and the LangChain chain is built correctly,
without incurring costs by making actual API calls.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.generation.llm import RAGGenerator

@pytest.mark.unit
class TestRAGGenerator:
    """Tests for LLM initialization and Chain construction."""

    @patch("src.generation.llm.ChatOpenAI")
    def test_get_chain(self, mock_chat):
        """
        Test that the LCEL (LangChain Expression Language) chain is constructed.
        
        This verifies that 'get_chain' returns a runnable object capable of
        processing inputs. We verify the structure of the return object 
        rather than its output, as the output depends on the mocked LLM.
        """
        generator = RAGGenerator()
        mock_retriever = MagicMock()
        
        chain = generator.get_chain(mock_retriever)
        
        # The chain should be a Runnable (LangChain object)
        # We check for the 'invoke' method, which is the standard interface for LCEL chains
        assert hasattr(chain, "invoke")