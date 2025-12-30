"""Unit tests for Utility modules.

This module verifies the behavior of shared utility functions, primarily the logger.
Reliable logging is critical for observability in production environments.
"""

import logging
import pytest
from src.utils import setup_logger

@pytest.mark.unit
class TestUtils:
    """Tests for the logging utility."""

    def test_setup_logger_creation(self):
        """
        Verify that the logger is correctly initialized.
        
        Ensures that calling setup_logger returns a standard Python logging object
        configured with the specified name, allowing for consistent log tagging.
        """
        logger = setup_logger("test_logger")
        
        # Assert type and name
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
        
        # Verify it has handlers attached (console handler)
        # Without handlers, logs would be swallowed silently
        assert len(logger.handlers) > 0