"""Centralized logging configuration.

This module provides a consistent logger setup across the application.
It ensures logs are formatted correctly and respect the global log level
setting defined in the application configuration.
"""

import logging
import sys
from src.config import settings

def setup_logger(name: str) -> logging.Logger:
    """Configures and returns a logger instance.

    Args:
        name (str): The name of the logger, typically __name__ of the module.

    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Check if handlers are already set to prevent duplicate logs
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        
        # Define the log format
        # Example: 2023-10-27 10:00:00 - src.ingestion.loader - INFO - Loading docs...
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Set the log level from config (default to INFO)
        log_level = settings.log_level.upper()
        logger.setLevel(log_level)

    return logger