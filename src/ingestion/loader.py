"""Document loading logic for the ingestion pipeline."""

import os
from typing import List, Type
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from src.utils import setup_logger
from src.ingestion.base import BaseIngestion
from src.config import settings

# Initialize logger for this module
logger = setup_logger(__name__)

def load_documents(
    directory: str, 
    glob_pattern: str = "*.pdf", 
    loader_cls: Type = PyPDFLoader
) -> List[Document]:
    """
    Loads documents from a directory matching a specific pattern.

    Args:
        directory (str): Path to the directory containing files.
        glob_pattern (str): The file pattern to match (e.g., "*.pdf", "*.txt").
                            Defaults to "*.pdf".
        loader_cls (Type): The LangChain loader class to use (e.g., PyPDFLoader).
                           Defaults to PyPDFLoader.

    Returns:
        List[Document]: A list of loaded documents.
    """
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        raise FileNotFoundError(f"Directory not found: {directory}")

    logger.info(f"Loading documents from {directory} matching '{glob_pattern}'...")

    try:
        # Initialize DirectoryLoader with the dynamic loader class
        loader = DirectoryLoader(
            directory, 
            glob=glob_pattern, 
            loader_cls=loader_cls
        )
        documents = loader.load()
        
        if not documents:
            logger.warning(f"No documents found in {directory} matching {glob_pattern}")
        else:
            logger.info(f"Successfully loaded {len(documents)} documents.")
            
        return documents

    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise e
