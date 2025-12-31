"""Document loading logic for the ingestion pipeline.

This module handles the loading of raw data files from the disk.
It supports multiple file formats (PDF, CSV) to satisfy assignment requirements.
"""

import os
import glob
from typing import List
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_core.documents import Document
from src.utils import setup_logger

# Initialize logger for this module
logger = setup_logger(__name__)

def load_documents(directory: str) -> List[Document]:
    """
    Loads all supported documents (PDF and CSV) from a directory.

    Args:
        directory (str): Path to the directory containing raw files.

    Returns:
        List[Document]: A combined list of loaded documents.
    
    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        raise FileNotFoundError(f"Directory not found: {directory}")

    documents = []
    
    # 1. Load PDFs
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    for file_path in pdf_files:
        try:
            logger.info(f"Loading PDF: {file_path}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")

    # 2. Load CSVs (FIX: Added encoding='utf-8')
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    for file_path in csv_files:
        try:
            logger.info(f"Loading CSV: {file_path}")
            # FIX: explicit encoding handles Windows issues with special chars
            loader = CSVLoader(
                file_path=file_path, 
                source_column="Job Title", 
                encoding="utf-8"
            )
            documents.extend(loader.load())
        except Exception as e:
            logger.error(f"Failed to load CSV {file_path}: {e}")

    if not documents:
        logger.warning(f"No PDF or CSV documents found in {directory}")
    else:
        logger.info(f"Successfully loaded {len(documents)} total documents.")
        
    return documents