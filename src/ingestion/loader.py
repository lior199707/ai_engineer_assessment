"""Document ingestion and processing module.

This module handles the loading of raw documents (currently PDFs) from disk
and splitting them into smaller, manageable chunks for embedding.
"""

from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings
from src.utils import setup_logger

# Initialize logger for this module
logger = setup_logger(__name__)

def load_documents(directory_path: str) -> List[Document]:
    """Loads PDF documents from a specified directory.

    Args:
        directory_path (str): The relative or absolute path to the directory
            containing PDF files.

    Returns:
        List[Document]: A list of LangChain Document objects containing
            the text content and metadata of the loaded files.

    Raises:
        FileNotFoundError: If the directory_path does not exist.
        RuntimeError: If document loading fails (e.g., corrupted files).
    """
    logger.info(f"Scanning directory for PDFs: {directory_path}")
    
    try:
        loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        
        if not docs:
            logger.warning(f"No documents found or loaded from {directory_path}")
        else:
            logger.info(f"Successfully loaded {len(docs)} documents from {directory_path}")
            
        return docs
        
    except Exception as e:
        logger.error(f"Failed to load documents from {directory_path}. Error: {e}")
        # Re-raise the exception so the calling function knows it failed
        raise e

def split_documents(documents: List[Document]) -> List[Document]:
    """Splits a list of documents into smaller text chunks.

    This function uses a RecursiveCharacterTextSplitter, which attempts to 
    keep paragraphs and sentences together. Configuration for chunk size 
    and overlap is pulled from global settings.

    Args:
        documents (List[Document]): The list of original documents to split.

    Returns:
        List[Document]: A list of smaller Document chunks ready for embedding.
    """
    if not documents:
        logger.warning("No documents provided to split.")
        return []

    logger.debug(
        f"Splitting documents with chunk_size={settings.chunk_size} "
        f"and chunk_overlap={settings.chunk_overlap}"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    
    chunks = splitter.split_documents(documents)
    
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks