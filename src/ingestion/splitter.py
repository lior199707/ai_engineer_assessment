"""Text splitting logic for the ingestion pipeline.

This module contains pure functions for breaking down large documents 
into smaller, semantically meaningful chunks suitable for vector embedding.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings
from src.utils import setup_logger

logger = setup_logger(__name__)

def split_documents(documents: List[Document], chunk_size: Optional[int] = None) -> List[Document]:
    """
    Splits documents into smaller chunks for vector embedding.

    It uses a recursive character splitter to maintain semantic context 
    by keeping paragraphs and sentences together where possible.

    Args:
        documents (List[Document]): The raw documents loaded from disk.
        chunk_size (int, optional): The target size for each chunk. 
                                    If None, defaults to `settings.chunk_size`.

    Returns:
        List[Document]: A list of smaller, chunked documents.
    """
    # Prioritize the passed argument, fallback to global settings
    final_chunk_size = chunk_size or settings.chunk_size
    
    logger.info(f"Splitting {len(documents)} documents with chunk_size={final_chunk_size}...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=final_chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")
    
    return chunks