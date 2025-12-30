"""Document ingestion and processing module.

This module handles the loading of raw documents (currently PDFs) from disk
and splitting them into smaller, manageable chunks for embedding.
"""

from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings

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
    loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs

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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    return splitter.split_documents(documents)