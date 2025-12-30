"""Main entry point for the RAG Application CLI.

This script orchestrates the RAG pipeline by connecting the Ingestion, 
Retrieval, and Generation layers. It uses a Factory Pattern to load 
the configured Vector Database, making the system agnostic to the 
underlying storage technology (Chroma, Pinecone, etc.).

Usage:
    python main.py ingest --data data/raw
    python main.py query --q "What is the summary?"
"""

import argparse
import sys
from src.ingestion.manager import IngestionManager
from src.retrieval.vector_db import get_vector_db  # <--- UPDATED: Import Factory
from src.generation.llm import RAGGenerator
from src.utils import setup_logger

# Initialize logger
logger = setup_logger(__name__)

def ingest(data_dir: str) -> None:
    """Executes the data ingestion pipeline.
    
    This function:
    1. Loads and chunks documents using the IngestionManager.
    2. Persists them using the configured Vector Database.

    Args:
        data_dir (str): Path to the directory containing raw documents.
    """
    logger.info(f"Starting ingestion from {data_dir}...")
    
    # 1. Initialize the Ingestion Manager
    ingestion_manager = IngestionManager()
    
    try:
        # 2. Load Documents
        docs = ingestion_manager.load(data_dir)
        if not docs:
            logger.warning("No documents loaded. Aborting ingestion.")
            return

        # 3. Chunk Documents
        chunks = ingestion_manager.chunk(docs)
        
        # 4. Persist to Vector DB using Factory
        logger.info(f"Creating vector store with {len(chunks)} chunks...")
        
        # We ask the factory for the DB instance (e.g., ChromaVectorDB)
        vsm = get_vector_db() 
        vsm.create_vector_store(chunks)
        
        logger.info("Ingestion pipeline completed successfully.")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)

def query(question: str) -> None:
    """Executes the retrieval and generation pipeline.
    
    This function:
    1. Gets the configured Vector DB from the factory.
    2. initializes the Retriever.
    3. Runs the RAG chain to generate an answer.

    Args:
        question (str): The user's query string.
    """
    # 1. Get DB from Factory
    vsm = get_vector_db()
    
    try:
        # 2. Initialize Retriever
        # This encapsulates the specific DB logic (checking paths, connecting, etc.)
        retriever = vsm.as_retriever()
    except FileNotFoundError as e:
        logger.error(f"Cannot run query: {e}")
        print(f"Error: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error initializing retriever: {e}")
        return

    # 3. Initialize RAG Generator & Chain
    rag = RAGGenerator()
    chain = rag.get_chain(retriever)
    
    print("Thinking...")
    try:
        response = chain.invoke(question)
        print(f"\nAnswer: {response}")
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        print("An error occurred while generating the answer.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG System CLI")
    parser.add_argument("mode", choices=["ingest", "query"], help="Operation mode")
    parser.add_argument("--data", help="Path to data directory (for ingest)", default="data/raw")
    parser.add_argument("--q", help="Question to ask (for query)")
    
    args = parser.parse_args()
    
    if args.mode == "ingest":
        ingest(args.data)
    elif args.mode == "query":
        if not args.q:
            print("Error: Please provide a question using --q")
        else:
            query(args.q)