"""Main entry point for the RAG Application CLI.

This script provides a Command Line Interface (CLI) to interact with the 
application. It supports two primary modes:
1. 'ingest': Loads documents and builds the vector database.
2. 'query': accepts a question and generates an answer using RAG.

Usage:
    python main.py ingest --data path/to/docs
    python main.py query --q "What is the summary?"
"""

import argparse
from src.ingestion.loader import load_documents, split_documents
from src.retrieval.vector_db import VectorStoreManager
from src.generation.llm import RAGGenerator

def ingest(data_dir: str) -> None:
    """Executes the data ingestion pipeline.

    Args:
        data_dir (str): Path to the directory containing raw documents.
    """
    print(f"Loading data from {data_dir}...")
    docs = load_documents(data_dir)
    chunks = split_documents(docs)
    
    print(f"Creating vector store with {len(chunks)} chunks...")
    vsm = VectorStoreManager()
    vsm.create_vector_store(chunks)
    print("Ingestion complete. Vector store persisted.")

def query(question: str) -> None:
    """Executes the retrieval and generation pipeline.

    Args:
        question (str): The user's query string.
    """
    vsm = VectorStoreManager()
    
    try:
        vectorstore = vsm.load_vector_store()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    retriever = vectorstore.as_retriever()
    
    rag = RAGGenerator()
    chain = rag.get_chain(retriever)
    
    print("Thinking...")
    response = chain.invoke(question)
    print(f"\nAnswer: {response}")

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