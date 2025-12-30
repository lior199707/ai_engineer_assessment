"""LLM Generation and RAG Pipeline module.

This module constructs the Retrieval-Augmented Generation (RAG) chain by
combining the retriever, prompt templates, and the Large Language Model.

It is designed to be agnostic to the underlying Vector Database implementation,
relying solely on the standard LangChain `VectorStoreRetriever` interface.
"""

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnablePassthrough
from src.config import settings, LLMProvider
from src.utils import setup_logger

logger = setup_logger(__name__)

class RAGGenerator:
    """
    Orchestrates the LLM generation process.

    Attributes:
        llm: The configured LLM instance (OpenAI or Google).
        prompt (ChatPromptTemplate): The prompt template for Q&A.
    """

    def __init__(self):
        """Initializes the RAGGenerator with model settings."""
        self.llm = self._get_llm_model()
        self.prompt = ChatPromptTemplate.from_template("""
            Answer the question based only on the following context:
            {context}

            Question: {question}
        """)

    def _get_llm_model(self):
        """
        Selects and initializes the LLM based on global configuration.
        
        Returns:
            ChatOpenAI | ChatGoogleGenerativeAI: The initialized LLM object.
        """
        if settings.llm_provider == LLMProvider.GOOGLE:
            logger.info(f"Initializing Google Gemini Model: {settings.google_model_name}")
            return ChatGoogleGenerativeAI(
                model=settings.google_model_name,
                temperature=0,
                google_api_key=settings.google_api_key
            )
        else: # LLMProvider.OPENAI
            logger.info(f"Initializing OpenAI Model: {settings.openai_model_name}")
            return ChatOpenAI(
                model=settings.openai_model_name,
                temperature=0,
                openai_api_key=settings.openai_api_key
            )

    def get_chain(self, retriever: VectorStoreRetriever) -> RunnableSerializable:
        """
        Constructs the standard LCEL (LangChain Expression Language) chain.

        The chain performs the following steps:
        1. Retrieves relevant documents using the passed `retriever`.
        2. Formats the documents into a single string.
        3. Passes the context and question to the prompt template.
        4. Sends the formatted prompt to the LLM.
        5. Parses the output string.

        Args:
            retriever (VectorStoreRetriever): The generic retriever interface 
                                              (works with Chroma, Pinecone, etc.).

        Returns:
            RunnableSerializable: A compiled LangChain runnable ready for invocation.
        """
        def format_docs(docs):
            """Helper to join document content into a single string."""
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {
                "context": retriever | format_docs, 
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain