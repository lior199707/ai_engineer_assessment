"""LLM Generation and RAG Pipeline module.

This module constructs the Retrieval-Augmented Generation (RAG) chain by
combining the retriever, prompt templates, and the Large Language Model.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable
from langchain_core.vectorstores import VectorStoreRetriever
from src.config import settings

class RAGGenerator:
    """Orchestrates the LLM generation process.

    Attributes:
        llm (ChatOpenAI): The configured ChatOpenAI instance.
        prompt (ChatPromptTemplate): The prompt template for Q&A.
    """

    def __init__(self):
        """Initializes the RAGGenerator with model settings."""
        self.llm = ChatOpenAI(
            model=settings.model_name,
            temperature=0  # Deterministic output
        )
        self.prompt = ChatPromptTemplate.from_template("""
            Answer the question based only on the following context:
            {context}

            Question: {question}
        """)

    def get_chain(self, retriever: VectorStoreRetriever) -> RunnableSerializable:
        """Constructs the standard LCEL (LangChain Expression Language) chain.

        The chain performs the following steps:
        1. Retrieves relevant documents based on the question.
        2. Formats the documents into a string.
        3. Passes context and question to the prompt.
        4. Sends the formatted prompt to the LLM.
        5. Parses the output string.

        Args:
            retriever (VectorStoreRetriever): The retriever object from the vector store.

        Returns:
            RunnableSerializable: A compiled LangChain runnable ready for invocation.
        """
        from langchain_core.runnables import RunnablePassthrough

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain