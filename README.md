# Modular RAG System (AI Engineer Assessment)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Conda](https://img.shields.io/badge/Conda-Env-green)
![Status](https://img.shields.io/badge/Status-Development-orange)

A production-ready, modular RAG (Retrieval-Augmented Generation) pipeline designed for extensibility, maintainability, and clean separation of concerns. This project serves as a template for building robust AI applications involving document ingestion, vector retrieval, and LLM-based generation.

## 🏗 Architecture

The system is designed with a clear separation of concerns to ensure modularity and ease of testing:

- **Ingestion Layer (`src/ingestion`):** Handles loading of raw documents (PDFs) and intelligent chunking using recursive character splitting.
- **Retrieval Layer (`src/retrieval`):** Manages vector embeddings and persistence using ChromaDB. Supports **OpenAI** and **Google Gemini** embeddings.
- **Generation Layer (`src/generation`):** Orchestrates the LLM and manages prompt templates via LangChain. Swappable support for **GPT-4o** and **Gemini 1.5 Flash**.
- **Configuration:** Centralized `pydantic` settings management for strict typing and environment variable validation.

## 🚀 Setup & Installation

### Prerequisites
- **Python 3.10+**
- **Conda** (Anaconda or Miniconda)
- **API Key:** Either OpenAI (Paid) or Google Gemini (Free Tier available)

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd ai_rag_assignment

```

### 2. Create the Environment

This project uses a Conda environment to manage dependencies.

```bash
conda env create -f environment.yml
conda activate ai_rag_assignment

```

### 3. Configure Environment Variables

Copy the example configuration file.

```bash
cp .env.example .env

```

Open `.env` and configure your provider.

**Option A: Use Google Gemini (Free Tier)**

```ini
LLM_PROVIDER=google
GOOGLE_API_KEY=AIzaSy...

```

**Option B: Use OpenAI**

```ini
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

```

---

## 🏃 Usage

The application exposes a Command Line Interface (CLI) via `main.py` for easy interaction.

### 1. Ingest Documents

Place your source PDF files into the `data/raw/` directory. Then, run the ingestion pipeline to parse, chunk, and index the data.

```bash
python main.py ingest --data data/raw

```

*Output: Vector store will be created in `data/vector_store/`.*

### 2. Query the System (RAG)

Ask questions based on the ingested documents. The system will retrieve relevant context and generate an answer using the configured provider.

```bash
python main.py query --q "What are the key findings in the document?"

```

---

## 🧪 Testing

The project includes a test suite configured with `pytest`.

To run all tests:

```bash
pytest tests/

```

To run a specific test file:

```bash
pytest tests/unit/test_config.py

```

---

## 📂 Project Structure

```text
ai_rag_assignment/
├── config/                 # Static configuration files
├── data/
│   ├── raw/                # Input documents (PDFs) go here
│   └── vector_store/       # Persisted ChromaDB files
├── src/
│   ├── ingestion/          # Data loading & splitting logic
│   ├── retrieval/          # Vector DB & Embedding management
│   ├── generation/         # LLM interaction & Prompt templates
│   ├── utils/              # Helper utilities (Logger)
│   ├── config.py           # Pydantic settings & Enum definitions
│   └── main.py             # CLI Entry point
├── tests/                  # Unit and Integration tests
├── .env.example            # Template for environment variables
├── .gitignore              # Git ignore rules
├── environment.yml         # Conda environment definition
└── README.md               # Project documentation

```

## 🛠 Tech Stack

* **Orchestration:** LangChain
* **LLM Support:** OpenAI GPT-4o, Google Gemini 1.5 Flash
* **Vector Database:** ChromaDB (Local)
* **Configuration:** Pydantic Settings
* **Testing:** Pytest

## 📝 License

This project is intended for educational and assessment purposes.