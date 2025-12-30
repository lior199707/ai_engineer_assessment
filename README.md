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

### 2. Initial Environment Setup

Create the environment and install all dependencies (including `make` and `pre-commit`).

```bash
conda env update --file environment.yml --prune
conda activate ai_rag_assignment

```

### 3. Install Git Hooks

This ensures code quality checks run automatically before every commit.

```bash
make install

```

### 4. Configure Environment Variables

Copy the example configuration file to create your local `.env` file.

**Linux / Mac / PowerShell / Git Bash:**

```bash
cp .env.example .env

```

**Windows (Command Prompt):**

```cmd
copy .env.example .env

```

Open `.env` and configure your provider:

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

## 🛠 Development & Automation

This project uses a `Makefile` to automate common development tasks.

| Command | Description |
| --- | --- |
| `make install` | Updates Conda environment and installs pre-commit hooks |
| `make format` | Formats code using Ruff |
| `make lint` | Checks for linting errors |
| `make test` | Runs the test suite |
| `make clean` | Removes cache files (Cross-platform safe) |

---

## 🏃 Usage

### 1. Ingest Documents

Place your source PDF files into the `data/raw/` directory.

```bash
make ingest

```

*Output: Vector store will be created in `data/vector_store/`.*

### 2. Query the System

Ask questions based on the ingested documents.

```bash
make query Q="What are the key findings?"

```

---

## 🧪 Testing

The project includes a test suite configured with `pytest`.

To run all tests:

```bash
make test

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
├── Makefile                # Task automation
└── README.md               # Project documentation

```

## 🛠 Tech Stack

* **Orchestration:** LangChain
* **LLM Support:** OpenAI GPT-4o, Google Gemini 1.5 Flash
* **Vector Database:** ChromaDB (Local)
* **Configuration:** Pydantic Settings
* **Quality Control:** Ruff, Pre-commit

## 📝 License

This project is intended for educational and assessment purposes.