# 🤖 SoluGen AI: Enterprise RAG Matching Engine

## 📋 Project Overview

This system is a high-fidelity **Retrieval-Augmented Generation (RAG)** platform designed to solve the "Semantic Gap" in recruitment. Traditional keyword searches often fail to find a "Data Scientist" when a query asks for a "Predictive Modeling Expert." This engine uses vector embeddings to understand underlying intent, effectively matching complex candidate descriptions to the specific job roles in the database.

---

## 🏗 Architectural Rationale (Senior Engineering Insights)

### 1. Choice of Vector Database: Why ChromaDB?

For this specific assignment, **ChromaDB** was selected over cloud-native solutions for the following strategic reasons:

* **Zero-Latency Local Persistence:** By co-locating the DB on the same infrastructure as the API, we eliminate network overhead during retrieval, providing sub-100ms response times.
* **Explainable AI (XAI):** Chroma provides direct access to `relevance_scores` and document indices. We have exposed these in the UI to build trust with recruiters by showing *why* a candidate was matched.
* **Privacy by Design:** Recruitment data is sensitive. By using a local instance, data remains within the controlled environment during the retrieval phase.

### 2. Quality Control: Similarity Thresholding

A signature feature of this implementation is the **Similarity Threshold (0.3 floor)**.

* **The Problem:** Standard RAG systems always return the "top-k" results, even if they are completely irrelevant to the query.
* **The Solution:** Our API calculates the Cosine Similarity and rejects any chunk with a score below 0.3. This acts as a "Guardrail," ensuring the LLM never receives "junk" data that could lead to hallucinations.

### 3. Hybrid Strategy & Factory Pattern

We utilize a **Local-First Embedding Layer** (`all-MiniLM-L6-v2`) combined with a **Cloud-Intelligence Generation Layer** (Gemini/OpenAI). The system is built using the **Factory Pattern**, allowing the embedding provider or the database backend to be swapped via `.env` configuration without changing business logic.

---

## 📂 Dataset & Usage Documentation

### Why this Dataset (`jobs.csv`)?

I chose the job description dataset because recruitment is a high-stakes domain where **context matters more than keywords**.

* It contains unstructured text with overlapping terminology (e.g., Data Science vs. Data Engineering).
* It allows for the demonstration of metadata mapping (mapping the "source" column to "Job Title" in the UI).

### Expected User Questions

The system is optimized for high-value semantic queries:

1. **Requirement Matching:** *"Who is the ideal candidate for a role requiring 5+ years of T-SQL and Predictive Modeling?"*
2. **Gap Analysis:** *"What specific certifications are required for the Clinical Lab roles?"*
3. **Cross-Functional Search:** *"Find me roles that combine Healthcare knowledge with Machine Learning."*

---

## 🛠 Setup & Developer Experience (DX)

### 1. Prerequisites

* **Conda** (Anaconda or Miniconda)
* **Google Gemini API Key** (Set in `.env`)

### 2. Environment Installation

The project uses a `environment.yml` file for reproducible dependency management.

```bash
# Create the environment from the yaml file
conda env create -f environment.yml

# Activate the environment
conda activate ai_rag_assignment

```

### 3. Automation via Makefile

We provide a simplified entry point for all major operations to ensure a seamless evaluation experience:

* **Build the Index (Ingestion):**
```bash
make ingest

```


* **Launch the Application (UI + API):**
```bash
make run

```


*The UI is accessible at: **[http://127.0.0.1:8000*](http://127.0.0.1:8000)**
* **Run Test Suite:**
```bash
make test

```



---

## 📂 Project Structure

```text
├── data/
│   ├── raw/                # Source CSV/PDF documents (jobs.csv)
│   └── vector_store/       # Persistent ChromaDB files
├── src/
│   ├── api/                # FastAPI logic & Pydantic schemas
│   ├── ingestion/          # Multi-format loaders & splitters
│   ├── retrieval/          # Vector DB wrappers & Property decorators
│   ├── static/             # Modern Green UI (HTML/CSS/JS)
│   └── config.py           # Type-safe environment management
├── tests/                  # Unit and Integration suites
├── environment.yml         # Conda environment definition
└── Makefile                # Developer shortcuts (run, ingest, test)

```

---

## 📈 Scalability Roadmap

1. **Hybrid Search (BM25 + Vector):** Combining keyword frequency with semantic meaning to improve matching for specific technical acronyms (e.g., "VB.NET").
2. **Reranking Layer:** Adding a Cross-Encoder stage to re-score results for even higher precision.
3. **Observability:** Integrating **LangSmith** to monitor cost-per-query and retrieval latency in production.

---

**Author:** Lior Shilon
**Date:** December 2025