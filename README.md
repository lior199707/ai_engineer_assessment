# 🤖 SoluGen AI: Semantic Job Matching Engine

## 📋 Project Overview

This system is a specialized **Semantic Retrieval Engine** designed to bridge the gap between complex job requirements and recruitment data. While traditional keyword searches often fail to find a "Data Scientist" when a query asks for a "Predictive Modeling Expert," this engine uses vector embeddings to understand underlying intent. It provides a sub-100ms search experience with explainable similarity scores.

---

## 🏗 Architectural Rationale (Senior Engineering Insights)

### 1. Choice of Vector Database: Why ChromaDB?

For this specific assignment, **ChromaDB** was selected for the following strategic reasons:

* **Zero-Latency Local Persistence:** By co-locating the database on the same infrastructure as the API, we eliminate network overhead, providing instant response times.
* **Explainable AI (XAI):** Chroma provides direct access to `relevance_scores` and document indices. We have exposed these in the UI to build trust with users by showing *why* a candidate was matched.
* **Privacy by Design:** Recruitment data is sensitive. By using a local instance, search data remains within the controlled environment.

### 2. Quality Control: Similarity Thresholding

A signature feature of this implementation is the **Similarity Threshold (0.3 floor)**.

* **The Problem:** Standard retrieval systems always return the "top-k" results, even if they are completely irrelevant to the query.
* **The Solution:** Our API calculates the Cosine Similarity and rejects any chunk with a score below 0.3. This acts as a quality "Guardrail," ensuring users only see high-confidence matches.

### 3. Pure Retrieval Focus

This system is optimized as a **Retrieval-only engine**. By focusing exclusively on high-fidelity vector search without an LLM generation layer, we ensure:

* **Maximum Reliability:** No risk of "hallucinations" in job descriptions.
* **Cost Efficiency:** Zero API costs for text generation.
* **Speed:** Faster response times by avoiding the latency of cloud-based LLM calls.

---

## 📂 Dataset & Usage Documentation

### Chosen Dataset

**Name:** [Data Science Job Posting on Glassdoor](https://www.kaggle.com/datasets/rashikrahmanpritom/data-science-job-posting-on-glassdoor?resource=download)

**Source:** Kaggle

**Rationale:** I chose this dataset because it contains unstructured, complex technical text with overlapping terminology. It is a perfect testbed for measuring how well a vector model can generalize across different data engineering and scientific disciplines.

### Expected User Questions

The system is optimized for high-value semantic queries:

1. **Requirement Matching:** *"Who is the ideal candidate for a role requiring 5+ years of T-SQL and Predictive Modeling?"*
2. **Gap Analysis:** *"What specific certifications are required for the Clinical Lab roles?"*
3. **Cross-Functional Search:** *"Find me roles that combine Healthcare knowledge with Machine Learning."*

---

## 🛠 Setup & Developer Experience (DX)

### 1. Prerequisites

* **Conda** (Anaconda or Miniconda)

### 2. Environment Installation

```bash
# Create the environment from the yaml file
conda env create -f environment.yml

# Activate the environment
conda activate ai_rag_assignment

```

### 3. Automation via Makefile

* **Build the Index (Ingestion):** `make ingest`
* **Launch the Application (UI + API):** `make run`
*Accessible at: **[http://127.0.0.1:8000*](http://127.0.0.1:8000)**
* **Run Test Suite:** `make test`

---

## 📂 Project Structure

```text
ai_engineer_assessment/
├── data/
│   ├── raw/                # Source jobs.csv (Kaggle Dataset)
│   └── vector_store/       # Persistent ChromaDB files
├── src/
│   ├── api/                # FastAPI logic & Pydantic schemas
│   ├── ingestion/          # Multi-format loaders & splitters
│   ├── retrieval/          # Vector DB wrappers & Property decorators
│   ├── static/             # Modern Green UI (HTML/CSS/JS)
│   ├── config.py           # Type-safe environment management
│   └── utils.py            # Logger and shared utilities
├── tests/                  # Unit and Integration suites
├── environment.yml         # Conda environment definition
└── Makefile                # Developer shortcuts

```

---

## 📈 Scalability Roadmap

1. **Hybrid Search (BM25 + Vector):** Combining keyword frequency with semantic meaning for specific technical acronyms (e.g., "VB.NET").
2. **Reranking Layer:** Adding a Cross-Encoder stage to re-score results for even higher precision.
3. **Real-time Indexing:** Enabling dynamic CSV uploads to update the vector store without restarting the service.

---

**Author:** [Your Name]
**Date:** December 2025

---