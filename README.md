# Personal Research Assistant (RAG Example)

This repository contains a small retrieval-augmented generation (RAG) pipeline that I built to showcase my ability to work with modern LLM tooling for prospective data-scientist roles. The documents in the `data/` directory (not included in this repo because they are my own publications) are ingested, chunked, embedded into a Chroma vector database, and then queried through an Ollama-hosted Llama 3.2 model.

Although the data is personal, the code path demonstrates the full pipeline you would follow to build a grounded Q&A assistant over any small PDF/TXT corpus and optionally expose it via a Streamlit UI.

## Features

- Ingestion script that reads PDFs/TXT, splits them with LangChain text splitters, and stores embeddings in a persistent Chroma vector database.
- CLI query interface plus a Streamlit-powered GUI that reload the vector store and send grounded prompts to Llama 3.2 via Ollama.
- Pytest suite that validates ingestion, retrieval, and prompt construction logic with lightweight stubs.
- Clear separation between data preparation (`ingest.py`), inference (`query.py`), and presentation (`app.py`).

## Project Structure

| Path | Description |
| --- | --- |
| `data/` | Place PDFs/TXT files here before running ingestion (not bundled in the repo). |
| `vectorstore/` | Folder where the persisted Chroma DB is written; created after ingestion. |
| `src/ingest.py` | Orchestrates the ingestion pipeline. See method breakdown below. |
| `src/query.py` | Command-line interface for grounded question answering. |
| `src/app.py` | Streamlit front-end that wraps the query pipeline with a lightweight UI. |
| `src/utils.py` | Placeholder for helper functions (currently unused). |
| `tests/test_ingest.py` | Pytest coverage for document loading, chunking, and vector store creation. |
| `tests/test_query.py` | Tests prompt formatting, model initialization, and the query loop. |
| `tests/test_retrieval.py` | Ensures retrieved snippets and sources are surfaced when querying. |
| `requirements.txt` | Python dependencies used across ingestion and query steps. |

## Method Walkthrough

### `src/ingest.py`

- `load_documents(data_dir="./data")`: Iterates through PDFs/TXT files, loads their contents using LangChain loaders, and keeps a running count for visibility.
- `chunk_documents(documents)`: Applies `RecursiveCharacterTextSplitter` (chunk size 1000, overlap 200) so downstream retrieval captures enough context while staying efficient.
- `create_vecstore(chunks, persist_dir="./vectorstore")`: Generates embeddings using HuggingFace `sentence-transformers/all-MiniLM-L6-v2`, builds a Chroma vector database, and persists it for reuse during querying.
- `main()`: High-level wrapper that calls the three steps in sequence, prints progress, and guides the user to run `query.py` next.

### `src/query.py`

- `load_vecstore(persist_dir="./vectorstore")`: Reloads the Chroma store with the same embedding function so similarity search returns vector-compatible results.
- `init_llm()`: Spins up an Ollama client pointed at `llama3.2` (temperature 0.1) for low-variance answers.
- `format_docs(docs)`: Converts retrieved documents into a human-readable context string, including per-document metadata (source, page).
- `create_prompt(context, question)`: Crafts instructions that force the LLM to stay grounded in retrieved context and to cite document numbers or state when information is unavailable.
- `query(vectorstore, llm, question, k=3)`: Implements the RAG loop—retrieve similar chunks, build prompt, invoke the LLM, print the answer, and list sources with short excerpts.
- `main()`: CLI loop that keeps answering questions until the user types `quit`/`exit`.

### `src/app.py`

Streamlit app that caches the vector store/LLM, collects a user question, reuses the same retrieval + prompt-building logic from `query.py`, and renders answers with expandable source panels. Run it with `streamlit run src/app.py`.

## Getting Started

1. **Create & activate an environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add your documents**

   - Place PDF/TXT files into `./data`. In my case these were personal publications, so they were intentionally left out of the repository.

4. **Run ingestion**

   ```bash
   python src/ingest.py
   ```

   This loads documents, chunks them, creates embeddings, and persists the Chroma vector store under `./vectorstore`.

5. **Start querying**

   ```bash
   python src/query.py
   ```

   Type questions at the prompt. The script will retrieve the top 3 relevant chunks, build a grounded prompt for Llama 3.2, and display both the answer and cited sources.

6. **Optional: launch the Streamlit UI**

   ```bash
   streamlit run src/app.py
   ```

   This provides a simple web interface for the same retrieval pipeline.

7. **Run tests**

   ```bash
   pytest
   ```

   Tests rely on stubs/mocks, so they can run without downloading large models.

## Notes & Next Steps

- Make sure [Ollama](https://ollama.com/) is installed locally and has the `llama3.2` model pulled before running `query.py`.
- Because this project is part of my personal portfolio, the bundled code is intentionally lightweight and showcases end-to-end comprehension rather than production hardening.
- Potential future improvements: swap in a GPU-backed embedding model (it's hardcoded for CPU at the moment), add document upload to the Streamlit app, or package the ingestion/query steps into a small API service.
