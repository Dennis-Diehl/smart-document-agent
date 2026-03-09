<h1 align="center">Smart Document Agent</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python" />
  <img src="https://img.shields.io/badge/LangChain-1.2-green?style=flat" />
  <img src="https://img.shields.io/badge/ChromaDB-1.5-orange?style=flat" />
  <img src="https://img.shields.io/badge/Streamlit-1.55-red?style=flat&logo=streamlit" />
</p>

<p align="center">
  A local RAG (Retrieval-Augmented Generation) application that lets you upload PDF documents and ask questions about them — powered by LangChain, ChromaDB, and free LLM APIs.
</p>

<p align="center">
  <a href="https://smart-document-agent.streamlit.app/"><strong>Live Demo →</strong></a><br/>
  <sub>The app may take ~30 seconds to start on first load (Streamlit Community Cloud cold start).</sub>
</p>

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| AI Framework | LangChain |
| Vector Database | ChromaDB (local, persistent) |
| Embeddings | Google Generative AI (`gemini-embedding-001`) |
| LLM (primary) | Groq — Llama 3.3 70B Versatile |
| LLM (secondary) | OpenRouter — Step 3.5 Flash (free) |
| Frontend | Streamlit |
| Document Loading | PyPDF |
| Evaluation | RAGAS |

---

## Features

- Upload one or multiple PDF files via the sidebar
- Duplicate PDFs are automatically detected and skipped
- Three retrieval strategies selectable at runtime:
  - **Similarity** — top-k most similar chunks (cosine similarity)
  - **MMR** — Maximal Marginal Relevance, balances relevance and diversity
  - **Multi-Query** — LLM rephrases the query into multiple variations for broader recall
- Switch between LLMs (Groq / OpenRouter) without restarting
- Source chunks displayed in an expandable panel below each answer
- Chat history persists across interactions within the session
- Manual RAG pipeline (no high-level `RetrievalQA` abstraction) — built step-by-step for transparency

---

## Project Structure

```
smart-document-agent/
├── app.py                    # Streamlit entry point
├── requirements.txt
├── .env                      # API keys (not committed)
├── .gitignore
├── rag/
│   ├── loader.py             # PDF loading (PyPDFLoader)
│   ├── chunking.py           # Text splitting (RecursiveCharacterTextSplitter)
│   ├── embeddings.py         # Google GenAI embeddings
│   ├── vectorstore.py        # ChromaDB store + retrieval strategies
│   └── chain.py              # Manual RAG chain (retrieve → context → prompt → LLM)
├── evaluation/
│   └── evaluate.py           # RAGAS evaluation (stub)
├── tests/
│   └── rag_test.ipynb        # Jupyter test notebook
└── vector_store/             # ChromaDB persisted data (auto-created, gitignored)
```

---

## Installation & Setup

### Prerequisites

- Python 3.11+
- API keys for Google GenAI and at least one LLM provider (Groq or OpenRouter)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Dennis-Diehl/smart-document-agent.git
cd smart-document-agent

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# → Open .env and fill in your API keys

# 5. Run the app
streamlit run app.py
```

---

## Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_genai_key_here
GROQ_API_KEY=your_groq_key_here
OPENROUTER_API_KEY=your_openrouter_key_here   # optional, only needed for OpenRouter LLM
```

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Used for `gemini-embedding-001` embeddings |
| `GROQ_API_KEY` | Yes | Used for Llama 3.3 70B inference + Multi-Query retriever |
| `OPENROUTER_API_KEY` | Optional | Used for Step 3.5 Flash via OpenRouter |

> Never commit your `.env` file. It is listed in `.gitignore`.

---

## Architecture

Manual RAG pipeline — each step is implemented explicitly for learning purposes:

```
PDF file
  → rag/loader.py       DocumentLoader       (PyPDFLoader)
  → rag/chunking.py     DocumentChunker      (RecursiveCharacterTextSplitter, chunk=1000, overlap=200)
  → rag/embeddings.py   EmbeddingGenerator   (Google GenAI, gemini-embedding-001)
  → rag/vectorstore.py  VectorStore          (ChromaDB, persisted to ./vector_store/)
  → rag/chain.py        RAGChain             (retrieve → build context → prompt → LLM → answer)
```

The `RAGChain.run()` method executes these four steps manually instead of relying on LangChain's `RetrievalQA` abstraction, making each stage visible and easy to understand.

---

## Implementation Status

### Implemented

| File | Description |
|---|---|
| `rag/loader.py` | PDF loading via PyPDFLoader — extracts pages with full metadata (source, page number, creation date) |
| `rag/chunking.py` | Text splitting with RecursiveCharacterTextSplitter (chunk=1000, overlap=200) |
| `rag/embeddings.py` | Google GenAI embeddings (`gemini-embedding-001`, 3072-dim vectors) |
| `rag/vectorstore.py` | ChromaDB with 3 retrieval strategies + automatic deduplication by source file |
| `rag/chain.py` | Manual 4-step RAG pipeline (retrieve → context → prompt → LLM) — Groq & OpenRouter |
| `app.py` | Streamlit UI — PDF upload, chat interface, strategy & LLM selector, source viewer |
| `tests/rag_test.ipynb` | E2E tests for all pipeline stages and both LLM backends |

### Planned

| File | Description |
|---|---|
| `agent/agent.py` | LangChain agent with `rag_search` and `load_document` tools for autonomous document interaction |
| `evaluation/evaluate.py` | RAGAS evaluation — automatic quality metrics for retrieval and answer generation |

---

## License

This project is licensed under the MIT License.
