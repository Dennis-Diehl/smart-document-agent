<h1 align="center">Smart Document Agent</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python" />
  <img src="https://img.shields.io/badge/LangChain-1.2-green?style=flat" />
  <img src="https://img.shields.io/badge/ChromaDB-1.5-orange?style=flat" />
  <img src="https://img.shields.io/badge/Streamlit-1.55-red?style=flat&logo=streamlit" />
  <img src="https://img.shields.io/badge/RAGAS-0.4.3-purple?style=flat" />
</p>

<p align="center">
  A local RAG (Retrieval-Augmented Generation) application that lets you upload PDF documents and ask questions about them — powered by LangChain, ChromaDB, and free LLM APIs. Includes automatic answer quality evaluation via RAGAS.
</p>

<p align="center">
  <a href="https://smart-document-agent.streamlit.app/"><strong>Live Demo →</strong></a><br/>
  <sub>The app may take ~30 seconds to start on first load (Streamlit Community Cloud cold start).</sub>
</p>

---

## Demo

> Screenshot or GIF here
>
> ![Demo](assets/demo.png)

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
| Web Search Fallback | DuckDuckGo Search |
| Evaluation | RAGAS 0.4.3 (Faithfulness + Answer Relevancy) |

---

## Features

- Upload one or multiple PDF files via the sidebar
- Duplicate PDFs are automatically detected and skipped
- Three retrieval strategies selectable at runtime:
  - **Similarity** — top-k most similar chunks (cosine similarity)
  - **MMR** — Maximal Marginal Relevance, balances relevance and diversity
  - **Multi-Query** — LLM rephrases the query into multiple variations for broader recall
- Switch between LLMs (Groq / OpenRouter) without restarting
- **Web search fallback** — when documents don't contain enough information, the app automatically falls back to DuckDuckGo web search
- Source chunks displayed in an expandable panel below each answer
- **RAGAS evaluation** — toggle in sidebar to score each answer on:
  - **Faithfulness** (0.0–1.0): Are all claims in the answer backed by the retrieved context?
  - **Answer Relevancy** (0.0–1.0): Does the answer actually address the question?
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
│   └── chain.py              # Manual RAG chain + web search fallback
├── evaluation/
│   └── evaluate.py           # RAGAS evaluation (Faithfulness + Answer Relevancy)
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
| `GOOGLE_API_KEY` | Yes | Used for `gemini-embedding-001` embeddings + RAGAS evaluation |
| `GROQ_API_KEY` | Yes | Used for Llama 3.3 70B inference, Multi-Query retriever + RAGAS evaluation |
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
                                              ↓ if context insufficient
                                              DuckDuckGo web search fallback
  → evaluation/evaluate.py  RAGASEvaluator   (Faithfulness + Answer Relevancy scoring)
```

The `RAGChain.run()` method executes these steps manually instead of relying on LangChain's `RetrievalQA` abstraction, making each stage visible and easy to understand.

---

## License

This project is licensed under the MIT License.
