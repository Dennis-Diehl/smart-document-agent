# Entry point for the Smart Document Agent application
import streamlit as st
import tempfile
import os
from collections import defaultdict
from rag.loader import DocumentLoader
from rag.chunking import DocumentChunker
from rag.vectorstore import VectorStore
from rag.chain import RAGChain
from evaluation.evaluate import RAGASEvaluator

# ──────────────────────────────────────────────
# Session State — bleibt zwischen Streamlit-Reruns bestehen
# ──────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "rag_evaluator" not in st.session_state:
    st.session_state.rag_evaluator = RAGASEvaluator()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = set()

# ──────────────────────────────────────────────
# Page Header
# ──────────────────────────────────────────────
st.title("Smart Document Agent")
st.caption("Upload a PDF, ask a question, get an answer.")

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:

    # --- PDF Upload & Processing ---
    st.header("Load Document")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file is not None:
        file_name = uploaded_file.name

        if file_name not in st.session_state.loaded_files:
            with st.spinner(f"Processing '{file_name}'..."):
                # PyPDFLoader braucht einen Dateipfad → temp file erstellen
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # RAG Pipeline: load → split → embed & store
                docs = DocumentLoader(tmp_path).load()
                chunks = DocumentChunker().chunk(docs)
                for chunk in chunks:
                    chunk.metadata["source"] = file_name  # temp-Pfad durch echten Namen ersetzen
                st.session_state.vector_store.store(chunks)

                os.unlink(tmp_path)

            st.session_state.loaded_files.add(file_name)
            st.success(f"'{file_name}' loaded successfully.")
        else:
            st.info(f"'{file_name}' is already loaded.")

    st.divider()

    # --- Loaded Documents ---
    st.subheader("Loaded Documents", help="These documents are part of the knowledge base and can be asked about in the chat. Upload more PDFs to expand the knowledge base.")
    if st.session_state.loaded_files:
        for fname in sorted(st.session_state.loaded_files):
            st.write(f"- {fname}")
    else:
        st.caption("No documents loaded yet.")

    st.divider()

    # --- Settings ---
    st.subheader("Settings")
    strategy = st.selectbox(
        "Retrieval Strategy",
        ["similarity", "mmr", "multi_query"],
        help=(
            "similarity — the k most similar chunks\n\n"
            "mmr — similar AND diverse (less redundancy)\n\n"
            "multi_query — LLM generates multiple search queries"
        ),
    )

    llm_choice = st.selectbox(
        "LLM",
        ["groq", "openrouter"],
        help="groq: Llama 3.3 70B via Groq | openrouter: Step 3.5 Flash (free)",
    )

    rag_toggle = st.toggle(
        "RAG Evaluation",
        key="evaluate_rag",
        help="Toggle to run RAGAS evaluation on each response (adds latency)",
    )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ──────────────────────────────────────────────
# Helper: Quellen aus Document-Objekten gruppieren
# ──────────────────────────────────────────────
def _extract_doc_sources(sources):
    """Gruppiert Document-Chunks nach Dateiname und sammelt Seitenzahlen."""
    doc_pages = defaultdict(set)
    for chunk in sources:
        source = os.path.basename(chunk.metadata.get("source", "Unknown"))
        page = chunk.metadata.get("page", "?")
        doc_pages[source].add(page)
    return [
        {
            "source": source,
            "pages": sorted(p for p in pages if isinstance(p, int))
                   + sorted(str(p) for p in pages if not isinstance(p, int)),
        }
        for source, pages in doc_pages.items()
    ]


def _render_scores(eval_scores):
    """Zeigt Faithfulness und Answer Relevancy als zwei Metriken nebeneinander."""
    col1, col2 = st.columns(2)
    col1.metric("Faithfulness", f"{eval_scores['faithfulness']:.2f}")
    col2.metric("Answer Relevancy", f"{eval_scores['answer_relevancy']:.2f}")


def _render_sources(doc_sources):
    """Zeigt die Quellen-Dokumente in einem aufklappbaren Expander."""
    with st.expander(f"Sources ({len(doc_sources)} documents)"):
        for src in doc_sources:
            page_str = ", ".join(str(p) for p in src["pages"])
            st.caption(f"**{src['source']}** — Pages: {page_str}")


# ──────────────────────────────────────────────
# Chat History Replay — zeigt alte Nachrichten nach einem Rerun
# ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("web_search"):
            st.info("🌐 Answer is based on web search (Documents did not contain enough information)")

        st.write(msg["content"])

        if msg.get("web_sources"):
            with st.expander(f"🌐 Web Sources ({len(msg['web_sources'])})"):
                for ws in msg["web_sources"]:
                    st.markdown(f"- [{ws['title']}]({ws['url']})")

        if msg.get("eval_scores"):
            _render_scores(msg["eval_scores"])

        if msg.get("doc_sources") and not msg.get("web_search"):
            _render_sources(msg["doc_sources"])


# ──────────────────────────────────────────────
# Chat Input — neue Frage verarbeiten
# ──────────────────────────────────────────────
if query := st.chat_input("Ask a question..."):

    if not st.session_state.loaded_files:
        st.warning("Please upload a PDF first.")
    else:
        # User-Nachricht anzeigen
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Antwort generieren
        with st.chat_message("assistant"):

            # --- RAG Pipeline ---
            with st.spinner("Generating answer..."):
                try:
                    retriever = st.session_state.vector_store.get_retriever(strategy=strategy)
                    result = RAGChain(retriever, llm=llm_choice).run(query)
                    answer = result["answer"]
                    sources = result["sources"]
                    web_search = result.get("web_search", False)
                    web_sources = result.get("web_sources", [])
                except Exception as e:
                    answer = f"Error: {e}"
                    sources = []
                    web_search = False
                    web_sources = []

            # --- Antwort anzeigen ---
            if web_search:
                st.info("🌐 Answer is based on web search (Documents did not contain enough information)")

            st.write(answer)

            if web_search and web_sources:
                with st.expander(f"🌐 Web Sources ({len(web_sources)})"):
                    for ws in web_sources:
                        st.markdown(f"- [{ws['title']}]({ws['url']})")

            # --- RAGAS Evaluation (nur bei RAG-Antworten, nicht bei Web Search) ---
            eval_scores = None
            if rag_toggle and sources and not web_search:
                with st.spinner("Evaluating response..."):
                    try:
                        eval_scores = st.session_state.rag_evaluator.evaluate_response(query, result)
                    except Exception as e:
                        st.warning(f"Evaluation failed: {e}")

            if eval_scores:
                _render_scores(eval_scores)

            # --- Quellen anzeigen ---
            doc_sources = _extract_doc_sources(sources) if sources and not web_search else []
            if doc_sources:
                _render_sources(doc_sources)

        # --- Chat History persistieren ---
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "web_search": web_search,
            "web_sources": web_sources,
            "doc_sources": doc_sources,
            "eval_scores": eval_scores,
        })
