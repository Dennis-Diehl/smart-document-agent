# Entry point for the Smart Document Agent application
import streamlit as st
import tempfile
import os
from rag.loader import DocumentLoader
from rag.chunking import DocumentChunker
from rag.vectorstore import VectorStore
from rag.chain import RAGChain

# Streamlit reruns the entire script on every interaction, so we use session_state
# to keep the vector store, chat history, and loaded file list alive across reruns.
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()

if "messages" not in st.session_state:
    # List of dicts: [{"role": "user", "content": "..."}, ...]
    st.session_state.messages = []

if "loaded_files" not in st.session_state:
    # Prevents the same file from being processed more than once
    st.session_state.loaded_files = set()

st.title("Smart Document Agent")
st.caption("Upload a PDF, ask a question, get an answer.")

# Sidebar: let the user upload a PDF and push it through the RAG pipeline.
# st.file_uploader returns bytes in memory, but PyPDFLoader needs a real path —
# so we write to a temp file, process it, then delete it.
with st.sidebar:
    st.header("Load Document")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file is not None:
        file_name = uploaded_file.name

        if file_name not in st.session_state.loaded_files:
            with st.spinner(f"Processing '{file_name}'..."):
                # Write bytes to disk so PyPDFLoader gets a file path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # RAG pipeline: load → split → embed & store
                docs = DocumentLoader(tmp_path).load()
                chunks = DocumentChunker().chunk(docs)
                st.session_state.vector_store.store(chunks)

                # Clean up the temp file
                os.unlink(tmp_path)

            st.session_state.loaded_files.add(file_name)
            st.success(f"'{file_name}' loaded successfully.")
        else:
            st.info(f"'{file_name}' is already loaded.")

    st.divider()

    # Choose retrieval strategy
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

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Replay previous messages so the chat history stays visible after reruns.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# The walrus operator (:=) assigns and tests in one step — we only enter the block
# if the user actually submitted a message. Bail early if no PDF is loaded yet.
if query := st.chat_input("Ask a question..."):

    if not st.session_state.loaded_files:
        st.warning("Please upload a PDF first.")
    else:
        # show the user's message right away
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # run the RAG pipeline and stream back an answer
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                try:
                    retriever = st.session_state.vector_store.get_retriever(strategy=strategy)
                    result = RAGChain(retriever, llm=llm_choice).run(query)
                    answer = result["answer"]
                    sources = result["sources"]
                except Exception as e:
                    answer = f"Error: {e}"
                    sources = []

            st.write(answer)

            # show which chunks the answer was built from
            if sources:
                with st.expander(f"Sources ({len(sources)} chunks)"):
                    for i, chunk in enumerate(sources, 1):
                        st.caption(f"**Chunk {i}** — Page {chunk.metadata.get('page', '?')}")
                        st.text(chunk.page_content[:300])
                        st.divider()

        # persist the answer so it shows up on the next rerun
        st.session_state.messages.append({"role": "assistant", "content": answer})
