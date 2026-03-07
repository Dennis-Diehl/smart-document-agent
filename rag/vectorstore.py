# Manages vector store operations (store, retrieve, query)
from langchain_chroma import Chroma
from rag.embeddings import EmbeddingGenerator
from langchain_groq import ChatGroq
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
import dotenv
import os

dotenv.load_dotenv()


class VectorStore:
    def __init__(self, collection_name="smart_documents", persist_directory="./vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=EmbeddingGenerator().get_embedding_model(),
            persist_directory=self.persist_directory
        )

    def store(self, chunks):
        # All chunks from one PDF share the same source value
        new_source = chunks[0].metadata.get("source")

        # Fetch all stored metadata and collect existing source values
        existing = self.vector_store.get()["metadatas"]
        existing_sources = {m.get("source") for m in existing}

        if new_source in existing_sources:
            print(f"'{new_source}' already in vector store — skipping.")
            return

        self.vector_store.add_documents(chunks)
        print(f"'{new_source}': {len(chunks)} chunks stored successfully.")

    def get_retriever(self, strategy="similarity", k=3):
        if strategy == "similarity":
            # Returns the k most similar chunks to the query using cosine similarity on embeddings.
            return self.vector_store.as_retriever(search_kwargs={"k": k})
        elif strategy == "mmr":
            # Maximal Marginal Relevance: fetches fetch_k candidates, then picks k that are
            # both relevant to the query AND diverse from each other — reduces redundant results.
            return self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 10})
        elif strategy == "multi_query":
            # Uses an LLM to rephrase the original query into multiple variations,
            # runs each against the vector store, and merges the results.
            # Helps when a single query wording might miss relevant chunks.
            base_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
            llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
            return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
        

    def get_all_retrievers(self, k=3):
        return {
            "similarity": self.get_retriever(strategy="similarity", k=k),
            "mmr": self.get_retriever(strategy="mmr", k=k),
            "multi_query": self.get_retriever(strategy="multi_query", k=k)
        }