# Manages vector store operations (store, retrieve, query)
from langchain_chroma import Chroma
from rag.embeddings import EmbeddingGenerator

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
        self.vector_store.add_documents(chunks)

    def get_retriever(self, k=3):
        return self.vector_store.as_retriever(search_kwargs={"k": k})
    
