# Generates vector embeddings from text chunks
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()


class EmbeddingGenerator:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.embeddings_model = "models/gemini-embedding-001"

    def get_embedding_model(self):
        return GoogleGenerativeAIEmbeddings(model=self.embeddings_model, api_key=self.api_key)