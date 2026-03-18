# RAG pipeline evaluation using RAGAS v0.4.x metrics
import asyncio
from openai import AsyncOpenAI
from ragas.metrics.collections import Faithfulness, AnswerRelevancy
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from dotenv import load_dotenv
import os

load_dotenv()


class RAGASEvaluator:
    """Bewertet RAG-Antworten mit zwei Metriken:
    - Faithfulness: Nutzt die Antwort nur Fakten aus dem Kontext? (Halluzinations-Check)
    - AnswerRelevancy: Beantwortet die Antwort tatsaechlich die gestellte Frage?
    """

    def __init__(self):
        # LLM fuer Evaluation — Groq via OpenAI-kompatible API (RAGAS native llm_factory)
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY is not set in .env")
        client = AsyncOpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        self.llm = llm_factory("llama-3.3-70b-versatile", provider="openai", client=client)

        # Embeddings fuer AnswerRelevancy — RAGAS native embedding_factory (liest GOOGLE_API_KEY aus env)
        self.embeddings = embedding_factory("google", model="gemini-embedding-001")

        # Faithfulness: Zerlegt die Antwort in einzelne Aussagen und prueft jede gegen den Kontext.
        #   1.0 = alle Aussagen sind durch den Kontext belegt, 0.0 = alles halluziniert
        self.faithfulness_metric = Faithfulness(llm=self.llm)
        # AnswerRelevancy: Generiert aus der Antwort Rueckfragen und vergleicht per Embedding,
        #   ob diese zur Originalfrage passen. 1.0 = voll relevant, 0.0 = geht am Thema vorbei
        self.answer_relevancy_metric = AnswerRelevancy(llm=self.llm, embeddings=self.embeddings)

    def evaluate_response(self, query: str, rag_result: dict) -> dict:
        """Bewertet eine einzelne RAG-Antwort.

        Args:
            query: Die Frage des Users
            rag_result: RAGChain-Result mit "answer" (str) und "sources" (List[Document])

        Returns:
            {"faithfulness": 0.0-1.0, "answer_relevancy": 0.0-1.0}
        """
        # Document-Objekte aus ChromaDB → einfache Strings fuer RAGAS
        retrieved_contexts = [doc.page_content for doc in rag_result["sources"]]

        # .ascore() ist async → asyncio.run() als Bruecke
        # Gibt MetricResult zurueck → .value fuer den float Score
        faith_result = asyncio.run(self.faithfulness_metric.ascore(
            user_input=query,
            response=rag_result["answer"],
            retrieved_contexts=retrieved_contexts,
        ))

        relevancy_result = asyncio.run(self.answer_relevancy_metric.ascore(
            user_input=query,
            response=rag_result["answer"],
        ))

        return {
            "faithfulness": faith_result.value,
            "answer_relevancy": relevancy_result.value,
        }
