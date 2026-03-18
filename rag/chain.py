# Assembles the RAG chain connecting retriever and LLM
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from duckduckgo_search import DDGS
from dotenv import load_dotenv
import os

load_dotenv()

# Sentinel string the LLM returns when the provided context cannot answer the question
_NO_CONTEXT = "[NO_CONTEXT]"


class RAGChain:
    def __init__(self, retriever, llm):
        self.retriever = retriever

        if llm == "groq":
            groq_key = os.getenv("GROQ_API_KEY")
            if not groq_key:
                raise ValueError("GROQ_API_KEY is not set in .env")
            self.llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key)
        elif llm == "openrouter":
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_key:
                raise ValueError("OPENROUTER_API_KEY is not set in .env")
            self.llm = ChatOpenAI(
                model="stepfun/step-3.5-flash:free",
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
            )
        else:
            raise ValueError(f"Unknown LLM choice: {llm}")

    def _build_rag_prompt(self, context, query):
        """Build the RAG prompt that instructs the LLM to reply with [NO_CONTEXT]
        when the provided context does not contain enough information."""
        return (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Instructions: Answer the question based ONLY on the context above. "
            f"If the context does not contain enough information to answer the question, "
            f"respond with exactly {_NO_CONTEXT} and nothing else.\n"
            "Answer:"
        )

    def _web_search(self, query, max_results=5):
        """Run a DuckDuckGo text search and return results + structured source list."""
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return "", []
        # Build context string for the LLM
        parts = []
        for r in results:
            parts.append(f"**{r['title']}**\n{r['body']}\nSource: {r['href']}")
        context = "\n\n".join(parts)
        # Build structured source list for the frontend
        web_sources = [{"title": r["title"], "url": r["href"]} for r in results]
        return context, web_sources

    def run(self, query):
        """
        Manual RAG implementation with Web Search Fallback.

        Steps:
        1. Fetch relevant chunks from ChromaDB
        2. Build context from chunks
        3. Build prompt with [NO_CONTEXT] instruction
        4. Call LLM
        5. If LLM responds with [NO_CONTEXT] → DuckDuckGo web search fallback
        """
        # Step 1: fetch relevant chunks from ChromaDB based on the query
        relevant_chunks = self.retriever.invoke(query)

        # Step 2: combine all chunk texts into a single context string
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

        # Step 3: build the prompt with [NO_CONTEXT] instruction
        prompt = self._build_rag_prompt(context, query)

        # Step 4: send the prompt to the LLM and extract the response text
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}") from e

        # Step 5: if the LLM says context is insufficient, fall back to web search
        if _NO_CONTEXT in response.content:
            web_context, web_sources = self._web_search(query)
            if web_context:
                web_prompt = (
                    f"Context (from web search):\n{web_context}\n\n"
                    f"Question: {query}\n"
                    "Answer the question based on the web search results above.\n"
                    "Answer:"
                )
                try:
                    web_response = self.llm.invoke([HumanMessage(content=web_prompt)])
                except Exception as e:
                    raise RuntimeError(f"LLM call (web search) failed: {e}") from e
                return {
                    "answer": web_response.content,
                    "sources": relevant_chunks,
                    "web_search": True,
                    "web_sources": web_sources,
                }
            # Web search returned nothing — return a fallback message
            return {
                "answer": "Sorry, I couldn't find an answer in the documents or via web search.",
                "sources": relevant_chunks,
                "web_search": True,
                "web_sources": [],
            }

        return {"answer": response.content, "sources": relevant_chunks, "web_search": False, "web_sources": []}

