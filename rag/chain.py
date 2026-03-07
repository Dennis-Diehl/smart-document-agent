# Assembles the RAG chain connecting retriever and LLM
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

class RAGChain:
    def __init__(self, retriever, llm):
        self.retriever = retriever

        if llm == "groq":
            self.llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
        elif llm == "openrouter":
            self.llm = ChatOpenAI(
                model="stepfun/step-3.5-flash:free",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                #base_url="https://openrouter.ai/api/v1",
            )

    def run(self, query):
        """
        Manual RAG implementation for learning purposes.

        This could have been done automatically using LangChain's RetrievalQA:

            from langchain.chains import RetrievalQA
            chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.retriever,
                return_source_documents=True
            )
            result = chain.invoke({"query": query})
            # result["result"]           -> answer text
            # result["source_documents"] -> chunks used to generate the answer

        RetrievalQA handles these 4 steps automatically under the hood:
        fetch chunks -> build context -> build prompt -> call LLM
        """
        # Step 1: fetch relevant chunks from ChromaDB based on the query
        relevant_chunks = self.retriever.invoke(query)

        # Step 2: combine all chunk texts into a single context string
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

        # Step 3: build the prompt from context and the user's question
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        # Step 4: send the prompt to the LLM and extract the response text
        response = self.llm.invoke([HumanMessage(content=prompt)])

        return {"answer": response.content, "sources": relevant_chunks}
