#!/usr/bin/env python3
"""
pdf_agentic_rag.py
Agentic RAG over the Chroma collection created by build_pdf_index.py
"""

import os, getpass
from typing import Literal
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from pathlib import Path
from guardrails.rag_guardrails import compute_rag_confidence
# ── Environment & API key ────────────────────────────────────────────────
from dotenv import load_dotenv          # pip install python-dotenv
load_dotenv()                           # reads variables from .env into os.environ

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY not found. "
        "Add it to a .env file or export it in your shell."
    )


# ── Open existing Chroma collection ────────────────────────────────────────
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

PERSIST_DIR  = r"C:\Users\AnandP\Desktop\assesment\chroma_db"   # ← save Chroma DB here
COLLECTION   = Path(PERSIST_DIR).stem                           # "chroma_db"

vectordb = Chroma(
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large"),
    collection_name    = COLLECTION,
    persist_directory  = PERSIST_DIR,
)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# Turn retriever into a LangChain tool
from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(
    retriever,
    "pdf_search",
    "Search and return information from the ingested PDFs."
)

# ── LLM model used in all nodes ────────────────────────────────────────────
from langchain.chat_models import init_chat_model
LLM = init_chat_model("openai:gpt-4o", temperature=0)

# ── Graph nodes ────────────────────────────────────────────────────────────
def generate_query_or_respond(state: MessagesState):
    resp = LLM.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [resp]}

# —— grade retrieved docs
GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question.\n"
    "Document:\n{context}\nQuestion: {question}\nReturn only 'yes' or 'no'."
)
class Grade(BaseModel):
    binary_score: str = Field(...)

GRADER = init_chat_model("openai:gpt-4o", temperature=0)

def grade_docs(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    q   = state["messages"][0].content
    ctx = state["messages"][-1].content
    res = GRADER.with_structured_output(Grade).invoke(
        [{"role": "user", "content": GRADE_PROMPT.format(question=q, context=ctx)}])
    return "generate_answer" if res.binary_score == "yes" else "rewrite_question"

# —— optional rewrite
REWRITE_PROMPT = "Rewrite this question to improve search recall:\n{question}"
def rewrite_question(state: MessagesState):
    q = state["messages"][0].content
    better = LLM.invoke([{"role": "user", "content": REWRITE_PROMPT.format(question=q)}])
    return {"messages": [{"role": "user", "content": better.content}]}

# —— final answer
ANSWER_PROMPT = (
    "You are a concise Q&A assistant.\n"
    "Use the context below to answer in ≤3 sentences.\n"
    "Question: {question}\nContext: {context}"
)
def generate_answer(state: MessagesState):
    q = state["messages"][0].content
    ctx = state["messages"][-1].content
    ans = LLM.invoke([{"role": "user",
                       "content": ANSWER_PROMPT.format(question=q, context=ctx)}])
    return {"messages": [ans]}

# ── Build LangGraph workflow ───────────────────────────────────────────────
workflow = StateGraph(MessagesState)
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")
workflow.add_conditional_edges("generate_query_or_respond",
                               tools_condition,
                               {"tools": "retrieve", END: END})
workflow.add_conditional_edges("retrieve", grade_docs)
workflow.add_edge("rewrite_question", "generate_query_or_respond")
workflow.add_edge("generate_answer", END)

graph = workflow.compile()


def retrieve_sim_scores(query: str, k: int = 4) -> list[float]:
    """Direct Chroma call; returns the k similarity scores."""
    pairs = vectordb.similarity_search_with_relevance_scores(query, k=k)
    return [score for _, score in pairs]

def ask_pdf(query: str, k: int = 4) -> tuple[str, float]:
    """
    1. Run the LangGraph agent → get the answer text.
    2. Hit the vector DB again → get similarity scores.
    3. Compute and return confidence.
    """
    result = graph.invoke({"messages": [{"role": "user", "content": query}]})
    answer_text = result["messages"][-1].content

    sim_scores = retrieve_sim_scores(query, k=k)
    conf = compute_rag_confidence(query, answer_text, sim_scores)

    return answer_text, conf

# ── Simple CLI demo ─────────────────────────────────────────────────────
if __name__ == "__main__":
    while True:
        try:
            q = input("\n🔎 Ask about your PDFs (Enter to quit): ").strip()
            if not q:
                break

            answer, conf = ask_pdf(q)
            print(f"\n🟢  Answer  (confidence ≈ {conf:.2f}):\n{answer}")

        except KeyboardInterrupt:
            break