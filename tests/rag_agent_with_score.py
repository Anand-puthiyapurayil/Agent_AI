#!/usr/bin/env python3
"""
rag_agent_with_score.py
───────────────────────
• Graph does:   user Q → retrieve (docs, scores) → answer.
• Outside graph: CLI computes and prints confidence.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Extra
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

from guardrails.rag_confidence import compute_rag_confidence

# ── env + LLM ──────────────────────────────────────────────────────────
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Set OPENAI_API_KEY first")

LLM = init_chat_model("openai:gpt-4o", temperature=0)

# ── vector store ──────────────────────────────────────────────────────
PERSIST_DIR = r"C:\Users\AnandP\Desktop\assesment\chroma_db"
COLLECTION  = Path(PERSIST_DIR).stem

vectordb = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    collection_name   = COLLECTION,
    persist_directory = PERSIST_DIR,
)

# ── custom state ──────────────────────────────────────────────────────
class RagState(BaseModel, extra=Extra.allow):
    messages: List[dict]
    docs:      List[Document] | None = None         # retrieved context

# ── graph nodes ───────────────────────────────────────────────────────
def retrieve_docs(state: RagState) -> dict:
    q = state.messages[0]["content"]
    k = 4
    pairs: List[Tuple[Document, float]] = (
        vectordb.similarity_search_with_relevance_scores(q, k=k)
    )

    docs = []
    for doc, score in pairs:
        doc.metadata["similarity_score"] = score
        docs.append(doc)

    return {"docs": docs}  # only docs travel through the graph

PROMPT = (
    "You are a concise Q&A assistant.\n"
    "Use the context below to answer in ≤3 sentences.\n"
    "Question: {question}\nContext:\n{context}"
)
def generate_answer(state: RagState) -> dict:
    q    = state.messages[0]["content"]
    ctx  = "\n\n".join(d.page_content for d in (state.docs or []))
    ans  = LLM.invoke([{
        "role": "user",
        "content": PROMPT.format(question=q, context=ctx)
    }])
    return {"messages": state.messages + [ans]}

# ── build graph ───────────────────────────────────────────────────────
workflow = StateGraph(RagState)
workflow.add_node(retrieve_docs)
workflow.add_node(generate_answer)
workflow.add_edge(START, "retrieve_docs")
workflow.add_edge("retrieve_docs", "generate_answer")
workflow.add_edge("generate_answer", END)
graph = workflow.compile()

# ── CLI loop ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    while True:
        try:
            q = input("\n🔎 Ask about your PDFs (Enter to quit): ").strip()
            if not q:
                break

            result: RagState = graph.invoke(
                {"messages": [{"role": "user", "content": q}]}
            )

            # — compute confidence OUTSIDE the graph —
            sim_scores = [d.metadata["similarity_score"] for d in result.docs]
            confidence = compute_rag_confidence(q, result.messages[-1].content, sim_scores)

            # — display —
            print(f"\n🟢  Answer  (confidence ≈ {confidence:.2f}):\n",
                  result.messages[-1].content)

            print("\n📑 Retrieved passages:")
            for i, d in enumerate(result.docs, 1):
                snippet = d.page_content.replace("\n", " ")[:120]
                print(f"[{i}] {d.metadata['similarity_score']:.3f} | {snippet} …")

        except KeyboardInterrupt:
            break
