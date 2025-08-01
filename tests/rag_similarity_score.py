import os, getpass
from typing import Literal
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from pathlib import Path

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

q = "runs scored by virat kohli against kkr?"
k = 4  
results = vectordb.similarity_search_with_relevance_scores(q, k=k)

for i, (doc, score) in enumerate(results, 1):
    print(f"[{i}] score = {score:.3f}")
    print(doc.page_content[:300], "...")
    print(doc.metadata)  