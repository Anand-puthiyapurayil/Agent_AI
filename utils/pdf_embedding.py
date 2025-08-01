
import os, glob
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


# ── Define paths relative to the project root ────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent   # <-- adjust as needed
PDF_DIR      = ROOT_DIR / "data"
PERSIST_DIR  = ROOT_DIR / "chroma_db"
COLLECTION   = PERSIST_DIR.name

# ── Load + split PDFs ────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_paths = glob.glob(f"{PDF_DIR}/*.pdf")
assert pdf_paths, f"No PDFs found in {PDF_DIR}"

docs = []
for path in pdf_paths:
    # PyPDFLoader returns one Document per page
    docs.extend(PyPDFLoader(path).load())

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks   = splitter.split_documents(docs)

print(f"Indexed {len(chunks)} chunks from {len(pdf_paths)} PDFs.")

# ── Embed + store in Chroma ──────────────────────────────────────────────
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=OPENAI_API_KEY
)

vectordb = Chroma(
    collection_name   = COLLECTION,
    embedding_function= embeddings,      # note the parameter name
    persist_directory = PERSIST_DIR,
)

vectordb.add_documents(chunks)           # add the pages you split

print(f"✅  Collection “{COLLECTION}” written to {PERSIST_DIR}")