Assesment 


This project implements an Intelligent Query Router that dynamically routes natural language queries to the most relevant data source—SQL or RAG (vector store)—based on intelligent confidence scoring. If both paths fail, the system rephrases and retries the query using an LLM.

🚀 Project Overview 
SQL Agent Tool: LangGraph + LangChain SQL agent wrapped as a tool, executes validated SQL queries over PostgreSQL.

RAG Agent Tool: Embedding-based document retriever (ChromaDB) wrapped as a tool, handles unstructured data (PDFs).


Router Agents:

LangChain ReAct Router: A standard ReAct agent that decides between tools (SQL or RAG) using tool-calling logic.

LangGraph Router: A graph-based pipeline that:

Calls both SQL and RAG tools

Scores both responses using heuristics + cross-encoder

Picks the better result based on confidence

Optionally rephrases and retries the query if needed

Tool Server (MCP): Hosts both SQL and RAG agents as callable MCP tools. 



📁 Repository Structure
text
Copy
Edit
├── agents/
│   ├── rag_agent.py           # RAG retrieval agent + tool
│   ├── prebuilt_sql_agent.py  # LangGraph SQL graph + tool
│   
│
├── MCP/
│   ├── tools.py               # Tool loader/registrar (RAG + SQL)
│   └── mcp_server.py          # Launches MCP tool server
│
├── utils/
│   ├── Pdf_embedding.py          # Embeds PDFs to ChromaDB
│   ├── setup_postgres.py         # Loads CSVs into PostgreSQL
│   
│
├── guardrails/
│   ├── sql_guardrails.py      # SQL row-level confidence scoring
│   └── rag_guardrails.py      # Answer string-level scoring
│
├── prompts/
│   ├── sql_prompts.py         # Prompts for SQL generation/checking
│   └── rag_prompts.py         # Prompts for RAG retrieval
│
├── tests/
│   ├── test_sql.py            # Unit tests for SQL agent   ## DONT USE
│   ├── test_rag.py            # Unit tests for RAG agent
│   └── test_router.py         # Full router tests
│
├── gradio_ui/
│   └── app.py                 # Gradio UI interface for user input
│
├── scripts/
│   └── example_run.py         # Optional runners / experiments
│
├── data/
│   ├── csv_data.csv           # Structured tabular example
│   └── pdf_docs/              # Sample PDFs for RAG
│
├── chroma_db/                 # Local ChromaDB storage
├── .env                       # API keys and DB URL config
├── requirements.txt           # Python dependencies
└── README.md                  # You are here
🛠️ Setup
Create & activate environment (Python 3.11)

Conda:

bash
Copy
Edit
conda create -n "env Name" python=3.11 -y
conda activate venv
venv:

bash
Copy
Edit
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Install optional guard-rail models (for confidence scoring):

bash
Copy
Edit
pip install sentence-transformers torch numpy
Configure .env:

Create a .env file with:

dotenv
Copy
Edit
DATABASE_URL=postgresql://user:pass@localhost:5432/your_db
OPENAI_API_KEY=sk-...
CE_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2   


⚙️ Running the System
Step 1: Host the MCP Tool Server
This serves the SQL and RAG agents as callable tools.

bash
Copy
Edit
python -m MCP.mcp_server
Keep it running in one terminal.

Step 2: Run the Query Router
In a new terminal, run the router interface:

bash
Copy
Edit
python -m agents.router
# ❓  Enter your question: Who submitted the compliance report?
Output will show the selected tool, answer, and confidence score.

Optional: Run Gradio UI
bash
Copy
Edit
python gradio_ui/app.py
This launches a browser-based interface to test the system interactively.

🔍 Confidence Scoring
1. SQL Confidence (compute_sql_confidence)
Row volume: min(#rows / 10, 1.0)

Null coverage: 1 − (null_cells / total_cells)

Cross-encoder score for top-6 rows

Weighted sum: 0.3 × volume + 0.2 × coverage + 0.5 × encoder

2. Answer Confidence (compute_answer_confidence)
Cross-encoder similarity between question and answer

Penalizes overly short or lengthy answers

Checks if numeric tokens match

Final Confidence
text
Copy
Edit
confidence = 0.7 × sql_score + 0.3 × answer_score
📖 Architecture Overview
🧠 SQL Agent
LangGraph-based

Nodes: list_tables → get_schema → generate → check → run

Exposed via MCP as a tool

📄 RAG Agent
Embeds PDF/CSV to ChromaDB

Retrieves top-k docs

Synthesizes answer via LLM

Exposed via MCP as a tool

🔀 Query Router
Step 1: Try SQL Agent → compute confidence

Step 2: Fallback to RAG → compare score

Step 3: Rephrase and retry if both are low

🧪 Running Tests
bash
Copy
Edit
pytest tests/
Make sure SQL and Chroma data are preloaded.

📜 License
This project is licensed under the MIT License. See LICENSE for details.