# 🧠 Intelligent Query Router with Sub-Agents

This project implements an **Intelligent Query Router** that dynamically routes natural language queries to the most relevant data source—**SQL** or **RAG (Vector Store)**—based on confidence scoring. If both tools fail to confidently respond, the system rephrases and retries the query using an LLM.

---

## 🚀 Project Overview

### 🔧 Tools

* **SQL Agent Tool**: LangGraph + LangChain SQL agent wrapped as a tool, executes validated SQL queries over PostgreSQL.
* **RAG Agent Tool**: Embedding-based document retriever (ChromaDB), handles unstructured data (PDFs), exposed as a tool.

### 🧠 Router Agents

* **LangChain ReAct Router**: Uses tool-calling logic to choose between SQL and RAG tools.
* **LangGraph Router**:

  * Calls both tools in parallel
  * Scores both responses using heuristics + cross-encoder
  * Picks the best response
  * Optionally rephrases and retries the query

### 🛠️ Tool Server (MCP)

Hosts both SQL and RAG tools as callable Model Context Protocol (MCP) endpoints.

---

## 📁 Repository Structure

```
├── agents/
│   ├── rag_agent.py              # RAG retrieval agent + tool
│   ├── prebuilt_sql_agent.py     # LangGraph SQL agent as a tool
│
├── MCP/
│   ├── tools.py                  # Tool loader for SQL & RAG
│   └── mcp_server.py             # Launches MCP tool server
│
├── utils/
│   ├── Pdf_embedding.py          # Embeds PDFs to ChromaDB
│   ├── setup_postgres.py         # Loads CSV into PostgreSQL
│
├── guardrails/
│   ├── sql_guardrails.py         # SQL confidence scoring
│   └── rag_guardrails.py         # RAG answer scoring
│
├── prompts/
│   ├── sql_prompts.py            # Prompt templates for SQL
│   └── rag_prompts.py            # Prompt templates for RAG
│
├── tests/
│   ├── test_sql.py               # (Do not use)
│   ├── test_rag.py               # RAG unit tests
│   └── test_router.py            # Router integration tests
│
├── gradio_ui/
│   └── app.py                    # Gradio interface
│
├── scripts/
│   └── example_run.py            # Optional runner
│
├── data/
│   ├── csv_data.csv              # Sample structured data
│   └── pdf_docs/                 # Sample PDFs
│
├── chroma_db/                    # Chroma vector DB (ignored)
├── .env                          # API keys + DB config
├── requirements.txt              # Project dependencies
└── README.md                     # You are here
```

---

## 🛠️ Setup

### ✅ Create Environment (Python 3.11)

**With Conda:**

```bash
conda create -n agent-router python=3.11 -y
conda activate agent-router
```

**With venv:**

```bash
python3.11 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

---

### ✅ Install Dependencies

```bash
pip install -r requirements.txt
pip install sentence-transformers torch numpy   # (optional scoring models)
```

---

### ✅ Configure `.env`

Create a `.env` file at the root:

```env
DATABASE_URL=postgresql://user:pass@localhost:5432/your_db
OPENAI_API_KEY=sk-...
CE_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## ⚙️ Running the System

### ▶️ Step 1: Start MCP Tool Server

```bash
python -m MCP.mcp_server
```

> Keep this terminal open — it hosts the tools.

---

### ▶️ Step 2: Run the Query Router

```bash
python -m agents.router
# Example input:
# ❓  Who submitted the compliance report?
```

> Output includes selected tool, answer, and confidence score.

---

### 🖼 Optional: Launch Gradio UI

```bash
python gradio_ui/app.py
```

> Opens a browser UI for testing the system interactively.

---

## 🔍 Confidence Scoring Logic

### ✅ 1. SQL Confidence

* **Row volume**: `min(num_rows / 10, 1.0)`
* **Null coverage**: `1 - (null_cells / total_cells)`
* **Cross-encoder**: Similarity over top rows

```python
sql_conf = 0.3 * volume + 0.2 * coverage + 0.5 * encoder
```

---

### ✅ 2. Answer Confidence

* Cross-encoder similarity between question and answer
* Penalizes overly short or long responses
* Checks numeric token alignment

---

### ✅ Final Score

```python
confidence = 0.7 * sql_conf + 0.3 * answer_conf
```

---

## 🧠 Architecture Summary

### 📃 SQL Agent (LangGraph)

* Nodes: `list_tables → get_schema → generate → validate → run`
* Exposed via MCP tool server

### 📄 RAG Agent (LangChain + Chroma)

* Embeds PDF & CSV to vector DB
* Retrieves top-k documents
* Synthesizes answer with LLM

### 🔀 Query Router

1. Try SQL tool → score
2. Try RAG tool → score
3. Pick highest score
4. Rephrase + retry if both are weak

---

## 🧪 Running Tests

Make sure `chroma_db/` and PostgreSQL are preloaded:

```bash
pytest tests/
```

---

## 📜 License

MIT License — see `LICENSE` file for details.
