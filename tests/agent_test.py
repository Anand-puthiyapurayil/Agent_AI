"""
LangGraph SQL agent
───────────────────
• ReAct flow: list → schema → generate → (checker) → run
• Aggregate awareness (SUM / AVG … when question asks for totals)
• Confidence score via guardrails‑ai when available, else heuristic
"""

import os, math
from typing import Literal, Union

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

# ── 1. env & LLM ────────────────────────────────────────────────────────────
load_dotenv()                                  # needs DATABASE_URL + OPENAI_API_KEY
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ── 2. DB & toolkit ─────────────────────────────────────────────────────────
db       = SQLDatabase.from_uri(os.environ["DATABASE_URL"])
toolkit  = SQLDatabaseToolkit(db=db, llm=llm)
TOOLS    = {t.name: t for t in toolkit.get_tools()}
LIST_TOOL, SCHEMA_TOOL, RUN_TOOL = (
    TOOLS["sql_db_list_tables"],
    TOOLS["sql_db_schema"],
    TOOLS["sql_db_query"],
)
CHECK_TOOL = TOOLS.get("sql_db_query_checker")   # optional

# ── 3. tool nodes ───────────────────────────────────────────────────────────
SCHEMA_NODE = ToolNode([SCHEMA_TOOL], name="get_schema")
RUN_NODE    = ToolNode([RUN_TOOL],    name="run_query")

# ── 4. custom nodes ─────────────────────────────────────────────────────────
def list_tables(state: MessagesState):
    call = {"name": LIST_TOOL.name, "args": {}, "id": "list_tables", "type": "tool_call"}
    call_msg = AIMessage(content="", tool_calls=[call])
    tool_msg = LIST_TOOL.invoke(call)
    summary  = AIMessage(content=f"Available tables: {tool_msg.content}")
    return {"messages": [call_msg, tool_msg, summary]}

def call_get_schema(state: MessagesState):
    return {"messages": [
        llm.bind_tools([SCHEMA_TOOL], tool_choice="any").invoke(state["messages"])
    ]}

GEN_QUERY_PROMPT = f"""
You are an agent designed to interact with a SQL database.
If the question requests a total, average, minimum, maximum, or count, **use the
corresponding aggregate function (SUM, AVG, MIN, MAX, COUNT) and do NOT LIMIT**.
Otherwise limit to 5 rows.  Generate valid {db.dialect}; never use DML.
"""
def generate_query(state: MessagesState):
    sys = {"role": "system", "content": GEN_QUERY_PROMPT}
    resp = llm.bind_tools([RUN_TOOL]).invoke([sys] + state["messages"])
    return {"messages": [resp]}

CHECK_PROMPT = f"""
You are a SQL expert. Review the {db.dialect} query for common mistakes; fix if
needed, else return unchanged, then call `sql_db_query`.
"""
def check_query(state: MessagesState):
    if CHECK_TOOL is None:
        return {"messages": []}
    tool_call = state["messages"][-1].tool_calls[0]
    user = {"role": "user", "content": tool_call["args"]["query"]}
    sys  = {"role": "system", "content": CHECK_PROMPT}
    resp = llm.bind_tools([RUN_TOOL], tool_choice="any").invoke([sys, user])
    resp.id = state["messages"][-1].id
    return {"messages": [resp]}

try:
    from guardrails.sql_guardrails import compute_sql_confidence
except ImportError:                           # fallback heuristic
    def compute_sql_confidence(**_): return 0.0

def add_confidence(state: MessagesState):
    msgs = state["messages"]
    user = next(m for m in reversed(msgs) if isinstance(m, HumanMessage))
    ai   = next(m for m in reversed(msgs) if isinstance(m, AIMessage) and m.tool_calls)
    tool = next(m for m in reversed(msgs) if isinstance(m, ToolMessage))
    sql  = ai.tool_calls[0]["args"]["query"]
    rows = tool.content
    if isinstance(rows, str):
        try: rows = eval(rows)
        except Exception: pass
    conf = compute_sql_confidence(
        user_query=user.content, generated_sql=sql,
        execution_results=rows, dialect=db.dialect,
    ) or math.tanh(len(rows)/10)*0.95
    return {"messages": [], "confidence": round(conf, 3)}

# routing
def after_generate(state: MessagesState) -> Literal["check_query", "confidence"]:
    return "check_query" if state["messages"][-1].tool_calls else "confidence"

# ── 5. build graph  (NODE‑FIRST, EDGES AFTER) ───────────────────────────────
builder = StateGraph(MessagesState)
builder.add_node(list_tables)
builder.add_node(call_get_schema)
builder.add_node(SCHEMA_NODE, "get_schema")
builder.add_node(generate_query)
builder.add_node(check_query)
builder.add_node(RUN_NODE, "run_query")
builder.add_node(add_confidence, name="confidence")   # ← ensure registered early

builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", "generate_query")
builder.add_edge("confidence", END)
builder.add_conditional_edges("generate_query", after_generate)  # add **last**

AGENT = builder.compile()

# ── 6. convenience wrapper ──────────────────────────────────────────────────
def sql_agent(question: Union[str, dict], *, live: bool = False):
    if isinstance(question, dict):
        question = question.get("query", "")
    init = {"messages": [{"role": "user", "content": question}]}
    stream = AGENT.stream(init, stream_mode="values")
    if live:
        yield from stream
        return
    final = None
    for chunk in stream: final = chunk
    if final is None:
        return {"answer": "(no output)", "confidence": None}
    return {
        "answer": final["messages"][-1].content,
        "confidence": final.get("confidence"),
    }

# ── 7. demo ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(sql_agent("total runs scored by V Kohli?"))
