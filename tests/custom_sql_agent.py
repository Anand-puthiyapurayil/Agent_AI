from typing import Literal
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI  # legacy import retains bind_tools
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

from prompts.sql_prompts import CHECK_QUERY_PROMPT, GEN_QUERY_PROMPT

# ---------------------------------------------------------------------------
# LLM & ENV
# ---------------------------------------------------------------------------
load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ---------------------------------------------------------------------------
# DB StructuredTools
# ---------------------------------------------------------------------------
from tools.sql_db_tools import (
    sql_db_list_tables,
    sql_db_schema,
    sql_db_query_checker,
    sql_db_query,
)

list_tool = sql_db_list_tables
schema_tool = sql_db_schema
checker_tool = sql_db_query_checker
run_tool = sql_db_query

# Nodes that simply wrap a single StructuredTool ----------------------------
schema_node = ToolNode([schema_tool], name="get_schema")
run_node    = ToolNode([run_tool],    name="run_query")

# ---------------------------------------------------------------------------
# Custom graph nodes
# ---------------------------------------------------------------------------

def list_tables(state: MessagesState):
    """Always start by listing tables so the LLM knows the schema space."""
    # Assistant triggers the tool
    assistant_call = AIMessage(
        content="",
        tool_calls=[{"name": list_tool.name, "args": {}, "id": "list000", "type": "tool_call"}],
    )
    # Tool executes
    tables_str: str = list_tool.invoke({})  # no args
    tool_response = ToolMessage(content=tables_str, tool_call_id="list000")
    # Assistant summarises
    summary = AIMessage(content=f"Available tables: {tables_str}")
    return {"messages": [assistant_call, tool_response, summary]}


def call_get_schema(state: MessagesState):
    """Let the model decide which table schemas it needs by calling the schema tool."""
    bound = llm.bind_tools([schema_tool], tool_choice="any")
    assistant_msg = bound.invoke(state["messages"])
    return {"messages": [assistant_msg]}


def generate_query(state: MessagesState):
    """Generate a SELECT query using the run_query tool (no forced call)."""
    sys = {"role": "system", "content": GEN_QUERY_PROMPT}
    bound = llm.bind_tools([run_tool])
    assistant_msg = bound.invoke([sys] + state["messages"])
    return {"messages": [assistant_msg]}


def check_query(state: MessagesState):
    """Run the checker tool on the latest generated SQL before executing it."""
    tool_call = state["messages"][-1].tool_calls[0]
    user_msg = {"role": "user", "content": tool_call["args"]["query"]}
    sys_msg  = {"role": "system", "content": CHECK_QUERY_PROMPT}
    bound = llm.bind_tools([checker_tool], tool_choice="any")
    assistant_msg = bound.invoke([sys_msg, user_msg])
    assistant_msg.id = state["messages"][-1].id  # keep same id for correlation
    return {"messages": [assistant_msg]}


def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
    """Route to checker only if a tool_call was created."""
    last = state["messages"][-1]

    # No tool call → the LLM already gave a final answer
    if not last.tool_calls:
        return END

    # A tool call for run_query means we already executed the SQL
    if last.tool_calls[0]["name"] == run_tool.name:
        return END

    # Otherwise we still need to check the proposed SQL
    return "check_query"

# ---------------------------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------------------------

builder = StateGraph(MessagesState)

builder.add_node(list_tables)
builder.add_node(call_get_schema)
builder.add_node(schema_node, "get_schema")
builder.add_node(generate_query)
builder.add_node(check_query)
builder.add_node(run_node, "run_query")

builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")

builder.add_conditional_edges("generate_query", should_continue)
builder.add_edge("check_query", "run_query")

_graph = builder.compile()

# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------
from guardrails.sql_guardrails import rows_to_confidence

def sql_app(state: dict):
    """Run a natural‑language query through the LangGraph SQL agent."""
    messages = [{"role": "user", "content": state["query"]}]
    trace = _graph.stream({"messages": messages})

    # Grab the first ToolMessage produced by the run_query tool
    from langchain_core.messages import ToolMessage

    rows = None
    for m in reversed(trace["messages"]):
        if isinstance(m, ToolMessage):
            try:
                rows = eval(m.content)          # rows as list[dict]
            except Exception:
                rows = m.content               # fallback to raw string
            break

    return {
        "result": rows or trace["messages"][-1].content,
        "confidence": rows_to_confidence(rows),
        "source": "SQL",
    }


if __name__ == "__main__":
    print(sql_app({"query": "Show 3 rows from dataset1"}))
