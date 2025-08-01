import os
from typing import Literal
from dotenv import load_dotenv
from typing import List, Tuple,Dict, Any
from decimal import Decimal 
from langchain_openai import ChatOpenAI  # keeps .bind_tools
from langchain_core.messages import AIMessage ,HumanMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode  
from guardrails.sql_guardrails import compute_answer_confidence,compute_sql_confidence

# ---------------------------------------------------------------------------
# 1.  Environment & LLM
# ---------------------------------------------------------------------------
load_dotenv()  # loads DATABASE_URL + OPENAI_API_KEY from .env
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ---------------------------------------------------------------------------
# 2.  Database & toolkit
# ---------------------------------------------------------------------------
db = SQLDatabase.from_uri(os.environ["DATABASE_URL"])

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
# Map tools by name for quick access
tools = {t.name: t for t in toolkit.get_tools()}

list_tables_tool = tools["sql_db_list_tables"]
get_schema_tool = tools["sql_db_schema"]
run_query_tool = tools["sql_db_query"]
checker_tool = tools.get("sql_db_query_checker", None)  # optional

# ---------------------------------------------------------------------------
# 3.  Tool nodes (direct wrappers around individual tools)
# ---------------------------------------------------------------------------
get_schema_node = ToolNode([get_schema_tool], name="get_schema")
run_query_node = ToolNode([run_query_tool], name="run_query")

# ---------------------------------------------------------------------------
# 4.  Custom ReAct‑style agent nodes
# ---------------------------------------------------------------------------

def list_tables(state: MessagesState):
    """Always start by listing tables so the LLM knows the DB layout."""
    tool_call = {
        "name": list_tables_tool.name,
        "args": {},
        "id": "call_list_tables",
        "type": "tool_call",
    }
    # Fabricate the assistant tool‑call message
    tool_call_msg = AIMessage(content="", tool_calls=[tool_call])
    # Execute the actual tool
    tool_msg = list_tables_tool.invoke(tool_call)
    # Summarise the result for the LLM
    summary_msg = AIMessage(content=f"Available tables: {tool_msg.content}")
    return {"messages": [tool_call_msg, tool_msg, summary_msg]}


def call_get_schema(state: MessagesState):
    """Force the model to call the schema tool for the first table."""
    llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# Prompt used every time we generate a new SQL query
generate_query_system_prompt = f"""
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {db.dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most 5 results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
"""


def generate_query(state: MessagesState):
    system_msg = {"role": "system", "content": generate_query_system_prompt}
    llm_with_tools = llm.bind_tools([run_query_tool])  # no forced tool choice
    response = llm_with_tools.invoke([system_msg] + state["messages"])
    return {"messages": [response]}


# Prompt for the (optional) checker tool
check_query_system_prompt = f"""
You are a SQL expert with a strong attention to detail.
Double check the {db.dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes,
just reproduce the original query.

You will call the appropriate tool to execute the query after running this check.
"""


def check_query(state: MessagesState):
    # If the toolkit doesn't provide a checker, just fall through
    if checker_tool is None:
        return {"messages": []}

    tool_call = state["messages"][-1].tool_calls[0]
    user_msg = {"role": "user", "content": tool_call["args"]["query"]}
    system_msg = {"role": "system", "content": check_query_system_prompt}

    llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_msg, user_msg])
    # Preserve the original tool‑call ID so the executor knows to replace it
    response.id = state["messages"][-1].id
    return {"messages": [response]}


# Determine where to send the flow after the generate_query node

def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
    last_msg = state["messages"][-1]
    if not last_msg.tool_calls:
        return END  # model already answered with plain text
    return "check_query"  # a query was generated → check & run it


# ---------------------------------------------------------------------------
# 5.  Build the LangGraph workflow
# ---------------------------------------------------------------------------

builder = StateGraph(MessagesState)

builder.add_node(list_tables)
builder.add_node(call_get_schema)
builder.add_node(get_schema_node, "get_schema")
builder.add_node(generate_query)
builder.add_node(check_query)
builder.add_node(run_query_node, "run_query")

builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")

builder.add_conditional_edges("generate_query", should_continue)
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", "generate_query")  # look at results & answer

# Compile the graph into a runnable agent
agent = builder.compile()

# ---------------------------------------------------------------------------
# 6.  Convenience wrapper
# ---------------------------------------------------------------------------


def _extract_rows(messages) -> List[Tuple]:
    for m in reversed(messages):
        if getattr(m, "name", "") == "sql_db_query":   # ToolMessage
            try:
                rows = eval(m.content, {"Decimal": Decimal, "__builtins__": {}})
                print("🔎  Extracted rows:", rows[:3], "...")
                return rows
            except Exception as e:
                print("⚠️  eval failed:", e)
                return []
    print("⚠️  No sql_db_query ToolMessage found")
    return []


def sql_app(query: str) -> Dict[str, Any]:
    """
    One-shot execution of the LangGraph SQL agent.

    Returns
    -------
    {
        "answer":     str,
        "confidence": float,   # 0-1
        "messages":   list[BaseMessage],
        "source":    "SQL+LLM"
    }
    """
    state = agent.invoke({"messages": [HumanMessage(content=query)]})

    messages   = state["messages"]
    answer_str = messages[-1].content

    rows = _extract_rows(messages)                 # evidence
    # --- compute individual scores ---
    row_score  = compute_sql_confidence(rows, query) if rows else 0.0
    ans_score  = compute_answer_confidence(query, answer_str)

    # Blend (weights can be tuned)
    confidence = round(0.7 * row_score + 0.3 * ans_score, 3)

    return {
        "answer":     answer_str,
        "confidence": confidence,
        "messages":   messages,
        "source":    "SQL+LLM",
    }


if __name__ == "__main__":
    q = input("❓  Enter your question: ").strip()
    res = sql_app(q)
    print("\n📝  Answer:\n", res["answer"])
    print(f"\n✅  Confidence: {res['confidence']}")