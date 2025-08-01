#!/usr/bin/env python3
# router2.py

import os, sys, re, asyncio
from typing import Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_mcp_adapters.client import MultiServerMCPClient
from fastmcp import FastMCP

# ─── 0. LOAD ENVIRONMENT ──────────────────────────────────────────────
load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("OPENAI_API_KEY not set")

# ─── 1. DISCOVER TOOLS ────────────────────────────────────────────────
async def _discover_tools():
    client = MultiServerMCPClient({
        "pdf_sql_server": {
            "url": "http://localhost:8080/mcp/",
            "transport": "streamable_http",
        }
    })
    tools = await client.get_tools()
    return {t.name: t for t in tools}

remote_tools = asyncio.run(_discover_tools())
ask_pdf_tool = remote_tools["ask_pdf"]
ask_sql_tool = remote_tools["ask_sql"]

# ─── 2. SETUP STATE AND MODEL ─────────────────────────────────────────
class RouterState(BaseModel):
    question: str
    original_question: str = ""
    pdf_raw: str = ""
    pdf_conf: float = 0.0
    sql_raw: str = ""
    sql_conf: float = 0.0
    best_tool: str = ""
    best_ans: str = ""
    best_conf: float = 0.0
    output: str = ""

chat_model = init_chat_model(temperature=0, model="gpt-4o")

# ─── 3. HELPER ────────────────────────────────────────────────────────
def parse_response_with_confidence(response: str) -> tuple[str, float]:
    match = re.search(r'\(confidence≈([0-9.]+)\)', response)
    if match:
        conf = float(match.group(1))
        text = response[:match.start()].strip()
    else:
        conf = 0.0
        text = response.strip()
    return text, conf

# ─── 4. NODE FUNCTIONS ────────────────────────────────────────────────
async def rewrite_question_node(state: RouterState) -> RouterState:
    state.original_question = state.question
    response = await chat_model.ainvoke([
        SystemMessage(content="You are a helpful assistant that rewrites vague or unclear user questions."),
        HumanMessage(content=f"Rewrite the question clearly and completely: {state.question}")
    ])
    state.question = response.content.strip()
    return state

async def call_pdf_node(state: RouterState) -> RouterState:
    response = await ask_pdf_tool.ainvoke({"question": state.question})
    print(f"[DEBUG] PDF tool response: {response}")
    state.pdf_raw, state.pdf_conf = parse_response_with_confidence(response)
    return state

async def call_sql_node(state: RouterState) -> RouterState:
    response = await ask_sql_tool.ainvoke({"question": state.question})
    print(f"[DEBUG] SQL tool response: {response}")
    state.sql_raw, state.sql_conf = parse_response_with_confidence(response)
    return state

async def compare_node(state: RouterState) -> RouterState:
    if state.pdf_conf >= state.sql_conf:
        state.best_tool = "PDF"
        state.best_ans = state.pdf_raw
        state.best_conf = state.pdf_conf
    else:
        state.best_tool = "SQL"
        state.best_ans = state.sql_raw
        state.best_conf = state.sql_conf
    return state

async def generate_answer_node(state: RouterState) -> RouterState:
    response = await chat_model.ainvoke([
        SystemMessage(content="You are a smart assistant. Use the best available answer to respond clearly to the user's question."),
        HumanMessage(content=(
            f"Original question: {state.original_question or state.question}\n\n"
            f"PDF tool answer (conf={state.pdf_conf:.2f}):\n{state.pdf_raw}\n\n"
            f"SQL tool answer (conf={state.sql_conf:.2f}):\n{state.sql_raw}\n\n"
            f"Chosen answer from {state.best_tool}:\n{state.best_ans}\n\n"
            f"Now generate a clear and complete final answer."
        ))
    ])
    state.output = response.content.strip()
    return state

# ─── 5. BUILD GRAPH ───────────────────────────────────────────────────
workflow = StateGraph(RouterState)

workflow.add_node(rewrite_question_node)
workflow.add_node(call_pdf_node)
workflow.add_node(call_sql_node)
workflow.add_node(compare_node)
workflow.add_node(generate_answer_node)

workflow.add_edge(START,"rewrite_question_node")
workflow.add_edge("rewrite_question_node", "call_pdf_node")
workflow.add_edge("call_pdf_node", "call_sql_node")
workflow.add_edge("call_sql_node", "compare_node")
workflow.add_edge("compare_node", "generate_answer_node")
workflow.add_edge("generate_answer_node", END)

graph = workflow.compile()

# ─── 6. EXPOSE VIA MCP ────────────────────────────────────────────────
mcp = FastMCP("Router-Graph-Final")

@mcp.tool(
    name="route_query",
    description="Rewrite the query, use PDF+SQL tools, compare answers, generate final response."
)
async def route_query_tool(question: str) -> str:
    result = await graph.ainvoke({"question": question})
    return result["output"]

# ─── 7. CLI ENTRYPOINT ────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].lower() == "serve":
        mcp.run(transport="http", host="0.0.0.0", port=9000)
    else:
        print("▶️  CLI: returns the highest-confidence answer\n")
        while True:
            q = input("❓  ").strip()
            if not q:
                break
            out = asyncio.run(graph.ainvoke({"question": q}))
            print("\n" + out["output"] + "\n")
