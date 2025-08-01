# router.py ─────────────────────────────────────────────────────────────
"""
ReAct router that streams step‐by‐step updates + tokens.

Usage:

  # 1. Make sure your sub-agent MCP server is running:
  python mcp_tool.py   # listens on http://localhost:8080/mcp/

  # 2a. Interactive debug CLI:
  python router.py

  # 2b. Run as an HTTP microservice:
  python router.py serve   # serves on http://localhost:9000/mcp/
"""

from __future__ import annotations
import asyncio
import os
import sys
import pprint
from typing import List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from fastmcp import FastMCP

pp = pprint.PrettyPrinter(depth=4, compact=True)

# ─── 0. Load environment & validate ─────────────────────────────────────
load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("OPENAI_API_KEY not set in .env or environment")

# ─── 1. Discover remote tools via MCP ──────────────────────────────────
async def _discover_tools() -> dict[str, any]:
    client = MultiServerMCPClient({
        "pdf_sql_server": {
            "url": "http://localhost:8080/mcp/",
            "transport": "streamable_http",
        }
    })
    tools = await client.get_tools()           # coroutine → list of StructuredTool
    return {t.name: t for t in tools}

remote_tools = asyncio.run(_discover_tools())
tools_list: List = list(remote_tools.values())

# ─── 2. Initialize LLM & create ReAct agent ─────────────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0)

router_agent = create_react_agent(
    model=llm,
    tools=tools_list,
    prompt=(
        "You are a routing agent. You have two tools:\n"
        "  - ask_pdf(question): queries a PDF knowledge base\n"
        "  - ask_sql(question): queries a SQL database\n"
        "Use ReAct format with steps:\n"
        "  Thought: your reasoning\n"
        "  Action: the tool call (e.g. <tool_call name=\"ask_pdf\" args=\"...\"/>)\n"
        "  Observation: the tool's result\n"
        "After reasoning, produce only the final concise answer, with no extra text."
    ),
    name="router_agent",
)

# ─── 3. Stream helper: yield both updates & messages ───────────────────
async def route_query_stream(question: str):
    """Yield (mode, chunk) from the agent for updates+messages."""
    async for mode, chunk in router_agent.astream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode=["updates", "messages"],
    ):
        yield mode, chunk

# ─── 4. FastMCP tool: stream same updates+messages over HTTP ────────────
mcp = FastMCP("Router-ReAct")

@mcp.tool(
    name="route_query",
    description="Stream step-by-step updates and tokens for a given question.",
)
async def route_query_tool(question: str):
    async for mode, chunk in route_query_stream(question):
        yield mode, chunk

# ─── 5. CLI: print every update & token ────────────────────────────────
async def _cli():
    while True:
        q = input("\n❓  ").strip()
        if not q:
            break

        async for mode, chunk in route_query_stream(q):
            if mode == "updates":
                # chunk is a dict with the node name, state, etc.
                node = chunk.get("langgraph_node", "?")
                print(f"\n🔄  STEP: {node}")
                if "messages" in chunk:
                    # show the last message (tool output or final reply)
                    print("   ↳", chunk["messages"][-1].content)
            elif mode == "messages":
                # chunk is a (ChatMessageChunk, metadata) tuple
                token, meta = chunk
                if token.content:
                    # stream tokens in-line
                    print(token.content, end="", flush=True)

# ─── 6. Entry-point: serve or CLI ──────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].lower() == "serve":
        # Start the MCP HTTP server on port 9000 (no SSE session needed)
        mcp.run(transport="http", host="0.0.0.0", port=9000)
    else:
        # Interactive CLI
        try:
            asyncio.run(_cli())
        except KeyboardInterrupt:
            pass
