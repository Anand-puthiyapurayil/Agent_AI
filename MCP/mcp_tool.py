# subagent_server.py  (run once per machine or container)
from fastmcp import FastMCP
from agents.rag_agent import ask_pdf
from agents.prebuilt_sql_agent import sql_app

mcp = FastMCP("RAG+SQL server")

@mcp.tool(name="ask_pdf", description="Query the PDF collection")
def ask_pdf_tool(question: str) -> str:
    print(f"[ask_pdf_tool] Received question: {question!r}")
    ans, conf = ask_pdf(question)
    print(f"[ask_pdf_tool] ask_pdf returned answer={ans!r}, confidence={conf!r}")
    result = f"{ans}  (confidence≈{conf:.2f})"
    print(f"[ask_pdf_tool] Sending back: {result!r}")
    return result

@mcp.tool(name="ask_sql", description="Query the relational DB")
def ask_sql_tool(question: str) -> str:
    print(f"[ask_sql_tool] Received question: {question!r}")
    out = sql_app(question)
    print(f"[ask_sql_tool] sql_app returned: {out!r}")
    answer = out.get("answer")
    conf   = out.get("confidence")
    result = f"{answer}  (confidence≈{conf:.2f})"
    print(f"[ask_sql_tool] Sending back: {result!r}")
    return result

if __name__ == "__main__":
    print("Starting subagent_server on http://127.0.0.1:8080/mcp/")
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8080,
        path="/mcp/",
    )
