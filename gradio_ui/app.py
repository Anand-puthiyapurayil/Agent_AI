# gradio_ui/app.py

import gradio as gr
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

# Connect to FastMCP Router server
client = MultiServerMCPClient({
    "router_server": {
        "url": "http://localhost:9000/mcp/",
        "transport": "streamable_http"
    }
})

tools = asyncio.run(client.get_tools())
router_tool = next(t for t in tools if t.name == "route_query")

# gradio_ui/app.py

import gradio as gr
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

# Connect to FastMCP Router server
client = MultiServerMCPClient({
    "router_server": {
        "url": "http://localhost:9000/mcp/",
        "transport": "streamable_http"
    }
})

# Fetch tools and get "route_query"
tools = asyncio.run(client.get_tools())
router_tool = next(t for t in tools if t.name == "route_query")

# Async-compatible function for Gradio
async def ask_router(question):
    try:
        response = await router_tool.ainvoke({"question": question})
        return response
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Gradio Interface
iface = gr.Interface(
    fn=ask_router,
    inputs=gr.Textbox(label="Ask your question", placeholder="e.g. What is the revenue in 2023?"),
    outputs=gr.Textbox(label="Answer"),
    title="🧠 Intelligent Query Router",
    description="Uses LangGraph + MCP to route questions to SQL or PDF tools."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)


# Launch Gradio UI
iface = gr.Interface(
    fn=ask_router,
    inputs=gr.Textbox(label="Ask your question", placeholder="e.g. What is the revenue in 2022?"),
    outputs=gr.Textbox(label="Answer"),
    title="🧠 Intelligent Query Router",
    description="Routes your query to SQL or PDF tool using LangGraph + MCP."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
