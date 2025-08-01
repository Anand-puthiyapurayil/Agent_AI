from agents.prebuilt_sql_agent import sql_agent
for step in sql_agent({"query": "tell me total runs scored by virat kohli "}, live=True):
    step["messages"][-1].pretty_print()


#for chunk, meta in agent.stream(
#        {"messages": [HumanMessage(content=question)]},
        stream_mode="messages",               # yields (chunk, meta) pairs
):
    # Every chunk carries the node name that produced it
    if meta.get("langgraph_node") == "generate_query":   # ← last answer node
        # Guard against system/empty chunks
        if chunk.content:
            print(chunk.content, end="", flush=True)

print()  # newline after the stream finishes