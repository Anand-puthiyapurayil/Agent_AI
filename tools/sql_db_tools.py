from langchain.tools import tool
# LangChain 0.2+ moved community models; keep this if you’re on <=0.1.x
# from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI

from tools.sql_tool import SQLTool
from guardrails.sql_guardrails import ensure_read_only        # ← fixed spelling
from prompts.sql_prompts import CHECK_QUERY_PROMPT

_sql = SQLTool()
_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ------------------------------------------------------------------ #
@tool(
    "sql_db_list_tables",
    return_direct=True,
    description="Return a comma‑separated list of all tables in the SQL database."
)
def sql_db_list_tables() -> str:      # ← remove the '_' parameter
    return ", ".join(_sql.list_tables())


# ------------------------------------------------------------------ #
@tool(
    "sql_db_schema",
    return_direct=True,
    description="Given one or more table names (comma‑separated), show each table’s schema and three sample rows."
)
def sql_db_schema(table_names: str) -> str:
    """Return CREATE TABLE skeleton + sample rows for each table."""
    outs = []
    for name in map(str.strip, table_names.split(",")):
        schema = _sql.get_schema(name)
        sample = _sql.execute_query(f"SELECT * FROM {name} LIMIT 3;")
        outs.append(
            f"CREATE TABLE {name} (...);\n-- columns: {schema}\n"
            f"/* sample rows:\n{sample} */"
        )
    return "\n\n".join(outs)

# ------------------------------------------------------------------ #
@tool(
    "sql_db_query_checker",
    return_direct=True,
    description="Pass an SQL query to the LLM for a safety/content check and return its verdict."
)
def sql_db_query_checker(query: str) -> str:
    """Ask the LLM to check whether the query is safe / syntactically valid."""
    return _llm.invoke(CHECK_QUERY_PROMPT.format(query=query)).content.strip()

# ------------------------------------------------------------------ #
@tool(
    "sql_db_query",
    return_direct=True,
    description="Execute a read‑only (SELECT) SQL query after guardrail validation."
)
def sql_db_query(query: str) -> str:
    """Run a validated read‑only SQL query and return the results as a string."""
    ensure_read_only(query)               # raises if not SELECT
    return str(_sql.execute_query(query))
