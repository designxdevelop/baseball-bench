from __future__ import annotations

import json
from pathlib import Path

from inspect_ai.tool import Tool, ToolError, tool

from baseball_bench.data import execute_read_only_query, schema_text
from baseball_bench.paths import DATA_DB_PATH


@tool
def sql_tool(database_path: str | None = None, row_limit: int = 50) -> Tool:
    """Execute a read-only SQL query against the benchmark DuckDB database."""

    async def execute(sql: str) -> str:
        try:
            result = execute_read_only_query(
                sql,
                database_path=Path(database_path) if database_path else DATA_DB_PATH,
                row_limit=row_limit,
            )
        except Exception as exc:  # noqa: BLE001
            raise ToolError(str(exc)) from exc
        return json.dumps(result, indent=2)

    return execute


@tool
def schema_lookup_tool() -> Tool:
    """Return the documented database schema and common derived metrics."""

    async def execute() -> str:
        return schema_text()

    return execute

