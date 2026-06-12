from __future__ import annotations

from pathlib import Path

import duckdb

from baseball_bench.paths import DATA_DB_PATH


def connect_read_only(database_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    path = database_path or DATA_DB_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Database not found at {path}. Run `uv run scripts/build-data` first."
        )
    return duckdb.connect(str(path), read_only=True)


def _validate_select_query(sql: str) -> str:
    cleaned = sql.strip().rstrip(";")
    lowered = cleaned.lower()
    if not lowered.startswith(("select", "with")):
        raise ValueError("Only SELECT and CTE queries are allowed.")
    blocked = (" insert ", " update ", " delete ", " create ", " drop ", " alter ", " copy ")
    wrapped = f" {lowered} "
    if any(token in wrapped for token in blocked):
        raise ValueError("Mutating SQL is not allowed.")
    return cleaned


def execute_read_only_query(
    sql: str,
    database_path: Path | None = None,
    row_limit: int = 50,
) -> dict[str, object]:
    cleaned = _validate_select_query(sql)
    query = f"select * from ({cleaned}) as bench_query limit {int(row_limit)}"
    with connect_read_only(database_path) as conn:
        cursor = conn.execute(query)
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchall()
    return {
        "columns": columns,
        "rows": [list(row) for row in rows],
        "row_count": len(rows),
        "applied_limit": row_limit,
    }


def schema_text() -> str:
    return (Path(__file__).resolve().parent / "schema.md").read_text()

