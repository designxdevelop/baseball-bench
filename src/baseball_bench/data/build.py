from __future__ import annotations

import importlib.resources
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from baseball_bench.paths import ARTIFACTS_DIR, DATA_CHECKSUM_PATH, DATA_DB_PATH
from baseball_bench.utils import stable_hash, write_json


@dataclass(frozen=True)
class BuildResult:
    database_path: Path
    checksum: str
    seed_hash: str
    table_counts: dict[str, int]


def load_seed_data() -> dict[str, Any]:
    payload = importlib.resources.files("baseball_bench.data.seed").joinpath(
        "seed_data.json"
    )
    return __import__("json").loads(payload.read_text())


def _create_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        create table teams (
          team_id varchar primary key,
          city varchar not null,
          nickname varchar not null,
          league varchar not null,
          venue_name varchar not null
        );

        create table players (
          player_id varchar primary key,
          first_name varchar not null,
          last_name varchar not null,
          bats varchar not null,
          throws varchar not null,
          primary_position varchar not null
        );

        create table batting (
          season integer not null,
          team_id varchar not null,
          player_id varchar not null,
          games integer not null,
          plate_appearances integer not null,
          at_bats integer not null,
          hits integer not null,
          doubles integer not null,
          triples integer not null,
          home_runs integer not null,
          walks integer not null,
          strikeouts integer not null,
          hit_by_pitch integer not null,
          sacrifice_flies integer not null
        );

        create table pitching (
          season integer not null,
          team_id varchar not null,
          player_id varchar not null,
          games integer not null,
          games_started integer not null,
          innings_pitched double not null,
          hits_allowed integer not null,
          earned_runs integer not null,
          home_runs_allowed integer not null,
          walks integer not null,
          strikeouts integer not null
        );

        create table games (
          game_id varchar primary key,
          season integer not null,
          game_date date not null,
          home_team_id varchar not null,
          away_team_id varchar not null,
          home_score integer not null,
          away_score integer not null
        );

        create table win_probabilities (
          inning integer not null,
          half varchar not null,
          outs integer not null,
          runner_state varchar not null,
          score_diff integer not null,
          home_win_prob double not null
        );
        """
    )


def _insert_rows(
    conn: duckdb.DuckDBPyConnection, table: str, rows: list[dict[str, Any]]
) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    placeholders = ", ".join(["?"] * len(columns))
    conn.executemany(
        f"insert into {table} ({', '.join(columns)}) values ({placeholders})",
        [tuple(row[column] for column in columns) for row in rows],
    )


def build_database(database_path: Path | None = None) -> BuildResult:
    database_path = database_path or DATA_DB_PATH
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()

    seed_data = load_seed_data()
    with duckdb.connect(str(database_path)) as conn:
        _create_tables(conn)
        for table in ("teams", "players", "batting", "pitching", "games", "win_probabilities"):
            _insert_rows(conn, table, seed_data[table])

    checksum = __import__("hashlib").sha256(database_path.read_bytes()).hexdigest()
    seed_hash = stable_hash(seed_data)
    table_counts = {table: len(seed_data[table]) for table in seed_data if isinstance(seed_data[table], list)}
    write_json(
        DATA_CHECKSUM_PATH,
        {
            "database_path": str(database_path.relative_to(database_path.parent.parent)),
            "checksum": checksum,
            "seed_hash": seed_hash,
            "table_counts": table_counts,
        },
    )
    return BuildResult(
        database_path=database_path,
        checksum=checksum,
        seed_hash=seed_hash,
        table_counts=table_counts,
    )


def main() -> int:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    result = build_database()
    print(f"built {result.database_path}")
    print(f"checksum {result.checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
