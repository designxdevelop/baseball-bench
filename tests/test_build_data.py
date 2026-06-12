from baseball_bench.data import build_database, connect_read_only


def test_build_database_creates_expected_tables(tmp_path):
    database_path = tmp_path / "baseball.duckdb"
    result = build_database(database_path)

    assert database_path.exists()
    assert len(result.checksum) == 64
    assert result.table_counts["batting"] >= 20

    with connect_read_only(database_path) as conn:
        batting_rows = conn.execute("select count(*) from batting").fetchone()[0]
        team_rows = conn.execute("select count(*) from teams").fetchone()[0]

    assert batting_rows == result.table_counts["batting"]
    assert team_rows == 30
