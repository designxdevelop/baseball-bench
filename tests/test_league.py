from baseball_bench.data import build_database
from baseball_bench.tracks.league import run_league


def test_league_is_deterministic_for_same_seed(tmp_path):
    database_path = tmp_path / "baseball.duckdb"
    build_database(database_path)
    output_dir = tmp_path / "results"
    first = run_league(
        ["rulebook", "aggressive"],
        games_per_matchup=1,
        seed=17,
        database_path=database_path,
        output_dir=output_dir,
        write_latest=False,
    )
    second = run_league(
        ["rulebook", "aggressive"],
        games_per_matchup=1,
        seed=17,
        database_path=database_path,
        output_dir=output_dir,
        write_latest=False,
    )

    assert first["standings"] == second["standings"]
    assert first["games"] == second["games"]
    assert first["head_to_head"] == second["head_to_head"]
    assert first["head_to_head"]["rulebook"]["aggressive"]["games"] == 2
