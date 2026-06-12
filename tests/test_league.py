import json
from random import Random

from baseball_bench.data import build_database
from baseball_bench.tracks.league.engine import (
    HitterProfile,
    PitcherProfile,
    TeamRoster,
    _bullpen_candidates,
    _plate_appearance_outcome,
    _platoon_factor,
    _platoon_lineup,
)
from baseball_bench.tracks.league.plan import roster_from_plan
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


def test_league_writes_progress_snapshot(tmp_path):
    database_path = tmp_path / "baseball.duckdb"
    build_database(database_path)
    output_dir = tmp_path / "results"

    run_league(
        ["rulebook", "aggressive"],
        games_per_matchup=1,
        seed=17,
        database_path=database_path,
        output_dir=output_dir,
        write_latest=False,
    )

    progress_files = list(output_dir.glob("controlled-league-progress-*.json"))
    assert len(progress_files) == 1
    progress = json.loads(progress_files[0].read_text())
    assert progress["kind"] == "controlled_league_progress"
    assert progress["status"] == "complete"
    assert progress["completed_games"] == 2
    assert progress["remaining_games"] == 0
    assert progress["current_game"] is None
    assert progress["last_completed_game"]["game_number"] == 2


def test_league_games_caps_sampled_schedule(tmp_path):
    database_path = tmp_path / "baseball.duckdb"
    build_database(database_path)
    output_dir = tmp_path / "results"

    summary = run_league(
        ["rulebook", "aggressive", "conservative"],
        games_per_matchup=3,
        league_games=5,
        seed=17,
        database_path=database_path,
        output_dir=output_dir,
        write_latest=False,
    )

    progress = json.loads(next(output_dir.glob("controlled-league-progress-*.json")).read_text())
    assert summary["schedule_mode"] == "sampled"
    assert summary["game_count"] == 5
    assert progress["completed_games"] == 5
    assert progress["total_games"] == 5


def test_matchup_model_uses_platoon_park_fatigue_and_pitcher_profile():
    hitter = HitterProfile("h1", "Power Bat", 600, 70, 120, 105, 35, 3, 38, bats="L", position="RF")
    pitcher = PitcherProfile("p1", "Tired Righty", 180.0, 60, 170, 170, 30, "SP", throws="R")
    power_pitcher = PitcherProfile("p2", "Power Righty", 180.0, 35, 240, 125, 15, "SP", throws="R")

    low_pressure_homers = sum(
        _plate_appearance_outcome(hitter, pitcher, Random(seed), fatigue=0, times_through_order=1, park_factor=0.92)
        == "home_run"
        for seed in range(1000)
    )
    high_pressure_homers = sum(
        _plate_appearance_outcome(hitter, pitcher, Random(seed), fatigue=34, times_through_order=4, park_factor=1.16)
        == "home_run"
        for seed in range(1000)
    )
    tired_strikeouts = sum(
        _plate_appearance_outcome(hitter, pitcher, Random(seed), fatigue=34, times_through_order=4, park_factor=1.0)
        == "strikeout"
        for seed in range(1000)
    )
    power_strikeouts = sum(
        _plate_appearance_outcome(hitter, power_pitcher, Random(seed), fatigue=0, times_through_order=1, park_factor=1.0)
        == "strikeout"
        for seed in range(1000)
    )

    assert _platoon_factor(hitter, pitcher) > _platoon_factor(hitter, PitcherProfile("p3", "Lefty", 180.0, 60, 170, 170, 30, "SP", throws="L"))
    assert high_pressure_homers > low_pressure_homers
    assert power_strikeouts > tired_strikeouts


def test_platoon_lineup_uses_ordered_bench_for_matchup_swaps():
    starter = PitcherProfile("p1", "Right Starter", 180.0, 50, 180, 160, 20, "SP", throws="R")
    weak_same_side = HitterProfile("h1", "Same Side", 500, 35, 150, 80, 12, 1, 8, bats="R", position="RF")
    lineup = [weak_same_side] + [
        HitterProfile(f"h{index}", f"Hitter {index}", 500, 45, 100, 95, 20, 2, 18, bats="R", position="DH")
        for index in range(2, 10)
    ]
    bench = [HitterProfile("hb", "Left Bench", 260, 35, 45, 60, 18, 1, 14, bats="L", position="OF")]
    roster = TeamRoster(lineup=lineup, bench=bench, starter=starter, bullpen=[])

    adjusted_lineup, adjusted_bench = _platoon_lineup(roster, starter)

    assert adjusted_lineup[0].player_id == "hb"
    assert adjusted_bench[-1].player_id == "h1"


def test_manager_plan_preserves_bullpen_roles_and_roster_traits(tmp_path):
    database_path = tmp_path / "baseball.duckdb"
    build_database(database_path)
    output_dir = tmp_path / "results"
    run_league(
        ["rulebook", "aggressive"],
        games_per_matchup=2,
        league_games=4,
        seed=17,
        database_path=database_path,
        output_dir=output_dir,
        write_latest=False,
    )
    summary_path = next(path for path in output_dir.glob("controlled-league-*.json") if "progress" not in path.name)
    summary = json.loads(summary_path.read_text())
    game = summary["games"][0]

    assert "home_bullpen_min_rest" in game
    assert "away_bullpen_min_rest" in game
    assert min(item["home_bullpen_min_rest"] for item in summary["games"]) < 100

    roster = TeamRoster(
        lineup=[
            HitterProfile(f"h{index}", f"Hitter {index}", 500, 50, 90, 100, 20, 2, 18, bats="L" if index == 1 else "R", position="DH")
            for index in range(1, 10)
        ],
        bench=[HitterProfile("h10", "Bench Switch", 300, 35, 55, 70, 14, 1, 9, bats="S", position="UT")],
        starter=PitcherProfile("sp", "Starter", 180.0, 50, 180, 160, 20, "SP", throws="L"),
        bullpen=[
            PitcherProfile("rp1", "Closer", 65.0, 20, 85, 58, 6, "RP", throws="R"),
            PitcherProfile("rp2", "Long", 65.0, 25, 75, 55, 7, "RP", throws="L"),
        ],
    )
    planned = roster_from_plan(
        roster,
        {
            "lineup": [hitter.player_id for hitter in roster.lineup],
            "bench": ["h10"],
            "starter": "sp",
            "bullpen": ["rp2", "rp1"],
            "bullpen_roles": {"rp2": "closer", "rp1": "long"},
        },
    )

    assert planned.lineup[0].bats == "L"
    assert planned.bench[0].position == "UT"
    assert planned.starter.throws == "L"
    assert planned.bullpen_roles == {"rp2": "closer", "rp1": "long"}


def test_bullpen_candidates_respect_roles_and_rest():
    closer = PitcherProfile("rp1", "Closer", 65.0, 20, 85, 58, 6, "RP")
    setup = PitcherProfile("rp2", "Setup", 65.0, 20, 82, 58, 6, "RP")
    long = PitcherProfile("rp3", "Long", 70.0, 22, 70, 60, 7, "RP")

    high_leverage = _bullpen_candidates(
        [long, setup, closer],
        {"rp1": "closer", "rp2": "setup", "rp3": "long"},
        {"rp1": 100, "rp2": 100, "rp3": 100},
        leverage_index=1.8,
    )
    tired_high_leverage = _bullpen_candidates(
        [long, setup, closer],
        {"rp1": "closer", "rp2": "setup", "rp3": "long"},
        {"rp1": 10, "rp2": 100, "rp3": 100},
        leverage_index=1.8,
    )

    assert high_leverage[0].player_id == "rp1"
    assert tired_high_leverage[0].player_id == "rp2"
