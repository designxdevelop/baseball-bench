import json
from random import Random

from baseball_bench.data import build_database
from baseball_bench.core import ActionType, BenchOption, BullpenOption, DecisionOption, GameState, ManagerDecision
from baseball_bench.tracks.league.eval_modes import LeagueEvalConfig
from baseball_bench.tracks.league.engine import (
    HitterProfile,
    PitcherProfile,
    TeamRoster,
    _bullpen_candidates,
    _decide_with_policy,
    _plate_appearance_outcome,
    _platoon_factor,
    _platoon_lineup,
)
from baseball_bench.tracks.league.plan import roster_from_plan
from baseball_bench.tracks.league import run_league
from baseball_bench.tracks.league.season import _league_progress_filename, _league_result_filename


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
    assert progress["last_completed_game"]["park_name"]


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
    assert "park_name" in game
    assert "park_factor" in game
    assert all(0 <= item["home_bullpen_min_rest"] <= 100 for item in summary["games"])

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


def test_league_uses_compact_filenames_for_large_model_lists():
    models = [f"openrouter/provider/super-long-model-name-{index}" for index in range(12)]
    progress_name = _league_progress_filename(models, "controlled_league")
    result_name = _league_result_filename(models, "controlled_league")

    assert len(progress_name) < 255
    assert len(result_name) < 255


def test_eval_policy_caps_external_live_calls():
    class CountingManager:
        name = "openrouter/test-model"

        def __init__(self):
            self.calls = 0

        def decide(self, state, options):
            self.calls += 1
            return ManagerDecision(action_id=options[-1].action_id, rationale="called")

    manager = CountingManager()
    state = GameState(
        inning=7,
        half="top",
        outs=0,
        home_team="home",
        away_team="away",
        batting_team="away",
        fielding_team="home",
        home_score=3,
        away_score=2,
        batter_name="Batter",
        pitcher_name="Pitcher",
    )
    options = [
        DecisionOption(action_id=ActionType.LET_HIT.value, action_type=ActionType.LET_HIT, label="Let hit"),
        DecisionOption(action_id=ActionType.STEAL.value, action_type=ActionType.STEAL, label="Steal"),
    ]

    public_decision, public_live = _decide_with_policy(
        manager,
        state,
        options,
        eval_config=LeagueEvalConfig(),
        live_call_counts={manager.name: 0},
        team_name=manager.name,
    )
    deep_decision, deep_live = _decide_with_policy(
        manager,
        state,
        options,
        eval_config=LeagueEvalConfig(
            evaluation_mode="deep-eval",
            live_call_start_inning=7,
            live_call_max_score_gap=3,
            max_live_calls_per_team=1,
        ),
        live_call_counts={manager.name: 0},
        team_name=manager.name,
    )

    assert public_live is False
    assert public_decision.action_id == ActionType.LET_HIT.value
    assert deep_live is True
    assert deep_decision.action_id == ActionType.STEAL.value
    assert manager.calls == 1


def test_manager_prompt_context_includes_baseball_decision_inputs():
    state = GameState(
        inning=9,
        half="bottom",
        outs=0,
        home_team="home",
        away_team="away",
        batting_team="home",
        fielding_team="away",
        home_score=2,
        away_score=3,
        batter_name="Charles Hicklen",
        batter_bats="R",
        batter_summary="2025 line approx AVG .280, OBP .340, SLG .470",
        pitcher_name="Cole Ragans",
        pitcher_throws="L",
        pitcher_role="SP",
        pitcher_summary="2025 rates K/9 10.2, BB/9 3.0, HR/9 0.9",
        pitcher_fatigue=27,
        times_through_order=4,
        park_name="Kauffman Stadium",
        park_factor=0.97,
        bench=[
            BenchOption(
                player_id="h1",
                name="Bench Bat",
                bats="L",
                position="OF",
                summary="2025 line approx AVG .260, OBP .360, SLG .500",
            )
        ],
        bullpen=[
            BullpenOption(
                player_id="p1",
                name="Closer",
                throws="R",
                role="closer",
                stamina=84,
                summary="2025 rates K/9 12.0, BB/9 2.4, HR/9 0.6",
            )
        ],
    )

    prompt_context = state.manager_prompt_context()

    assert "Score context:" in prompt_context
    assert "tying run at the plate" in prompt_context
    assert "Batter context: Charles Hicklen, bats R" in prompt_context
    assert "Pitcher context: Cole Ragans, throws L, role SP" in prompt_context
    assert "fatigue batters faced in game 27" in prompt_context
    assert "Park context: Kauffman Stadium" in prompt_context
    assert "Bullpen options:" in prompt_context
    assert "Bench options:" in prompt_context
