from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from baseball_bench import cli


def test_resolve_model_names_supports_repeated_and_comma_args():
    args = Namespace(
        model=["openai/gpt-5", "anthropic/claude-4.1"],
        models="openai/gpt-5,google/gemini-2.5-pro",
    )

    assert cli._resolve_model_names(args) == [
        "openai/gpt-5",
        "anthropic/claude-4.1",
        "google/gemini-2.5-pro",
    ]


def test_resolve_eval_models_defaults_to_curated_openrouter_pack():
    args = Namespace(model=None, models=None)

    assert cli._resolve_eval_models(args) == [
        "openrouter/anthropic/claude-fable-5",
        "openrouter/anthropic/claude-opus-4.8",
        "openrouter/openai/gpt-5.5",
        "openrouter/deepseek/deepseek-v4-pro",
        "openrouter/nvidia/nemotron-3-ultra-550b-a55b",
        "openrouter/qwen/qwen3.6-35b-a3b",
        "openrouter/google/gemini-3.1-pro-preview",
    ]


def test_run_bench_batches_models_into_single_inspect_call(monkeypatch):
    inspect_calls: list[tuple[str, list[str]]] = []
    summarized: list[tuple[str, str, Path]] = []
    league_calls: list[tuple[list[str], int | None, str]] = []
    snapshots: list[Path] = []
    manifest_updates: list[tuple[str, list[str]]] = []
    finalized: list[tuple[Path, list[str]]] = []

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(cli, "build_database", lambda: None)
    monkeypatch.setattr(cli, "generate_questions", lambda: None)
    monkeypatch.setattr(cli, "analysis_task", lambda: "analysis-task")
    monkeypatch.setattr(cli, "decisions_task", lambda: "decisions-task")
    monkeypatch.setattr(cli, "build_site", lambda: None)
    monkeypatch.setattr(
        cli,
        "_create_run_manifest",
        lambda **kwargs: (Path("/tmp/test-run"), {"label": "test"}),
    )
    monkeypatch.setattr(cli, "_build_run_snapshot", lambda run_dir: snapshots.append(run_dir))
    monkeypatch.setattr(
        cli,
        "_update_run_manifest",
        lambda run_dir, manifest, active_track, tracks_completed: manifest_updates.append(
            (active_track, tracks_completed)
        ),
    )
    monkeypatch.setattr(
        cli,
        "_finalize_run_manifest",
        lambda run_dir, manifest, tracks_completed: finalized.append((run_dir, tracks_completed)),
    )

    def fake_inspect_eval(task, *, model, log_dir, log_format):
        inspect_calls.append((task, list(model)))
        if task == "analysis-task":
            return ["analysis-log-1", "analysis-log-2"]
        return ["decisions-log-1", "decisions-log-2"]

    monkeypatch.setattr(cli, "inspect_eval", fake_inspect_eval)
    monkeypatch.setattr(
        cli,
        "_summarize_eval_log",
        lambda log, kind, output_dir: summarized.append((kind, log, output_dir)),
    )
    monkeypatch.setattr(cli, "run_gm_rosters", lambda models, output_dir, allow_model_call=True: [])
    monkeypatch.setattr(cli, "build_controlled_roster", lambda: "controlled-roster")
    monkeypatch.setattr(
        cli,
        "run_manager_plans",
        lambda models, roster, output_dir, allow_model_call=True: {
            model: f"planned-{model}" for model in models
        },
    )
    monkeypatch.setattr(
        cli,
        "run_league",
        lambda models, games_per_matchup, seed, league_games, league_kind, rosters_by_model, output_dir: league_calls.append(
            (list(models), league_games, league_kind)
        ),
    )

    args = Namespace(
        baseline=False,
        model=["openai/gpt-5"],
        models="anthropic/claude-4.1",
        games=2,
        league_games=12,
        full_league=False,
        seed=7,
    )

    result = cli.run_bench(args)

    assert result == 0
    assert inspect_calls == [
        (
            "analysis-task",
            [
                "openrouter/openai/gpt-5",
                "openrouter/anthropic/claude-4.1",
            ],
        ),
        (
            "decisions-task",
            [
                "openrouter/openai/gpt-5",
                "openrouter/anthropic/claude-4.1",
            ],
        ),
    ]
    assert summarized == [
        ("analysis", "analysis-log-1", Path("/tmp/test-run")),
        ("analysis", "analysis-log-2", Path("/tmp/test-run")),
        ("decisions", "decisions-log-1", Path("/tmp/test-run")),
        ("decisions", "decisions-log-2", Path("/tmp/test-run")),
    ]
    assert league_calls == [(
        [
            "openrouter/openai/gpt-5",
            "openrouter/anthropic/claude-4.1",
            "rulebook",
        ],
        12,
        "controlled_league",
    )]
    assert manifest_updates == [
        ("analysis", []),
        ("decisions", ["analysis"]),
        ("gm", ["analysis", "decisions"]),
        ("manager_plan", ["analysis", "decisions", "gm"]),
        ("controlled_league", ["analysis", "decisions", "gm", "manager_plan"]),
    ]
    assert snapshots == [Path("/tmp/test-run")]
    assert finalized == [(Path("/tmp/test-run"), ["analysis", "decisions", "gm", "controlled_league"])]


def test_run_cost_estimate_uses_estimator_and_prints_summary(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "estimate_costs",
        lambda model_names, games_per_matchup, league_games, allow_network: {
            "pricing_source": "test_source",
            "total_cost_usd": 12.34,
            "per_model": [
                {
                    "model": "openrouter/openai/gpt-5.5",
                    "total_cost_usd": 10.0,
                    "analysis_cost_usd": 1.0,
                    "decisions_cost_usd": 2.0,
                    "league_cost_usd": 7.0,
                }
            ],
        },
    )
    written: list[dict[str, object]] = []
    monkeypatch.setattr(cli, "write_cost_estimate", lambda summary: written.append(summary))

    result = cli.run_cost_estimate(
        Namespace(model=["openai/gpt-5.5"], models=None, games=3, league_games=12, offline=True)
    )

    assert result == 0
    assert written[0]["total_cost_usd"] == 12.34
    stdout = capsys.readouterr().out
    assert "pricing_source=test_source" in stdout
    assert "total_cost_usd=12.3400" in stdout
