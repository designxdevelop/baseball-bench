from __future__ import annotations

from argparse import Namespace

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
    ]


def test_run_bench_batches_models_into_single_inspect_call(monkeypatch):
    inspect_calls: list[tuple[str, list[str]]] = []
    summarized: list[tuple[str, str]] = []
    league_calls: list[list[str]] = []

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(cli, "build_database", lambda: None)
    monkeypatch.setattr(cli, "generate_questions", lambda: None)
    monkeypatch.setattr(cli, "analysis_task", lambda: "analysis-task")
    monkeypatch.setattr(cli, "decisions_task", lambda: "decisions-task")
    monkeypatch.setattr(cli, "build_site", lambda: None)

    def fake_inspect_eval(task, *, model, log_dir, log_format):
        inspect_calls.append((task, list(model)))
        if task == "analysis-task":
            return ["analysis-log-1", "analysis-log-2"]
        return ["decisions-log-1", "decisions-log-2"]

    monkeypatch.setattr(cli, "inspect_eval", fake_inspect_eval)
    monkeypatch.setattr(cli, "_summarize_eval_log", lambda log, kind: summarized.append((kind, log)))
    monkeypatch.setattr(
        cli,
        "run_league",
        lambda models, games_per_matchup, seed: league_calls.append(list(models)),
    )

    args = Namespace(
        baseline=False,
        model=["openai/gpt-5"],
        models="anthropic/claude-4.1",
        games=2,
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
        ("analysis", "analysis-log-1"),
        ("analysis", "analysis-log-2"),
        ("decisions", "decisions-log-1"),
        ("decisions", "decisions-log-2"),
    ]
    assert league_calls == [[
        "openrouter/openai/gpt-5",
        "openrouter/anthropic/claude-4.1",
        "rulebook",
    ]]


def test_run_cost_estimate_uses_estimator_and_prints_summary(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "estimate_costs",
        lambda model_names, games_per_matchup, allow_network: {
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
        Namespace(model=["openai/gpt-5.5"], models=None, games=3, offline=True)
    )

    assert result == 0
    assert written[0]["total_cost_usd"] == 12.34
    stdout = capsys.readouterr().out
    assert "pricing_source=test_source" in stdout
    assert "total_cost_usd=12.3400" in stdout
