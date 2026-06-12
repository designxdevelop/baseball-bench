from __future__ import annotations

import json

from baseball_bench.leaderboard import site


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def test_build_site_renders_english_latest_view_and_history(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    run_dir = results_dir / "runs" / "20260611T120000-000001"
    run_dir.mkdir(parents=True)

    analysis_samples = [{"score": 1.0}] * 3 + [{"score": 0.0}] * 7
    _write_json(
        results_dir / "analysis-openrouter-openai-gpt-5.5.json",
        {
            "kind": "analysis",
            "model": "openrouter/openai/gpt-5.5",
            "sample_count": 10,
            "overall_accuracy": 0.3,
            "samples": analysis_samples,
        },
    )
    _write_json(
        results_dir / "decisions-openrouter-openai-gpt-5.5.json",
        {
            "kind": "decisions",
            "model": "openrouter/openai/gpt-5.5",
            "sample_count": 8,
            "mean_wp_delta": 0.0,
            "near_optimal_rate": 1.0,
            "samples": [{"metadata": {"near_optimal": True}} for _ in range(8)],
        },
    )
    _write_json(
        run_dir / "manifest.json",
        {
            "run_id": "20260611T120000-000001",
            "label": "OpenRouter benchmark",
            "started_at": "2026-06-11T12:00:00-06:00",
            "status": "complete",
            "models": ["openrouter/openai/gpt-5.5"],
            "tracks_completed": ["analysis", "decisions", "league"],
        },
    )

    leaderboard_path = results_dir / "leaderboard.json"
    site_dir = results_dir / "site"
    site_index_path = site_dir / "index.html"

    monkeypatch.setattr(site, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(site, "RUNS_DIR", results_dir / "runs")

    leaderboard = site.build_site(
        results_dir=results_dir,
        leaderboard_json_path=leaderboard_path,
        site_dir=site_dir,
        site_index_path=site_index_path,
    )

    assert leaderboard["models"][0]["display_name"] == "openai/gpt-5.5"
    assert leaderboard["models"][0]["tracks"]["analysis"]["summary"] == (
        "Answered 3 of 10 baseball research questions correctly."
    )
    assert leaderboard["models"][0]["tracks"]["league"]["summary"] == (
        "Controlled manager league play has not run yet for this model."
    )
    assert leaderboard["models"][0]["overall_read"] == "Strong start"
    assert leaderboard["run_history"][0]["label"] == "OpenRouter benchmark"

    html = site_index_path.read_text()
    assert "AI manager league" in html
    assert "Run History" in html
    assert "snapshot" in html
    assert "../runs/20260611T120000-000001/site/index.html" in html


def test_build_site_renders_public_model_league_matchups(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    _write_json(
        results_dir / "league-openrouter-openai-gpt-5.5-openrouter-anthropic-claude-fable-5-rulebook.json",
        {
            "kind": "league",
            "models": [
                "openrouter/openai/gpt-5.5",
                "openrouter/anthropic/claude-fable-5",
                "rulebook",
            ],
            "games_per_matchup": 1,
            "game_count": 6,
            "average_decisions_per_game": 12,
            "standings": [
                {
                    "model": "openrouter/openai/gpt-5.5",
                    "wins": 3,
                    "losses": 1,
                    "run_diff": 9,
                    "elo": 1518.2,
                    "win_pct": 0.75,
                },
                {
                    "model": "rulebook",
                    "wins": 2,
                    "losses": 2,
                    "run_diff": 0,
                    "elo": 1500,
                    "win_pct": 0.5,
                },
                {
                    "model": "openrouter/anthropic/claude-fable-5",
                    "wins": 1,
                    "losses": 3,
                    "run_diff": -9,
                    "elo": 1481.8,
                    "win_pct": 0.25,
                },
            ],
            "head_to_head": {
                "openrouter/openai/gpt-5.5": {
                    "openrouter/anthropic/claude-fable-5": {
                        "wins": 2,
                        "losses": 0,
                        "runs_for": 11,
                        "runs_against": 5,
                        "games": 2,
                        "win_pct": 1.0,
                        "run_diff": 6,
                    }
                },
                "openrouter/anthropic/claude-fable-5": {
                    "openrouter/openai/gpt-5.5": {
                        "wins": 0,
                        "losses": 2,
                        "runs_for": 5,
                        "runs_against": 11,
                        "games": 2,
                        "win_pct": 0.0,
                        "run_diff": -6,
                    }
                },
            },
            "games": [],
        },
    )

    leaderboard = site.build_site(
        results_dir=results_dir,
        leaderboard_json_path=results_dir / "leaderboard.json",
        site_dir=results_dir / "site",
        site_index_path=results_dir / "site" / "index.html",
    )

    assert [entry["display_name"] for entry in leaderboard["models"]] == [
        "openai/gpt-5.5",
        "anthropic/claude-fable-5",
    ]
    assert leaderboard["models"][0]["tracks"]["league"]["summary"] == (
        "Finished 1st of 2 managers and went 3-1 across 6 league games."
    )
    html = (results_dir / "site" / "index.html").read_text()
    assert "Head-to-head model standings" in html
    assert "2-0 <span>+6</span>" in html
    assert "rulebook" not in html


def test_build_site_reads_baseline_decision_samples_without_metadata(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    _write_json(
        results_dir / "decisions-wp-baseline.json",
        {
            "kind": "decisions",
            "model": "wp-baseline",
            "sample_count": 2,
            "mean_wp_delta": 0.0,
            "near_optimal_rate": 1.0,
            "samples": [
                {"near_optimal": True},
                {"near_optimal": True},
            ],
        },
    )

    leaderboard = site.build_site(
        results_dir=results_dir,
        leaderboard_json_path=results_dir / "leaderboard.json",
        site_dir=results_dir / "site",
        site_index_path=results_dir / "site" / "index.html",
    )

    assert leaderboard["models"] == []
    assert leaderboard["internal_models"][0]["tracks"]["decisions"]["summary"] == (
        "Chose the best or near-best move in 2 of 2 late-game situations."
    )
