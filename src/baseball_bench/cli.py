from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from inspect_ai import eval as inspect_eval

from baseball_bench.costs import (
    DEFAULT_OPENROUTER_MODELS,
    estimate_costs,
    normalize_openrouter_model_name,
    require_openrouter_api_key,
    write_cost_estimate,
)
from baseball_bench.data.build import build_database
from baseball_bench.leaderboard import build_site
from baseball_bench.paths import LOGS_DIR, RESULTS_DIR, RUNS_DIR
from baseball_bench.tracks.analysis import (
    analysis_task,
    generate_questions,
    run_sql_baseline,
    write_analysis_summary,
)
from baseball_bench.tracks.decisions import (
    decisions_task,
    run_wp_baseline,
    write_decisions_summary,
)
from baseball_bench.tracks.gm import run_gm_roster, run_gm_rosters, team_roster_from_build
from baseball_bench.tracks.league import build_controlled_roster, run_league, run_manager_plans
from baseball_bench.tracks.league.eval_modes import LeagueEvalConfig, resolve_league_eval_config
from baseball_bench.utils import write_json


def _resolve_model_names(args: Any) -> list[str]:
    model_names: list[str] = []

    if getattr(args, "model", None):
        for model_name in args.model:
            model_name = model_name.strip()
            if model_name:
                model_names.append(model_name)

    if getattr(args, "models", None):
        model_names.extend(
            [item.strip() for item in args.models.split(",") if item.strip()]
        )

    return list(dict.fromkeys(model_names))


def _resolve_eval_models(args: Any) -> list[str]:
    model_names = _resolve_model_names(args)
    if not model_names:
        model_names = DEFAULT_OPENROUTER_MODELS
    return [normalize_openrouter_model_name(model) for model in model_names]


def _create_run_manifest(
    *,
    label: str,
    models: list[str],
    games: int,
    seed: int,
    baseline: bool,
    eval_config: LeagueEvalConfig | None = None,
) -> tuple[Path, dict[str, object]]:
    started_at = datetime.now().astimezone()
    run_id = started_at.strftime("%Y%m%dT%H%M%S-%f")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id,
        "label": label,
        "started_at": started_at.isoformat(timespec="seconds"),
        "status": "running",
        "baseline": baseline,
        "models": models,
        "games_per_matchup": games,
        "seed": seed,
        "tracks_completed": [],
    }
    if eval_config is not None:
        manifest["evaluation_mode"] = eval_config.evaluation_mode
        manifest["eval_config"] = eval_config.as_metadata()
    write_json(run_dir / "manifest.json", manifest)
    return run_dir, manifest


def _finalize_run_manifest(
    run_dir: Path,
    manifest: dict[str, object],
    *,
    tracks_completed: list[str],
) -> None:
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest.pop("active_track", None)
    manifest["tracks_completed"] = tracks_completed
    manifest["artifacts"] = {
        "leaderboard_json": "leaderboard.json",
        "site_index": "site/index.html",
        "result_files": sorted(
            path.name for path in run_dir.glob("*.json") if path.name != "manifest.json"
        ),
    }
    write_json(run_dir / "manifest.json", manifest)


def _update_run_manifest(
    run_dir: Path,
    manifest: dict[str, object],
    *,
    active_track: str,
    tracks_completed: list[str],
) -> None:
    completed = ", ".join(tracks_completed) if tracks_completed else "none"
    print(
        f"[baseball-bench] active_track={active_track} completed={completed} run_dir={run_dir}",
        flush=True,
    )
    manifest["status"] = "running"
    manifest["active_track"] = active_track
    manifest["tracks_completed"] = tracks_completed
    manifest["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_json(run_dir / "manifest.json", manifest)


def _resolve_eval_config_from_args(args: Any) -> LeagueEvalConfig:
    return resolve_league_eval_config(
        mode=getattr(args, "mode", None),
        league_games=None if getattr(args, "full_league", False) else getattr(args, "league_games", None),
        live_call_start_inning=getattr(args, "live_call_start_inning", None),
        live_call_max_score_gap=getattr(args, "live_call_max_score_gap", None),
        max_live_calls_per_team=getattr(args, "max_live_calls_per_team", None),
        enable_open_league=getattr(args, "enable_open_league", None),
    )


def _league_games_arg(args: Any, eval_config: LeagueEvalConfig) -> int | None:
    return None if getattr(args, "full_league", False) else eval_config.league_games


def _build_run_snapshot(run_dir: Path) -> None:
    build_site(
        results_dir=run_dir,
        leaderboard_json_path=run_dir / "leaderboard.json",
        site_dir=run_dir / "site",
        site_index_path=run_dir / "site" / "index.html",
    )


def _summarize_eval_log(log, kind: str, output_dir: Path = RESULTS_DIR) -> dict[str, object]:
    scorer = log.results.scores[0] if log.results and log.results.scores else None
    samples = []
    if log.samples:
        for sample in log.samples:
            score = None
            if sample.scores:
                score = next(iter(sample.scores.values()))
            samples.append(
                {
                    "id": sample.id,
                    "target": sample.target,
                    "output": sample.output.completion,
                    "score": score.value if score else None,
                    "metadata": score.metadata if score else {},
                }
            )
    if kind == "analysis":
        summary = {
            "kind": "analysis",
            "model": log.eval.model,
            "sample_count": len(samples),
            "overall_accuracy": scorer.metrics["mean"].value if scorer and "mean" in scorer.metrics else 0.0,
            "samples": samples,
        }
    else:
        deltas = [float(sample["metadata"].get("wp_delta", 0.0)) for sample in samples]
        summary = {
            "kind": "decisions",
            "model": log.eval.model,
            "sample_count": len(samples),
            "mean_wp_delta": mean(deltas) if deltas else 0.0,
            "near_optimal_rate": mean(1.0 if sample["metadata"].get("near_optimal") else 0.0 for sample in samples) if samples else 0.0,
            "samples": samples,
        }
    if kind == "analysis":
        write_analysis_summary(summary, output_dir=output_dir)
    else:
        write_decisions_summary(summary, output_dir=output_dir)
    return summary


def run_bench(args) -> int:
    build_database()
    generate_questions()
    eval_config = _resolve_eval_config_from_args(args)
    if args.baseline:
        baseline_models = ["sql-baseline", "wp-baseline", "rulebook", "aggressive", "conservative"]
        run_dir, manifest = _create_run_manifest(
            label="Baseline benchmark",
            models=baseline_models,
            games=args.games,
            seed=args.seed,
            baseline=True,
            eval_config=eval_config,
        )
        _update_run_manifest(run_dir, manifest, active_track="analysis", tracks_completed=[])
        analysis_summary = run_sql_baseline()
        write_analysis_summary(analysis_summary, output_dir=run_dir)
        _update_run_manifest(run_dir, manifest, active_track="decisions", tracks_completed=["analysis"])
        decisions_summary = run_wp_baseline()
        write_decisions_summary(decisions_summary, output_dir=run_dir)
        _update_run_manifest(run_dir, manifest, active_track="gm", tracks_completed=["analysis", "decisions"])
        gm_models = ["rulebook", "aggressive", "conservative"]
        run_gm_rosters(gm_models, output_dir=run_dir, allow_model_call=False)
        _update_run_manifest(run_dir, manifest, active_track="manager_plan", tracks_completed=["analysis", "decisions", "gm"])
        controlled_roster = build_controlled_roster()
        planned_rosters = run_manager_plans(
            gm_models,
            controlled_roster,
            output_dir=run_dir,
            allow_model_call=False,
        )
        _update_run_manifest(run_dir, manifest, active_track="controlled_league", tracks_completed=["analysis", "decisions", "gm", "manager_plan"])
        run_league(
            gm_models,
            games_per_matchup=args.games,
            seed=args.seed,
            league_games=_league_games_arg(args, eval_config),
            league_kind="controlled_league",
            rosters_by_model=planned_rosters,
            output_dir=run_dir,
            eval_config=eval_config,
        )
        _build_run_snapshot(run_dir)
        _finalize_run_manifest(
            run_dir,
            manifest,
            tracks_completed=["analysis", "decisions", "gm", "controlled_league"],
        )
        build_site()
        return 0

    model_names = _resolve_eval_models(args)
    require_openrouter_api_key(model_names)
    run_dir, manifest = _create_run_manifest(
        label="OpenRouter benchmark",
        models=model_names,
        games=args.games,
        seed=args.seed,
        baseline=False,
        eval_config=eval_config,
    )
    _update_run_manifest(run_dir, manifest, active_track="analysis", tracks_completed=[])

    analysis_logs = inspect_eval(
        analysis_task(),
        model=model_names,
        log_dir=str(LOGS_DIR / "analysis"),
        log_format="json",
    )
    for log in analysis_logs:
        _summarize_eval_log(log, "analysis", output_dir=run_dir)
    _update_run_manifest(run_dir, manifest, active_track="decisions", tracks_completed=["analysis"])
    decisions_logs = inspect_eval(
        decisions_task(),
        model=model_names,
        log_dir=str(LOGS_DIR / "decisions"),
        log_format="json",
    )
    for log in decisions_logs:
        _summarize_eval_log(log, "decisions", output_dir=run_dir)

    _update_run_manifest(run_dir, manifest, active_track="gm", tracks_completed=["analysis", "decisions"])
    gm_builds = run_gm_rosters(model_names, output_dir=run_dir)
    league_models = list(dict.fromkeys(model_names + ["rulebook"]))
    _update_run_manifest(run_dir, manifest, active_track="manager_plan", tracks_completed=["analysis", "decisions", "gm"])
    controlled_roster = build_controlled_roster()
    planned_rosters = run_manager_plans(league_models, controlled_roster, output_dir=run_dir)
    _update_run_manifest(run_dir, manifest, active_track="controlled_league", tracks_completed=["analysis", "decisions", "gm", "manager_plan"])
    run_league(
        league_models,
        games_per_matchup=args.games,
        seed=args.seed,
        league_games=_league_games_arg(args, eval_config),
        league_kind="controlled_league",
        rosters_by_model=planned_rosters,
        output_dir=run_dir,
        eval_config=eval_config,
    )
    tracks_completed = ["analysis", "decisions", "gm", "controlled_league"]
    if eval_config.enable_open_league:
        _update_run_manifest(run_dir, manifest, active_track="open_league", tracks_completed=tracks_completed)
        rosters_by_model = {
            str(build["model"]): team_roster_from_build(build)
            for build in gm_builds
            if build.get("roster", {}).get("validation", {}).get("valid")
        }
        run_league(
            model_names,
            games_per_matchup=args.games,
            seed=args.seed,
            league_games=_league_games_arg(args, eval_config),
            league_kind="open_league",
            rosters_by_model=rosters_by_model,
            output_dir=run_dir,
            eval_config=eval_config,
        )
        tracks_completed.append("open_league")
    _build_run_snapshot(run_dir)
    _finalize_run_manifest(
        run_dir,
        manifest,
        tracks_completed=tracks_completed,
    )
    build_site()
    return 0


def run_cost_estimate(args: Any) -> int:
    model_names = _resolve_eval_models(args)
    eval_config = _resolve_eval_config_from_args(args)
    summary = estimate_costs(
        model_names,
        games_per_matchup=args.games,
        league_games=_league_games_arg(args, eval_config),
        estimated_decisions_per_game=eval_config.max_live_calls_per_team * 2,
        allow_network=not getattr(args, "offline", False),
    )
    write_cost_estimate(summary)
    print(f"pricing_source={summary['pricing_source']}")
    print(f"total_cost_usd={summary['total_cost_usd']:.4f}")
    for item in summary["per_model"]:
        print(
            f"{item['model']} total=${item['total_cost_usd']:.4f} "
            f"(analysis=${item['analysis_cost_usd']:.4f}, "
            f"decisions=${item['decisions_cost_usd']:.4f}, "
            f"league=${item['league_cost_usd']:.4f})"
        )
    return 0


def _add_eval_mode_args(parser: argparse.ArgumentParser, *, include_open_league: bool = False) -> None:
    parser.add_argument("--mode", choices=["public-refresh", "deep-eval"], default="public-refresh")
    parser.add_argument("--live-call-start-inning", type=int)
    parser.add_argument("--live-call-max-score-gap", type=int)
    parser.add_argument("--max-live-calls-per-team", type=int)
    if include_open_league:
        parser.add_argument("--enable-open-league", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="baseball-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build-data")
    subparsers.add_parser("generate-questions")
    subparsers.add_parser("run-analysis-baseline")
    subparsers.add_parser("run-decisions-baseline")
    subparsers.add_parser("build-site")

    league_parser = subparsers.add_parser("run-league")
    league_parser.add_argument("--model", action="append")
    league_parser.add_argument("--models")
    league_parser.add_argument("--games", type=int, default=6)
    league_parser.add_argument("--league-games", type=int)
    league_parser.add_argument("--full-league", action="store_true")
    league_parser.add_argument("--seed", type=int, default=7)
    _add_eval_mode_args(league_parser)

    gm_parser = subparsers.add_parser("run-gm")
    gm_parser.add_argument("--model", action="append")
    gm_parser.add_argument("--models")
    gm_parser.add_argument("--offline", action="store_true")

    open_league_parser = subparsers.add_parser("run-open-league")
    open_league_parser.add_argument("--model", action="append")
    open_league_parser.add_argument("--models")
    open_league_parser.add_argument("--games", type=int, default=6)
    open_league_parser.add_argument("--league-games", type=int)
    open_league_parser.add_argument("--full-league", action="store_true")
    open_league_parser.add_argument("--seed", type=int, default=7)
    open_league_parser.add_argument("--offline-gm", action="store_true")
    _add_eval_mode_args(open_league_parser)

    estimate_parser = subparsers.add_parser("estimate-cost")
    estimate_parser.add_argument("--model", action="append")
    estimate_parser.add_argument("--models")
    estimate_parser.add_argument("--games", type=int, default=6)
    estimate_parser.add_argument("--league-games", type=int)
    estimate_parser.add_argument("--offline", action="store_true")
    _add_eval_mode_args(estimate_parser)

    bench_parser = subparsers.add_parser("run-bench")
    bench_parser.add_argument("--baseline", action="store_true")
    bench_parser.add_argument("--model", action="append")
    bench_parser.add_argument("--models")
    bench_parser.add_argument("--games", type=int, default=6)
    bench_parser.add_argument("--league-games", type=int)
    bench_parser.add_argument("--full-league", action="store_true")
    bench_parser.add_argument("--seed", type=int, default=7)
    _add_eval_mode_args(bench_parser, include_open_league=True)

    args = parser.parse_args(argv)
    if args.command == "build-data":
        build_database()
        return 0
    if args.command == "generate-questions":
        generate_questions()
        return 0
    if args.command == "run-analysis-baseline":
        run_sql_baseline()
        return 0
    if args.command == "run-decisions-baseline":
        run_wp_baseline()
        return 0
    if args.command == "run-league":
        model_names = _resolve_eval_models(args)
        if len(model_names) < 2:
            raise ValueError("League needs at least two resolved models.")
        require_openrouter_api_key(model_names)
        eval_config = _resolve_eval_config_from_args(args)
        run_league(
            model_names,
            games_per_matchup=args.games,
            seed=args.seed,
            league_games=_league_games_arg(args, eval_config),
            league_kind="controlled_league",
            eval_config=eval_config,
        )
        return 0
    if args.command == "run-gm":
        model_names = _resolve_eval_models(args)
        require_openrouter_api_key(model_names)
        run_gm_rosters(model_names, allow_model_call=not args.offline)
        return 0
    if args.command == "run-open-league":
        model_names = _resolve_eval_models(args)
        require_openrouter_api_key(model_names)
        builds = run_gm_rosters(model_names, allow_model_call=not args.offline_gm)
        rosters_by_model = {
            str(build["model"]): team_roster_from_build(build)
            for build in builds
            if build.get("roster", {}).get("validation", {}).get("valid")
        }
        eval_config = _resolve_eval_config_from_args(args)
        run_league(
            model_names,
            games_per_matchup=args.games,
            seed=args.seed,
            league_games=_league_games_arg(args, eval_config),
            league_kind="open_league",
            rosters_by_model=rosters_by_model,
            eval_config=eval_config,
        )
        return 0
    if args.command == "build-site":
        build_site()
        return 0
    if args.command == "estimate-cost":
        return run_cost_estimate(args)
    if args.command == "run-bench":
        return run_bench(args)
    raise AssertionError("unreachable")
