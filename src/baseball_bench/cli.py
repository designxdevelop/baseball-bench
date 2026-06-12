from __future__ import annotations

import argparse
from statistics import mean
from typing import Any

from inspect_ai import eval as inspect_eval

from baseball_bench.costs import (
    BUNDLED_MANAGERS,
    DEFAULT_OPENROUTER_MODELS,
    estimate_costs,
    normalize_openrouter_model_name,
    require_openrouter_api_key,
    write_cost_estimate,
)
from baseball_bench.data.build import build_database
from baseball_bench.leaderboard import build_site
from baseball_bench.paths import LOGS_DIR, RESULTS_DIR
from baseball_bench.tracks.analysis import analysis_task, generate_questions, run_sql_baseline
from baseball_bench.tracks.decisions import decisions_task, run_wp_baseline
from baseball_bench.tracks.league import run_league
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


def _summarize_eval_log(log, kind: str) -> dict[str, object]:
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
    write_json(RESULTS_DIR / f"{kind}-{log.eval.model.replace('/', '-')}.json", summary)
    return summary


def run_bench(args) -> int:
    build_database()
    generate_questions()
    if args.baseline:
        run_sql_baseline()
        run_wp_baseline()
        run_league(["rulebook", "aggressive", "conservative"], games_per_matchup=args.games, seed=args.seed)
        build_site()
        return 0

    model_names = _resolve_eval_models(args)
    require_openrouter_api_key(model_names)

    analysis_logs = inspect_eval(
        analysis_task(),
        model=model_names,
        log_dir=str(LOGS_DIR / "analysis"),
        log_format="json",
    )
    decisions_logs = inspect_eval(
        decisions_task(),
        model=model_names,
        log_dir=str(LOGS_DIR / "decisions"),
        log_format="json",
    )
    for log in analysis_logs:
        _summarize_eval_log(log, "analysis")
    for log in decisions_logs:
        _summarize_eval_log(log, "decisions")

    league_models = list(dict.fromkeys(model_names + ["rulebook"]))
    run_league(league_models, games_per_matchup=args.games, seed=args.seed)
    build_site()
    return 0


def run_cost_estimate(args: Any) -> int:
    model_names = _resolve_eval_models(args)
    summary = estimate_costs(
        model_names,
        games_per_matchup=args.games,
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
    league_parser.add_argument("--seed", type=int, default=7)

    estimate_parser = subparsers.add_parser("estimate-cost")
    estimate_parser.add_argument("--model", action="append")
    estimate_parser.add_argument("--models")
    estimate_parser.add_argument("--games", type=int, default=6)
    estimate_parser.add_argument("--offline", action="store_true")

    bench_parser = subparsers.add_parser("run-bench")
    bench_parser.add_argument("--baseline", action="store_true")
    bench_parser.add_argument("--model", action="append")
    bench_parser.add_argument("--models")
    bench_parser.add_argument("--games", type=int, default=6)
    bench_parser.add_argument("--seed", type=int, default=7)

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
        run_league(model_names, games_per_matchup=args.games, seed=args.seed)
        return 0
    if args.command == "build-site":
        build_site()
        return 0
    if args.command == "estimate-cost":
        return run_cost_estimate(args)
    if args.command == "run-bench":
        return run_bench(args)
    raise AssertionError("unreachable")
