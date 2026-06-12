from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from statistics import mean
from typing import Any

from baseball_bench.paths import RESULTS_DIR
from baseball_bench.utils import write_json

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
BUNDLED_MANAGERS = {"rulebook", "aggressive", "conservative"}
DEFAULT_OPENROUTER_MODELS = [
    "anthropic/claude-fable-5",
    "anthropic/claude-opus-4.8",
    "openai/gpt-5.5",
    "deepseek/deepseek-v4-pro",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "qwen/qwen3.6-35b-a3b",
    "google/gemini-3.1-pro-preview",
]

# Fallback prices are USD per token, copied from live OpenRouter model data on 2026-06-11.
FALLBACK_PRICING: dict[str, dict[str, float]] = {
    "openrouter/anthropic/claude-fable-5": {"prompt": 0.00001, "completion": 0.00005},
    "openrouter/anthropic/claude-opus-4.8": {"prompt": 0.000005, "completion": 0.000025},
    "openrouter/openai/gpt-5.5": {"prompt": 0.000005, "completion": 0.00003},
    "openrouter/deepseek/deepseek-v4-pro": {"prompt": 0.000000435, "completion": 0.00000087},
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b": {"prompt": 0.0000005, "completion": 0.0000025},
    "openrouter/qwen/qwen3.6-35b-a3b": {"prompt": 0.00000015, "completion": 0.000001},
    "openrouter/google/gemini-3.1-pro-preview": {"prompt": 0.000002, "completion": 0.000012},
}


@dataclass(frozen=True)
class TrackAssumptions:
    analysis_input_tokens: int = 900
    analysis_output_tokens: int = 120
    decisions_input_tokens: int = 160
    decisions_output_tokens: int = 40
    league_input_tokens: int = 140
    league_output_tokens: int = 35


def normalize_openrouter_model_name(model_name: str) -> str:
    if model_name in BUNDLED_MANAGERS:
        return model_name
    if model_name.startswith("openrouter/"):
        return model_name
    return f"openrouter/{model_name.lstrip('/')}"


def require_openrouter_api_key(model_names: list[str]) -> None:
    requires_openrouter = any(model not in BUNDLED_MANAGERS for model in model_names)
    if requires_openrouter and not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError(
            "OPENROUTER_API_KEY is required for model runs. "
            "Set it in your environment and pass model names like "
            "'openai/gpt-5' or 'anthropic/claude-opus-4.8'; the CLI will route them through OpenRouter."
        )


def fetch_openrouter_catalog(timeout_seconds: float = 20.0) -> dict[str, Any]:
    request = urllib.request.Request(
        OPENROUTER_MODELS_URL,
        headers={"User-Agent": "baseball-bench/0.1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def build_pricing_lookup(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {}
    for item in payload.get("data", []):
        model_id = item.get("id")
        pricing = item.get("pricing", {})
        if not model_id or "prompt" not in pricing or "completion" not in pricing:
            continue
        try:
            lookup[f"openrouter/{model_id}"] = {
                "prompt": float(pricing["prompt"]),
                "completion": float(pricing["completion"]),
            }
        except (TypeError, ValueError):
            continue
    return lookup


def get_pricing_lookup(allow_network: bool = True) -> tuple[dict[str, dict[str, float]], str]:
    if allow_network:
        try:
            payload = fetch_openrouter_catalog()
            lookup = build_pricing_lookup(payload)
            if lookup:
                return lookup, "live_openrouter"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            pass
    return FALLBACK_PRICING.copy(), "fallback_2026_06_11"


def count_analysis_samples() -> int:
    from baseball_bench.tracks.analysis.question_bank import load_questions

    return len(load_questions())


def count_decision_samples() -> int:
    from baseball_bench.tracks.decisions.task import load_situations

    return len(load_situations())


def estimate_costs(
    model_names: list[str],
    *,
    games_per_matchup: int = 6,
    league_games: int | None = None,
    assumptions: TrackAssumptions | None = None,
    allow_network: bool = True,
) -> dict[str, Any]:
    assumptions = assumptions or TrackAssumptions()
    normalized_models = [normalize_openrouter_model_name(model) for model in model_names]
    external_models = [model for model in normalized_models if model not in BUNDLED_MANAGERS]
    pricing_lookup, pricing_source = get_pricing_lookup(allow_network=allow_network)

    analysis_samples = count_analysis_samples()
    decision_samples = count_decision_samples()
    manager_count = len(external_models)
    if manager_count == 0:
        raise ValueError("No external OpenRouter models were provided.")

    # Controlled league adds the rulebook manager. A sampled schedule gives each
    # external model roughly an equal share of home/away slots.
    total_managers = manager_count + 1
    if league_games is None:
        league_games_per_model = 2 * manager_count * games_per_matchup
        estimated_total_league_games = total_managers * (total_managers - 1) * games_per_matchup
    else:
        estimated_total_league_games = league_games
        league_games_per_model = (2 * league_games) / total_managers
    estimated_decisions_per_game = 32.5
    league_decisions_per_model = league_games_per_model * estimated_decisions_per_game

    per_model: list[dict[str, Any]] = []
    total_cost = 0.0

    for model_name in external_models:
        pricing = pricing_lookup.get(model_name)
        if pricing is None:
            raise ValueError(f"Missing pricing for model {model_name}.")

        analysis_cost = (
            analysis_samples
            * (
                assumptions.analysis_input_tokens * pricing["prompt"]
                + assumptions.analysis_output_tokens * pricing["completion"]
            )
        )
        decisions_cost = (
            decision_samples
            * (
                assumptions.decisions_input_tokens * pricing["prompt"]
                + assumptions.decisions_output_tokens * pricing["completion"]
            )
        )
        league_cost = league_decisions_per_model * (
            assumptions.league_input_tokens * pricing["prompt"]
            + assumptions.league_output_tokens * pricing["completion"]
        )
        model_total = analysis_cost + decisions_cost + league_cost
        total_cost += model_total

        per_model.append(
            {
                "model": model_name,
                "pricing_per_token": pricing,
                "analysis_cost_usd": round(analysis_cost, 4),
                "decisions_cost_usd": round(decisions_cost, 4),
                "league_cost_usd": round(league_cost, 4),
                "total_cost_usd": round(model_total, 4),
                "league_games_per_model": league_games_per_model,
                "estimated_league_decisions_per_model": round(league_decisions_per_model, 1),
            }
        )

    summary = {
        "kind": "cost_estimate",
        "pricing_source": pricing_source,
        "models": external_models,
        "games_per_matchup": games_per_matchup,
        "league_games": league_games,
        "estimated_total_league_games": round(estimated_total_league_games, 1),
        "analysis_samples": analysis_samples,
        "decision_samples": decision_samples,
        "estimated_decisions_per_game": estimated_decisions_per_game,
        "assumptions": {
            "analysis_input_tokens": assumptions.analysis_input_tokens,
            "analysis_output_tokens": assumptions.analysis_output_tokens,
            "decisions_input_tokens": assumptions.decisions_input_tokens,
            "decisions_output_tokens": assumptions.decisions_output_tokens,
            "league_input_tokens": assumptions.league_input_tokens,
            "league_output_tokens": assumptions.league_output_tokens,
        },
        "per_model": per_model,
        "total_cost_usd": round(total_cost, 4),
        "average_cost_usd_per_model": round(mean(item["total_cost_usd"] for item in per_model), 4),
    }
    return summary


def write_cost_estimate(summary: dict[str, Any]) -> None:
    write_json(RESULTS_DIR / "cost-estimate.json", summary)
