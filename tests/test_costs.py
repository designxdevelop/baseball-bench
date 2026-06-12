from baseball_bench.costs import (
    FALLBACK_PRICING,
    TrackAssumptions,
    build_pricing_lookup,
    estimate_costs,
    normalize_openrouter_model_name,
)


def test_normalize_openrouter_model_name_keeps_builtin_managers():
    assert normalize_openrouter_model_name("rulebook") == "rulebook"
    assert normalize_openrouter_model_name("openai/gpt-5.5") == "openrouter/openai/gpt-5.5"


def test_build_pricing_lookup_reads_openrouter_payload():
    payload = {
        "data": [
            {
                "id": "openai/gpt-5.5",
                "pricing": {"prompt": "0.1", "completion": "0.2"},
            }
        ]
    }

    assert build_pricing_lookup(payload) == {
        "openrouter/openai/gpt-5.5": {"prompt": 0.1, "completion": 0.2}
    }


def test_estimate_costs_uses_fallback_prices(monkeypatch):
    monkeypatch.setattr("baseball_bench.costs.count_analysis_samples", lambda: 10)
    monkeypatch.setattr("baseball_bench.costs.count_decision_samples", lambda: 8)
    monkeypatch.setattr(
        "baseball_bench.costs.get_pricing_lookup",
        lambda allow_network=True: (FALLBACK_PRICING.copy(), "test_fallback"),
    )

    summary = estimate_costs(
        ["openrouter/openai/gpt-5.5", "openrouter/qwen/qwen3.6-35b-a3b"],
        games_per_matchup=2,
        assumptions=TrackAssumptions(
            analysis_input_tokens=100,
            analysis_output_tokens=10,
            decisions_input_tokens=50,
            decisions_output_tokens=5,
            league_input_tokens=25,
            league_output_tokens=5,
        ),
        allow_network=False,
    )

    assert summary["pricing_source"] == "test_fallback"
    assert summary["analysis_samples"] == 10
    assert summary["decision_samples"] == 8
    assert len(summary["per_model"]) == 2
    assert summary["total_cost_usd"] > 0

