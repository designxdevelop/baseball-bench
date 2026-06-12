from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model

from baseball_bench.paths import RESULTS_DIR
from baseball_bench.tracks.league.engine import HitterProfile, PitcherProfile, TeamRoster
from baseball_bench.utils import parse_jsonish, slugify, write_json

PLAN_TIMEOUT_SECONDS = 45


def _hitter_rating(hitter: HitterProfile) -> float:
    rates = hitter.rates
    return (
        rates["walk"]
        + rates["single"]
        + (1.6 * rates["double"])
        + (2.1 * rates["triple"])
        + (2.8 * rates["home_run"])
        - (0.3 * rates["strikeout"])
    )


def _pitcher_rating(pitcher: PitcherProfile) -> float:
    innings = max(pitcher.innings_pitched, 1.0)
    return (pitcher.strikeouts / innings) - (0.6 * pitcher.walks / innings) - (
        0.5 * pitcher.home_runs_allowed / innings
    )


def _heuristic_plan(model_name: str, roster: TeamRoster) -> dict[str, Any]:
    if "aggressive" in model_name:
        lineup = sorted(
            roster.lineup + roster.bench,
            key=lambda hitter: (hitter.rates["home_run"], _hitter_rating(hitter)),
            reverse=True,
        )
    elif "conservative" in model_name:
        lineup = sorted(
            roster.lineup + roster.bench,
            key=lambda hitter: (hitter.rates["walk"], -hitter.rates["strikeout"], _hitter_rating(hitter)),
            reverse=True,
        )
    else:
        lineup = sorted(roster.lineup + roster.bench, key=_hitter_rating, reverse=True)

    pitchers = [roster.starter, *roster.bullpen]
    starters = sorted(
        [pitcher for pitcher in pitchers if pitcher.role == "SP"],
        key=_pitcher_rating,
        reverse=True,
    )
    bullpen = sorted(
        [pitcher for pitcher in pitchers if pitcher.role == "RP"],
        key=_pitcher_rating,
        reverse=True,
    )
    return {
        "lineup": [hitter.player_id for hitter in lineup[:9]],
        "bench": [hitter.player_id for hitter in lineup[9:]],
        "starter": starters[0].player_id if starters else roster.starter.player_id,
        "bullpen": [pitcher.player_id for pitcher in bullpen],
        "bullpen_roles": {
            pitcher.player_id: (
                "closer"
                if index == 0
                else "setup"
                if index == 1
                else "middle"
                if index <= 3
                else "long"
            )
            for index, pitcher in enumerate(bullpen)
        },
        "tendencies": {
            "steal": "aggressive" if "aggressive" in model_name else "selective",
            "bunt": "low" if "aggressive" in model_name else "situational",
            "starter_hook": "quick" if "aggressive" in model_name else "standard",
        },
    }


def _format_roster_for_prompt(roster: TeamRoster) -> str:
    hitters = roster.lineup + roster.bench
    pitcher_pool = [roster.starter, *roster.bullpen]
    hitter_lines = [
        f"{hitter.player_id}: {hitter.name}, {hitter.position}, bats {hitter.bats}, PA {hitter.pa}, HR {hitter.home_runs}, BB {hitter.walks}, SO {hitter.strikeouts}"
        for hitter in hitters
    ]
    pitcher_lines = [
        f"{pitcher.player_id}: {pitcher.name}, {pitcher.role}, throws {pitcher.throws}, IP {pitcher.innings_pitched:.1f}, K {pitcher.strikeouts}, BB {pitcher.walks}, HR {pitcher.home_runs_allowed}"
        for pitcher in pitcher_pool
    ]
    return "Hitters:\n" + "\n".join(hitter_lines) + "\n\nPitchers:\n" + "\n".join(pitcher_lines)


async def _generate_plan(model_name: str, roster: TeamRoster) -> dict[str, Any]:
    prompt = "\n".join(
        [
            "You are managing the same roster as every other model in a baseball benchmark.",
            "Set the batting order, bench order, starting pitcher, bullpen hierarchy, and tendencies.",
            "Return strict JSON with keys: lineup, bench, starter, bullpen, bullpen_roles, tendencies.",
            "lineup must contain exactly 9 hitter player_id values.",
            "bench must contain remaining hitter player_id values in preferred platoon and pinch-hit order.",
            "starter must be one pitcher player_id.",
            "bullpen must contain reliever player_id values in preferred usage order.",
            "bullpen_roles may map reliever ids to closer, setup, middle, or long.",
            "",
            _format_roster_for_prompt(roster),
        ]
    )
    model = get_model(model_name)
    result = await asyncio.wait_for(
        model.generate(
            [
                ChatMessageSystem(content="Return only strict JSON for a baseball manager plan."),
                ChatMessageUser(content=prompt),
            ]
        ),
        timeout=PLAN_TIMEOUT_SECONDS,
    )
    return parse_jsonish(result.completion)


def _ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _validate_plan(raw: dict[str, Any], roster: TeamRoster) -> tuple[dict[str, Any], dict[str, Any]]:
    hitters = roster.lineup + roster.bench
    hitter_ids = {hitter.player_id for hitter in hitters}
    pitcher_pool = [roster.starter, *roster.bullpen]
    pitcher_ids = {pitcher.player_id for pitcher in pitcher_pool}
    reliever_ids = {pitcher.player_id for pitcher in roster.bullpen}

    lineup = _ids(raw.get("lineup"))
    bench = _ids(raw.get("bench"))
    starter = str(raw.get("starter", "")).strip()
    bullpen = _ids(raw.get("bullpen"))
    raw_roles = raw.get("bullpen_roles", {})
    bullpen_roles = {}
    if isinstance(raw_roles, dict):
        bullpen_roles = {
            str(player_id): str(role).lower()
            for player_id, role in raw_roles.items()
            if str(role).lower() in {"closer", "setup", "middle", "long"}
        }
    selected_hitters = lineup + bench
    errors: list[str] = []
    if len(lineup) != 9:
        errors.append("wrong_lineup_size")
    if set(selected_hitters) != hitter_ids:
        errors.append("hitter_pool_mismatch")
    if len(selected_hitters) != len(set(selected_hitters)):
        errors.append("duplicate_hitters")
    if starter not in pitcher_ids:
        errors.append("invalid_starter")
    if any(player_id not in reliever_ids for player_id in bullpen):
        errors.append("invalid_bullpen")
    if len(bullpen) != len(set(bullpen)):
        errors.append("duplicate_relievers")

    validity_score = 1.0 if not errors else max(0.0, 1.0 - (0.2 * len(errors)))
    plan = {
        "lineup": lineup,
        "bench": bench,
        "starter": starter,
        "bullpen": bullpen,
        "bullpen_roles": bullpen_roles,
        "tendencies": raw.get("tendencies", {}),
        "validation": {"valid": not errors, "errors": errors},
    }
    return plan, {"validity_score": round(validity_score, 3)}


def roster_from_plan(roster: TeamRoster, plan: dict[str, Any]) -> TeamRoster:
    hitters = {hitter.player_id: hitter for hitter in roster.lineup + roster.bench}
    pitchers = {pitcher.player_id: pitcher for pitcher in [roster.starter, *roster.bullpen]}
    lineup = [hitters[player_id] for player_id in plan["lineup"] if player_id in hitters][:9]
    bench = [hitters[player_id] for player_id in plan["bench"] if player_id in hitters]
    starter = pitchers.get(str(plan.get("starter")), roster.starter)
    bullpen = [
        pitchers[player_id]
        for player_id in plan["bullpen"]
        if player_id in pitchers and pitchers[player_id].role == "RP"
    ]
    if len(lineup) < 9:
        used = {hitter.player_id for hitter in lineup}
        lineup.extend([hitter for hitter in roster.lineup + roster.bench if hitter.player_id not in used][: 9 - len(lineup)])
    if not bullpen:
        bullpen = list(roster.bullpen)
    bullpen_roles = {
        player_id: role
        for player_id, role in dict(plan.get("bullpen_roles", {})).items()
        if player_id in {pitcher.player_id for pitcher in bullpen}
    }
    if not bullpen_roles:
        bullpen_roles = {
            pitcher.player_id: (
                "closer"
                if index == 0
                else "setup"
                if index == 1
                else "middle"
                if index <= 3
                else "long"
            )
            for index, pitcher in enumerate(bullpen)
        }
    return TeamRoster(
        lineup=lineup,
        bench=bench,
        starter=starter,
        bullpen=bullpen,
        park_factor=roster.park_factor,
        bullpen_roles=bullpen_roles,
    )


def run_manager_plan(
    model_name: str,
    roster: TeamRoster,
    *,
    output_dir: Path = RESULTS_DIR,
    allow_model_call: bool = True,
) -> dict[str, Any]:
    source = "model"
    fallback_reason = None
    if model_name in {"rulebook", "aggressive", "conservative"} or not allow_model_call:
        raw = _heuristic_plan(model_name, roster)
        source = "heuristic"
    else:
        try:
            raw = asyncio.run(_generate_plan(model_name, roster))
        except Exception as exc:
            raw = _heuristic_plan(model_name, roster)
            source = "fallback"
            fallback_reason = type(exc).__name__
    plan, scores = _validate_plan(raw, roster)
    payload = {
        "kind": "manager_plan",
        "model": model_name,
        "source": source,
        "fallback_reason": fallback_reason,
        "plan": plan,
        "scores": scores,
    }
    write_json(output_dir / f"manager-plan-{slugify(model_name)}.json", payload)
    if output_dir != RESULTS_DIR:
        write_json(RESULTS_DIR / f"manager-plan-{slugify(model_name)}.json", payload)
    return payload


def run_manager_plans(
    model_names: list[str],
    roster: TeamRoster,
    *,
    output_dir: Path = RESULTS_DIR,
    allow_model_call: bool = True,
) -> dict[str, TeamRoster]:
    planned_rosters: dict[str, TeamRoster] = {}
    for model_name in model_names:
        payload = run_manager_plan(
            model_name,
            roster,
            output_dir=output_dir,
            allow_model_call=allow_model_call,
        )
        planned_rosters[model_name] = roster_from_plan(roster, payload["plan"])
    return planned_rosters
