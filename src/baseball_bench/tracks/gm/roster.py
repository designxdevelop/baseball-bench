from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model

from baseball_bench.data import connect_read_only
from baseball_bench.paths import DATA_DB_PATH, RESULTS_DIR
from baseball_bench.tracks.league.engine import HitterProfile, PitcherProfile, TeamRoster
from baseball_bench.utils import parse_jsonish, slugify, write_json

GM_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class HitterPoolRow:
    player_id: str
    name: str
    bats: str
    position: str
    plate_appearances: int
    hits: int
    doubles: int
    triples: int
    home_runs: int
    walks: int
    strikeouts: int
    hit_by_pitch: int

    @property
    def obp(self) -> float:
        return (self.hits + self.walks + self.hit_by_pitch) / max(self.plate_appearances, 1)

    @property
    def slug(self) -> float:
        singles = self.hits - self.doubles - self.triples - self.home_runs
        total_bases = singles + 2 * self.doubles + 3 * self.triples + 4 * self.home_runs
        at_bats = max(self.plate_appearances - self.walks - self.hit_by_pitch, 1)
        return total_bases / at_bats

    @property
    def rating(self) -> float:
        contact = self.hits / max(self.plate_appearances, 1)
        discipline = self.walks / max(self.plate_appearances, 1)
        power = (self.doubles + 2 * self.triples + 3 * self.home_runs) / max(self.plate_appearances, 1)
        return contact + discipline + power


@dataclass(frozen=True)
class PitcherPoolRow:
    player_id: str
    name: str
    throws: str
    role: str
    innings_pitched: float
    earned_runs: int
    walks: int
    strikeouts: int
    hits_allowed: int
    home_runs_allowed: int

    @property
    def rating(self) -> float:
        innings = max(self.innings_pitched, 1.0)
        strikeout_rate = self.strikeouts / innings
        walk_rate = self.walks / innings
        run_rate = self.earned_runs / innings
        homer_rate = self.home_runs_allowed / innings
        return strikeout_rate - (0.6 * walk_rate) - (0.8 * run_rate) - (0.5 * homer_rate)


def _player_pool(database_path: Path | None = None) -> tuple[list[HitterPoolRow], list[PitcherPoolRow]]:
    with connect_read_only(database_path or DATA_DB_PATH) as conn:
        hitters = [
            HitterPoolRow(*row)
            for row in conn.execute(
                """
                select
                  b.player_id,
                  p.first_name || ' ' || p.last_name as name,
                  p.bats,
                  p.primary_position,
                  b.plate_appearances,
                  b.hits,
                  b.doubles,
                  b.triples,
                  b.home_runs,
                  b.walks,
                  b.strikeouts,
                  b.hit_by_pitch
                from batting b
                join players p on p.player_id = b.player_id
                where b.season = 2025 and p.primary_position not in ('SP', 'RP')
                order by b.plate_appearances desc
                """
            ).fetchall()
        ]
        pitchers = [
            PitcherPoolRow(*row)
            for row in conn.execute(
                """
                select
                  pi.player_id,
                  p.first_name || ' ' || p.last_name as name,
                  p.throws,
                  case when pi.games_started > 0 then 'SP' else 'RP' end as role,
                  pi.innings_pitched,
                  pi.earned_runs,
                  pi.walks,
                  pi.strikeouts,
                  pi.hits_allowed,
                  pi.home_runs_allowed
                from pitching pi
                join players p on p.player_id = pi.player_id
                where pi.season = 2025
                """
            ).fetchall()
        ]
    return hitters, pitchers


def _roster_requirements(hitters: list[HitterPoolRow], pitchers: list[PitcherPoolRow]) -> dict[str, int]:
    starters = [pitcher for pitcher in pitchers if pitcher.role == "SP"]
    relievers = [pitcher for pitcher in pitchers if pitcher.role == "RP"]
    return {
        "lineup_size": min(9, len(hitters)),
        "bench_size": max(0, len(hitters) - min(9, len(hitters))),
        "rotation_size": min(5, len(starters)),
        "bullpen_size": min(8, len(relievers)),
    }


def _format_pool_for_prompt(hitters: list[HitterPoolRow], pitchers: list[PitcherPoolRow]) -> str:
    hitter_lines = [
        f"{h.player_id}: {h.name}, {h.position}, bats {h.bats}, OBP {h.obp:.3f}, SLG {h.slug:.3f}, HR {h.home_runs}, SO {h.strikeouts}"
        for h in hitters
    ]
    pitcher_lines = [
        f"{p.player_id}: {p.name}, {p.role}, throws {p.throws}, IP {p.innings_pitched:.1f}, K {p.strikeouts}, BB {p.walks}, ER {p.earned_runs}"
        for p in pitchers
    ]
    return "Hitters:\n" + "\n".join(hitter_lines) + "\n\nPitchers:\n" + "\n".join(pitcher_lines)


def _heuristic_roster(
    model_name: str,
    hitters: list[HitterPoolRow],
    pitchers: list[PitcherPoolRow],
) -> dict[str, Any]:
    requirements = _roster_requirements(hitters, pitchers)
    if "aggressive" in model_name:
        hitter_key = lambda hitter: (hitter.slug, hitter.home_runs, hitter.rating)
    elif "conservative" in model_name:
        hitter_key = lambda hitter: (hitter.obp, -hitter.strikeouts, hitter.rating)
    else:
        hitter_key = lambda hitter: (hitter.rating, hitter.obp, hitter.slug)

    ordered_hitters = sorted(hitters, key=hitter_key, reverse=True)
    starters = sorted(
        (pitcher for pitcher in pitchers if pitcher.role == "SP"),
        key=lambda pitcher: pitcher.rating,
        reverse=True,
    )
    relievers = sorted(
        (pitcher for pitcher in pitchers if pitcher.role == "RP"),
        key=lambda pitcher: pitcher.rating,
        reverse=True,
    )
    return {
        "lineup": [hitter.player_id for hitter in ordered_hitters[: requirements["lineup_size"]]],
        "bench": [
            hitter.player_id
            for hitter in ordered_hitters[
                requirements["lineup_size"] : requirements["lineup_size"] + requirements["bench_size"]
            ]
        ],
        "rotation": [pitcher.player_id for pitcher in starters[: requirements["rotation_size"]]],
        "bullpen": [pitcher.player_id for pitcher in relievers[: requirements["bullpen_size"]]],
        "rationale": f"{model_name} prioritized projected run creation and pitcher run prevention.",
    }


async def _generate_roster(
    model_name: str,
    hitters: list[HitterPoolRow],
    pitchers: list[PitcherPoolRow],
) -> dict[str, Any]:
    requirements = _roster_requirements(hitters, pitchers)
    prompt = "\n".join(
        [
            "Build a baseball roster from this player pool.",
            "Return strict JSON with keys: lineup, bench, rotation, bullpen, rationale.",
            f"lineup must contain {requirements['lineup_size']} hitter player_id values.",
            f"bench must contain {requirements['bench_size']} hitter player_id values.",
            f"rotation must contain {requirements['rotation_size']} SP player_id values.",
            f"bullpen must contain {requirements['bullpen_size']} RP player_id values.",
            "Use each player at most once. Prefer baseball reasoning over alphabetical order.",
            "",
            _format_pool_for_prompt(hitters, pitchers),
        ]
    )
    model = get_model(model_name)
    result = await asyncio.wait_for(
        model.generate(
            [
                ChatMessageSystem(content="Return only strict JSON for roster construction."),
                ChatMessageUser(content=prompt),
            ]
        ),
        timeout=GM_TIMEOUT_SECONDS,
    )
    return parse_jsonish(result.completion)


def _normalize_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _ordered_score(selected: list[str], ideal: list[str]) -> float:
    if not selected:
        return 0.0
    ideal_rank = {player_id: index for index, player_id in enumerate(ideal)}
    max_distance = max(len(ideal) - 1, 1)
    penalties = [
        abs(index - ideal_rank.get(player_id, len(ideal))) / max_distance
        for index, player_id in enumerate(selected)
    ]
    return max(0.0, 1.0 - mean(penalties))


def _score_roster(
    raw: dict[str, Any],
    hitters: list[HitterPoolRow],
    pitchers: list[PitcherPoolRow],
) -> tuple[dict[str, Any], dict[str, Any]]:
    hitter_ids = {hitter.player_id for hitter in hitters}
    starter_ids = {pitcher.player_id for pitcher in pitchers if pitcher.role == "SP"}
    reliever_ids = {pitcher.player_id for pitcher in pitchers if pitcher.role == "RP"}
    requirements = _roster_requirements(hitters, pitchers)

    lineup = _normalize_ids(raw.get("lineup"))
    bench = _normalize_ids(raw.get("bench"))
    rotation = _normalize_ids(raw.get("rotation"))
    bullpen = _normalize_ids(raw.get("bullpen"))
    selected = lineup + bench + rotation + bullpen
    duplicate_count = len(selected) - len(set(selected))

    errors: list[str] = []
    if len(lineup) != requirements["lineup_size"]:
        errors.append("wrong_lineup_size")
    if len(bench) != requirements["bench_size"]:
        errors.append("wrong_bench_size")
    if len(rotation) != requirements["rotation_size"]:
        errors.append("wrong_rotation_size")
    if len(bullpen) != requirements["bullpen_size"]:
        errors.append("wrong_bullpen_size")
    if duplicate_count:
        errors.append("duplicate_players")
    if any(player_id not in hitter_ids for player_id in lineup + bench):
        errors.append("invalid_hitter")
    if any(player_id not in starter_ids for player_id in rotation):
        errors.append("invalid_starter")
    if any(player_id not in reliever_ids for player_id in bullpen):
        errors.append("invalid_reliever")

    ideal_hitters = [
        hitter.player_id for hitter in sorted(hitters, key=lambda hitter: hitter.rating, reverse=True)
    ]
    ideal_starters = [
        pitcher.player_id
        for pitcher in sorted(
            (pitcher for pitcher in pitchers if pitcher.role == "SP"),
            key=lambda pitcher: pitcher.rating,
            reverse=True,
        )
    ]
    ideal_relievers = [
        pitcher.player_id
        for pitcher in sorted(
            (pitcher for pitcher in pitchers if pitcher.role == "RP"),
            key=lambda pitcher: pitcher.rating,
            reverse=True,
        )
    ]

    lineup_score = _ordered_score(lineup, ideal_hitters)
    pitching_score = mean(
        [
            _ordered_score(rotation, ideal_starters),
            _ordered_score(bullpen, ideal_relievers),
        ]
    )
    selected_hitters = [hitter for hitter in hitters if hitter.player_id in set(lineup + bench)]
    positions = {hitter.position for hitter in selected_hitters}
    bats = {hitter.bats for hitter in selected_hitters}
    position_score = min(len(positions) / max(len({hitter.position for hitter in hitters}), 1), 1.0)
    handedness_score = min(len(bats) / 3, 1.0)
    balance_score = mean([position_score, handedness_score])
    validity_score = 1.0 if not errors else max(0.0, 1.0 - (0.2 * len(errors)))
    overall_score = round(
        (0.4 * validity_score)
        + (0.25 * lineup_score)
        + (0.2 * pitching_score)
        + (0.15 * balance_score),
        3,
    )

    roster = {
        "lineup": lineup,
        "bench": bench,
        "rotation": rotation,
        "bullpen": bullpen,
        "rationale": str(raw.get("rationale", "")),
    }
    validation = {
        "valid": not errors,
        "errors": errors,
        "requirements": requirements,
    }
    scores = {
        "overall_score": overall_score,
        "validity_score": round(validity_score, 3),
        "lineup_score": round(lineup_score, 3),
        "pitching_score": round(pitching_score, 3),
        "balance_score": round(balance_score, 3),
    }
    return roster | {"validation": validation}, scores


def run_gm_roster(
    model_name: str,
    *,
    database_path: Path | None = None,
    output_dir: Path = RESULTS_DIR,
    allow_model_call: bool = True,
) -> dict[str, Any]:
    hitters, pitchers = _player_pool(database_path=database_path)
    source = "model"
    fallback_reason = None
    if model_name in {"rulebook", "aggressive", "conservative"} or not allow_model_call:
        raw = _heuristic_roster(model_name, hitters, pitchers)
        source = "heuristic"
    else:
        try:
            raw = asyncio.run(_generate_roster(model_name, hitters, pitchers))
        except Exception as exc:  # provider failures are scored but should not stop the bench
            raw = _heuristic_roster(model_name, hitters, pitchers)
            source = "fallback"
            fallback_reason = type(exc).__name__

    roster, scores = _score_roster(raw, hitters, pitchers)
    payload = {
        "kind": "gm_roster",
        "model": model_name,
        "source": source,
        "fallback_reason": fallback_reason,
        "roster": roster,
        "scores": scores,
    }
    write_json(output_dir / f"gm-roster-{slugify(model_name)}.json", payload)
    if output_dir != RESULTS_DIR:
        write_json(RESULTS_DIR / f"gm-roster-{slugify(model_name)}.json", payload)
    return payload


def run_gm_rosters(
    model_names: list[str],
    *,
    database_path: Path | None = None,
    output_dir: Path = RESULTS_DIR,
    allow_model_call: bool = True,
) -> list[dict[str, Any]]:
    return [
        run_gm_roster(
            model_name,
            database_path=database_path,
            output_dir=output_dir,
            allow_model_call=allow_model_call,
        )
        for model_name in model_names
    ]


def team_roster_from_build(
    build: dict[str, Any],
    *,
    database_path: Path | None = None,
) -> TeamRoster:
    hitters, pitchers = _player_pool(database_path=database_path)
    hitter_lookup = {hitter.player_id: hitter for hitter in hitters}
    pitcher_lookup = {pitcher.player_id: pitcher for pitcher in pitchers}
    roster = build["roster"]
    lineup = [
        hitter_lookup[player_id]
        for player_id in roster["lineup"]
        if player_id in hitter_lookup
    ]
    bench = [
        hitter_lookup[player_id]
        for player_id in roster["bench"]
        if player_id in hitter_lookup
    ]
    rotation = [
        pitcher_lookup[player_id]
        for player_id in roster["rotation"]
        if player_id in pitcher_lookup
    ]
    bullpen = [
        pitcher_lookup[player_id]
        for player_id in roster["bullpen"]
        if player_id in pitcher_lookup
    ]
    starter = rotation[0]
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
        lineup=[
            HitterProfile(
                player_id=hitter.player_id,
                name=hitter.name,
                pa=hitter.plate_appearances,
                walks=hitter.walks,
                strikeouts=hitter.strikeouts,
                singles=hitter.hits - hitter.doubles - hitter.triples - hitter.home_runs,
                doubles=hitter.doubles,
                triples=hitter.triples,
                home_runs=hitter.home_runs,
                bats=hitter.bats,
                position=hitter.position,
            )
            for hitter in lineup
        ],
        bench=[
            HitterProfile(
                player_id=hitter.player_id,
                name=hitter.name,
                pa=hitter.plate_appearances,
                walks=hitter.walks,
                strikeouts=hitter.strikeouts,
                singles=hitter.hits - hitter.doubles - hitter.triples - hitter.home_runs,
                doubles=hitter.doubles,
                triples=hitter.triples,
                home_runs=hitter.home_runs,
                bats=hitter.bats,
                position=hitter.position,
            )
            for hitter in bench
        ],
        starter=PitcherProfile(
            player_id=starter.player_id,
            name=starter.name,
            innings_pitched=starter.innings_pitched,
            walks=starter.walks,
            strikeouts=starter.strikeouts,
            hits_allowed=starter.hits_allowed,
            home_runs_allowed=starter.home_runs_allowed,
            role=starter.role,
            throws=starter.throws,
        ),
        bullpen=[
            PitcherProfile(
                player_id=pitcher.player_id,
                name=pitcher.name,
                innings_pitched=pitcher.innings_pitched,
                walks=pitcher.walks,
                strikeouts=pitcher.strikeouts,
                hits_allowed=pitcher.hits_allowed,
                home_runs_allowed=pitcher.home_runs_allowed,
                role=pitcher.role,
                throws=pitcher.throws,
            )
            for pitcher in bullpen
        ],
        bullpen_roles=bullpen_roles,
    )
