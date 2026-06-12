from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

EvalMode = Literal["public-refresh", "deep-eval"]


@dataclass(frozen=True)
class LeagueEvalConfig:
    evaluation_mode: EvalMode = "public-refresh"
    league_games: int = 6
    live_call_start_inning: int = 8
    live_call_max_score_gap: int = 2
    max_live_calls_per_team: int = 2
    enable_open_league: bool = False

    def should_call_live(self, *, inning: int, score_gap: int, team_live_calls: int) -> bool:
        if self.max_live_calls_per_team <= 0:
            return False
        if team_live_calls >= self.max_live_calls_per_team:
            return False
        if inning > 9:
            return True
        return inning >= self.live_call_start_inning and score_gap <= self.live_call_max_score_gap

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)


MODE_DEFAULTS: dict[EvalMode, LeagueEvalConfig] = {
    "public-refresh": LeagueEvalConfig(),
    "deep-eval": LeagueEvalConfig(
        evaluation_mode="deep-eval",
        league_games=12,
        live_call_start_inning=7,
        live_call_max_score_gap=3,
        max_live_calls_per_team=5,
        enable_open_league=False,
    ),
}


def resolve_league_eval_config(
    *,
    mode: str | None = None,
    league_games: int | None = None,
    live_call_start_inning: int | None = None,
    live_call_max_score_gap: int | None = None,
    max_live_calls_per_team: int | None = None,
    enable_open_league: bool | None = None,
) -> LeagueEvalConfig:
    selected_mode = mode or "public-refresh"
    if selected_mode not in MODE_DEFAULTS:
        raise ValueError(f"Unsupported evaluation mode: {selected_mode}")
    base = MODE_DEFAULTS[selected_mode]  # type: ignore[index]
    return LeagueEvalConfig(
        evaluation_mode=base.evaluation_mode,
        league_games=base.league_games if league_games is None else league_games,
        live_call_start_inning=base.live_call_start_inning
        if live_call_start_inning is None
        else live_call_start_inning,
        live_call_max_score_gap=base.live_call_max_score_gap
        if live_call_max_score_gap is None
        else live_call_max_score_gap,
        max_live_calls_per_team=base.max_live_calls_per_team
        if max_live_calls_per_team is None
        else max_live_calls_per_team,
        enable_open_league=base.enable_open_league
        if enable_open_league is None
        else enable_open_league,
    )
