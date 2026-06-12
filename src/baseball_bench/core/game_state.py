from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RunnerState(BaseModel):
    first: bool = False
    second: bool = False
    third: bool = False

    def compact(self) -> str:
        return f"{int(self.first)}{int(self.second)}{int(self.third)}"


class BenchOption(BaseModel):
    player_id: str
    name: str
    bats: str
    position: str | None = None
    summary: str | None = None


class BullpenOption(BaseModel):
    player_id: str
    name: str
    throws: str
    role: str | None = None
    stamina: int = 100
    summary: str | None = None


class GameState(BaseModel):
    inning: int
    half: Literal["top", "bottom"]
    outs: int
    runners: RunnerState = Field(default_factory=RunnerState)
    home_team: str
    away_team: str
    batting_team: str
    fielding_team: str
    home_score: int
    away_score: int
    batter_name: str
    batter_bats: str | None = None
    batter_summary: str | None = None
    pitcher_name: str
    pitcher_throws: str | None = None
    pitcher_role: str | None = None
    pitcher_summary: str | None = None
    pitcher_fatigue: int | None = None
    times_through_order: int | None = None
    park_name: str | None = None
    park_factor: float | None = None
    leverage_index: float = 1.0
    bench: list[BenchOption] = Field(default_factory=list)
    bullpen: list[BullpenOption] = Field(default_factory=list)

    @property
    def score_diff_home(self) -> int:
        return self.home_score - self.away_score

    @property
    def score_diff_batting(self) -> int:
        if self.batting_team == self.home_team:
            return self.home_score - self.away_score
        return self.away_score - self.home_score

    def summary(self) -> str:
        bases = []
        if self.runners.first:
            bases.append("1st")
        if self.runners.second:
            bases.append("2nd")
        if self.runners.third:
            bases.append("3rd")
        runners = ", ".join(bases) if bases else "bases empty"
        return (
            f"{self.half.title()} {self.inning}, {self.outs} out(s), {runners}. "
            f"{self.away_team} {self.away_score}, {self.home_team} {self.home_score}. "
            f"Batter: {self.batter_name}. Pitcher: {self.pitcher_name}."
        )

    def manager_prompt_context(self) -> str:
        lines = [
            self.summary(),
            f"Batting team: {self.batting_team}. Fielding team: {self.fielding_team}.",
            f"Score context: {self._score_context()}.",
            f"Leverage index: {self.leverage_index:.1f}.",
        ]
        if self.park_name or self.park_factor is not None:
            park = self.park_name or "current park"
            factor = f"{self.park_factor:.3f}" if self.park_factor is not None else "unknown"
            lines.append(f"Park context: {park}, run-scoring factor {factor}.")

        batter = self.batter_name
        if self.batter_bats:
            batter += f", bats {self.batter_bats}"
        if self.batter_summary:
            batter += f", {self.batter_summary}"
        lines.append(f"Batter context: {batter}.")

        pitcher = self.pitcher_name
        if self.pitcher_throws:
            pitcher += f", throws {self.pitcher_throws}"
        if self.pitcher_role:
            pitcher += f", role {self.pitcher_role}"
        if self.pitcher_summary:
            pitcher += f", {self.pitcher_summary}"
        if self.pitcher_fatigue is not None:
            pitcher += f", fatigue batters faced in game {self.pitcher_fatigue}"
        if self.times_through_order is not None:
            pitcher += f", lineup turn {self.times_through_order}"
        lines.append(f"Pitcher context: {pitcher}.")

        if self.bullpen:
            lines.append("Bullpen options:")
            for arm in self.bullpen:
                role = f", role {arm.role}" if arm.role else ""
                summary = f", {arm.summary}" if arm.summary else ""
                lines.append(
                    f"- {arm.name}: throws {arm.throws}{role}, stamina {arm.stamina}/100{summary}."
                )
        if self.bench:
            lines.append("Bench options:")
            for hitter in self.bench:
                position = f", {hitter.position}" if hitter.position else ""
                summary = f", {hitter.summary}" if hitter.summary else ""
                lines.append(f"- {hitter.name}: bats {hitter.bats}{position}{summary}.")
        return "\n".join(lines)

    def _score_context(self) -> str:
        diff = self.score_diff_batting
        if diff == 0:
            state = "tie game"
        elif diff > 0:
            state = f"{self.batting_team} leads by {diff}"
        else:
            state = f"{self.batting_team} trails by {abs(diff)}"
        base_runners = int(self.runners.first) + int(self.runners.second) + int(self.runners.third)
        tying_run = "tying run already on base" if diff < 0 and base_runners >= abs(diff) else None
        tying_run_at_plate = "tying run at the plate" if diff < 0 and base_runners + 1 >= abs(diff) else None
        go_ahead_run = "go-ahead run at the plate" if diff <= 0 and base_runners + 1 > abs(diff) else None
        save_spot = "save-style late lead" if self.inning >= 9 and diff < 0 and abs(diff) <= 3 else None
        extras = [item for item in [tying_run, tying_run_at_plate, go_ahead_run, save_spot] if item]
        return state if not extras else f"{state}; {'; '.join(extras)}"
