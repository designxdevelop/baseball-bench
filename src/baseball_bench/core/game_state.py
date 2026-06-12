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


class BullpenOption(BaseModel):
    player_id: str
    name: str
    throws: str
    stamina: int = 100


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
    pitcher_name: str
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

