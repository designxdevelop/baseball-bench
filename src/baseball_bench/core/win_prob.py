from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from baseball_bench.core.game_state import GameState
from baseball_bench.data import connect_read_only


@dataclass(frozen=True)
class WinProbRow:
    inning: int
    half: str
    outs: int
    runner_state: str
    score_diff: int
    home_win_prob: float


class WinProbabilityTable:
    def __init__(self, rows: Iterable[WinProbRow]) -> None:
        self.rows = list(rows)

    @classmethod
    def from_database(cls, database_path: Path | None = None) -> "WinProbabilityTable":
        with connect_read_only(database_path) as conn:
            rows = conn.execute(
                """
                select inning, half, outs, runner_state, score_diff, home_win_prob
                from win_probabilities
                """
            ).fetchall()
        return cls(WinProbRow(*row) for row in rows)

    def lookup(self, state: GameState) -> float:
        target = (
            state.inning,
            state.half,
            state.outs,
            state.runners.compact(),
            state.score_diff_home,
        )
        def distance(row: WinProbRow) -> tuple[int, int, int]:
            return (
                abs(row.inning - target[0]) + abs(row.score_diff - target[4]),
                abs(row.outs - target[2]),
                sum(int(a != b) for a, b in zip(row.runner_state, target[3], strict=True)),
            )

        best = min(
            (
                row
                for row in self.rows
                if row.half == target[1]
            ),
            key=distance,
        )
        return best.home_win_prob

