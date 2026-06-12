from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from baseball_bench.data import connect_read_only
from pathlib import Path

from baseball_bench.paths import DATA_DB_PATH, RESULTS_DIR
from baseball_bench.tracks.league.engine import HitterProfile, PitcherProfile, TeamRoster, simulate_game
from baseball_bench.tracks.league.manager import build_manager
from baseball_bench.utils import write_json


@dataclass
class Standing:
    name: str
    wins: int = 0
    losses: int = 0
    runs_for: int = 0
    runs_against: int = 0
    elo: float = 1500.0


def _build_identical_roster(database_path: Path | None = None) -> TeamRoster:
    with connect_read_only(database_path or DATA_DB_PATH) as conn:
        hitters = conn.execute(
            """
            with batting_2025 as (
              select
                b.player_id,
                p.first_name || ' ' || p.last_name as name,
                b.plate_appearances,
                b.walks,
                b.strikeouts,
                (b.hits - b.doubles - b.triples - b.home_runs) as singles,
                b.doubles,
                b.triples,
                b.home_runs
              from batting b
              join players p on p.player_id = b.player_id
              where b.season = 2025 and p.primary_position not in ('SP', 'RP')
              order by b.plate_appearances desc
            )
            select * from batting_2025
            """
        ).fetchall()
        pitchers = conn.execute(
            """
            with pitching_2025 as (
              select
                pi.player_id,
                p.first_name || ' ' || p.last_name as name,
                pi.innings_pitched,
                pi.walks,
                pi.strikeouts,
                pi.hits_allowed,
                pi.home_runs_allowed,
                case when pi.games_started > 0 then 'SP' else 'RP' end as role
              from pitching pi
              join players p on p.player_id = pi.player_id
              where pi.season = 2025
            )
            select * from pitching_2025
            """
        ).fetchall()
    hitter_profiles = [HitterProfile(*row) for row in hitters]
    pitcher_profiles = [PitcherProfile(*row) for row in pitchers]
    lineup = hitter_profiles[:9]
    bench = hitter_profiles[9:11]
    starter = max((pitcher for pitcher in pitcher_profiles if pitcher.role == "SP"), key=lambda pitcher: pitcher.strikeouts)
    bullpen = sorted((pitcher for pitcher in pitcher_profiles if pitcher.role == "RP"), key=lambda pitcher: pitcher.strikeouts, reverse=True)
    return TeamRoster(lineup=lineup, bench=bench, starter=starter, bullpen=bullpen)


def _update_elo(winner: Standing, loser: Standing, k_factor: float = 20.0) -> None:
    expected = 1.0 / (1.0 + 10 ** ((loser.elo - winner.elo) / 400))
    winner.elo += k_factor * (1.0 - expected)
    loser.elo += k_factor * (0.0 - (1.0 - expected))


def run_league(
    model_names: list[str],
    games_per_matchup: int = 6,
    seed: int = 7,
    database_path: Path | None = None,
) -> dict[str, object]:
    if len(model_names) < 2:
        raise ValueError("League requires at least two managers.")
    managers = [build_manager(name) for name in model_names]
    roster = _build_identical_roster(database_path=database_path)
    standings = {manager.name: Standing(name=manager.name) for manager in managers}
    games: list[dict[str, object]] = []
    game_index = 0
    for i, home_manager in enumerate(managers):
        for j, away_manager in enumerate(managers):
            if i == j:
                continue
            for _ in range(games_per_matchup):
                game_index += 1
                result = simulate_game(home_manager, away_manager, roster, roster, seed + game_index)
                home = standings[result.home_manager]
                away = standings[result.away_manager]
                home.runs_for += result.home_score
                home.runs_against += result.away_score
                away.runs_for += result.away_score
                away.runs_against += result.home_score
                if result.home_score > result.away_score:
                    home.wins += 1
                    away.losses += 1
                    _update_elo(home, away)
                    winner = home.name
                else:
                    away.wins += 1
                    home.losses += 1
                    _update_elo(away, home)
                    winner = away.name
                games.append(
                    {
                        "game_number": game_index,
                        "home": result.home_manager,
                        "away": result.away_manager,
                        "home_score": result.home_score,
                        "away_score": result.away_score,
                        "winner": winner,
                        "decision_count": len(result.decision_log),
                    }
                )
    standings_table = [
        {
            "model": standing.name,
            "wins": standing.wins,
            "losses": standing.losses,
            "run_diff": standing.runs_for - standing.runs_against,
            "elo": round(standing.elo, 2),
            "win_pct": round(standing.wins / max(standing.wins + standing.losses, 1), 3),
        }
        for standing in sorted(
            standings.values(),
            key=lambda item: (item.wins, item.elo, item.runs_for - item.runs_against),
            reverse=True,
        )
    ]
    summary = {
        "kind": "league",
        "models": model_names,
        "games_per_matchup": games_per_matchup,
        "game_count": len(games),
        "average_decisions_per_game": mean(game["decision_count"] for game in games) if games else 0.0,
        "standings": standings_table,
        "games": games,
    }
    write_json(RESULTS_DIR / f"league-{'-'.join(model_names)}.json", summary)
    return summary
