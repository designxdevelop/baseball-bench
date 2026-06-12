from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import Random
from statistics import mean
from baseball_bench.data import connect_read_only

from baseball_bench.paths import DATA_DB_PATH, RESULTS_DIR
from baseball_bench.tracks.league.engine import HitterProfile, PitcherProfile, TeamRoster, simulate_game
from baseball_bench.tracks.league.eval_modes import LeagueEvalConfig
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
                b.home_runs,
                p.bats,
                p.primary_position
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
                case when pi.games_started > 0 then 'SP' else 'RP' end as role,
                p.throws
              from pitching pi
              join players p on p.player_id = pi.player_id
              where pi.season = 2025
            )
            select * from pitching_2025
            """
        ).fetchall()
    hitter_profiles = [HitterProfile(*row) for row in hitters]
    pitcher_profiles = [PitcherProfile(*row) for row in pitchers]

    ordered_hitters = sorted(
        hitter_profiles,
        key=lambda hitter: (
            hitter.walks / max(hitter.pa, 1),
            hitter.home_runs / max(hitter.pa, 1),
            hitter.pa,
        ),
        reverse=True,
    )
    starters = sorted(
        (pitcher for pitcher in pitcher_profiles if pitcher.role == "SP"),
        key=lambda pitcher: (
            pitcher.strikeouts / max(pitcher.innings_pitched, 1.0),
            -pitcher.walks / max(pitcher.innings_pitched, 1.0),
            pitcher.innings_pitched,
        ),
        reverse=True,
    )
    relievers = sorted(
        (pitcher for pitcher in pitcher_profiles if pitcher.role == "RP"),
        key=lambda pitcher: (
            pitcher.strikeouts / max(pitcher.innings_pitched, 1.0),
            -pitcher.walks / max(pitcher.innings_pitched, 1.0),
            pitcher.innings_pitched,
        ),
        reverse=True,
    )
    lineup = ordered_hitters[:9]
    bench = ordered_hitters[9:13]
    starter = starters[0]
    bullpen = relievers[:8]
    bullpen_roles = _default_bullpen_roles(bullpen)
    return TeamRoster(lineup=lineup, bench=bench, starter=starter, bullpen=bullpen, bullpen_roles=bullpen_roles)


def build_controlled_roster(database_path: Path | None = None) -> TeamRoster:
    return _build_identical_roster(database_path=database_path)


def _with_park_factor(roster: TeamRoster, park_factor: float) -> TeamRoster:
    return TeamRoster(
        lineup=roster.lineup,
        bench=roster.bench,
        starter=roster.starter,
        bullpen=roster.bullpen,
        park_factor=park_factor,
        bullpen_roles=roster.bullpen_roles,
    )


def _default_bullpen_roles(bullpen: list[PitcherProfile]) -> dict[str, str]:
    return {
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


def _bullpen_rest_for_roster(
    bullpen_rest_by_model: dict[str, dict[str, int]],
    model_name: str,
    roster: TeamRoster,
) -> dict[str, int]:
    rest = bullpen_rest_by_model.setdefault(model_name, {})
    for arm in roster.bullpen:
        rest.setdefault(arm.player_id, 100)
    stale_ids = set(rest) - {arm.player_id for arm in roster.bullpen}
    for player_id in stale_ids:
        rest.pop(player_id, None)
    return rest


def _recover_bullpen(rest: dict[str, int], roster: TeamRoster, recovery: int = 18) -> dict[str, int]:
    recovered = dict(rest)
    for arm in roster.bullpen:
        recovered[arm.player_id] = min(100, recovered.get(arm.player_id, 100) + recovery)
    return recovered


def _update_elo(winner: Standing, loser: Standing, k_factor: float = 20.0) -> None:
    expected = 1.0 / (1.0 + 10 ** ((loser.elo - winner.elo) / 400))
    winner.elo += k_factor * (1.0 - expected)
    loser.elo += k_factor * (0.0 - (1.0 - expected))


def _build_head_to_head(
    model_names: list[str],
    games: list[dict[str, object]],
) -> dict[str, dict[str, dict[str, object]]]:
    head_to_head: dict[str, dict[str, dict[str, object]]] = {
        model: {
            opponent: {
                "wins": 0,
                "losses": 0,
                "runs_for": 0,
                "runs_against": 0,
                "games": 0,
            }
            for opponent in model_names
            if opponent != model
        }
        for model in model_names
    }
    for game in games:
        home = str(game["home"])
        away = str(game["away"])
        home_score = int(game["home_score"])
        away_score = int(game["away_score"])
        winner = str(game["winner"])

        home_row = head_to_head[home][away]
        away_row = head_to_head[away][home]
        home_row["games"] = int(home_row["games"]) + 1
        away_row["games"] = int(away_row["games"]) + 1
        home_row["runs_for"] = int(home_row["runs_for"]) + home_score
        home_row["runs_against"] = int(home_row["runs_against"]) + away_score
        away_row["runs_for"] = int(away_row["runs_for"]) + away_score
        away_row["runs_against"] = int(away_row["runs_against"]) + home_score
        if winner == home:
            home_row["wins"] = int(home_row["wins"]) + 1
            away_row["losses"] = int(away_row["losses"]) + 1
        else:
            away_row["wins"] = int(away_row["wins"]) + 1
            home_row["losses"] = int(home_row["losses"]) + 1

    for opponents in head_to_head.values():
        for row in opponents.values():
            games_played = int(row["games"])
            row["win_pct"] = round(int(row["wins"]) / games_played, 3) if games_played else 0.0
            row["run_diff"] = int(row["runs_for"]) - int(row["runs_against"])
    return head_to_head


def _league_file_suffix(model_names: list[str]) -> str:
    raw = "-".join(model.replace("/", "-").replace(".", "-") for model in model_names)
    if len(raw) <= 100:
        return raw
    digest = __import__("hashlib").sha256(raw.encode("utf-8")).hexdigest()[:12]
    preview = "-".join(
        model.replace("/", "-").replace(".", "-")[:18].strip("-") or "model"
        for model in model_names[:2]
    )
    return f"{len(model_names)}m-{preview}-{digest}"


def _league_result_filename(model_names: list[str], league_kind: str) -> str:
    prefix = league_kind.replace("_", "-")
    return f"{prefix}-{_league_file_suffix(model_names)}.json"


def _league_progress_filename(model_names: list[str], league_kind: str) -> str:
    prefix = league_kind.replace("_", "-")
    return f"{prefix}-progress-{_league_file_suffix(model_names)}.json"


def _park_pool(database_path: Path | None = None) -> list[dict[str, object]]:
    with connect_read_only(database_path or DATA_DB_PATH) as conn:
        rows = conn.execute(
            """
            with league_context as (
              select avg(home_score + away_score) as league_avg_runs
              from games
              where season = 2025
            ),
            home_context as (
              select
                home_team_id as team_id,
                avg(home_score + away_score) as home_avg_runs
              from games
              where season = 2025
              group by home_team_id
            )
            select
              t.team_id,
              t.venue_name,
              case
                when lc.league_avg_runs is null or lc.league_avg_runs = 0 then 1.0
                else hc.home_avg_runs / lc.league_avg_runs
              end as park_factor
            from teams t
            join home_context hc on hc.team_id = t.team_id
            cross join league_context lc
            order by t.team_id
            """
        ).fetchall()
    return [
        {
            "team_id": row[0],
            "venue_name": row[1],
            "park_factor": round(min(max(float(row[2]), 0.85), 1.15), 3),
        }
        for row in rows
    ]


def _standings_table(standings: dict[str, Standing]) -> list[dict[str, object]]:
    return [
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


def _write_league_progress(
    path: Path,
    *,
    model_names: list[str],
    games_per_matchup: int,
    league_kind: str,
    started_at: str,
    status: str,
    standings: dict[str, Standing],
    games: list[dict[str, object]],
    total_games: int,
    current_game: dict[str, object] | None,
    eval_config: LeagueEvalConfig | None = None,
) -> None:
    completed_games = len(games)
    payload = {
        "kind": f"{league_kind}_progress",
        "league_kind": league_kind,
        "status": status,
        "started_at": started_at,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "models": model_names,
        "games_per_matchup": games_per_matchup,
        "total_games": total_games,
        "completed_games": completed_games,
        "remaining_games": max(total_games - completed_games, 0),
        "current_game": current_game,
        "last_completed_game": games[-1] if games else None,
        "standings": _standings_table(standings),
        "head_to_head": _build_head_to_head(model_names, games),
    }
    if eval_config is not None:
        payload["evaluation_mode"] = eval_config.evaluation_mode
        payload["eval_config"] = eval_config.as_metadata()
    write_json(path, payload)


def _build_schedule(
    managers,
    *,
    games_per_matchup: int,
    seed: int,
    league_games: int | None,
):
    if games_per_matchup < 1:
        raise ValueError("games_per_matchup must be at least 1.")
    if league_games is not None and league_games < 1:
        raise ValueError("league_games must be at least 1 when provided.")

    full_schedule = [
        (home_manager, away_manager)
        for home_manager in managers
        for away_manager in managers
        if home_manager != away_manager
        for _ in range(games_per_matchup)
    ]
    if league_games is None:
        return full_schedule, "full"

    sampled_schedule = list(full_schedule)
    Random(seed).shuffle(sampled_schedule)
    return sampled_schedule[: min(league_games, len(sampled_schedule))], "sampled"


def run_league(
    model_names: list[str],
    games_per_matchup: int = 6,
    seed: int = 7,
    league_games: int | None = None,
    league_kind: str = "controlled_league",
    rosters_by_model: dict[str, TeamRoster] | None = None,
    database_path: Path | None = None,
    output_dir: Path = RESULTS_DIR,
    write_latest: bool = True,
    eval_config: LeagueEvalConfig | None = None,
) -> dict[str, object]:
    eval_config = eval_config or LeagueEvalConfig()
    if len(model_names) < 2:
        raise ValueError("League requires at least two managers.")
    managers = [build_manager(name) for name in model_names]
    default_roster = _build_identical_roster(database_path=database_path)
    standings = {manager.name: Standing(name=manager.name) for manager in managers}
    games: list[dict[str, object]] = []
    bullpen_rest_by_model: dict[str, dict[str, int]] = {}
    schedule, schedule_mode = _build_schedule(
        managers,
        games_per_matchup=games_per_matchup,
        seed=seed,
        league_games=league_games,
    )
    total_games = len(schedule)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    park_pool = _park_pool(database_path=database_path)
    progress_path = output_dir / _league_progress_filename(model_names, league_kind)
    _write_league_progress(
        progress_path,
        model_names=model_names,
        games_per_matchup=games_per_matchup,
        league_kind=league_kind,
        started_at=started_at,
        status="running",
        standings=standings,
        games=games,
        total_games=total_games,
        current_game=None,
        eval_config=eval_config,
    )
    for game_index, (home_manager, away_manager) in enumerate(schedule, start=1):
        park = park_pool[(seed + game_index - 1) % len(park_pool)] if park_pool else {
            "team_id": "NEUTRAL",
            "venue_name": "Neutral Park",
            "park_factor": 1.0,
        }
        print(
            f"[baseball-bench] {league_kind} game {game_index}/{total_games} "
            f"{away_manager.name} at {home_manager.name}",
            flush=True,
        )
        current_game = {
            "game_number": game_index,
            "home": home_manager.name,
            "away": away_manager.name,
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "park_team_id": park["team_id"],
            "park_name": park["venue_name"],
            "park_factor": park["park_factor"],
        }
        _write_league_progress(
            progress_path,
            model_names=model_names,
            games_per_matchup=games_per_matchup,
            league_kind=league_kind,
            started_at=started_at,
            status="running",
            standings=standings,
            games=games,
            total_games=total_games,
            current_game=current_game,
            eval_config=eval_config,
        )
        home_roster = _with_park_factor(
            (rosters_by_model or {}).get(home_manager.name, default_roster),
            float(park["park_factor"]),
        )
        away_roster = (rosters_by_model or {}).get(away_manager.name, default_roster)
        result = simulate_game(
            home_manager,
            away_manager,
            home_roster,
            away_roster,
            seed + game_index,
            home_bullpen_rest=_bullpen_rest_for_roster(bullpen_rest_by_model, home_manager.name, home_roster),
            away_bullpen_rest=_bullpen_rest_for_roster(bullpen_rest_by_model, away_manager.name, away_roster),
            eval_config=eval_config,
            park_name=str(park["venue_name"]),
        )
        bullpen_rest_by_model[home_manager.name] = _recover_bullpen(result.home_bullpen_rest, home_roster)
        bullpen_rest_by_model[away_manager.name] = _recover_bullpen(result.away_bullpen_rest, away_roster)
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
                "live_model_call_count": sum(1 for item in result.decision_log if item.get("live_model_call")),
                "park_team_id": park["team_id"],
                "park_name": park["venue_name"],
                "park_factor": park["park_factor"],
                "home_bullpen_min_rest": min(bullpen_rest_by_model[home_manager.name].values(), default=100),
                "away_bullpen_min_rest": min(bullpen_rest_by_model[away_manager.name].values(), default=100),
            }
        )
        _write_league_progress(
            progress_path,
            model_names=model_names,
            games_per_matchup=games_per_matchup,
            league_kind=league_kind,
            started_at=started_at,
            status="running",
            standings=standings,
            games=games,
            total_games=total_games,
            current_game=None,
            eval_config=eval_config,
        )
    standings_table = _standings_table(standings)
    _write_league_progress(
        progress_path,
        model_names=model_names,
        games_per_matchup=games_per_matchup,
        league_kind=league_kind,
        started_at=started_at,
        status="complete",
        standings=standings,
        games=games,
        total_games=total_games,
        current_game=None,
        eval_config=eval_config,
    )
    summary = {
        "kind": league_kind,
        "league_kind": league_kind,
        "evaluation_mode": eval_config.evaluation_mode,
        "eval_config": eval_config.as_metadata(),
        "models": model_names,
        "games_per_matchup": games_per_matchup,
        "league_games": league_games,
        "schedule_mode": schedule_mode,
        "game_count": len(games),
        "average_decisions_per_game": mean(game["decision_count"] for game in games) if games else 0.0,
        "average_live_model_calls_per_game": mean(game["live_model_call_count"] for game in games) if games else 0.0,
        "standings": standings_table,
        "head_to_head": _build_head_to_head(model_names, games),
        "games": games,
    }
    filename = _league_result_filename(model_names, league_kind)
    write_json(output_dir / filename, summary)
    if write_latest and output_dir != RESULTS_DIR:
        write_json(RESULTS_DIR / filename, summary)
    return summary
