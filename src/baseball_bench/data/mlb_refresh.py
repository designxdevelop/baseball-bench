from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from baseball_bench.data.build import load_seed_data

BASE_URL = "https://statsapi.mlb.com/api/v1"
SEED_PATH = Path(__file__).resolve().parent / "seed" / "seed_data.json"


def _fetch_json(path: str, params: dict[str, object] | None = None) -> dict[str, object]:
    query = urlencode(params or {}, doseq=True, safe="(),[]")
    url = f"{BASE_URL}{path}"
    if query:
        url = f"{url}?{query}"
    with urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _as_int(value: object) -> int:
    if value in (None, "", ".---", "-.--"):
        return 0
    return int(float(str(value)))


def _as_float(value: object) -> float:
    if value in (None, "", ".---", "-.--"):
        return 0.0
    return float(str(value))


def _team_rows(season: int) -> tuple[list[dict[str, object]], dict[int, dict[str, str]]]:
    payload = _fetch_json("/teams", {"sportId": 1, "season": season})
    teams: list[dict[str, object]] = []
    team_map: dict[int, dict[str, str]] = {}
    for team in payload.get("teams", []):
        if not team.get("active", False):
            continue
        team_id = int(team["id"])
        league_id = int(team["league"]["id"])
        row = {
            "team_id": str(team.get("abbreviation") or team.get("teamCode") or team_id),
            "city": str(team.get("locationName") or team.get("name") or ""),
            "nickname": str(team.get("teamName") or team.get("clubName") or ""),
            "league": "AL" if league_id == 103 else "NL" if league_id == 104 else str(team["league"]["name"]),
            "venue_name": str(team.get("venue", {}).get("name", "")),
        }
        teams.append(row)
        team_map[team_id] = {
            "team_id": row["team_id"],
            "league": row["league"],
            "venue_name": row["venue_name"],
        }
    teams.sort(key=lambda item: str(item["team_id"]))
    return teams, team_map


def _player_row(person: dict[str, object], roster_entry: dict[str, object]) -> dict[str, object]:
    bat_side = person.get("batSide", {}) if isinstance(person.get("batSide"), dict) else {}
    pitch_hand = person.get("pitchHand", {}) if isinstance(person.get("pitchHand"), dict) else {}
    primary_position = person.get("primaryPosition", {}) if isinstance(person.get("primaryPosition"), dict) else {}
    roster_position = roster_entry.get("position", {}) if isinstance(roster_entry.get("position"), dict) else {}
    return {
        "player_id": str(person["id"]),
        "first_name": str(person.get("firstName", "")),
        "last_name": str(person.get("lastName", "")),
        "bats": str(bat_side.get("code", "R") or "R"),
        "throws": str(pitch_hand.get("code", "R") or "R"),
        "primary_position": str(primary_position.get("abbreviation") or roster_position.get("abbreviation") or "UT"),
    }


def _extract_stat_split(person: dict[str, object]) -> dict[str, object] | None:
    stats = person.get("stats")
    if not isinstance(stats, list) or not stats:
        return None
    splits = stats[0].get("splits")
    if not isinstance(splits, list) or not splits:
        return None
    first_split = splits[0]
    return first_split if isinstance(first_split, dict) else None


def _batting_rows_for_team(
    season: int,
    numeric_team_id: int,
    team_id: str,
    players: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    payload = _fetch_json(
        f"/teams/{numeric_team_id}/roster",
        {
            "season": season,
            "rosterType": "fullSeason",
            "hydrate": f"person(stats(type=[season],group=[hitting],season={season}))",
        },
    )
    rows: list[dict[str, object]] = []
    for roster_entry in payload.get("roster", []):
        person = roster_entry.get("person")
        if not isinstance(person, dict):
            continue
        player_id = str(person["id"])
        players.setdefault(player_id, _player_row(person, roster_entry))
        split = _extract_stat_split(person)
        if split is None:
            continue
        stat = split.get("stat", {})
        if not isinstance(stat, dict):
            continue
        plate_appearances = _as_int(stat.get("plateAppearances"))
        at_bats = _as_int(stat.get("atBats"))
        if plate_appearances <= 0 and at_bats <= 0:
            continue
        rows.append(
            {
                "season": season,
                "team_id": team_id,
                "player_id": player_id,
                "games": _as_int(stat.get("gamesPlayed")),
                "plate_appearances": plate_appearances,
                "at_bats": at_bats,
                "hits": _as_int(stat.get("hits")),
                "doubles": _as_int(stat.get("doubles")),
                "triples": _as_int(stat.get("triples")),
                "home_runs": _as_int(stat.get("homeRuns")),
                "walks": _as_int(stat.get("baseOnBalls")),
                "strikeouts": _as_int(stat.get("strikeOuts")),
                "hit_by_pitch": _as_int(stat.get("hitByPitch")),
                "sacrifice_flies": _as_int(stat.get("sacFlies")),
            }
        )
    return rows


def _pitching_rows_for_team(
    season: int,
    numeric_team_id: int,
    team_id: str,
    players: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    payload = _fetch_json(
        f"/teams/{numeric_team_id}/roster",
        {
            "season": season,
            "rosterType": "fullSeason",
            "hydrate": f"person(stats(type=[season],group=[pitching],season={season}))",
        },
    )
    rows: list[dict[str, object]] = []
    for roster_entry in payload.get("roster", []):
        person = roster_entry.get("person")
        if not isinstance(person, dict):
            continue
        player_id = str(person["id"])
        players.setdefault(player_id, _player_row(person, roster_entry))
        split = _extract_stat_split(person)
        if split is None:
            continue
        stat = split.get("stat", {})
        if not isinstance(stat, dict):
            continue
        innings_pitched = _as_float(stat.get("inningsPitched"))
        if innings_pitched <= 0:
            continue
        rows.append(
            {
                "season": season,
                "team_id": team_id,
                "player_id": player_id,
                "games": _as_int(stat.get("gamesPitched") or stat.get("gamesPlayed")),
                "games_started": _as_int(stat.get("gamesStarted")),
                "innings_pitched": innings_pitched,
                "hits_allowed": _as_int(stat.get("hits")),
                "earned_runs": _as_int(stat.get("earnedRuns")),
                "home_runs_allowed": _as_int(stat.get("homeRuns")),
                "walks": _as_int(stat.get("baseOnBalls")),
                "strikeouts": _as_int(stat.get("strikeOuts")),
            }
        )
    return rows


def _game_rows(season: int, team_map: dict[int, dict[str, str]]) -> list[dict[str, object]]:
    payload = _fetch_json("/schedule", {"sportId": 1, "season": season, "gameType": "R"})
    rows: list[dict[str, object]] = []
    seen_game_ids: set[str] = set()
    for date_entry in payload.get("dates", []):
        games = date_entry.get("games", [])
        if not isinstance(games, list):
            continue
        for game in games:
            status = game.get("status", {})
            if not isinstance(status, dict):
                continue
            if str(status.get("codedGameState")) != "F":
                continue
            game_id = str(game["gamePk"])
            if game_id in seen_game_ids:
                continue
            seen_game_ids.add(game_id)
            teams = game.get("teams", {})
            home = teams.get("home", {}) if isinstance(teams, dict) else {}
            away = teams.get("away", {}) if isinstance(teams, dict) else {}
            home_team = home.get("team", {}) if isinstance(home, dict) else {}
            away_team = away.get("team", {}) if isinstance(away, dict) else {}
            home_numeric = int(home_team["id"])
            away_numeric = int(away_team["id"])
            rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "game_date": str(game["officialDate"]),
                    "home_team_id": team_map[home_numeric]["team_id"],
                    "away_team_id": team_map[away_numeric]["team_id"],
                    "home_score": _as_int(home.get("score")),
                    "away_score": _as_int(away.get("score")),
                }
            )
    rows.sort(key=lambda item: (str(item["game_date"]), str(item["game_id"])))
    return rows


def build_mlb_seed_data(season: int = 2025) -> dict[str, object]:
    existing = load_seed_data()
    teams, team_map = _team_rows(season)
    players: dict[str, dict[str, object]] = {}
    batting: list[dict[str, object]] = []
    pitching: list[dict[str, object]] = []

    for numeric_team_id, team_info in sorted(team_map.items(), key=lambda item: item[1]["team_id"]):
        batting.extend(
            _batting_rows_for_team(season, numeric_team_id, team_info["team_id"], players)
        )
        pitching.extend(
            _pitching_rows_for_team(season, numeric_team_id, team_info["team_id"], players)
        )

    batting.sort(key=lambda item: (int(item["season"]), str(item["team_id"]), str(item["player_id"])))
    pitching.sort(key=lambda item: (int(item["season"]), str(item["team_id"]), str(item["player_id"])))
    games = _game_rows(season, team_map)
    player_rows = sorted(players.values(), key=lambda item: item["player_id"])

    return {
        "teams": teams,
        "players": player_rows,
        "batting": batting,
        "pitching": pitching,
        "games": games,
        "win_probabilities": existing["win_probabilities"],
    }


def refresh_seed_file(season: int = 2025, output_path: Path = SEED_PATH) -> Path:
    payload = build_mlb_seed_data(season=season)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path
