from __future__ import annotations

import json
from datetime import datetime
from html import escape
from math import ceil
from pathlib import Path

from baseball_bench.paths import LEADERBOARD_JSON_PATH, RESULTS_DIR, RUNS_DIR, SITE_DIR, SITE_INDEX_PATH
from baseball_bench.utils import write_json

TRACK_COUNT = 4
INTERNAL_MODELS = {"sql-baseline", "wp-baseline", "rulebook", "aggressive", "conservative"}


def _load_result_files(results_dir: Path) -> list[dict[str, object]]:
    payloads = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name in {"leaderboard.json", "data-checksum.json", "cost-estimate.json", "manifest.json"}:
            continue
        payloads.append(json.loads(path.read_text()))
    return payloads


def _ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _display_model_name(model: str) -> str:
    return model.removeprefix("openrouter/")


def _is_public_model(model: str) -> bool:
    return model not in INTERNAL_MODELS


def _analysis_track(payload: dict[str, object]) -> dict[str, object]:
    samples = payload.get("samples", [])
    correct = sum(1 for sample in samples if float(sample.get("score") or 0.0) >= 1.0)
    total = int(payload.get("sample_count", len(samples)))
    accuracy = float(payload.get("overall_accuracy", correct / total if total else 0.0))
    if accuracy >= 0.8:
        headline = "Strong baseball research"
    elif accuracy >= 0.6:
        headline = "Mostly reliable research"
    elif accuracy >= 0.4:
        headline = "Usable, but check the details"
    elif accuracy > 0.0:
        headline = "Needs fact-checking"
    else:
        headline = "Missed the research test"
    summary = f"Answered {correct} of {total} baseball research questions correctly."
    return {
        "status": "complete",
        "headline": headline,
        "summary": summary,
        "score": round(accuracy, 3),
        "metrics": {
            "accuracy": round(accuracy, 3),
            "correct": correct,
            "total": total,
        },
    }


def _decisions_track(payload: dict[str, object]) -> dict[str, object]:
    samples = payload.get("samples", [])
    near_optimal_count = sum(
        1
        for sample in samples
        if bool(
            sample.get("near_optimal")
            if "near_optimal" in sample
            else sample.get("metadata", {}).get("near_optimal")
        )
    )
    total = int(payload.get("sample_count", len(samples)))
    near_optimal_rate = float(
        payload.get(
            "near_optimal_rate",
            near_optimal_count / total if total else 0.0,
        )
    )
    mean_wp_delta = float(payload.get("mean_wp_delta", 0.0))
    if near_optimal_rate >= 0.99:
        headline = "Flawless late-game choices"
    elif near_optimal_rate >= 0.875:
        headline = "Very steady under pressure"
    elif near_optimal_rate >= 0.75:
        headline = "Usually found the right move"
    elif near_optimal_rate >= 0.5:
        headline = "Mixed results in big spots"
    else:
        headline = "Often missed the best move"
    summary = f"Chose the best or near-best move in {near_optimal_count} of {total} late-game situations."
    return {
        "status": "complete",
        "headline": headline,
        "summary": summary,
        "score": round(near_optimal_rate, 3),
        "metrics": {
            "near_optimal_rate": round(near_optimal_rate, 3),
            "near_optimal_count": near_optimal_count,
            "total": total,
            "mean_wp_delta": round(mean_wp_delta, 3),
        },
    }


def _league_track(
    row: dict[str, object],
    rank: int,
    team_count: int,
    run: dict[str, object],
) -> dict[str, object]:
    wins = int(row["wins"])
    losses = int(row["losses"])
    win_pct = float(row["win_pct"])
    run_diff = int(row["run_diff"])
    if rank == 1:
        headline = "Won the sim league"
    elif rank <= max(2, ceil(team_count / 3)):
        headline = "Looked like a contender"
    elif rank <= ceil(team_count * 2 / 3):
        headline = "Held the middle of the pack"
    else:
        headline = "Finished near the back"
    summary = (
        f"Finished {_ordinal(rank)} of {team_count} managers and went {wins}-{losses} "
        f"across {int(run['game_count'])} league games."
    )
    rank_score = 1.0 if team_count <= 1 else 1.0 - ((rank - 1) / (team_count - 1))
    return {
        "status": "complete",
        "headline": headline,
        "summary": summary,
        "score": round(rank_score, 3),
        "metrics": {
            "rank": rank,
            "team_count": team_count,
            "wins": wins,
            "losses": losses,
            "win_pct": round(win_pct, 3),
            "elo": float(row["elo"]),
            "run_diff": run_diff,
            "game_count": int(run["game_count"]),
        },
    }


def _gm_track(payload: dict[str, object]) -> dict[str, object]:
    scores = payload.get("scores", {})
    roster = payload.get("roster", {})
    validation = roster.get("validation", {}) if isinstance(roster, dict) else {}
    overall = float(scores.get("overall_score", 0.0))
    valid = bool(validation.get("valid", False))
    lineup = roster.get("lineup", []) if isinstance(roster, dict) else []
    rotation = roster.get("rotation", []) if isinstance(roster, dict) else []
    if overall >= 0.9 and valid:
        headline = "Built a balanced contender"
    elif overall >= 0.75 and valid:
        headline = "Strong roster construction"
    elif valid:
        headline = "Legal roster, mixed shape"
    else:
        headline = "Roster needs cleanup"
    summary = (
        f"Built a roster with {len(lineup)} lineup slots and {len(rotation)} rotation arms. "
        f"Validity score {float(scores.get('validity_score', 0.0)):.3f}."
    )
    return {
        "status": "complete",
        "headline": headline,
        "summary": summary,
        "score": round(overall, 3),
        "metrics": {
            "overall_score": round(overall, 3),
            "validity_score": round(float(scores.get("validity_score", 0.0)), 3),
            "lineup_score": round(float(scores.get("lineup_score", 0.0)), 3),
            "pitching_score": round(float(scores.get("pitching_score", 0.0)), 3),
            "balance_score": round(float(scores.get("balance_score", 0.0)), 3),
            "valid": valid,
        },
    }


def _missing_track(summary: str) -> dict[str, object]:
    return {
        "status": "missing",
        "headline": "Not run yet",
        "summary": summary,
        "score": None,
        "metrics": {},
    }


def _league_run_summary(run: dict[str, object]) -> dict[str, object]:
    standings = run.get("standings", [])
    public_standings = [
        row for row in standings if _is_public_model(str(row.get("model", "")))
    ]
    if not standings:
        headline = "League run has no standings yet."
    elif public_standings:
        leader = public_standings[0]
        headline = (
            f"{_display_model_name(str(leader['model']))} finished first in a "
            f"{len(public_standings)}-model league."
        )
    else:
        headline = "League run only contains internal calibration managers."
    return {
        "kind": str(run.get("kind", "league")),
        "league_kind": str(run.get("league_kind", run.get("kind", "league"))),
        "models": run.get("models", []),
        "public_models": [
            model for model in run.get("models", []) if _is_public_model(str(model))
        ],
        "games_per_matchup": run.get("games_per_matchup", 0),
        "league_games": run.get("league_games"),
        "schedule_mode": run.get("schedule_mode", "full"),
        "game_count": run.get("game_count", 0),
        "headline": headline,
        "standings": standings,
        "public_standings": public_standings,
        "head_to_head": run.get("head_to_head", {}),
    }


def _format_started_at(value: str) -> str:
    try:
        return datetime.fromisoformat(value).strftime("%b %d, %Y at %I:%M %p")
    except ValueError:
        return value


def _load_run_history(results_dir: Path) -> list[dict[str, object]]:
    if results_dir != RESULTS_DIR or not RUNS_DIR.exists():
        return []

    history = []
    for manifest_path in sorted(RUNS_DIR.glob("*/manifest.json"), reverse=True):
        manifest = json.loads(manifest_path.read_text())
        run_dir = manifest_path.parent
        history.append(
            {
                **manifest,
                "started_at_label": _format_started_at(str(manifest.get("started_at", ""))),
                "site_href": f"../runs/{run_dir.name}/site/index.html",
                "json_href": f"../runs/{run_dir.name}/leaderboard.json",
            }
        )
    return history


def _overall_copy(
    entry: dict[str, object],
    model_count: int,
    ranking_ready: bool,
) -> tuple[str, str]:
    completed_tracks = int(entry["completed_tracks"])
    missing_tracks = [name for name, track in entry["tracks"].items() if track["status"] != "complete"]
    best_track_name, best_track = max(
        entry["tracks"].items(),
        key=lambda item: item[1]["score"] if item[1]["score"] is not None else -1.0,
    )
    best_track_label = {
        "analysis": "research",
        "decisions": "game moves",
        "gm": "roster building",
        "league": "league play",
    }[best_track_name]

    if completed_tracks == 0:
        return "No results yet", "This model has not completed any tracks yet."
    if missing_tracks:
        headline = "Strong start" if best_track["score"] is not None and float(best_track["score"]) >= 0.75 else "Run in progress"
        summary = (
            f"Completed {completed_tracks} of {TRACK_COUNT} tracks. Best showing so far came in "
            f"{best_track_label}. Overall ranking is on hold until the controlled manager league finishes."
        )
        return headline, summary
    if not ranking_ready:
        return (
            "Run in progress",
            f"All public models have not completed every track yet. Best showing here came in {best_track_label}.",
        )

    rank = int(entry["rank"])
    if rank == 1:
        headline = "Best all-around model in this run"
    elif rank <= max(2, ceil(model_count / 3)):
        headline = "Strong all-around contender"
    elif rank <= ceil(model_count * 2 / 3):
        headline = "Competitive, but uneven"
    else:
        headline = "Fell behind the field"
    summary = (
        f"Finished {_ordinal(rank)} overall across research, game moves, and league play. "
        f"Best showing came in {best_track_label}."
    )
    return headline, summary


def _leader_stat_value(track_name: str, track: dict[str, object]) -> str:
    metrics = track["metrics"]
    if track_name == "analysis":
        return _avg(metrics["accuracy"])
    if track_name == "decisions":
        return _avg(metrics["near_optimal_rate"])
    if track_name == "gm":
        return _avg(metrics["overall_score"])
    if track_name == "league":
        return _avg(metrics["win_pct"])
    return "&mdash;"


def _track_leader_summary(entries: list[dict[str, object]], track_name: str) -> dict[str, str]:
    label = {
        "analysis": "Research AVG",
        "decisions": "Decision AVG",
        "gm": "GM Score",
        "league": "League PCT",
    }[track_name]
    complete = [entry for entry in entries if entry["tracks"][track_name]["status"] == "complete"]
    if not complete:
        return {
            "label": label,
            "value": "&mdash;",
            "leader": "Not finished yet",
            "headline": "Not finished yet",
            "summary": f"{label} has not finished for the current public-model snapshot.",
        }
    top_score = max(float(entry["tracks"][track_name]["score"]) for entry in complete)
    leaders = [entry for entry in complete if float(entry["tracks"][track_name]["score"]) == top_score]
    value = _leader_stat_value(track_name, leaders[0]["tracks"][track_name])
    if len(leaders) == 1:
        leader = leaders[0]
        return {
            "label": label,
            "value": value,
            "leader": str(leader["display_name"]),
            "headline": f"Leader: {leader['display_name']}",
            "summary": str(leader["tracks"][track_name]["summary"]),
        }
    return {
        "label": label,
        "value": value,
        "leader": f"{len(leaders)}-way tie",
        "headline": f"{len(leaders)}-way tie",
        "summary": str(leaders[0]["tracks"][track_name]["summary"]),
    }


def _aggregate(payloads: list[dict[str, object]]) -> dict[str, object]:
    models: dict[str, dict[str, object]] = {}
    league_runs: list[dict[str, object]] = []
    open_league_runs: list[dict[str, object]] = []
    for payload in payloads:
        kind = payload.get("kind")
        if kind == "analysis":
            model = str(payload["model"])
            entry = models.setdefault(model, {"model": model})
            entry["analysis"] = _analysis_track(payload)
        elif kind == "decisions":
            model = str(payload["model"])
            entry = models.setdefault(model, {"model": model})
            entry["decisions"] = _decisions_track(payload)
        elif kind == "gm_roster":
            model = str(payload["model"])
            entry = models.setdefault(model, {"model": model})
            entry["gm"] = _gm_track(payload)
        elif kind in {"league", "controlled_league"}:
            league_runs.append(_league_run_summary(payload))
            standings = payload.get("standings", [])
            public_standings = [
                row for row in standings if _is_public_model(str(row.get("model", "")))
            ]
            public_ranks = {
                str(row["model"]): index
                for index, row in enumerate(public_standings, start=1)
            }
            for index, row in enumerate(standings, start=1):
                model = str(row["model"])
                entry = models.setdefault(model, {"model": model})
                if _is_public_model(model):
                    rank = public_ranks[model]
                    team_count = len(public_standings)
                else:
                    rank = index
                    team_count = len(standings)
                entry["league"] = _league_track(row, rank, team_count, payload)
        elif kind == "open_league":
            open_league_runs.append(_league_run_summary(payload))

    table = []
    for model, entry in models.items():
        tracks = {
            "analysis": entry.get(
                "analysis",
                _missing_track("This model has not completed the research track yet."),
            ),
            "decisions": entry.get(
                "decisions",
                _missing_track("This model has not completed the late-game decision track yet."),
            ),
            "gm": entry.get(
                "gm",
                _missing_track("This model has not completed the GM roster-building track yet."),
            ),
            "league": entry.get(
                "league",
                _missing_track("Controlled manager league play has not run yet for this model."),
            ),
        }
        completed_scores = [
            float(track["score"])
            for track in tracks.values()
            if track["score"] is not None
        ]
        completed_tracks = len(completed_scores)
        overall_score = round(sum(completed_scores) / completed_tracks, 3) if completed_scores else 0.0
        table.append(
            {
                "model": model,
                "display_name": _display_model_name(model),
                "tracks": tracks,
                "completed_tracks": completed_tracks,
                "overall_score": overall_score,
            }
        )

    table.sort(
        key=lambda item: (
            int(item["completed_tracks"]),
            float(item["overall_score"]),
            item["display_name"],
        ),
        reverse=True,
    )

    public_models = [entry for entry in table if _is_public_model(str(entry["model"]))]
    internal_models = [entry for entry in table if not _is_public_model(str(entry["model"]))]
    ranking_ready = bool(public_models) and all(
        int(entry["completed_tracks"]) == TRACK_COUNT for entry in public_models
    )

    for index, entry in enumerate(public_models, start=1):
        entry["rank"] = index if ranking_ready else None
    for entry in public_models:
        headline, summary = _overall_copy(entry, len(public_models), ranking_ready)
        entry["overall_read"] = headline
        entry["overall_summary"] = summary
    for index, entry in enumerate(internal_models, start=1):
        entry["rank"] = index
    for entry in internal_models:
        headline, summary = _overall_copy(entry, len(internal_models), True)
        entry["overall_read"] = headline
        entry["overall_summary"] = summary

    public_league_runs = [
        run for run in league_runs if any(_is_public_model(str(model)) for model in run.get("models", []))
    ]

    return {
        "models": public_models,
        "internal_models": internal_models,
        "league_runs": public_league_runs,
        "track_labels": {
            "analysis": "Research",
            "decisions": "Game Moves",
            "gm": "GM Build",
            "league": "Manager League",
        },
        "overall_ranking_ready": ranking_ready,
        "track_leaders": [
            _track_leader_summary(public_models, "analysis"),
            _track_leader_summary(public_models, "decisions"),
            _track_leader_summary(public_models, "gm"),
            _track_leader_summary(public_models, "league"),
        ],
        "open_league_runs": open_league_runs,
    }


def _avg(value: float | None) -> str:
    """Format a 0-1 rate as an MLB-style average: .800, 1.000, .000."""
    if value is None:
        return "&mdash;"
    text = f"{float(value):.3f}"
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    return text


def _signed(value: float | int | None) -> str:
    if value is None:
        return "&mdash;"
    return f"{int(value):+d}"


def _track_stat(name: str, track: dict[str, object]) -> tuple[str, str]:
    """Return (primary stat value, supporting detail) for a track."""
    if track["status"] != "complete":
        return "&mdash;", "Not run yet"
    metrics = track["metrics"]
    if name == "analysis":
        return _avg(metrics["accuracy"]), f"{int(metrics['correct'])}/{int(metrics['total'])} correct"
    if name == "decisions":
        detail = f"{int(metrics['near_optimal_count'])}/{int(metrics['total'])} optimal moves"
        return _avg(metrics["near_optimal_rate"]), detail
    if name == "gm":
        detail = (
            f"{_avg(metrics['lineup_score'])} lineup &middot; "
            f"{_avg(metrics['pitching_score'])} pitching"
        )
        return _avg(metrics["overall_score"]), detail
    if name == "league":
        record = f"{int(metrics['wins'])}-{int(metrics['losses'])}"
        detail = (
            f"{_avg(metrics['win_pct'])} PCT &middot; {_signed(metrics['run_diff'])} run diff "
            f"&middot; {round(float(metrics['elo']))} ELO"
        )
        return record, detail
    return "&mdash;", ""


def _stat_columns(spec: list[tuple[str, bool]]) -> str:
    return "".join(
        f'<th class="num">{escape(label)}</th>' if numeric else f"<th>{escape(label)}</th>"
        for label, numeric in spec
    )


def _stat_cell(value: str, numeric: bool) -> str:
    return f'<td class="num">{value}</td>' if numeric else f"<td>{value}</td>"


def _render_stat_table(
    columns: list[tuple[str, bool]],
    rows: list[list[str]],
    *,
    empty: str = "No results yet.",
) -> str:
    if not rows:
        return f'<div class="surface"><p class="empty-state">{escape(empty)}</p></div>'
    body = "".join(
        "<tr>" + "".join(_stat_cell(cell, columns[i][1]) for i, cell in enumerate(row)) + "</tr>"
        for row in rows
    )
    return (
        '<div class="surface table-wrap">'
        f"<table><thead><tr>{_stat_columns(columns)}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def _render_track_card(name: str, label: str, track: dict[str, object]) -> str:
    complete = track["status"] == "complete"
    status_class = "track track--complete" if complete else "track track--missing"
    value, detail = _track_stat(name, track)
    return (
        f'<section class="{status_class}">'
        f'<p class="track__label">{escape(label)}</p>'
        f'<p class="track__stat">{value}</p>'
        f'<p class="track__detail">{detail}</p>'
        f'<p class="track__read">{escape(str(track["headline"]))}</p>'
        "</section>"
    )


def _render_model_card(entry: dict[str, object], track_labels: dict[str, str]) -> str:
    tracks_html = "\n".join(
        _render_track_card(name, track_labels[name], entry["tracks"][name])
        for name in ("analysis", "decisions", "gm", "league")
    )
    if entry["rank"] is not None:
        rank = int(entry["rank"])
        medal = {1: "gold", 2: "silver", 3: "bronze"}.get(rank, "default")
        rank_markup = f'<span class="rank-badge rank-badge--{medal}">{rank}</span>'
    else:
        rank_markup = '<span class="rank-badge rank-badge--pending">TBD</span>'
    return f"""
      <article class="card">
        <div class="card__header">
          <div class="card__id">
            {rank_markup}
            <div>
              <h2>{escape(str(entry['display_name']))}</h2>
              <p class="card__verdict">{escape(str(entry['overall_read']))}</p>
            </div>
          </div>
          <div class="card__ovr">
            <p class="card__ovr-value">{_avg(entry['overall_score'])}</p>
            <p class="card__ovr-label">OVR &middot; {entry['completed_tracks']}/{TRACK_COUNT} tracks</p>
          </div>
        </div>
        <p class="card__summary">{escape(str(entry['overall_summary']))}</p>
        <div class="track-grid">
          {tracks_html}
        </div>
      </article>
    """


def _render_comparison_table(entries: list[dict[str, object]], ranking_ready: bool) -> str:
    columns = [
        ("RK", True),
        ("Manager", False),
        ("OVR", True),
        ("RES", True),
        ("DEC", True),
        ("GM", True),
        ("W-L", True),
        ("PCT", True),
        ("DIFF", True),
        ("ELO", True),
        ("TRK", True),
    ]
    rows: list[list[str]] = []
    for index, entry in enumerate(entries, start=1):
        rank = entry["rank"]
        rank_cell = f"{int(rank)}" if rank is not None else f'<span class="tbd">{index}</span>'
        analysis = entry["tracks"]["analysis"]
        decisions = entry["tracks"]["decisions"]
        gm = entry["tracks"]["gm"]
        league = entry["tracks"]["league"]
        res = _avg(analysis["metrics"]["accuracy"]) if analysis["status"] == "complete" else "&mdash;"
        dec = _avg(decisions["metrics"]["near_optimal_rate"]) if decisions["status"] == "complete" else "&mdash;"
        gm_score = _avg(gm["metrics"]["overall_score"]) if gm["status"] == "complete" else "&mdash;"
        if league["status"] == "complete":
            lm = league["metrics"]
            record = f"{int(lm['wins'])}-{int(lm['losses'])}"
            pct = _avg(lm["win_pct"])
            diff = _signed(lm["run_diff"])
            elo = f"{round(float(lm['elo']))}"
        else:
            record = pct = diff = elo = "&mdash;"
        rows.append(
            [
                rank_cell,
                escape(str(entry["display_name"])),
                f'<strong>{_avg(entry["overall_score"])}</strong>',
                res,
                dec,
                gm_score,
                record,
                pct,
                diff,
                elo,
                f"{int(entry['completed_tracks'])}/{TRACK_COUNT}",
            ]
        )
    rank_note = (
        "<p class=\"section__note\" style=\"margin:0 0 14px\">Final order &mdash; every public model has completed research, decisions, GM build, and controlled manager league. OVR is the mean of those ratings.</p>"
        if ranking_ready
        else "<p class=\"section__note\" style=\"margin:0 0 14px\">Provisional order &mdash; controlled manager league or GM scoring has not finished. OVR is the mean of completed track ratings.</p>"
    )
    return rank_note + _render_stat_table(columns, rows, empty="No model results yet.")


def _latest_public_league_run(league_runs: list[dict[str, object]]) -> dict[str, object] | None:
    public_runs = [run for run in league_runs if run.get("public_standings")]
    return public_runs[-1] if public_runs else None


def _format_matchup_cell(row: dict[str, object] | None) -> str:
    if not row or int(row.get("games", 0)) == 0:
        return "-"
    return (
        f"{int(row['wins'])}-{int(row['losses'])} "
        f"<span>{int(row['run_diff']):+d}</span>"
    )


def _render_league_comparison(run: dict[str, object] | None) -> str:
    if not run:
        return """
        <div class="section__head">
          <div>
            <p class="track__label">Manager League</p>
            <h2 class="section__title">Head-to-head matchups pending</h2>
          </div>
        </div>
        <div class="notes">
          <p>Run the OpenRouter pack through league play to compare each model as a manager against the rest of the field.</p>
        </div>
        """

    standings = run["public_standings"]
    models = [str(row["model"]) for row in standings]
    head_to_head = run.get("head_to_head", {})
    matchup_headers = "".join(f"<th>{escape(_display_model_name(model))}</th>" for model in models)
    rows = []
    for index, standing in enumerate(standings, start=1):
        model = str(standing["model"])
        matchup_cells = []
        for opponent in models:
            if opponent == model:
                matchup_cells.append("<td class=\"self-matchup\">-</td>")
            else:
                row = head_to_head.get(model, {}).get(opponent)
                matchup_cells.append(f"<td>{_format_matchup_cell(row)}</td>")
        rows.append(
            "<tr>"
            f"<td>#{index}</td>"
            f"<td>{escape(_display_model_name(model))}</td>"
            f"<td>{int(standing['wins'])}-{int(standing['losses'])}</td>"
            f"<td>{int(standing['run_diff']):+d}</td>"
            f"{''.join(matchup_cells)}"
            "</tr>"
        )
    return f"""
      <div class="league-board__header">
        <div>
          <p class="track__label">Manager League</p>
          <h2>Head-to-head model standings</h2>
          <p>{escape(str(run['headline']))} Matchup cells show record and run differential against that opponent.</p>
        </div>
        <span class="badge-games">{int(run['game_count'])} games</span>
      </div>
      <div class="surface table-wrap">
        <table>
          <thead><tr><th>Rank</th><th>Model</th><th>Record</th><th>Run Diff</th>{matchup_headers}</tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    """


def _render_legend() -> str:
    terms = [
        ("OVR", "Overall rating", "Mean of a model's completed track ratings, shown like a batting average (.000&ndash;1.000)."),
        ("RES", "Research", "Share of baseball research questions answered correctly."),
        ("DEC", "Decisions", "Share of late-game situations where the model picked the best or near-best move."),
        ("W-L", "League record", "Wins and losses as a manager in head-to-head league play."),
        ("PCT", "Win percentage", "League winning percentage (wins &divide; games)."),
        ("DIFF", "Run differential", "Runs scored minus runs allowed across league games."),
        ("ELO", "Manager rating", "Skill rating from league results; everyone starts at 1500 and higher is stronger."),
        ("TRK", "Tracks complete", "How many of the 3 benchmark tracks this model has finished."),
    ]
    items = "".join(
        (
            '<div class="legend__item">'
            f'<span class="legend__abbr">{abbr}</span>'
            f'<div><p class="legend__term">{term}</p>'
            f'<p class="legend__desc">{desc}</p></div>'
            "</div>"
        )
        for abbr, term, desc in terms
    )
    return (
        '<details class="legend surface" open>'
        "<summary>How to read the stats</summary>"
        f'<div class="legend__grid">{items}</div>'
        "</details>"
    )


def _render_history_entry(run: dict[str, object]) -> str:
    model_count = len(run.get("models", []))
    models_label = f"{model_count} models" if model_count != 1 else "1 model"
    tracks = run.get("tracks_completed", [])
    tracks_label = ", ".join(tracks) if tracks else "no completed tracks recorded"
    return (
        "<li>"
        f"<strong>{escape(str(run.get('started_at_label', 'Unknown time')))}</strong> "
        f"<span>{escape(str(run.get('label', 'Benchmark run')))} | "
        f"{escape(models_label)} | {escape(tracks_label)}</span> "
        f"<a href=\"{escape(str(run['site_href']))}\">snapshot</a> "
        f"<a href=\"{escape(str(run['json_href']))}\">json</a>"
        "</li>"
    )


def build_site(
    results_dir: Path = RESULTS_DIR,
    leaderboard_json_path: Path = LEADERBOARD_JSON_PATH,
    site_dir: Path = SITE_DIR,
    site_index_path: Path = SITE_INDEX_PATH,
) -> dict[str, object]:
    payloads = _load_result_files(results_dir)
    leaderboard = _aggregate(payloads)
    history = _load_run_history(results_dir)
    leaderboard["run_history"] = history

    write_json(leaderboard_json_path, leaderboard)
    site_dir.mkdir(parents=True, exist_ok=True)

    cards = "\n".join(
        _render_model_card(entry, leaderboard["track_labels"])
        for entry in leaderboard["models"]
    )
    comparison_table = _render_comparison_table(
        leaderboard["models"],
        bool(leaderboard["overall_ranking_ready"]),
    )
    legend = _render_legend()
    leader_panels = "\n".join(
        (
            "<section class=\"leader-panel\">"
            f"<p class=\"track__label\">{escape(str(item['label']))}</p>"
            f"<p class=\"leader-panel__value\">{item['value']}</p>"
            f"<p class=\"leader-panel__name\">{escape(str(item['leader']))}</p>"
            f"<p class=\"leader-panel__note\">{escape(str(item['summary']))}</p>"
            "</section>"
        )
        for item in leaderboard["track_leaders"]
    )
    league_notes = "\n".join(
        (
            f"<li><strong>{escape(str(run['headline']))}</strong> "
            f"<span>{int(run['game_count'])} games</span></li>"
        )
        for run in leaderboard["league_runs"]
    ) or "<li><strong>No completed public-model league snapshot yet.</strong></li>"
    history_notes = "\n".join(_render_history_entry(run) for run in history) or "<li><strong>No saved benchmark snapshots yet.</strong></li>"
    league_comparison = _render_league_comparison(
        _latest_public_league_run(leaderboard["league_runs"])
    )
    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>baseball-bench leaderboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700;800;900&family=Roboto+Condensed:wght@600;700;800&display=swap" rel="stylesheet">
    <style>
      :root {{
        color-scheme: light;
        --navy: #041e42;
        --navy-2: #002d72;
        --red: #bf0d3e;
        --red-2: #e4002b;
        --ink: #11161d;
        --muted: #5c6b7a;
        --line: #e3e8ef;
        --line-strong: #cdd6e0;
        --bg: #eef1f6;
        --surface: #ffffff;
        --gold: #c89b3c;
        --silver: #9aa6b2;
        --bronze: #b06a3a;
        font-family: "Archivo", "Helvetica Neue", Arial, sans-serif;
      }}
      * {{ box-sizing: border-box; }}
      html {{ scroll-behavior: smooth; }}
      body {{
        margin: 0;
        color: var(--ink);
        background: var(--bg);
        -webkit-font-smoothing: antialiased;
      }}
      h1, h2, h3 {{ font-family: "Archivo", "Helvetica Neue", Arial, sans-serif; }}
      .display {{ font-family: "Roboto Condensed", "Archivo", sans-serif; }}

      /* ---- Top bar ---- */
      .topbar {{
        position: sticky;
        top: 0;
        z-index: 20;
        background: var(--navy);
        border-bottom: 3px solid var(--red);
      }}
      .topbar__inner {{
        max-width: 1180px;
        margin: 0 auto;
        padding: 0 20px;
        height: 58px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
      }}
      .wordmark {{
        display: flex;
        align-items: center;
        gap: 12px;
        color: #fff;
        font-weight: 800;
        letter-spacing: 0.16em;
        font-size: 0.95rem;
        text-transform: uppercase;
      }}
      .wordmark__mark {{
        display: inline-grid;
        place-items: center;
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: #fff;
        color: var(--navy);
        font-family: "Roboto Condensed", sans-serif;
        font-weight: 800;
        font-size: 0.9rem;
        letter-spacing: 0;
        box-shadow: inset 0 0 0 2px var(--red);
      }}
      .topbar__nav {{ display: flex; gap: 22px; }}
      .topbar__nav a {{
        color: rgba(255,255,255,0.78);
        text-decoration: none;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        padding: 4px 0;
        border-bottom: 2px solid transparent;
        transition: color .15s, border-color .15s;
      }}
      .topbar__nav a:hover {{ color: #fff; border-color: var(--red); }}

      main {{ max-width: 1180px; margin: 0 auto; padding: 0 20px 72px; }}

      /* ---- Hero ---- */
      .hero {{
        position: relative;
        overflow: hidden;
        margin: 24px 0 8px;
        padding: 56px 40px 60px;
        border-radius: 14px;
        color: #fff;
        background:
          radial-gradient(circle at 88% -20%, rgba(191,13,62,0.55), transparent 42%),
          linear-gradient(135deg, var(--navy) 0%, var(--navy-2) 60%, #00204f 100%);
        box-shadow: 0 24px 60px rgba(4, 30, 66, 0.28);
      }}
      .hero::after {{
        content: "";
        position: absolute;
        right: -120px;
        top: -120px;
        width: 360px;
        height: 360px;
        border-radius: 50%;
        background: radial-gradient(circle, transparent 58%, rgba(255,255,255,0.06) 59%, transparent 61%),
                    radial-gradient(circle, transparent 70%, rgba(255,255,255,0.05) 71%, transparent 73%);
        pointer-events: none;
      }}
      .hero__content {{ position: relative; z-index: 1; }}
      .eyebrow {{
        margin: 0 0 14px;
        color: #fff;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        gap: 10px;
      }}
      .eyebrow::before {{
        content: "";
        width: 26px; height: 3px;
        background: var(--red-2);
        display: inline-block;
      }}
      h1 {{
        margin: 0;
        font-family: "Roboto Condensed", sans-serif;
        font-weight: 800;
        font-size: clamp(2.6rem, 6vw, 5.2rem);
        line-height: 0.92;
        letter-spacing: -0.01em;
        text-transform: uppercase;
      }}
      .hero__sub {{ margin: 18px 0 0; max-width: 64ch; color: rgba(255,255,255,0.82); font-size: 1.05rem; line-height: 1.5; }}

      /* ---- Section scaffolding ---- */
      .section {{ margin-top: 40px; }}
      .section__head {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 2px solid var(--line-strong);
      }}
      .section__title {{
        margin: 0;
        font-family: "Roboto Condensed", sans-serif;
        font-weight: 800;
        font-size: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 0.02em;
        position: relative;
        padding-left: 16px;
      }}
      .section__title::before {{
        content: "";
        position: absolute;
        left: 0; top: 4px; bottom: 4px;
        width: 5px;
        background: var(--red);
        border-radius: 2px;
      }}
      .section__note {{ margin: 0; color: var(--muted); font-size: 0.9rem; }}

      /* ---- Tables (MLB stat style) ---- */
      .surface {{
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 12px;
        box-shadow: 0 6px 22px rgba(17, 22, 29, 0.05);
        overflow: hidden;
      }}
      .table-wrap {{ overflow-x: auto; }}
      table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
      thead th {{
        background: var(--navy);
        color: #fff;
        text-align: left;
        padding: 13px 14px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        white-space: nowrap;
        position: sticky;
        top: 0;
      }}
      tbody td {{
        padding: 13px 14px;
        border-bottom: 1px solid var(--line);
        vertical-align: middle;
      }}
      tbody tr:nth-child(even) {{ background: #f7f9fc; }}
      tbody tr:hover {{ background: #eef3fb; }}
      tbody tr:last-child td {{ border-bottom: none; }}
      td span {{ color: var(--muted); font-variant-numeric: tabular-nums; }}
      .self-matchup {{ color: #b9c2cd; text-align: center; }}

      /* ---- Leader panels ---- */
      .leader-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 16px;
        margin-bottom: 18px;
      }}
      .leader-panel {{
        padding: 18px 18px 20px;
        border-radius: 12px;
        background: var(--surface);
        border: 1px solid var(--line);
        border-top: 4px solid var(--red);
        box-shadow: 0 6px 22px rgba(17, 22, 29, 0.05);
      }}
      .leader-panel .track__label {{ color: var(--navy-2); }}
      .leader-panel h3 {{ margin: 6px 0 8px; font-size: 1.15rem; }}
      .leader-panel p {{ margin: 0; color: var(--muted); font-size: 0.9rem; line-height: 1.45; }}

      .track__label {{
        margin: 0 0 6px;
        color: var(--red);
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
      }}

      /* ---- League board ---- */
      .league-board__header {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 18px;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 2px solid var(--line-strong);
      }}
      .league-board__header h2 {{
        margin: 4px 0 6px;
        font-family: "Roboto Condensed", sans-serif;
        font-weight: 800;
        font-size: 1.5rem;
        text-transform: uppercase;
      }}
      .league-board__header p {{ margin: 0; color: var(--muted); max-width: 62ch; font-size: 0.9rem; }}

      .badge-games {{
        flex: none;
        align-self: flex-start;
        background: var(--navy);
        color: #fff;
        font-family: "Roboto Condensed", sans-serif;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 8px 14px;
        border-radius: 999px;
        white-space: nowrap;
      }}

      /* ---- Model cards ---- */
      .cards {{ display: grid; gap: 18px; }}
      .card {{
        position: relative;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 24px 24px 26px;
        background: var(--surface);
        box-shadow: 0 8px 26px rgba(17, 22, 29, 0.06);
        border-left: 5px solid var(--navy);
      }}
      .card__header {{
        display: flex;
        gap: 18px;
        justify-content: space-between;
        align-items: flex-start;
      }}
      .card__id {{ display: flex; gap: 16px; align-items: center; }}
      .rank-badge {{
        flex: none;
        display: inline-grid;
        place-items: center;
        width: 52px; height: 52px;
        border-radius: 50%;
        font-family: "Roboto Condensed", sans-serif;
        font-weight: 800;
        font-size: 1.5rem;
        color: #fff;
        background: var(--navy);
        box-shadow: inset 0 0 0 3px rgba(255,255,255,0.85), 0 4px 10px rgba(4,30,66,0.25);
      }}
      .rank-badge--gold {{ background: linear-gradient(160deg, #e6c25a, #b8862a); color: #2c2200; }}
      .rank-badge--silver {{ background: linear-gradient(160deg, #cdd6df, #93a0ad); color: #1d2630; }}
      .rank-badge--bronze {{ background: linear-gradient(160deg, #d08a52, #97552c); color: #2a1607; }}
      .rank-badge--pending {{
        background: #fff; color: var(--muted);
        box-shadow: inset 0 0 0 2px var(--line-strong);
        font-size: 0.78rem; letter-spacing: 0.08em;
      }}
      h2 {{ margin: 0; font-size: 1.55rem; font-weight: 800; letter-spacing: -0.01em; }}
      .card__verdict {{
        margin: 4px 0 0;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--red);
      }}
      .card__coverage {{
        margin: 0;
        flex: none;
        padding: 8px 14px;
        border-radius: 999px;
        background: #f1f4f9;
        border: 1px solid var(--line);
        color: var(--navy-2);
        font-size: 0.78rem;
        font-weight: 700;
        white-space: nowrap;
      }}
      .card__summary {{ margin: 16px 0 0; color: var(--muted); max-width: 78ch; line-height: 1.5; }}
      .track-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin-top: 20px;
      }}
      .track {{
        padding: 16px;
        border-radius: 10px;
        border: 1px solid var(--line);
        background: #f7f9fc;
      }}
      .track--complete {{ border-top: 3px solid #1f8a4c; }}
      .track--missing {{ border-top: 3px solid var(--line-strong); opacity: 0.85; }}
      .track h3 {{ margin: 0; font-size: 1.02rem; font-weight: 700; }}
      .track p:last-child {{ margin: 8px 0 0; color: var(--muted); font-size: 0.86rem; line-height: 1.45; }}

      /* ---- Notes ---- */
      .notes {{
        padding: 22px 24px;
        border-radius: 12px;
        background: var(--surface);
        border: 1px solid var(--line);
        box-shadow: 0 6px 22px rgba(17, 22, 29, 0.05);
      }}
      .notes ul {{ margin: 0; padding: 0; list-style: none; }}
      .notes li {{
        padding: 12px 0;
        border-bottom: 1px solid var(--line);
        color: var(--muted);
        display: flex;
        flex-wrap: wrap;
        align-items: baseline;
        gap: 8px;
      }}
      .notes li:last-child {{ border-bottom: none; }}
      .notes li strong {{ color: var(--ink); }}
      .notes a {{
        color: var(--navy-2);
        font-weight: 700;
        text-decoration: none;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-left: auto;
        padding: 3px 10px;
        border: 1px solid var(--line-strong);
        border-radius: 999px;
      }}
      .notes a:hover {{ background: var(--navy); color: #fff; border-color: var(--navy); }}
      .notes p {{ margin: 0; color: var(--muted); }}

      /* ---- Numeric stat display ---- */
      th.num, td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
      thead th.num {{ text-align: right; }}
      td.num {{ font-family: "Roboto Condensed", sans-serif; font-weight: 600; letter-spacing: 0.01em; }}
      td.num strong {{ font-weight: 800; color: var(--navy-2); }}
      .tbd {{ color: var(--muted); font-weight: 600; }}
      .empty-state {{ padding: 22px; margin: 0; color: var(--muted); }}

      .leader-panel__value {{
        margin: 8px 0 2px;
        font-family: "Roboto Condensed", sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        line-height: 1;
        color: var(--navy);
        font-variant-numeric: tabular-nums;
      }}
      .leader-panel__name {{ margin: 0; font-weight: 700; font-size: 0.95rem; }}
      .leader-panel__note {{ margin: 6px 0 0; color: var(--muted); font-size: 0.85rem; line-height: 1.4; }}

      .card__ovr {{ flex: none; text-align: right; }}
      .card__ovr-value {{
        margin: 0;
        font-family: "Roboto Condensed", sans-serif;
        font-weight: 800;
        font-size: 2.4rem;
        line-height: 0.9;
        color: var(--navy);
        font-variant-numeric: tabular-nums;
      }}
      .card__ovr-label {{
        margin: 4px 0 0;
        font-size: 0.66rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
      }}
      .track__stat {{
        margin: 6px 0 2px;
        font-family: "Roboto Condensed", sans-serif;
        font-weight: 800;
        font-size: 1.9rem;
        line-height: 1;
        color: var(--navy);
        font-variant-numeric: tabular-nums;
      }}
      .track--missing .track__stat {{ color: var(--line-strong); }}
      .track__detail {{ margin: 0; font-size: 0.82rem; color: var(--muted); }}
      .track__read {{
        margin: 10px 0 0;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--ink);
      }}
      .track--missing .track__read {{ color: var(--muted); }}

      /* ---- Stat legend ---- */
      .legend {{ margin-top: 16px; padding: 0; }}
      .legend > summary {{
        list-style: none;
        cursor: pointer;
        padding: 14px 20px;
        font-family: "Roboto Condensed", sans-serif;
        font-weight: 700;
        font-size: 0.82rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--navy);
        display: flex;
        align-items: center;
        gap: 10px;
      }}
      .legend > summary::-webkit-details-marker {{ display: none; }}
      .legend > summary::before {{
        content: "+";
        display: inline-grid;
        place-items: center;
        width: 18px; height: 18px;
        border-radius: 4px;
        background: var(--navy);
        color: #fff;
        font-size: 0.9rem;
        line-height: 1;
      }}
      .legend[open] > summary::before {{ content: "\\2212"; }}
      .legend__grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 2px 0;
        padding: 0 20px 18px;
      }}
      .legend__item {{ display: flex; gap: 12px; align-items: flex-start; padding: 10px 0; border-top: 1px solid var(--line); }}
      .legend__abbr {{
        flex: none;
        min-width: 46px;
        text-align: center;
        padding: 3px 8px;
        border-radius: 4px;
        background: var(--navy);
        color: #fff;
        font-family: "Roboto Condensed", sans-serif;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
      }}
      .legend__term {{ margin: 0; font-weight: 700; font-size: 0.9rem; }}
      .legend__desc {{ margin: 2px 0 0; color: var(--muted); font-size: 0.84rem; line-height: 1.4; }}

      @media (max-width: 820px) {{
        .topbar__nav {{ display: none; }}
        .hero {{ padding: 40px 24px 44px; }}
        .league-board__header {{ flex-direction: column; }}
        .card__header {{ flex-direction: column; }}
        .card__coverage {{ white-space: normal; }}
        .leader-grid {{ grid-template-columns: 1fr; }}
        .track-grid {{ grid-template-columns: 1fr; }}
        .legend__grid {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <header class="topbar">
      <div class="topbar__inner">
        <span class="wordmark"><span class="wordmark__mark">BB</span>Baseball Bench</span>
        <nav class="topbar__nav">
          <a href="#standings">Standings</a>
          <a href="#leaders">Leaders</a>
          <a href="#h2h">Head to Head</a>
          <a href="#models">Managers</a>
          <a href="#history">History</a>
        </nav>
      </div>
    </header>
    <main>
      <section class="hero">
        <div class="hero__content">
          <p class="eyebrow">Baseball Bench &middot; 2026 Season</p>
          <h1>AI manager league</h1>
          <p class="hero__sub">Models are compared as baseball managers against the rest of the field. Research and tactical decisions are scouting context; league play is the main result.</p>
        </div>
      </section>
      <section class="section" id="standings">
        <div class="section__head">
          <h2 class="section__title">Standings</h2>
          <p class="section__note">RES &middot; DEC &middot; PCT shown as batting-average rates. DIFF is run differential; ELO is the manager rating.</p>
        </div>
        {comparison_table}
        {legend}
      </section>
      <section class="section" id="leaders">
        <div class="section__head">
          <h2 class="section__title">League Leaders</h2>
          <p class="section__note">Top mark in each category across the public field.</p>
        </div>
        <div class="leader-grid">
          {leader_panels}
        </div>
      </section>
      <section class="section" id="h2h">
        {league_comparison}
      </section>
      <section class="section" id="models">
        <div class="section__head">
          <h2 class="section__title">Manager Cards</h2>
          <p class="section__note">Full stat line for every model in the field.</p>
        </div>
        <div class="cards">
          {cards}
        </div>
      </section>
      <section class="section">
        <div class="section__head">
          <h2 class="section__title">League Runs On File</h2>
        </div>
        <div class="notes">
          <ul>{league_notes}</ul>
        </div>
      </section>
      <section class="section">
        <div class="section__head">
          <h2 class="section__title">Benchmark Notes</h2>
        </div>
        <div class="notes">
          <p>Internal calibration baselines are excluded from this public comparison so the page only shows actual model entries.</p>
        </div>
      </section>
      <section class="section" id="history">
        <div class="section__head">
          <h2 class="section__title">Run History</h2>
        </div>
        <div class="notes">
          <ul>{history_notes}</ul>
        </div>
      </section>
    </main>
  </body>
</html>
"""
    site_index_path.write_text(html)
    return leaderboard
