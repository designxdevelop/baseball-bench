from __future__ import annotations

import json
from datetime import datetime
from html import escape
from math import ceil
from pathlib import Path

from baseball_bench.paths import LEADERBOARD_JSON_PATH, RESULTS_DIR, RUNS_DIR, SITE_DIR, SITE_INDEX_PATH
from baseball_bench.utils import write_json

TRACK_COUNT = 3
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
        "kind": "league",
        "models": run.get("models", []),
        "public_models": [
            model for model in run.get("models", []) if _is_public_model(str(model))
        ],
        "games_per_matchup": run.get("games_per_matchup", 0),
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
        "league": "league play",
    }[best_track_name]

    if completed_tracks == 0:
        return "No results yet", "This model has not completed any tracks yet."
    if missing_tracks:
        headline = "Strong start" if best_track["score"] is not None and float(best_track["score"]) >= 0.75 else "Run in progress"
        summary = (
            f"Completed {completed_tracks} of {TRACK_COUNT} tracks. Best showing so far came in "
            f"{best_track_label}. Overall ranking is on hold until league play finishes."
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


def _track_leader_summary(entries: list[dict[str, object]], track_name: str) -> dict[str, str]:
    label = {
        "analysis": "Research",
        "decisions": "Game Moves",
        "league": "League Play",
    }[track_name]
    complete = [entry for entry in entries if entry["tracks"][track_name]["status"] == "complete"]
    if not complete:
        return {
            "label": label,
            "headline": "Not finished yet",
            "summary": f"{label} has not finished for the current public-model snapshot.",
        }
    top_score = max(float(entry["tracks"][track_name]["score"]) for entry in complete)
    leaders = [entry for entry in complete if float(entry["tracks"][track_name]["score"]) == top_score]
    if len(leaders) == 1:
        leader = leaders[0]
        return {
            "label": label,
            "headline": f"Leader: {leader['display_name']}",
            "summary": str(leader["tracks"][track_name]["summary"]),
        }
    return {
        "label": label,
        "headline": f"{len(leaders)}-way tie",
        "summary": str(leaders[0]["tracks"][track_name]["summary"]),
    }


def _aggregate(payloads: list[dict[str, object]]) -> dict[str, object]:
    models: dict[str, dict[str, object]] = {}
    league_runs: list[dict[str, object]] = []
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
        elif kind == "league":
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
            "league": entry.get(
                "league",
                _missing_track("League play has not run yet for this model."),
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
            "league": "League Play",
        },
        "overall_ranking_ready": ranking_ready,
        "track_leaders": [
            _track_leader_summary(public_models, "analysis"),
            _track_leader_summary(public_models, "decisions"),
            _track_leader_summary(public_models, "league"),
        ],
    }


def _render_track_card(label: str, track: dict[str, object]) -> str:
    status_class = "track track--complete" if track["status"] == "complete" else "track track--missing"
    return (
        f'<section class="{status_class}">'
        f"<p class=\"track__label\">{escape(label)}</p>"
        f"<h3>{escape(str(track['headline']))}</h3>"
        f"<p>{escape(str(track['summary']))}</p>"
        "</section>"
    )


def _render_model_card(entry: dict[str, object], track_labels: dict[str, str]) -> str:
    tracks_html = "\n".join(
        _render_track_card(track_labels[name], entry["tracks"][name])
        for name in ("analysis", "decisions", "league")
    )
    rank_markup = (
        f"<p class=\"card__rank\">#{entry['rank']}</p>"
        if entry["rank"] is not None
        else "<p class=\"card__rank\">In Progress</p>"
    )
    return f"""
      <article class="card">
        <div class="card__header">
          <div>
            {rank_markup}
            <h2>{escape(str(entry['display_name']))}</h2>
            <p class="card__verdict">{escape(str(entry['overall_read']))}</p>
          </div>
          <p class="card__coverage">{entry['completed_tracks']}/{TRACK_COUNT} tracks complete</p>
        </div>
        <p class="card__summary">{escape(str(entry['overall_summary']))}</p>
        <div class="track-grid">
          {tracks_html}
        </div>
      </article>
    """


def _render_comparison_table(entries: list[dict[str, object]], ranking_ready: bool) -> str:
    if not entries:
        return "<p>No model results yet.</p>"
    rows = []
    for entry in entries:
        rows.append(
            "<tr>"
            f"<td>{escape(str(entry['display_name']))}</td>"
            f"<td>{escape(str(entry['overall_read']))}</td>"
            f"<td>{escape(str(entry['tracks']['analysis']['headline']))}</td>"
            f"<td>{escape(str(entry['tracks']['decisions']['headline']))}</td>"
            f"<td>{escape(str(entry['tracks']['league']['headline']))}</td>"
            "</tr>"
        )
    rank_note = (
        "<p>The overall order is final because every public model has completed all three tracks.</p>"
        if ranking_ready
        else "<p>The overall order is provisional. League play has not finished, so this is an in-progress read rather than a final ranking.</p>"
    )
    return (
        "<div class=\"comparison-table\">"
        f"{rank_note}"
        "<table>"
        "<thead><tr><th>Model</th><th>Overall Read</th><th>Research</th><th>Game Moves</th><th>League Play</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


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
        <section class="league-board">
          <div>
            <p class="track__label">Manager League</p>
            <h2>Head-to-head matchups pending</h2>
            <p>Run the OpenRouter pack through league play to compare each model as a manager against the rest of the field.</p>
          </div>
        </section>
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
      <section class="league-board">
        <div class="league-board__header">
          <div>
            <p class="track__label">Manager League</p>
            <h2>Head-to-head model standings</h2>
            <p>{escape(str(run['headline']))} Matchup cells show record and run differential against that opponent.</p>
          </div>
          <p class="card__coverage">{int(run['game_count'])} games</p>
        </div>
        <div class="comparison-table">
          <table>
            <thead><tr><th>Rank</th><th>Model</th><th>Record</th><th>Run Diff</th>{matchup_headers}</tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
      </section>
    """


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
    leader_panels = "\n".join(
        (
            "<section class=\"leader-panel\">"
            f"<p class=\"track__label\">{escape(str(item['label']))}</p>"
            f"<h3>{escape(str(item['headline']))}</h3>"
            f"<p>{escape(str(item['summary']))}</p>"
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
    <style>
      :root {{
        color-scheme: light;
        font-family: "Iowan Old Style", Georgia, serif;
        --ink: #14211d;
        --muted: #56645d;
        --line: rgba(20, 33, 29, 0.14);
        --paper: rgba(255, 252, 247, 0.84);
        --accent: #b5412d;
        --grass: #dbead8;
        --sand: #f3e2c6;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(181, 65, 45, 0.14), transparent 28%),
          radial-gradient(circle at top right, rgba(69, 128, 89, 0.16), transparent 24%),
          linear-gradient(180deg, #f6efe4 0%, #eef4ea 100%);
      }}
      main {{ max-width: 1120px; margin: 0 auto; padding: 40px 20px 64px; }}
      .hero {{
        padding: 28px;
        border: 1px solid var(--line);
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(255,255,255,0.82), rgba(243,226,198,0.74));
        box-shadow: 0 18px 50px rgba(20, 33, 29, 0.08);
      }}
      .eyebrow {{
        margin: 0 0 10px;
        color: var(--accent);
        font-size: 0.85rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }}
      h1 {{ margin: 0; font-size: clamp(2.4rem, 5vw, 4.8rem); line-height: 0.96; }}
      .hero p:last-child {{ margin-bottom: 0; max-width: 62ch; color: var(--muted); }}
      .cards {{
        display: grid;
        gap: 18px;
        margin-top: 24px;
      }}
      .comparison {{
        margin-top: 24px;
        padding: 20px 22px;
        border-radius: 22px;
        background: rgba(255,255,255,0.78);
        border: 1px solid var(--line);
      }}
      .comparison h2 {{ margin-top: 0; font-size: 1.35rem; }}
      .comparison p {{ color: var(--muted); }}
      .comparison-table {{ overflow-x: auto; }}
      .comparison table {{ width: 100%; border-collapse: collapse; }}
      .league-board {{
        margin-top: 24px;
        padding: 20px 22px;
        border-radius: 8px;
        background: rgba(255,255,255,0.82);
        border: 1px solid var(--line);
      }}
      .league-board__header {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 18px;
      }}
      .league-board h2 {{ margin: 0; font-size: 1.5rem; }}
      .league-board p {{ color: var(--muted); }}
      .league-board td span {{ color: var(--muted); }}
      .self-matchup {{ color: var(--muted); }}
      .leader-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin-top: 16px;
      }}
      .leader-panel {{
        padding: 16px;
        border-radius: 18px;
        background: rgba(219, 234, 216, 0.42);
        border: 1px solid var(--line);
      }}
      .comparison th, .comparison td {{
        text-align: left;
        padding: 12px 10px;
        border-bottom: 1px solid var(--line);
        vertical-align: top;
      }}
      .comparison th {{
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
      }}
      .card {{
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 22px;
        background: var(--paper);
        box-shadow: 0 12px 40px rgba(20, 33, 29, 0.06);
      }}
      .card__header {{
        display: flex;
        gap: 16px;
        justify-content: space-between;
        align-items: flex-start;
      }}
      .card__rank {{
        margin: 0 0 6px;
        color: var(--accent);
        font-size: 0.88rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }}
      h2 {{ margin: 0; font-size: 1.8rem; }}
      .card__verdict {{
        margin: 8px 0 0;
        font-size: 1.05rem;
        font-weight: 700;
      }}
      .card__coverage {{
        margin: 0;
        padding: 10px 12px;
        border-radius: 999px;
        background: rgba(20, 33, 29, 0.06);
        color: var(--muted);
        white-space: nowrap;
      }}
      .card__summary {{ margin: 14px 0 0; color: var(--muted); max-width: 70ch; }}
      .track-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin-top: 18px;
      }}
      .track {{
        padding: 16px;
        border-radius: 18px;
        border: 1px solid var(--line);
      }}
      .track--complete {{ background: rgba(219, 234, 216, 0.55); }}
      .track--missing {{ background: rgba(255, 255, 255, 0.72); }}
      .track__label {{
        margin: 0 0 8px;
        color: var(--muted);
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }}
      h3 {{ margin: 0; font-size: 1.08rem; }}
      .track p:last-child {{ margin-bottom: 0; color: var(--muted); }}
      .notes {{
        margin-top: 24px;
        padding: 20px 22px;
        border-radius: 22px;
        background: rgba(255,255,255,0.72);
        border: 1px solid var(--line);
      }}
      .notes h2 {{ font-size: 1.2rem; margin-bottom: 10px; }}
      .notes ul {{ margin: 0; padding-left: 18px; color: var(--muted); }}
      .notes li + li {{ margin-top: 8px; }}
      .notes a {{ color: var(--accent); margin-left: 10px; }}
      @media (max-width: 820px) {{
        .league-board__header {{ flex-direction: column; }}
        .card__header {{ flex-direction: column; }}
        .card__coverage {{ white-space: normal; }}
        .leader-grid {{ grid-template-columns: 1fr; }}
        .track-grid {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="hero">
        <p class="eyebrow">Baseball Bench</p>
        <h1>AI manager league</h1>
        <p>Models are compared as baseball managers against the rest of the field. Research and tactical decisions are scouting context; league play is the main result.</p>
      </section>
      {league_comparison}
      <section class="comparison">
        <h2>Scouting Report</h2>
        <p>Read across each row to compare the same model on research, game moves, and league play.</p>
        <div class="leader-grid">
          {leader_panels}
        </div>
        {comparison_table}
      </section>
      <section class="cards">
        {cards}
      </section>
      <section class="notes">
        <h2>League Runs On File</h2>
        <ul>{league_notes}</ul>
      </section>
      <section class="notes">
        <h2>Benchmark Notes</h2>
        <p>Internal calibration baselines are excluded from this public comparison so the page only shows actual model entries.</p>
      </section>
      <section class="notes">
        <h2>Run History</h2>
        <ul>{history_notes}</ul>
      </section>
    </main>
  </body>
</html>
"""
    site_index_path.write_text(html)
    return leaderboard
