from __future__ import annotations

import json
from pathlib import Path

from baseball_bench.paths import LEADERBOARD_JSON_PATH, RESULTS_DIR, SITE_DIR, SITE_INDEX_PATH
from baseball_bench.utils import write_json


def _load_result_files(results_dir: Path) -> list[dict[str, object]]:
    payloads = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name in {"leaderboard.json", "data-checksum.json"}:
            continue
        payloads.append(json.loads(path.read_text()))
    return payloads


def _aggregate(payloads: list[dict[str, object]]) -> dict[str, object]:
    models: dict[str, dict[str, object]] = {}
    leagues: list[dict[str, object]] = []
    for payload in payloads:
        kind = payload.get("kind")
        if kind == "analysis":
            model = str(payload["model"])
            entry = models.setdefault(model, {"model": model})
            entry["analysis_accuracy"] = payload["overall_accuracy"]
        elif kind == "decisions":
            model = str(payload["model"])
            entry = models.setdefault(model, {"model": model})
            entry["decision_wp_delta"] = payload["mean_wp_delta"]
            entry["decision_near_optimal_rate"] = payload["near_optimal_rate"]
        elif kind == "league":
            leagues.append(payload)
            for row in payload.get("standings", []):
                model = str(row["model"])
                entry = models.setdefault(model, {"model": model})
                entry["league_elo"] = row["elo"]
                entry["league_win_pct"] = row["win_pct"]
    table = []
    for model, entry in models.items():
        analysis_component = float(entry.get("analysis_accuracy", 0.0))
        decision_component = 1.0 - float(entry.get("decision_wp_delta", 1.0))
        league_component = (float(entry.get("league_elo", 1500.0)) - 1400.0) / 200.0
        entry["overall_score"] = round((analysis_component + decision_component + league_component) / 3.0, 3)
        table.append(entry)
    table.sort(key=lambda item: item["overall_score"], reverse=True)
    return {"models": table, "league_runs": leagues}


def build_site(results_dir: Path = RESULTS_DIR) -> dict[str, object]:
    payloads = _load_result_files(results_dir)
    leaderboard = _aggregate(payloads)
    write_json(LEADERBOARD_JSON_PATH, leaderboard)
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        [
            f"<tr><td>{entry['model']}</td><td>{entry.get('analysis_accuracy', '-')}</td><td>{entry.get('decision_wp_delta', '-')}</td><td>{entry.get('league_elo', '-')}</td><td>{entry['overall_score']}</td></tr>"
            for entry in leaderboard["models"]
        ]
    )
    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>baseball-bench leaderboard</title>
    <style>
      :root {{ color-scheme: light; font-family: "Iowan Old Style", Georgia, serif; }}
      body {{ margin: 0; padding: 40px; background: linear-gradient(180deg, #f6f0e7, #e7efe9); color: #1b2420; }}
      main {{ max-width: 980px; margin: 0 auto; }}
      h1 {{ font-size: 3rem; margin-bottom: 0.25rem; }}
      p {{ max-width: 60ch; }}
      table {{ width: 100%; border-collapse: collapse; margin-top: 32px; background: rgba(255,255,255,0.72); }}
      th, td {{ padding: 12px 14px; border-bottom: 1px solid rgba(27,36,32,0.12); text-align: left; }}
      th {{ font-size: 0.85rem; letter-spacing: 0.08em; text-transform: uppercase; }}
    </style>
  </head>
  <body>
    <main>
      <h1>baseball-bench</h1>
      <p>Deterministic benchmark snapshots across analysis, in-game decisions, and a seeded sim league.</p>
      <table>
        <thead>
          <tr><th>Model</th><th>Analysis Accuracy</th><th>Decision Mean WP Delta</th><th>League Elo</th><th>Overall</th></tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </main>
  </body>
</html>
"""
    SITE_INDEX_PATH.write_text(html)
    return leaderboard

