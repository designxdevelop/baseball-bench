from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
LOGS_DIR = REPO_ROOT / "logs"
DATA_DB_PATH = ARTIFACTS_DIR / "baseball.duckdb"
DATA_CHECKSUM_PATH = RESULTS_DIR / "data-checksum.json"
LEADERBOARD_JSON_PATH = RESULTS_DIR / "leaderboard.json"
SITE_DIR = RESULTS_DIR / "site"
SITE_INDEX_PATH = SITE_DIR / "index.html"
