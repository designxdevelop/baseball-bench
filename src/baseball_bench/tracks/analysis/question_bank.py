from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from baseball_bench.data import execute_read_only_query
from baseball_bench.paths import DATA_DB_PATH
from baseball_bench.utils import normalize_answer, write_json


QUESTIONS_DIR = Path(__file__).resolve().parent / "questions"


class AnalysisQuestion(BaseModel):
    id: str
    tier: str
    prompt: str
    sql: str
    answer_type: str
    expected_answer: str
    tolerance: float | None = None


QUESTION_SPECS = [
    {
        "id": "highest_batting_average_2025",
        "tier": "easy",
        "prompt": "Which player had the highest batting average in 2025 among hitters with at least 500 plate appearances?",
        "sql": """
            select p.first_name || ' ' || p.last_name as answer
            from batting b
            join players p on p.player_id = b.player_id
            where b.season = 2025 and b.plate_appearances >= 500
            order by cast(b.hits as double) / b.at_bats desc, answer asc
            limit 1
        """,
        "answer_type": "string",
    },
    {
        "id": "most_home_runs_2025",
        "tier": "easy",
        "prompt": "Which player hit the most home runs in 2025?",
        "sql": """
            select p.first_name || ' ' || p.last_name as answer
            from batting b
            join players p on p.player_id = b.player_id
            where b.season = 2025
            order by b.home_runs desc, answer asc
            limit 1
        """,
        "answer_type": "string",
    },
    {
        "id": "best_ops_2025",
        "tier": "medium",
        "prompt": "Which player posted the best OPS in 2025 among hitters with at least 500 plate appearances?",
        "sql": """
            with stats as (
              select
                p.first_name || ' ' || p.last_name as answer,
                ((b.hits + b.walks + b.hit_by_pitch) * 1.0 / (b.at_bats + b.walks + b.hit_by_pitch + b.sacrifice_flies))
                +
                (((b.hits - b.doubles - b.triples - b.home_runs) + 2 * b.doubles + 3 * b.triples + 4 * b.home_runs) * 1.0 / b.at_bats)
                as ops
              from batting b
              join players p on p.player_id = b.player_id
              where b.season = 2025 and b.plate_appearances >= 500
            )
            select answer from stats order by ops desc, answer asc limit 1
        """,
        "answer_type": "string",
    },
    {
        "id": "best_run_diff_2025",
        "tier": "medium",
        "prompt": "Which team had the best run differential in the 2025 regular season?",
        "sql": """
            with team_runs as (
              select home_team_id as team_id, sum(home_score) as runs_for, sum(away_score) as runs_against from games where season = 2025 group by home_team_id
              union all
              select away_team_id as team_id, sum(away_score) as runs_for, sum(home_score) as runs_against from games where season = 2025 group by away_team_id
            ),
            rolled as (
              select team_id, sum(runs_for) as runs_for, sum(runs_against) as runs_against
              from team_runs
              group by team_id
            )
            select city || ' ' || nickname as answer
            from rolled r
            join teams t on t.team_id = r.team_id
            order by (runs_for - runs_against) desc, answer asc
            limit 1
        """,
        "answer_type": "string",
    },
    {
        "id": "best_k_per_9_2025",
        "tier": "medium",
        "prompt": "Which 2025 starting pitcher had the highest strikeouts per nine innings among pitchers with at least 180 innings pitched?",
        "sql": """
            select p.first_name || ' ' || p.last_name as answer
            from pitching pi
            join players p on p.player_id = pi.player_id
            where pi.season = 2025 and pi.games_started > 0 and pi.innings_pitched >= 180
            order by (pi.strikeouts * 9.0 / pi.innings_pitched) desc, answer asc
            limit 1
        """,
        "answer_type": "string",
    },
    {
        "id": "hr_leader_obp_2025",
        "tier": "hard",
        "prompt": "What was the on-base percentage of the 2025 home run leader? Return just the decimal value.",
        "sql": """
            with hr_leader as (
              select player_id
              from batting
              where season = 2025
              order by home_runs desc, player_id asc
              limit 1
            )
            select round((hits + walks + hit_by_pitch) * 1.0 / (at_bats + walks + hit_by_pitch + sacrifice_flies), 3) as answer
            from batting
            where season = 2025 and player_id in (select player_id from hr_leader)
        """,
        "answer_type": "number",
        "tolerance": 0.001,
    },
    {
        "id": "batting_average_leader_home_runs_2025",
        "tier": "easy",
        "prompt": "How many home runs did the 2025 batting average leader hit?",
        "sql": """
            with batting_average_leader as (
              select player_id
              from batting
              where season = 2025 and plate_appearances >= 500
              order by cast(hits as double) / at_bats desc, player_id asc
              limit 1
            )
            select home_runs as answer
            from batting
            where season = 2025 and player_id in (select player_id from batting_average_leader)
        """,
        "answer_type": "number",
    },
    {
        "id": "count_ops_850_2025",
        "tier": "hard",
        "prompt": "How many 2025 hitters reached at least a .850 OPS while also recording at least 450 plate appearances?",
        "sql": """
            with stats as (
              select
                player_id,
                (((hits + walks + hit_by_pitch) * 1.0 / (at_bats + walks + hit_by_pitch + sacrifice_flies))
                +
                (((hits - doubles - triples - home_runs) + 2 * doubles + 3 * triples + 4 * home_runs) * 1.0 / at_bats)) as ops
              from batting
              where season = 2025 and plate_appearances >= 450
            )
            select count(*) as answer from stats where ops >= 0.850
        """,
        "answer_type": "number",
    },
    {
        "id": "best_reliever_era_2025",
        "tier": "medium",
        "prompt": "Which 2025 reliever had the lowest ERA among relievers with at least 60 innings pitched?",
        "sql": """
            select p.first_name || ' ' || p.last_name as answer
            from pitching pi
            join players p on p.player_id = pi.player_id
            where pi.season = 2025 and pi.games_started = 0 and pi.innings_pitched >= 60
            order by (pi.earned_runs * 9.0 / pi.innings_pitched) asc, answer asc
            limit 1
        """,
        "answer_type": "string",
    },
    {
        "id": "best_win_pct_2025",
        "tier": "hard",
        "prompt": "Which team had the best winning percentage in the 2025 regular season?",
        "sql": """
            with team_results as (
              select home_team_id as team_id, sum(case when home_score > away_score then 1 else 0 end) as wins, count(*) as games
              from games where season = 2025 group by home_team_id
              union all
              select away_team_id as team_id, sum(case when away_score > home_score then 1 else 0 end) as wins, count(*) as games
              from games where season = 2025 group by away_team_id
            ),
            rolled as (
              select team_id, sum(wins) as wins, sum(games) as games
              from team_results
              group by team_id
            )
            select city || ' ' || nickname as answer
            from rolled r join teams t on t.team_id = r.team_id
            order by (wins * 1.0 / games) desc, answer asc
            limit 1
        """,
        "answer_type": "string",
    }
]


def _query_scalar(sql: str, database_path: Path | None = None) -> str:
    result = execute_read_only_query(sql, database_path=database_path or DATA_DB_PATH, row_limit=1)
    value = result["rows"][0][0]
    return normalize_answer(value)


def generate_questions(version: str = "v1", database_path: Path | None = None) -> list[AnalysisQuestion]:
    questions: list[AnalysisQuestion] = []
    for spec in QUESTION_SPECS:
        expected_answer = _query_scalar(spec["sql"], database_path)
        questions.append(
            AnalysisQuestion(
                id=spec["id"],
                tier=spec["tier"],
                prompt=spec["prompt"],
                sql=spec["sql"].strip(),
                answer_type=spec["answer_type"],
                expected_answer=expected_answer,
                tolerance=spec.get("tolerance"),
            )
        )
    write_json(
        QUESTIONS_DIR / f"{version}.json",
        [question.model_dump(mode="json") for question in questions],
    )
    return questions


def load_questions(version: str = "v1") -> list[AnalysisQuestion]:
    path = QUESTIONS_DIR / f"{version}.json"
    if not path.exists():
        return generate_questions(version=version)
    return [AnalysisQuestion.model_validate(item) for item in __import__("json").loads(path.read_text())]
