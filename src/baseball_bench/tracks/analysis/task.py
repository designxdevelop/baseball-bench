from __future__ import annotations

from pathlib import Path
from statistics import mean

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, system_message, use_tools

from baseball_bench.paths import DATA_DB_PATH, RESULTS_DIR
from baseball_bench.scoring import analysis_scorer, score_analysis_answer
from baseball_bench.tracks.analysis.question_bank import AnalysisQuestion, load_questions
from baseball_bench.tracks.analysis.tools import schema_lookup_tool, sql_tool
from baseball_bench.utils import write_json


SYSTEM_PROMPT = """
You are taking the baseball-bench analysis track.

Use the provided SQL and schema tools when useful. Work from the benchmark database only.
Return your final answer as JSON with this shape:
{"answer": "...", "confidence": 0.0, "notes": "..."}
"""


def _sample_for_question(question: AnalysisQuestion) -> Sample:
    return Sample(
        id=question.id,
        input=question.prompt,
        target=question.expected_answer,
        metadata={
            "tier": question.tier,
            "sql": question.sql,
            "answer_type": question.answer_type,
            "tolerance": question.tolerance,
        },
    )


@task(name="analysis")
def analysis_task(question_set: str = "v1", database_path: str | None = None) -> Task:
    questions = load_questions(question_set)
    dataset = [_sample_for_question(question) for question in questions]
    db_path = database_path or str(DATA_DB_PATH)
    return Task(
        dataset=dataset,
        solver=[
            system_message(SYSTEM_PROMPT),
            use_tools(sql_tool(db_path), schema_lookup_tool()),
            generate(),
        ],
        scorer=analysis_scorer(),
        message_limit=20,
        name="analysis",
        version=question_set,
        metadata={"question_set": question_set, "database_path": db_path},
    )


def run_sql_baseline(question_set: str = "v1", model_name: str = "sql-baseline") -> dict[str, object]:
    questions = load_questions(question_set)
    samples: list[dict[str, object]] = []
    tier_scores: dict[str, list[float]] = {}
    for question in questions:
        observed = execute_baseline_sql(question, DATA_DB_PATH)
        value, metadata = score_analysis_answer(
            f'{{"answer": "{observed}"}}',
            question.expected_answer,
            tolerance=question.tolerance,
        )
        samples.append(
            {
                "id": question.id,
                "tier": question.tier,
                "prompt": question.prompt,
                "expected_answer": question.expected_answer,
                "observed_answer": observed,
                "score": value,
                "metadata": metadata,
            }
        )
        tier_scores.setdefault(question.tier, []).append(value)
    summary = {
        "kind": "analysis",
        "model": model_name,
        "question_set": question_set,
        "sample_count": len(samples),
        "overall_accuracy": mean(sample["score"] for sample in samples) if samples else 0.0,
        "tier_accuracy": {tier: mean(values) for tier, values in tier_scores.items()},
        "samples": samples,
    }
    write_json(RESULTS_DIR / f"analysis-{model_name}.json", summary)
    return summary


def write_analysis_summary(
    summary: dict[str, object],
    output_dir: Path = RESULTS_DIR,
    write_latest: bool = True,
) -> None:
    filename = f"analysis-{summary['model'].replace('/', '-')}.json"
    write_json(output_dir / filename, summary)
    if write_latest and output_dir != RESULTS_DIR:
        write_json(RESULTS_DIR / filename, summary)


def execute_baseline_sql(question: AnalysisQuestion, database_path: Path) -> str:
    from baseball_bench.data import execute_read_only_query

    result = execute_read_only_query(question.sql, database_path=database_path, row_limit=1)
    return str(result["rows"][0][0])
