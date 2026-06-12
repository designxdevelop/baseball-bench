from __future__ import annotations

from typing import Any

from inspect_ai.scorer import Score, Target, accuracy, mean, scorer, stderr
from inspect_ai.solver import TaskState

from baseball_bench.utils import normalize_answer, parse_jsonish


def _extract_answer(text: str) -> str:
    payload = parse_jsonish(text)
    if "answer" in payload:
        return normalize_answer(payload["answer"])
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.lower().startswith("answer:"):
            return normalize_answer(line.split(":", 1)[1].strip())
    return normalize_answer(text.strip())


def score_analysis_answer(
    output_text: str,
    expected_answer: str,
    *,
    tolerance: float | None = None,
) -> tuple[float, dict[str, Any]]:
    observed = _extract_answer(output_text)
    expected = normalize_answer(expected_answer)
    if tolerance is None:
        value = 1.0 if observed.lower() == expected.lower() else 0.0
    else:
        try:
            observed_value = float(observed)
            expected_value = float(expected)
            value = (
                1.0
                if abs(observed_value - expected_value) <= float(tolerance) + 1e-9
                else 0.0
            )
        except ValueError:
            value = 0.0
    return value, {
        "observed_answer": observed,
        "expected_answer": expected,
        "tolerance": tolerance,
    }


@scorer(metrics=[mean(), accuracy(), stderr()])
def analysis_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        tolerance = None
        if state.metadata:
            tolerance = state.metadata.get("tolerance")
        value, metadata = score_analysis_answer(
            state.output.completion,
            target.text,
            tolerance=tolerance,
        )
        return Score(
            value=value,
            answer=metadata["observed_answer"],
            explanation=f"expected {metadata['expected_answer']}",
            metadata=metadata,
        )

    return score
