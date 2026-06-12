from __future__ import annotations

from typing import Any

from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

from baseball_bench.utils import parse_jsonish


def _extract_action_id(text: str) -> str:
    payload = parse_jsonish(text)
    if "action_id" in payload:
        return str(payload["action_id"]).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.lower().startswith("action_id:"):
            return line.split(":", 1)[1].strip()
    return text.strip()


def score_decision_answer(
    output_text: str,
    action_values: dict[str, float],
) -> tuple[float, dict[str, Any]]:
    chosen = _extract_action_id(output_text)
    best_action, best_wp = max(action_values.items(), key=lambda item: item[1])
    chosen_wp = action_values.get(chosen)
    if chosen_wp is None:
        delta = best_wp
        value = 0.0
    else:
        delta = best_wp - chosen_wp
        value = max(0.0, 1.0 - delta)
    return value, {
        "chosen_action": chosen,
        "best_action": best_action,
        "chosen_wp": chosen_wp,
        "best_wp": best_wp,
        "wp_delta": delta,
        "near_optimal": delta <= 0.01 if chosen_wp is not None else False,
    }


@scorer(metrics=[mean(), stderr()])
def decision_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        if not state.metadata or "action_values" not in state.metadata:
            raise ValueError("decision scorer requires action_values metadata")
        value, metadata = score_decision_answer(
            state.output.completion,
            state.metadata["action_values"],
        )
        return Score(
            value=value,
            answer=metadata["chosen_action"],
            explanation=f"best action: {metadata['best_action']}",
            metadata=metadata,
        )

    return score
