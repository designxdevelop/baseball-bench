from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, system_message
from pydantic import BaseModel

from baseball_bench.core import GameState
from baseball_bench.paths import RESULTS_DIR
from baseball_bench.scoring import decision_scorer, score_decision_answer
from baseball_bench.utils import write_json


SITUATIONS_PATH = Path(__file__).resolve().parent / "situations" / "v1.json"

SYSTEM_PROMPT = """
You are taking the baseball-bench decision track.

Pick exactly one action from the provided menu. Return JSON only:
{"action_id": "...", "confidence": 0.0, "rationale": "..."}
"""


class DecisionSituation(BaseModel):
    id: str
    description: str
    state: GameState
    options: list[dict[str, object]]

    @property
    def action_values(self) -> dict[str, float]:
        return {str(option["action_id"]): float(option["expected_win_prob"]) for option in self.options}

    @property
    def optimal_action(self) -> str:
        return max(self.action_values.items(), key=lambda item: item[1])[0]

    def prompt(self) -> str:
        option_lines = [
            f"- {option['action_id']}: {option['label']} (expected_wp={float(option['expected_win_prob']):.3f})"
            for option in self.options
        ]
        return "\n".join(
            [
                self.description,
                self.state.summary(),
                "Available actions:",
                *option_lines,
            ]
        )


def load_situations(version: str = "v1") -> list[DecisionSituation]:
    path = SITUATIONS_PATH if version == "v1" else SITUATIONS_PATH.with_name(f"{version}.json")
    payload = json.loads(path.read_text())
    return [DecisionSituation.model_validate(item) for item in payload]


def _sample_for_situation(situation: DecisionSituation) -> Sample:
    return Sample(
        id=situation.id,
        input=situation.prompt(),
        target=situation.optimal_action,
        metadata={
            "action_values": situation.action_values,
            "description": situation.description,
        },
    )


@task(name="decisions")
def decisions_task(version: str = "v1") -> Task:
    situations = load_situations(version)
    return Task(
        dataset=[_sample_for_situation(situation) for situation in situations],
        solver=[system_message(SYSTEM_PROMPT), generate()],
        scorer=decision_scorer(),
        message_limit=12,
        name="decisions",
        version=version,
        metadata={"situation_set": version},
    )


def run_wp_baseline(version: str = "v1", model_name: str = "wp-baseline") -> dict[str, object]:
    situations = load_situations(version)
    samples: list[dict[str, object]] = []
    deltas: list[float] = []
    for situation in situations:
        chosen = situation.optimal_action
        value, metadata = score_decision_answer(
            json.dumps({"action_id": chosen}),
            situation.action_values,
        )
        deltas.append(float(metadata["wp_delta"]))
        samples.append(
            {
                "id": situation.id,
                "prompt": situation.prompt(),
                "chosen_action": chosen,
                "best_action": metadata["best_action"],
                "wp_delta": metadata["wp_delta"],
                "near_optimal": metadata["near_optimal"],
                "score": value,
            }
        )
    summary = {
        "kind": "decisions",
        "model": model_name,
        "situation_set": version,
        "sample_count": len(samples),
        "mean_wp_delta": mean(deltas) if deltas else 0.0,
        "near_optimal_rate": mean(1.0 if sample["near_optimal"] else 0.0 for sample in samples) if samples else 0.0,
        "samples": samples,
    }
    write_json(RESULTS_DIR / f"decisions-{model_name}.json", summary)
    return summary

