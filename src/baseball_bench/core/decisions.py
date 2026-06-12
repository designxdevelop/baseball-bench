from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class ActionType(StrEnum):
    LET_HIT = "let_hit"
    PINCH_HIT = "pinch_hit"
    STEAL = "steal"
    BUNT = "bunt"
    STAY_WITH_PITCHER = "stay_with_pitcher"
    GO_TO_BULLPEN = "go_to_bullpen"


class DecisionOption(BaseModel):
    action_id: str
    action_type: ActionType
    label: str
    detail: str | None = None


class ManagerDecision(BaseModel):
    action_id: str
    rationale: str | None = None


def make_action_id(action_type: ActionType, subject: str | None = None) -> str:
    return f"{action_type.value}:{subject}" if subject else action_type.value

