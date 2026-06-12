from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Protocol

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model

from baseball_bench.core import ActionType, DecisionOption, GameState, ManagerDecision, make_action_id
from baseball_bench.utils import parse_jsonish


class Manager(Protocol):
    name: str

    def decide(self, state: GameState, options: list[DecisionOption]) -> ManagerDecision: ...


@dataclass
class RulebookManager:
    name: str = "rulebook"

    def decide(self, state: GameState, options: list[DecisionOption]) -> ManagerDecision:
        ids = {option.action_id for option in options}
        if state.inning >= 8 and state.half == "top" and make_action_id(ActionType.GO_TO_BULLPEN, "p104") in ids:
            return ManagerDecision(action_id=make_action_id(ActionType.GO_TO_BULLPEN, "p104"), rationale="Shorten the game with the best rested reliever.")
        if state.inning >= 8 and state.runners.first and state.outs == 0 and ActionType.STEAL.value in ids:
            return ManagerDecision(action_id=ActionType.STEAL.value, rationale="Push the tying run into scoring position.")
        pinch_hit = next((option.action_id for option in options if option.action_type == ActionType.PINCH_HIT), None)
        if pinch_hit and state.inning >= 8 and state.outs <= 1:
            return ManagerDecision(action_id=pinch_hit, rationale="Use the best bench bat in a leverage spot.")
        return ManagerDecision(action_id=options[0].action_id, rationale="Default to the primary option.")


@dataclass
class AggressiveManager:
    name: str = "aggressive"

    def decide(self, state: GameState, options: list[DecisionOption]) -> ManagerDecision:
        for action_type in (ActionType.STEAL, ActionType.BUNT):
            action = next((option.action_id for option in options if option.action_type == action_type), None)
            if action:
                return ManagerDecision(action_id=action, rationale="Force action and trade variance for aggression.")
        bullpen = next((option.action_id for option in options if option.action_type == ActionType.GO_TO_BULLPEN), None)
        if bullpen:
            return ManagerDecision(action_id=bullpen, rationale="Go to the pen aggressively.")
        return ManagerDecision(action_id=options[0].action_id, rationale="Attack the leverage spot directly.")


@dataclass
class ConservativeManager:
    name: str = "conservative"

    def decide(self, state: GameState, options: list[DecisionOption]) -> ManagerDecision:
        stay = next((option.action_id for option in options if option.action_type == ActionType.STAY_WITH_PITCHER), None)
        if stay:
            return ManagerDecision(action_id=stay, rationale="Preserve bullpen flexibility.")
        let_hit = next((option.action_id for option in options if option.action_type == ActionType.LET_HIT), None)
        if let_hit:
            return ManagerDecision(action_id=let_hit, rationale="Avoid giving away outs.")
        return ManagerDecision(action_id=options[0].action_id, rationale="Take the least disruptive option.")


@dataclass
class LLMManager:
    model_name: str

    @property
    def name(self) -> str:
        return self.model_name

    def decide(self, state: GameState, options: list[DecisionOption]) -> ManagerDecision:
        prompt = "\n".join(
            [
                "You are managing a baseball team in a simulation.",
                state.manager_prompt_context(),
                "Choose exactly one action from this menu:",
                *[f"- {option.action_id}: {option.label}" for option in options],
                'Return JSON: {"action_id": "...", "rationale": "..."}',
            ]
        )
        return asyncio.run(self._generate(prompt, options))

    async def _generate(self, prompt: str, options: list[DecisionOption]) -> ManagerDecision:
        model = get_model(self.model_name)
        result = await model.generate(
            [
                ChatMessageSystem(content="Return strict JSON for each baseball decision."),
                ChatMessageUser(content=prompt),
            ]
        )
        payload = parse_jsonish(result.completion)
        action_id = str(payload.get("action_id", options[0].action_id))
        valid_ids = {option.action_id for option in options}
        if action_id not in valid_ids:
            action_id = options[0].action_id
        return ManagerDecision(
            action_id=action_id,
            rationale=str(payload.get("rationale", "")),
        )


def build_manager(name: str) -> Manager:
    if name == "rulebook":
        return RulebookManager()
    if name == "aggressive":
        return AggressiveManager()
    if name == "conservative":
        return ConservativeManager()
    return LLMManager(name)
