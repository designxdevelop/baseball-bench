from __future__ import annotations

from dataclasses import dataclass, replace
from random import Random

from baseball_bench.core import ActionType, BenchOption, BullpenOption, DecisionOption, GameState, ManagerDecision, RunnerState, make_action_id


@dataclass
class HitterProfile:
    player_id: str
    name: str
    pa: int
    walks: int
    strikeouts: int
    singles: int
    doubles: int
    triples: int
    home_runs: int

    @property
    def rates(self) -> dict[str, float]:
        return {
            "walk": self.walks / self.pa,
            "strikeout": self.strikeouts / self.pa,
            "single": self.singles / self.pa,
            "double": self.doubles / self.pa,
            "triple": self.triples / self.pa,
            "home_run": self.home_runs / self.pa,
        }


@dataclass
class PitcherProfile:
    player_id: str
    name: str
    innings_pitched: float
    walks: int
    strikeouts: int
    hits_allowed: int
    home_runs_allowed: int
    role: str


@dataclass
class TeamRoster:
    lineup: list[HitterProfile]
    bench: list[HitterProfile]
    starter: PitcherProfile
    bullpen: list[PitcherProfile]


@dataclass
class GameResult:
    home_manager: str
    away_manager: str
    home_score: int
    away_score: int
    decision_log: list[dict[str, object]]


def _apply_offensive_decision(
    decision: ManagerDecision,
    state: GameState,
    batter: HitterProfile,
    bench: list[HitterProfile],
    rng: Random,
) -> tuple[HitterProfile, list[str]]:
    notes: list[str] = []
    if decision.action_id == ActionType.STEAL.value and state.runners.first:
        if rng.random() < 0.68:
            state.runners.first = False
            state.runners.second = True
            notes.append("stolen_base")
        else:
            state.runners.first = False
            state.outs += 1
            notes.append("caught_stealing")
    elif decision.action_id == ActionType.BUNT.value and state.outs < 2:
        state.outs += 1
        if state.runners.first and not state.runners.second:
            state.runners.first = False
            state.runners.second = True
        elif state.runners.second and not state.runners.third:
            state.runners.second = False
            state.runners.third = True
        notes.append("sacrifice")
    elif decision.action_id.startswith("pinch_hit:"):
        player_id = decision.action_id.split(":", 1)[1]
        replacement = next((player for player in bench if player.player_id == player_id), None)
        if replacement:
            batter = replacement
            notes.append(f"pinch_hit:{replacement.name}")
    return batter, notes


def _advance_runners(state: GameState, outcome: str) -> int:
    runs = 0
    if outcome in {"out", "strikeout"}:
        state.outs += 1
        return runs
    if outcome == "walk":
        if state.runners.first and state.runners.second and state.runners.third:
            runs += 1
        state.runners.third = state.runners.third or state.runners.second
        state.runners.second = state.runners.second or state.runners.first
        state.runners.first = True
        return runs
    if outcome == "single":
        if state.runners.third:
            runs += 1
        if state.runners.second:
            runs += 1
        state.runners.third = state.runners.first
        state.runners.second = False
        state.runners.first = True
        return runs
    if outcome == "double":
        runs += int(state.runners.third) + int(state.runners.second)
        if state.runners.first:
            state.runners.third = True
        else:
            state.runners.third = False
        state.runners.second = True
        state.runners.first = False
        return runs
    if outcome == "triple":
        runs += int(state.runners.first) + int(state.runners.second) + int(state.runners.third)
        state.runners = RunnerState(third=True)
        return runs
    if outcome == "home_run":
        runs += 1 + int(state.runners.first) + int(state.runners.second) + int(state.runners.third)
        state.runners = RunnerState()
        return runs
    raise ValueError(f"Unknown outcome {outcome}")


def _plate_appearance_outcome(batter: HitterProfile, pitcher: PitcherProfile, rng: Random) -> str:
    rates = dict(batter.rates)
    walk_adjust = max(0.85, min(1.15, pitcher.walks / max(pitcher.innings_pitched * 4.0, 1.0)))
    strikeout_adjust = max(0.85, min(1.20, pitcher.strikeouts / max(pitcher.innings_pitched * 3.0, 1.0)))
    hr_adjust = max(0.85, min(1.15, pitcher.home_runs_allowed / max(pitcher.innings_pitched / 9.0, 1.0)))
    rates["walk"] *= walk_adjust
    rates["strikeout"] *= strikeout_adjust
    rates["home_run"] *= hr_adjust
    total_positive = sum(rates.values())
    rates["out"] = max(0.20, 1.0 - total_positive)
    threshold = rng.random()
    cumulative = 0.0
    for outcome in ("walk", "strikeout", "single", "double", "triple", "home_run", "out"):
        cumulative += rates.get(outcome, 0.0)
        if threshold <= cumulative:
            return outcome
    return "out"


def offensive_options(state: GameState, bench: list[HitterProfile]) -> list[DecisionOption]:
    options = [DecisionOption(action_id=ActionType.LET_HIT.value, action_type=ActionType.LET_HIT, label="Let the current hitter hit")]
    if state.inning >= 7 and state.runners.first and state.outs <= 1:
        options.append(DecisionOption(action_id=ActionType.STEAL.value, action_type=ActionType.STEAL, label="Attempt a steal"))
        options.append(DecisionOption(action_id=ActionType.BUNT.value, action_type=ActionType.BUNT, label="Sacrifice bunt"))
    if bench:
        options.append(
            DecisionOption(
                action_id=make_action_id(ActionType.PINCH_HIT, bench[0].player_id),
                action_type=ActionType.PINCH_HIT,
                label=f"Pinch hit with {bench[0].name}",
            )
        )
    return options


def pitching_options(bullpen: list[PitcherProfile], current_pitcher: PitcherProfile) -> list[DecisionOption]:
    options = [
        DecisionOption(
            action_id=ActionType.STAY_WITH_PITCHER.value,
            action_type=ActionType.STAY_WITH_PITCHER,
            label=f"Keep {current_pitcher.name} on the mound",
        )
    ]
    for arm in bullpen[:2]:
        options.append(
            DecisionOption(
                action_id=make_action_id(ActionType.GO_TO_BULLPEN, arm.player_id),
                action_type=ActionType.GO_TO_BULLPEN,
                label=f"Bring in {arm.name}",
            )
        )
    return options


def simulate_game(
    home_manager,
    away_manager,
    home_roster: TeamRoster,
    away_roster: TeamRoster,
    seed: int,
) -> GameResult:
    rng = Random(seed)
    home_score = 0
    away_score = 0
    home_pitcher = replace(home_roster.starter)
    away_pitcher = replace(away_roster.starter)
    home_bench = list(home_roster.bench)
    away_bench = list(away_roster.bench)
    home_bullpen = list(home_roster.bullpen)
    away_bullpen = list(away_roster.bullpen)
    decision_log: list[dict[str, object]] = []
    home_index = 0
    away_index = 0
    home_fatigue = 0
    away_fatigue = 0

    inning = 1
    while inning <= 9 or home_score == away_score:
        for half in ("top", "bottom"):
            runners = RunnerState()
            outs = 0
            batting_home = half == "bottom"
            batting_lineup = home_roster.lineup if batting_home else away_roster.lineup
            batting_bench = home_bench if batting_home else away_bench
            manager = home_manager if batting_home else away_manager
            fielding_manager = away_manager if batting_home else home_manager
            batting_team_name = home_manager.name if batting_home else away_manager.name
            fielding_team_name = away_manager.name if batting_home else home_manager.name
            current_pitcher = away_pitcher if batting_home else home_pitcher
            bullpen = away_bullpen if batting_home else home_bullpen

            if inning >= 6 and ((batting_home and away_fatigue >= 18) or ((not batting_home) and home_fatigue >= 18)):
                state = GameState(
                    inning=inning,
                    half=half,
                    outs=outs,
                    runners=runners,
                    home_team=home_manager.name,
                    away_team=away_manager.name,
                        batting_team=batting_team_name,
                        fielding_team=fielding_team_name,
                        home_score=home_score,
                        away_score=away_score,
                        batter_name=batting_lineup[(home_index if batting_home else away_index) % len(batting_lineup)].name,
                        pitcher_name=current_pitcher.name,
                        leverage_index=1.5 if abs(home_score - away_score) <= 2 and inning >= 7 else 1.0,
                        bullpen=[BullpenOption(player_id=arm.player_id, name=arm.name, throws="R") for arm in bullpen[:2]],
                    )
                options = pitching_options(bullpen, current_pitcher)
                decision = fielding_manager.decide(state, options)
                decision_log.append({"phase": "pitching", "manager": fielding_manager.name, "decision": decision.action_id, "inning": inning, "half": half})
                if decision.action_id.startswith("go_to_bullpen:"):
                    arm_id = decision.action_id.split(":", 1)[1]
                    replacement_pitcher = next((arm for arm in bullpen if arm.player_id == arm_id), None)
                    if replacement_pitcher:
                        if batting_home:
                            away_pitcher = replace(replacement_pitcher)
                            away_fatigue = 0
                            current_pitcher = away_pitcher
                        else:
                            home_pitcher = replace(replacement_pitcher)
                            home_fatigue = 0
                            current_pitcher = home_pitcher

            while outs < 3:
                batter_idx = home_index if batting_home else away_index
                batter = batting_lineup[batter_idx % len(batting_lineup)]
                state = GameState(
                    inning=inning,
                    half=half,
                    outs=outs,
                    runners=runners.model_copy(deep=True),
                    home_team=home_manager.name,
                    away_team=away_manager.name,
                    batting_team=batting_team_name,
                    fielding_team=fielding_team_name,
                    home_score=home_score,
                    away_score=away_score,
                    batter_name=batter.name,
                    pitcher_name=current_pitcher.name,
                    leverage_index=1.8 if inning >= 7 and abs(home_score - away_score) <= 1 else 1.0,
                    bench=[BenchOption(player_id=player.player_id, name=player.name, bats="S") for player in batting_bench[:1]],
                )
                notes: list[str] = []
                if inning >= 7 and (runners.first or batting_bench):
                    options = offensive_options(state, batting_bench)
                    decision = manager.decide(state, options)
                    decision_log.append({"phase": "offense", "manager": manager.name, "decision": decision.action_id, "inning": inning, "half": half})
                    batter, notes = _apply_offensive_decision(decision, state, batter, batting_bench, rng)
                    outs = state.outs
                    runners = state.runners
                    if outs >= 3:
                        break
                outcome = _plate_appearance_outcome(batter, current_pitcher, rng)
                runs = _advance_runners(state, outcome)
                outs = state.outs
                runners = state.runners
                if batting_home:
                    home_score += runs
                    away_fatigue += 1
                else:
                    away_score += runs
                    home_fatigue += 1
                if batting_home:
                    home_index += 1
                else:
                    away_index += 1
            if inning >= 9 and half == "bottom" and home_score != away_score:
                break
        if inning >= 9 and home_score != away_score:
            break
        inning += 1

    return GameResult(
        home_manager=home_manager.name,
        away_manager=away_manager.name,
        home_score=home_score,
        away_score=away_score,
        decision_log=decision_log,
    )
