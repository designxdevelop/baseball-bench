from __future__ import annotations

from dataclasses import dataclass, field, replace
from random import Random

from baseball_bench.core import ActionType, BenchOption, BullpenOption, DecisionOption, GameState, ManagerDecision, RunnerState, make_action_id
from baseball_bench.tracks.league.eval_modes import LeagueEvalConfig


BUNDLED_MANAGERS = {"rulebook", "aggressive", "conservative"}


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
    bats: str = "R"
    position: str = "DH"

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
    throws: str = "R"

    @property
    def rates(self) -> dict[str, float]:
        batters_faced_estimate = max(self.innings_pitched * 4.25, 1.0)
        return {
            "walk": self.walks / batters_faced_estimate,
            "strikeout": self.strikeouts / batters_faced_estimate,
            "hit": self.hits_allowed / batters_faced_estimate,
            "home_run": self.home_runs_allowed / batters_faced_estimate,
        }

    @property
    def pitch_profile(self) -> dict[str, float]:
        rates = self.rates
        return {
            "power": _clamp(rates["strikeout"] / 0.23, 0.72, 1.38),
            "command": _clamp(0.085 / max(rates["walk"], 0.001), 0.72, 1.35),
            "mistake": _clamp(rates["home_run"] / 0.032, 0.65, 1.45),
        }

    @property
    def contact_profile(self) -> dict[str, float]:
        rates = self.rates
        return {
            "contact_quality": _clamp(rates["hit"] / 0.24, 0.72, 1.28),
            "traffic": _clamp((rates["hit"] + rates["walk"]) / 0.39, 0.78, 1.25),
        }


@dataclass
class TeamRoster:
    lineup: list[HitterProfile]
    bench: list[HitterProfile]
    starter: PitcherProfile
    bullpen: list[PitcherProfile]
    park_factor: float = 1.0
    bullpen_roles: dict[str, str] = field(default_factory=dict)


@dataclass
class GameResult:
    home_manager: str
    away_manager: str
    home_score: int
    away_score: int
    decision_log: list[dict[str, object]]
    home_bullpen_rest: dict[str, int] = field(default_factory=dict)
    away_bullpen_rest: dict[str, int] = field(default_factory=dict)


def _hitter_summary(hitter: HitterProfile) -> str:
    avg = (hitter.singles + hitter.doubles + hitter.triples + hitter.home_runs) / max(hitter.pa, 1)
    obp = (hitter.singles + hitter.doubles + hitter.triples + hitter.home_runs + hitter.walks) / max(hitter.pa, 1)
    slg = (
        hitter.singles + (2 * hitter.doubles) + (3 * hitter.triples) + (4 * hitter.home_runs)
    ) / max(hitter.pa, 1)
    return (
        f"2025 line approx AVG {avg:.3f}, OBP {obp:.3f}, SLG {slg:.3f}, "
        f"HR {hitter.home_runs}, BB {hitter.walks}, SO {hitter.strikeouts}"
    )


def _pitcher_summary(pitcher: PitcherProfile) -> str:
    innings = max(pitcher.innings_pitched, 1.0)
    return (
        f"2025 rates K/9 {(pitcher.strikeouts * 9 / innings):.1f}, "
        f"BB/9 {(pitcher.walks * 9 / innings):.1f}, "
        f"HR/9 {(pitcher.home_runs_allowed * 9 / innings):.1f}, "
        f"H/9 {(pitcher.hits_allowed * 9 / innings):.1f}"
    )


def _fallback_decision(options: list[DecisionOption]) -> ManagerDecision:
    stay = next((option.action_id for option in options if option.action_type == ActionType.STAY_WITH_PITCHER), None)
    if stay:
        return ManagerDecision(action_id=stay, rationale="Routine spot outside the live-call policy.")
    let_hit = next((option.action_id for option in options if option.action_type == ActionType.LET_HIT), None)
    if let_hit:
        return ManagerDecision(action_id=let_hit, rationale="Routine spot outside the live-call policy.")
    return ManagerDecision(action_id=options[0].action_id, rationale="Routine spot outside the live-call policy.")


def _decide_with_policy(
    manager,
    state: GameState,
    options: list[DecisionOption],
    *,
    eval_config: LeagueEvalConfig,
    live_call_counts: dict[str, int],
    team_name: str,
) -> tuple[ManagerDecision, bool]:
    if manager.name in BUNDLED_MANAGERS:
        return manager.decide(state, options), False
    should_call = eval_config.should_call_live(
        inning=state.inning,
        score_gap=abs(state.home_score - state.away_score),
        team_live_calls=live_call_counts.get(team_name, 0),
    )
    if not should_call:
        return _fallback_decision(options), False
    live_call_counts[team_name] = live_call_counts.get(team_name, 0) + 1
    return manager.decide(state, options), True


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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _advance_runners(state: GameState, outcome: str, rng: Random | None = None) -> int:
    rng = rng or Random(0)
    runs = 0
    if outcome in {"out", "strikeout", "groundout", "flyout", "lineout", "popout"}:
        state.outs += 1
        if outcome == "flyout" and state.outs <= 2 and state.runners.third and rng.random() < 0.36:
            runs += 1
            state.runners.third = False
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


def _platoon_factor(batter: HitterProfile, pitcher: PitcherProfile) -> float:
    if batter.bats == "S":
        return 1.04
    if batter.bats != pitcher.throws:
        return 1.06
    return 0.96


def _pitcher_contact_factor(pitcher: PitcherProfile) -> float:
    pitch_profile = pitcher.pitch_profile
    contact_profile = pitcher.contact_profile
    return _clamp(
        (contact_profile["contact_quality"] * contact_profile["traffic"]) / pitch_profile["power"],
        0.75,
        1.28,
    )


def _fatigue_factor(fatigue: int, pitcher: PitcherProfile, times_through_order: int) -> float:
    starter_limit = 24 if pitcher.role == "SP" else 8
    fatigue_pressure = max(0, fatigue - starter_limit) * (0.012 if pitcher.role == "SP" else 0.025)
    tto_pressure = max(0, times_through_order - 2) * 0.055
    return _clamp(1.0 + fatigue_pressure + tto_pressure, 0.92, 1.38)


def _batted_ball_outcome(
    rates: dict[str, float],
    *,
    batter: HitterProfile,
    pitcher: PitcherProfile,
    park_factor: float,
    rng: Random,
) -> str:
    power_share = _clamp((batter.doubles + batter.triples + batter.home_runs) / max(batter.pa, 1) / 0.105, 0.65, 1.45)
    pitcher_contact = _pitcher_contact_factor(pitcher)
    line_drive = _clamp(0.21 * pitcher_contact, 0.15, 0.30)
    fly_ball = _clamp(0.34 * power_share * park_factor, 0.24, 0.47)
    ground_ball = _clamp(0.39 / max(power_share, 0.75), 0.28, 0.50)
    pop_up = max(0.04, 1.0 - line_drive - fly_ball - ground_ball)
    batted_ball = _weighted_choice(
        {
            "line_drive": line_drive,
            "fly_ball": fly_ball,
            "ground_ball": ground_ball,
            "pop_up": pop_up,
        },
        rng,
    )
    hit_rate = rates["single"] + rates["double"] + rates["triple"]
    extra_base_share = (rates["double"] + rates["triple"]) / max(hit_rate, 0.001)
    if batted_ball == "line_drive":
        if rng.random() < _clamp(0.62 * pitcher_contact, 0.42, 0.78):
            return "double" if rng.random() < extra_base_share else "single"
        return "lineout"
    if batted_ball == "fly_ball":
        if rng.random() < _clamp(0.18 * power_share * park_factor, 0.08, 0.33):
            return "double"
        return "flyout"
    if batted_ball == "ground_ball":
        if rng.random() < _clamp(0.23 * pitcher_contact, 0.12, 0.35):
            return "single"
        return "groundout"
    return "popout"


def _weighted_choice(weights: dict[str, float], rng: Random) -> str:
    total = sum(max(weight, 0.0) for weight in weights.values())
    if total <= 0:
        return next(iter(weights))
    threshold = rng.random() * total
    cumulative = 0.0
    for outcome, weight in weights.items():
        cumulative += max(weight, 0.0)
        if threshold <= cumulative:
            return outcome
    return next(reversed(weights))


def _hitter_matchup_rating(hitter: HitterProfile, pitcher: PitcherProfile) -> float:
    rates = hitter.rates
    base = (
        rates["walk"]
        + rates["single"]
        + (1.6 * rates["double"])
        + (2.1 * rates["triple"])
        + (2.8 * rates["home_run"])
        - (0.25 * rates["strikeout"])
    )
    return base * _platoon_factor(hitter, pitcher)


def _positions_compatible(candidate: HitterProfile, incumbent: HitterProfile) -> bool:
    if candidate.position in {"UT", "DH"} or incumbent.position == "DH":
        return True
    if candidate.position == incumbent.position:
        return True
    outfield = {"LF", "CF", "RF", "OF"}
    return candidate.position in outfield and incumbent.position in outfield


def _platoon_lineup(roster: TeamRoster, opposing_starter: PitcherProfile) -> tuple[list[HitterProfile], list[HitterProfile]]:
    lineup = list(roster.lineup)
    bench = list(roster.bench)
    swaps = 0
    for candidate in list(bench):
        if swaps >= 2:
            break
        if _platoon_factor(candidate, opposing_starter) <= 1.0:
            continue
        candidate_score = _hitter_matchup_rating(candidate, opposing_starter)
        replace_index = min(
            range(len(lineup)),
            key=lambda index: _hitter_matchup_rating(lineup[index], opposing_starter)
            if _positions_compatible(candidate, lineup[index])
            else float("inf"),
        )
        incumbent = lineup[replace_index]
        if not _positions_compatible(candidate, incumbent):
            continue
        if candidate_score <= _hitter_matchup_rating(incumbent, opposing_starter) + 0.015:
            continue
        lineup[replace_index] = candidate
        bench.remove(candidate)
        bench.append(incumbent)
        swaps += 1
    return lineup, bench


def _plate_appearance_outcome(
    batter: HitterProfile,
    pitcher: PitcherProfile,
    rng: Random,
    *,
    fatigue: int = 0,
    times_through_order: int = 1,
    park_factor: float = 1.0,
) -> str:
    batter_rates = dict(batter.rates)
    pitcher_rates = pitcher.rates
    pitch_profile = pitcher.pitch_profile
    platoon = _platoon_factor(batter, pitcher)
    fatigue_multiplier = _fatigue_factor(fatigue, pitcher, times_through_order)
    contact_factor = _pitcher_contact_factor(pitcher)

    rates = {
        "walk": batter_rates["walk"] * _clamp(pitcher_rates["walk"] / 0.085, 0.72, 1.35) / pitch_profile["command"],
        "strikeout": batter_rates["strikeout"] * pitch_profile["power"],
        "single": batter_rates["single"] * contact_factor * platoon * fatigue_multiplier,
        "double": batter_rates["double"] * platoon * fatigue_multiplier * _clamp(park_factor, 0.92, 1.1),
        "triple": batter_rates["triple"] * platoon * _clamp(park_factor, 0.9, 1.18),
        "home_run": batter_rates["home_run"]
        * pitch_profile["mistake"]
        * platoon
        * fatigue_multiplier
        * _clamp(park_factor, 0.82, 1.22),
    }
    rates["walk"] = _clamp(rates["walk"], 0.035, 0.18)
    rates["strikeout"] = _clamp(rates["strikeout"], 0.08, 0.38)
    rates["home_run"] = _clamp(rates["home_run"], 0.005, 0.095)

    direct_outcome = _weighted_choice(
        {
            "walk": rates["walk"],
            "strikeout": rates["strikeout"],
            "home_run": rates["home_run"],
            "ball_in_play": max(0.34, 1.0 - rates["walk"] - rates["strikeout"] - rates["home_run"]),
        },
        rng,
    )
    if direct_outcome != "ball_in_play":
        return direct_outcome
    return _batted_ball_outcome(
        rates,
        batter=batter,
        pitcher=pitcher,
        park_factor=park_factor,
        rng=rng,
    )


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


def _bullpen_candidates(
    bullpen: list[PitcherProfile],
    roles: dict[str, str],
    rest: dict[str, int],
    *,
    leverage_index: float,
) -> list[PitcherProfile]:
    role_priority = (
        {"closer": 0, "setup": 1, "middle": 2, "long": 3}
        if leverage_index >= 1.5
        else {"long": 0, "middle": 1, "setup": 2, "closer": 3}
    )
    return sorted(
        bullpen,
        key=lambda arm: (
            rest.get(arm.player_id, 100) < 25,
            role_priority.get(roles.get(arm.player_id, "middle"), 2),
            -rest.get(arm.player_id, 100),
        ),
    )


def simulate_game(
    home_manager,
    away_manager,
    home_roster: TeamRoster,
    away_roster: TeamRoster,
    seed: int,
    home_bullpen_rest: dict[str, int] | None = None,
    away_bullpen_rest: dict[str, int] | None = None,
    eval_config: LeagueEvalConfig | None = None,
    park_name: str | None = None,
) -> GameResult:
    eval_config = eval_config or LeagueEvalConfig()
    rng = Random(seed)
    home_score = 0
    away_score = 0
    home_pitcher = replace(home_roster.starter)
    away_pitcher = replace(away_roster.starter)
    home_lineup, home_bench = _platoon_lineup(home_roster, away_pitcher)
    away_lineup, away_bench = _platoon_lineup(away_roster, home_pitcher)
    home_bullpen = list(home_roster.bullpen)
    away_bullpen = list(away_roster.bullpen)
    home_bullpen_rest = dict(home_bullpen_rest or {arm.player_id: 100 for arm in home_bullpen})
    away_bullpen_rest = dict(away_bullpen_rest or {arm.player_id: 100 for arm in away_bullpen})
    decision_log: list[dict[str, object]] = []
    home_index = 0
    away_index = 0
    home_fatigue = 0
    away_fatigue = 0
    live_call_counts = {home_manager.name: 0, away_manager.name: 0}

    inning = 1
    while inning <= 9 or home_score == away_score:
        for half in ("top", "bottom"):
            runners = RunnerState()
            outs = 0
            batting_home = half == "bottom"
            batting_lineup = home_lineup if batting_home else away_lineup
            batting_bench = home_bench if batting_home else away_bench
            manager = home_manager if batting_home else away_manager
            fielding_manager = away_manager if batting_home else home_manager
            batting_team_name = home_manager.name if batting_home else away_manager.name
            fielding_team_name = away_manager.name if batting_home else home_manager.name
            current_pitcher = away_pitcher if batting_home else home_pitcher
            bullpen = away_bullpen if batting_home else home_bullpen
            bullpen_rest = away_bullpen_rest if batting_home else home_bullpen_rest
            bullpen_roles = away_roster.bullpen_roles if batting_home else home_roster.bullpen_roles
            park_factor = home_roster.park_factor

            if inning >= 6 and ((batting_home and away_fatigue >= 18) or ((not batting_home) and home_fatigue >= 18)):
                leverage_index = 1.5 if abs(home_score - away_score) <= 2 and inning >= 7 else 1.0
                bullpen_menu = _bullpen_candidates(
                    bullpen,
                    bullpen_roles,
                    bullpen_rest,
                    leverage_index=leverage_index,
                )
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
                    batter_bats=batting_lineup[(home_index if batting_home else away_index) % len(batting_lineup)].bats,
                    batter_summary=_hitter_summary(
                        batting_lineup[(home_index if batting_home else away_index) % len(batting_lineup)]
                    ),
                    pitcher_name=current_pitcher.name,
                    pitcher_throws=current_pitcher.throws,
                    pitcher_role=current_pitcher.role,
                    pitcher_summary=_pitcher_summary(current_pitcher),
                    pitcher_fatigue=away_fatigue if batting_home else home_fatigue,
                    times_through_order=1
                    + ((home_index if batting_home else away_index) // max(len(batting_lineup), 1)),
                    park_name=park_name,
                    park_factor=park_factor,
                    leverage_index=leverage_index,
                    bullpen=[
                        BullpenOption(
                            player_id=arm.player_id,
                            name=arm.name,
                            throws=arm.throws,
                            role=bullpen_roles.get(arm.player_id, arm.role),
                            stamina=bullpen_rest.get(arm.player_id, 100),
                            summary=_pitcher_summary(arm),
                        )
                        for arm in bullpen_menu[:2]
                    ],
                )
                options = pitching_options(bullpen_menu, current_pitcher)
                decision, live_model_call = _decide_with_policy(
                    fielding_manager,
                    state,
                    options,
                    eval_config=eval_config,
                    live_call_counts=live_call_counts,
                    team_name=fielding_manager.name,
                )
                decision_log.append(
                    {
                        "phase": "pitching",
                        "manager": fielding_manager.name,
                        "decision": decision.action_id,
                        "inning": inning,
                        "half": half,
                        "live_model_call": live_model_call,
                    }
                )
                if decision.action_id.startswith("go_to_bullpen:"):
                    arm_id = decision.action_id.split(":", 1)[1]
                    replacement_pitcher = next((arm for arm in bullpen if arm.player_id == arm_id), None)
                    if replacement_pitcher:
                        bullpen_rest[replacement_pitcher.player_id] = max(
                            0,
                            bullpen_rest.get(replacement_pitcher.player_id, 100) - 35,
                        )
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
                    batter_bats=batter.bats,
                    batter_summary=_hitter_summary(batter),
                    pitcher_name=current_pitcher.name,
                    pitcher_throws=current_pitcher.throws,
                    pitcher_role=current_pitcher.role,
                    pitcher_summary=_pitcher_summary(current_pitcher),
                    pitcher_fatigue=away_fatigue if batting_home else home_fatigue,
                    times_through_order=1 + (batter_idx // max(len(batting_lineup), 1)),
                    park_name=park_name,
                    park_factor=park_factor,
                    leverage_index=1.8 if inning >= 7 and abs(home_score - away_score) <= 1 else 1.0,
                    bench=[
                        BenchOption(
                            player_id=player.player_id,
                            name=player.name,
                            bats=player.bats,
                            position=player.position,
                            summary=_hitter_summary(player),
                        )
                        for player in batting_bench[:1]
                    ],
                )
                notes: list[str] = []
                if inning >= 7 and (runners.first or batting_bench):
                    options = offensive_options(state, batting_bench)
                    decision, live_model_call = _decide_with_policy(
                        manager,
                        state,
                        options,
                        eval_config=eval_config,
                        live_call_counts=live_call_counts,
                        team_name=manager.name,
                    )
                    decision_log.append(
                        {
                            "phase": "offense",
                            "manager": manager.name,
                            "decision": decision.action_id,
                            "inning": inning,
                            "half": half,
                            "live_model_call": live_model_call,
                        }
                    )
                    batter, notes = _apply_offensive_decision(decision, state, batter, batting_bench, rng)
                    outs = state.outs
                    runners = state.runners
                    if outs >= 3:
                        break
                pitcher_fatigue = away_fatigue if batting_home else home_fatigue
                lineup_index = home_index if batting_home else away_index
                times_through_order = 1 + (lineup_index // max(len(batting_lineup), 1))
                outcome = _plate_appearance_outcome(
                    batter,
                    current_pitcher,
                    rng,
                    fatigue=pitcher_fatigue,
                    times_through_order=times_through_order,
                    park_factor=park_factor,
                )
                runs = _advance_runners(state, outcome, rng)
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
        home_bullpen_rest=home_bullpen_rest,
        away_bullpen_rest=away_bullpen_rest,
    )
