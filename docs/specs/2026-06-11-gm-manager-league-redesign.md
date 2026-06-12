# baseball-bench - GM and Manager League Redesign

**Date:** 2026-06-11
**Status:** Active plan

## Problem

The current league track is too narrow and too expensive. Every model receives the same fixed roster, fixed lineup, fixed starter, fixed bench, and fixed bullpen ordering. The only model work happens through repeated in-game tactical calls. A 7-manager `--games 2` run creates 84 league games and can require thousands of sequential model calls.

That makes stalls hard to tolerate and makes the league less interesting than the benchmark idea. It mostly measures late-game tactical response latency and decision quality, not broader baseball judgment.

## Direction

Split the baseball benchmark into three distinct capabilities:

1. **GM Track**
   Models build a roster from a shared player pool under explicit constraints.

2. **Controlled Manager League**
   Models manage equivalent talent so tactical comparison stays clean.

3. **Open League**
   Models use their own GM-built rosters and manager plans in a head-to-head competition. This is the fun showpiece, but it is not treated as the cleanest apples-to-apples tactical score.

For runtime policy and default operating modes, see [2026-06-11-eval-modes-plan.md](/Users/austin/code/dxd/baseball-bench/docs/specs/2026-06-11-eval-modes-plan.md).

## Track A - GM Roster Build

**Purpose:** measure research, player evaluation, roster construction, constraint satisfaction, and explanation quality.

**Inputs**

- Shared hitter and pitcher pool from the pinned seed database.
- Roster constraints:
  - 13 position players
  - 13 pitchers
  - required defensive coverage
  - at least 5 starting-pitcher candidates
  - at least 7 bullpen candidates
  - optional budget or value cap once player valuation exists
- Required structured JSON output:
  - 26-man roster
  - primary lineup
  - bench roles
  - rotation
  - bullpen roles
  - short rationale for each major choice

**Primary scores**

- `validity_rate`: roster obeys all constraints.
- `projected_strength`: local deterministic roster projection, not LLM judged.
- `balance_score`: lineup handedness, position coverage, pitcher-role coverage, bench flexibility.
- `explanation_completeness`: rubric-based structural score, not subjective quality ranking.

**Output artifact**

`results/gm-roster-{model}.json`

## Track B - Controlled Manager League

**Purpose:** measure tactical management without roster-construction noise.

**Inputs**

- Same player pool and same baseline roster for every model.
- One pregame model call per manager to produce a `ManagerPlan`:
  - batting order
  - bench priority
  - bullpen hierarchy
  - starter hook rules
  - steal/bunt/pinch-hit tendencies
  - high-leverage aggression settings

**Runtime behavior**

- The simulator executes the plan locally for normal situations.
- Live LLM calls are reserved for high-leverage spots only.
- Hard cap live calls per game, default `3`.
- If a live call times out or returns invalid JSON, fall back to the manager plan and record a penalty.

**Primary scores**

- win percentage
- run differential
- Elo
- invalid-action rate
- timeout/fallback rate

**Default schedule**

Use a sampled seeded schedule instead of full directed round-robin by default.

- smoke: 3 games
- normal: 12 games
- stronger signal: 18 games
- full matrix: opt-in only

## Track C - Open League

**Purpose:** measure the full baseball agent loop: research, roster building, planning, and tactical management.

**Inputs**

- Each model's GM roster artifact.
- Each model's manager plan artifact.
- Same simulator and high-leverage call caps as the controlled league.

**Interpretation**

Open League is a showpiece ranking. It is not used as the sole proof that one model is better at in-game tactics, because roster strength and tactical quality are intentionally mixed.

## Data Model

Add these structured objects:

- `RosterBuild`
  - `model`
  - `players`
  - `lineups`
  - `rotation`
  - `bullpen_roles`
  - `bench_roles`
  - `rationale`
  - `validation`
  - `scores`

- `ManagerPlan`
  - `model`
  - `lineup`
  - `bench_priority`
  - `bullpen_hierarchy`
  - `starter_hook`
  - `steal_policy`
  - `bunt_policy`
  - `pinch_hit_policy`
  - `live_decision_cap`

- `LeagueSchedule`
  - `schedule_mode`
  - `game_count`
  - `seed`
  - `games`

## Implementation Plan

### Phase 1 - Stop Runaway League Runs

- Add explicit `--league-games` to cap sampled league schedules.
- Keep the current `--games` full-matrix behavior for compatibility, but do not use it as the recommended OpenRouter path.
- Preserve `league-progress-*.json` heartbeat snapshots.
- Add README guidance for smoke, normal, stronger, and full runs.

### Phase 2 - Richer Local Matchup Engine

- Carry hitter handedness and position plus pitcher throwing hand through controlled and GM-built rosters.
- Infer pitcher power, command, mistake, traffic, and contact-quality profiles from the pinned pitching data.
- Resolve plate appearances with platoon factors, park factors, fatigue, times-through-order penalties, and batted-ball distributions instead of fixed single/double/home-run buckets.
- Let ordered bench choices create limited pregame platoon lineup swaps against the opposing starter.
- Use manager-provided bullpen roles with rest and leverage so closer/setup/long-relief choices have consequences across sampled league games.

### Phase 2 - ManagerPlan Without Live Spam

- Add `ManagerPlan` schema and validator.
- Add baseline plans for `rulebook`, `aggressive`, and `conservative`.
- Let LLM managers generate one plan before league play.
- Modify the simulator to use plan policies for routine decisions.
- Add max live calls per game with timeout and fallback metadata.

### Phase 3 - GM Track

- Expand the player pool exposed to models.
- Add `roster-build` task.
- Validate roster legality.
- Add deterministic roster projection scoring.
- Write `gm-roster-{model}.json` artifacts.

### Phase 4 - Open League

- Let each model enter with its own GM roster and manager plan.
- Keep schedule sampled by default.
- Label the leaderboard separately from the controlled manager league.

## Recommended Near-Term CLI

```bash
.venv/bin/baseball-bench run-bench --league-games 12
.venv/bin/baseball-bench run-league --league-games 12
.venv/bin/baseball-bench run-league --schedule full --games 2
```

## Success Criteria

- A normal OpenRouter bucket run completes in less than 90 minutes.
- League progress updates at least once per game and eventually once per live decision.
- The leaderboard separates GM quality, controlled manager quality, and open competition.
- Full round-robin remains possible, but never accidental.
