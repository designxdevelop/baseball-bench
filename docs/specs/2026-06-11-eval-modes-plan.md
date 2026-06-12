# baseball-bench - Evaluation Modes Plan

**Date:** 2026-06-11
**Status:** Proposed plan

## Problem

The benchmark currently mixes two competing needs into one default run:

- a publishable public snapshot that should finish quickly enough to refresh often
- a deeper baseball-management evaluation that can tolerate longer runtimes

That makes the default controlled league too expensive for routine publishing. The local simulator is fast, but the current manager loop still makes many live model calls during league games. In practice, the league dominates runtime.

## Goal

Introduce two explicit operating modes:

1. **Public Refresh Mode**
   Fast, repeatable, deploy-oriented, still useful as a manager benchmark.

2. **Deep Eval Mode**
   Slower, more complete, used for stronger internal comparisons and release-grade evaluations.

The same codebase, same data snapshot, and same scoring model should support both modes. The difference is call budget, schedule size, and how much live tactical management is delegated to the LLM.

## Principles

- Keep `analysis` and `decisions` intact across both modes.
- Use the controlled league as a realism layer, not the only source of tactical signal.
- Reserve live LLM league calls for moments with real evaluation value.
- Make public refreshes easy to run and easy to publish.
- Keep deep eval opt-in and clearly labeled in results.

## Mode Definitions

### Mode A - Public Refresh

**Purpose**

Produce a credible public leaderboard snapshot quickly enough to rerun after code or prompt changes.

**Target runtime**

- Ideal: 10 to 25 minutes
- Upper bound target: under 35 minutes

**League shape**

- `league_games`: 4 to 6
- sampled schedule only
- controlled league only by default

**Live management policy**

- Use live LLM calls only in high-leverage spots:
  - inning `>= 8` and score gap `<= 2`
  - extra innings
  - optional ninth-inning save/tying-run spots
- Hard cap live calls per team per game:
  - default `2`
  - maximum `3`
- All other decisions use validated manager-plan heuristics.

**What gets published**

- analysis results
- decision-track results
- GM build results
- controlled league summary
- public site rebuild

**Use cases**

- public site refresh
- benchmark regression checks
- prompt iteration
- “new model just dropped” same-day snapshot

### Mode B - Deep Eval

**Purpose**

Run a stronger, slower evaluation when signal matters more than turnaround.

**Target runtime**

- Ideal: 45 to 120 minutes
- Acceptable when run intentionally, not by default

**League shape**

- `league_games`: 12 to 18
- sampled schedule by default
- full matrix still opt-in only

**Live management policy**

- Broader live decision coverage:
  - pitching decisions from inning `>= 7`
  - offensive decisions from inning `>= 7`
  - always allow extra-innings calls
- Hard cap live calls per team per game:
  - default `5`
  - optional `6`

**What gets produced**

- everything in Public Refresh Mode
- optionally an Open League run on top of the controlled league
- richer comparison artifacts for internal review

**Use cases**

- model bake-offs
- release notes / writeups
- deeper quality validation before changing public defaults

## Recommended Defaults

### Public Refresh Defaults

```bash
.venv/bin/baseball-bench run-bench --mode public-refresh
```

Expanded behavior:

- `league_games = 6`
- controlled league enabled
- open league disabled
- `max_live_calls_per_team = 2`
- `live_call_inning_threshold = 8`
- `live_call_max_score_gap = 2`

### Deep Eval Defaults

```bash
.venv/bin/baseball-bench run-bench --mode deep-eval
```

Expanded behavior:

- `league_games = 12`
- controlled league enabled
- open league optional via explicit flag
- `max_live_calls_per_team = 5`
- `live_call_inning_threshold = 7`
- `live_call_max_score_gap = 3`

## Proposed CLI

### New top-level mode selector

Add:

- `--mode public-refresh`
- `--mode deep-eval`

Rules:

- default mode becomes `public-refresh`
- explicit flags like `--league-games` can still override mode defaults
- explicit low-level live-call flags override mode defaults

### New league-call controls

Add:

- `--live-call-start-inning`
- `--live-call-max-score-gap`
- `--max-live-calls-per-team`
- `--enable-open-league`
- `--disable-controlled-league` only for specialist runs, not normal use

## Runtime Behavior Changes

### ManagerPlan Becomes Primary

Both modes should rely on the `ManagerPlan` for routine baseball management:

- batting order
- platoon preferences
- bullpen hierarchy
- starter hook tendencies
- steal / bunt / pinch-hit aggression

The live LLM should not be asked to re-decide every routine spot that the plan already covers.

### League Calls Become Escalations

Live model calls should be treated as escalations:

- only when leverage is high
- only when the heuristic plan cannot reasonably stand in
- always with timeout and fallback

This preserves the benchmark's management signal without making every game a long sequential dialogue.

## Output Labeling

Mode must be visible in artifacts and the site:

- manifest includes `evaluation_mode`
- league summaries include `evaluation_mode`
- leaderboard cards show whether a run is `Public Refresh` or `Deep Eval`
- public site defaults to showing the latest `Public Refresh` snapshot
- deep runs remain visible in run history

## Implementation Plan

### Phase 1 - Introduce Mode Config

- Add a typed mode config object.
- Map `public-refresh` and `deep-eval` to concrete defaults.
- Thread the selected mode through `run-bench`, `run-league`, and site metadata.

### Phase 2 - Gate Live League Decisions

- Move league decision policy behind explicit config.
- Add inning/leverage/call-budget checks before `manager.decide(...)`.
- Use manager-plan / heuristic fallback automatically outside live-call windows.

### Phase 3 - CLI and Manifest

- Add new CLI flags.
- Record mode settings in `manifest.json`.
- Record mode settings in league progress and league summary artifacts.

### Phase 4 - Leaderboard and Public Publish Defaults

- Show latest `public-refresh` run as the default public snapshot.
- Keep deep runs in history with their own labels.
- Ensure `npm run cf:build` and deploy workflows naturally publish the latest public-refresh output.

## Success Criteria

### Public Refresh

- A normal public refresh run finishes comfortably enough to be used regularly.
- Controlled league still produces believable standings and per-game tactical variation.
- Public deploy cadence feels routine, not painful.

### Deep Eval

- Deep mode provides measurably more league signal than public refresh mode.
- Runtime is slower but still bounded and intentional.
- Differences between models can be investigated without relying only on the short public run.

## Recommendation

Change the repo default to **Public Refresh Mode** and treat **Deep Eval Mode** as the stronger comparison path.

This keeps the benchmark useful in day-to-day operation while preserving a serious management evaluation path when you actually want it.
