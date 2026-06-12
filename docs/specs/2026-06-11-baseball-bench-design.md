# baseball-bench — Design

**Date:** 2026-06-11
**Status:** Approved design, pre-implementation

## Purpose

A benchmark for testing the capabilities of new AI models using baseball as the domain. Three evaluation tracks measure different capabilities — agentic data analysis, decision-making under uncertainty, and sustained multi-step agency — all built on one shared foundation. Running a new model through the bench should be a single command; results feed a static leaderboard.

## Goals

- Objective, automatically gradable scores (no LLM-as-judge for primary metrics).
- Easy to run against any new model the day it ships.
- Deterministic and reproducible: pinned data, seeded sampling, versioned question sets.
- Cheap track first (data analysis), expensive showpiece last (sim league).

## Non-Goals

- Real-time/live MLB data (the bench uses historical data only).
- A hosted service or API. This is a local-first tool with a static published leaderboard.
- Fantasy baseball, projections, or betting use cases.

## Architecture Overview

One Python monorepo built on **Inspect AI** (model adapters, eval orchestration, scoring, logging, log viewer come free).

```
baseball-bench/
├── pyproject.toml            # uv-managed; deps: inspect-ai, duckdb, pydantic, ...
├── src/baseball_bench/
│   ├── data/                 # data layer: download, build, query DuckDB
│   │   ├── build.py          # Lahman + Retrosheet → baseball.duckdb (pinned versions)
│   │   └── schema.md         # documented schema given to models in prompts
│   ├── core/
│   │   ├── game_state.py     # typed GameState: inning, outs, runners, score, lineups, bullpen
│   │   ├── win_prob.py       # WP table derived from Retrosheet (base/out/inning/score states)
│   │   └── decisions.py      # decision space: swing actions a manager can take
│   ├── tracks/
│   │   ├── analysis/         # Track 1: agentic data analysis
│   │   │   ├── task.py       # Inspect task definition
│   │   │   ├── tools.py      # sql tool (read-only DuckDB), schema lookup tool
│   │   │   └── questions/    # versioned question sets (JSON: question, answer, tolerance)
│   │   ├── decisions/        # Track 2: in-game decision-making
│   │   │   ├── task.py
│   │   │   └── situations/   # sampled high-leverage Retrosheet situations (versioned)
│   │   └── league/           # Track 3: head-to-head sim league
│   │       ├── engine.py     # plate-appearance simulator
│   │       ├── manager.py    # LLM manager agent (uses core.decisions interface)
│   │       └── season.py     # scheduling, Elo, standings
│   ├── scoring/              # scorers shared across tracks
│   └── leaderboard/          # reads Inspect logs → static HTML/JSON site
├── scripts/                  # build-data, generate-questions, run-bench, build-site
├── results/                  # committed eval summaries (not raw logs)
└── docs/specs/
```

## Data Layer

- **Sources:** Lahman Database (season-level stats, free) and Retrosheet play-by-play (event-level data back to ~1910s, free with attribution).
- **Storage:** a single `baseball.duckdb` file built locally by `scripts/build-data`. Data versions are pinned (e.g. "Lahman 2025 release, Retrosheet through 2025 season") so every run queries identical data.
- The DuckDB file is gitignored; the build script is the source of truth. A checksum is committed to verify reproducibility.

## Track 1 — Agentic Data Analysis (build first)

**What it measures:** tool use, SQL reasoning, multi-step analysis, knowing when to verify.

- The model gets a read-only `sql` tool against DuckDB plus the documented schema.
- Question sets are **generated from the data** (template + sampling scripts), so ground truth is computed, not hand-written. Examples: "Which pitcher since 2000 had the highest K/9 in games his team lost (min 50 IP)?"
- Difficulty tiers: single-table lookups → multi-table joins → derived-stat reasoning (e.g. computing OPS+ style normalizations from raw data).
- **Scoring:** exact match for names/counts, tolerance match for rates/averages. Score = % correct per tier + overall.
- Target: 100–200 questions, versioned as `questions/v1.json`. New versions never mutate old ones.

## Track 2 — In-Game Decision-Making

**What it measures:** decision-making under uncertainty, applied domain reasoning, calibration.

- Sample real high-leverage situations from Retrosheet (late innings, close score, runners on). Each situation is rendered as a `GameState` with full context: matchup stats, bullpen availability, bench options.
- The model chooses from an explicit action menu (e.g. let him hit / pinch-hit with X / steal / bunt / pull starter for reliever Y) and must give structured output.
- **Scoring:** each action's expected win probability is computed from the WP table + empirical transition probabilities. Score = mean WP delta vs. the best available action (0 = always optimal). Also report % of decisions within 1% WP of optimal.
- The WP model is the bench's referee, not a claim about perfect baseball strategy — it's consistent across models, which is what a benchmark needs.
- Target: ~300 situations, versioned like Track 1.

## Track 3 — GM + Sim League (showpiece, build last)

**Active plan:** see [2026-06-11-gm-manager-league-redesign.md](/Users/austin/code/dxd/baseball-bench/docs/specs/2026-06-11-gm-manager-league-redesign.md). The original sim league direction has been expanded into a GM Track, Controlled Manager League, and Open League so roster construction and in-game management can be scored separately.

**What it measures:** sustained agency over long horizons, strategy, adaptation.

- **Engine:** plate-appearance simulator using log5/Markov-chain outcomes from real player season stats (e.g. 2025 rosters). Seeded RNG: identical seeds + identical decisions → identical games.
- **GM Track:** each model builds a roster from the same broad player pool under explicit constraints.
- **Controlled Manager League:** each model manages equivalent talent with a pregame manager plan plus a hard cap on live high-leverage calls.
- **Open League:** each model uses its own GM-built roster and manager plan in a labeled showpiece competition.
- **Cost control:** default league schedules are sampled and capped; full round-robin is opt-in only.

## Running the Bench

```bash
uv run scripts/build-data            # one-time: build baseball.duckdb
uv run inspect eval tracks/analysis  --model openrouter/openai/gpt-5.5
uv run inspect eval tracks/decisions --model anthropic/claude-4.6-opus
uv run scripts/run-league --models gpt-5.5,claude-4.6-opus --games 20
uv run scripts/build-site            # regenerate leaderboard from results/
```

Model access via Inspect's native provider support (Anthropic/OpenAI/Google direct, or OpenRouter for everything else). API keys via env vars.

## Leaderboard

- `scripts/build-site` parses Inspect eval logs + league results into `results/*.json` (committed), then renders a static site: overall table, per-track scores, per-question drill-down, league standings.
- Published via GitHub Pages. Adding a model = run evals, commit results, push.

## Error Handling

- Model refusals/malformed outputs: one retry with format reminder, then scored as incorrect (Track 1/2) or replaced by the baseline bot's choice with a logged penalty (Track 3). Malformed-output rate is itself reported — it's signal.
- SQL tool: query timeouts (10s) and row limits; errors are returned to the model as tool output (recovering from a bad query is part of the eval).
- Provider failures: Inspect's built-in retry; runs are resumable.

## Testing

- Unit tests for WP table construction, scorers, and the sim engine (distribution sanity checks: simulated league-wide stats within tolerance of real-world rates).
- Golden tests: a fixed mock-model transcript per track must produce a known score, so harness changes can't silently shift scoring.
- Question-generation scripts validated by re-computing every answer from the database in CI.

## Build Order

1. **Phase 0:** repo scaffolding, data layer, `baseball.duckdb` build + checksum.
2. **Phase 1:** Track 1 end-to-end (tools, question generation v1, scoring, first model runs) + minimal leaderboard.
3. **Phase 2:** WP model, situation sampling, Track 2 end-to-end.
4. **Phase 3:** sim engine, manager agent, league runner, Elo, leaderboard league page.

Each phase ends with the bench in a runnable, useful state.

## Open Questions (deferred, not blockers)

- Exact GM roster projection formula and whether to add budget/player-value constraints in v1.
- Whether Open League contributes to the overall score or remains a separate showpiece leaderboard.
- Statcast pitch-level data as a future Track 1 difficulty tier.
