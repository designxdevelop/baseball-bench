# baseball-bench

`baseball-bench` is a local-first baseball benchmark for evaluating AI models on research, tactical decisions, roster construction, and manager-style league play.

The core idea is to separate different baseball skills instead of collapsing everything into one noisy league table:

- Can the model answer factual baseball questions from a database?
- Can it pick good late-game tactical moves?
- Can it build a coherent roster from a shared player pool?
- Can it manage equivalent talent better than other models?
- What happens when every model builds and manages its own team?

The benchmark uses a deterministic seed dataset and a local simulator. Model calls are used for research answers, isolated tactical decisions, GM roster construction, and manager plans. The simulator does not call a model for every pitch or plate appearance.

## Quickstart

Install dependencies and run the built-in baseline:

```bash
uv sync
uv run scripts/run-bench --baseline --league-games 3
open results/site/index.html
```

If you are using the checked-in virtualenv directly:

```bash
.venv/bin/baseball-bench run-bench --baseline --league-games 3
```

The baseline requires no API key. It rebuilds the deterministic DuckDB database, runs the SQL and win-probability baselines, creates heuristic GM/manager plans, runs a small controlled league, and writes a leaderboard.

## OpenRouter Runs

For real model runs, set `OPENROUTER_API_KEY` and pass OpenRouter model ids. The CLI automatically normalizes names, so `openai/gpt-5.5` becomes `openrouter/openai/gpt-5.5`.

```bash
export OPENROUTER_API_KEY=...

uv run scripts/run-bench \
  --model openai/gpt-5.5 \
  --model anthropic/claude-opus-4.8 \
  --league-games 12
```

If no model flags are passed, `run-bench` uses the curated OpenRouter bucket:

- `anthropic/claude-fable-5`
- `anthropic/claude-opus-4.8`
- `openai/gpt-5.5`
- `deepseek/deepseek-v4-pro`
- `nvidia/nemotron-3-ultra-550b-a55b`
- `qwen/qwen3.6-35b-a3b`
- `google/gemini-3.1-pro-preview`

Recommended full default bucket run:

```bash
uv run scripts/run-bench --league-games 12
```

Equivalent venv command:

```bash
.venv/bin/baseball-bench run-bench --league-games 12
```

## Track Overview

### Track 1: Research

Models answer baseball research questions against the pinned DuckDB database. This is the tool-use and factual-analysis track.

Inputs:

- Generated question set from the seed data.
- SQL-capable baseball database.

Outputs:

- `results/analysis-*.json`
- Per-run copies under `results/runs/<run-id>/`

Main score:

- `overall_accuracy`: fraction of research questions answered correctly.

Baseline:

- `sql-baseline`

### Track 2: Game Moves

Models choose one action from isolated late-game tactical situations. This is the cleanest in-game decision track because every model sees the same game states and action menus.

Inputs:

- Fixed late-game situations.
- Win-probability scoring model.

Outputs:

- `results/decisions-*.json`
- Per-run copies under `results/runs/<run-id>/`

Main scores:

- `mean_wp_delta`: average win-probability value relative to the best available move.
- `near_optimal_rate`: share of decisions close to the best move.

Baseline:

- `wp-baseline`

### Track 3: GM Build

Each model builds a roster from the same hitter and pitcher pool. This tests roster construction, baseball reasoning, constraint following, and balance.

Inputs:

- Shared player pool from the deterministic database.
- Hitter handedness and position.
- Pitcher role and throwing hand.

Roster artifact:

- `lineup`
- `bench`
- `rotation`
- `bullpen`
- `rationale`
- `validation`

Outputs:

- `results/gm-roster-*.json`

Scores:

- `validity_score`: correct sizes, valid player ids, no duplicates, correct pitcher roles.
- `lineup_score`: ordered hitter quality versus the benchmark heuristic.
- `pitching_score`: rotation and bullpen quality.
- `balance_score`: handedness and position coverage.
- `overall_score`: weighted GM score.

Run only GM construction:

```bash
uv run python -m baseball_bench.cli run-gm
```

Offline heuristic GM smoke:

```bash
uv run python -m baseball_bench.cli run-gm --offline
```

### Track 4: Controlled Manager League

Every model manages equivalent talent, but it gets to choose a manager plan first. This is the cleanest model-vs-model league comparison because roster strength is controlled.

Inputs:

- Same controlled roster pool for every model.
- Model-specific manager plan.

Manager plan artifact:

- `lineup`: batting order.
- `bench`: bench priority, also used for platoon and pinch-hit ordering.
- `starter`: selected starting pitcher.
- `bullpen`: preferred relief order.
- `bullpen_roles`: `closer`, `setup`, `middle`, or `long`.
- `tendencies`: tactical style hints.

Outputs:

- `results/manager-plan-*.json`
- `results/controlled-league-*.json`
- `results/controlled-league-progress-*.json`

League scores:

- Wins and losses.
- Run differential.
- Elo.
- Head-to-head table.
- Average decisions per game.

Progress files are written before, during, and after league play so a run can be checked for stalls. The progress JSON includes:

- `status`
- `total_games`
- `completed_games`
- `remaining_games`
- `current_game`
- `last_completed_game`
- live standings
- head-to-head summary

Run only the controlled league:

```bash
uv run scripts/run-league --models openai/gpt-5.5,anthropic/claude-opus-4.8 --league-games 12
```

### Track 5: Open League

The Open League uses each model's GM-built roster and then lets that model manage it. This is the full agent loop: research, roster construction, planning, and tactical management.

This track is useful and fun, but it is not the cleanest proof that one model is better at in-game tactics because roster quality and tactical quality are intentionally mixed.

Outputs:

- `results/gm-roster-*.json`
- `results/open-league-*.json`
- `results/open-league-progress-*.json`

Run the open competition:

```bash
uv run python -m baseball_bench.cli run-open-league --league-games 12
```

Offline GM roster construction with live/included managers:

```bash
uv run python -m baseball_bench.cli run-open-league --offline-gm --league-games 12
```

## League Sample Sizes

Avoid full matrix league runs for routine OpenRouter testing. With many models, full directed round-robin schedules grow quickly and can take a long time.

Use `--league-games` for sampled, seeded schedules:

- Smoke: `--league-games 3`
- Normal baseline: `--league-games 12`
- Stronger signal: `--league-games 18`
- Full matrix: `--full-league`

`--games` controls games per directed matchup when running the full matrix. By default, the CLI uses sampled league games unless `--full-league` is passed.

Examples:

```bash
uv run scripts/run-bench --league-games 3
uv run scripts/run-bench --league-games 12
uv run scripts/run-bench --league-games 18
uv run scripts/run-bench --full-league --games 2
```

## Local Matchup Simulator

The league simulator is deterministic and local. It uses model outputs for roster and manager setup, then simulates games without per-pitch model calls.

The current engine includes:

- Batter handedness versus pitcher throwing hand platoon effects.
- Hitter position and bench order for limited platoon lineup construction.
- Pitcher power, command, mistake, traffic, and contact-quality profiles inferred from available pitching stats.
- Park factors assigned deterministically by home manager.
- Fatigue and times-through-order penalties.
- Bullpen roles, rest, leverage ordering, and cross-game bullpen recovery.
- Batted-ball distribution for line drives, fly balls, ground balls, and popups instead of fixed singles/doubles/home-run rates.
- Sacrifice fly handling on fly-ball outs.

This gives the league more baseball texture while keeping runs cheap and reproducible.

## Cost Estimate

Estimate cost without launching a model run:

```bash
uv run scripts/estimate-cost
```

Estimate a custom lineup:

```bash
uv run python -m baseball_bench.cli estimate-cost \
  --model openai/gpt-5.5 \
  --model anthropic/claude-opus-4.8 \
  --league-games 12
```

The estimator writes `results/cost-estimate.json`. It uses live OpenRouter pricing when available and falls back to a pinned local snapshot when offline.

Force offline pricing:

```bash
uv run python -m baseball_bench.cli estimate-cost --offline
```

## Output Layout

Main generated files:

- `artifacts/baseball.duckdb`: deterministic local database.
- `results/data-checksum.json`: database checksum and seed hash.
- `results/analysis-*.json`: research track summaries.
- `results/decisions-*.json`: game-move summaries.
- `results/gm-roster-*.json`: roster construction artifacts.
- `results/manager-plan-*.json`: controlled manager plans.
- `results/controlled-league-*.json`: controlled league summary.
- `results/controlled-league-progress-*.json`: controlled league progress heartbeat.
- `results/open-league-*.json`: open league summary.
- `results/open-league-progress-*.json`: open league progress heartbeat.
- `results/leaderboard.json`: aggregated leaderboard data.
- `results/site/index.html`: rendered leaderboard.
- `results/runs/<run-id>/manifest.json`: per-run manifest.

The run manifest tracks:

- `status`
- `active_track`
- `tracks_completed`
- `started_at`
- `completed_at`
- models
- result files

## Common Commands

Build/reset deterministic data:

```bash
uv run scripts/build-data
```

Generate research questions:

```bash
uv run scripts/generate-questions
```

Run SQL research baseline:

```bash
uv run scripts/run-analysis-baseline
```

Run win-probability decision baseline:

```bash
uv run scripts/run-decisions-baseline
```

Build the leaderboard site from existing result JSON:

```bash
uv run scripts/build-site
open results/site/index.html
```

Run tests:

```bash
.venv/bin/pytest
```

## Interpreting Results

Use Track 1 and Track 2 for clean task-specific model quality.

Use GM Build to compare roster construction and constraint following.

Use Controlled Manager League for the cleanest model-vs-model league read because talent is controlled and manager planning is isolated.

Use Open League as a showpiece for the full baseball agent loop, not as the only final ranking.

Use progress files while long runs are active. If `completed_games` is not changing and `current_game.updated_at` is old, the run is likely stuck in a model call or league step.
