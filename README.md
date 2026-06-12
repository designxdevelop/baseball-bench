# baseball-bench

`baseball-bench` is a local-first benchmark for evaluating AI models on baseball-flavored analysis, decision-making, and multi-step agency.

## Quickstart

```bash
uv sync
uv run scripts/run-bench --baseline
open results/site/index.html
```

The repository ships with a deterministic seed dataset so the full bench is runnable without live data downloads or API keys. For real model runs, set `OPENROUTER_API_KEY` and use OpenRouter-sourced model names. If you pass `openai/gpt-5` or `anthropic/claude-opus-4.8`, the CLI automatically routes them through OpenRouter.

You can evaluate multiple models in one invocation:

```bash
uv run scripts/run-bench \
  --model openai/gpt-5 \
  --model anthropic/claude-4.1 \
  --games 4
```

If you do not pass any model flags, `run-bench` defaults to a curated OpenRouter pack:

- `anthropic/claude-fable-5`
- `anthropic/claude-opus-4.8`
- `openai/gpt-5.5`
- `deepseek/deepseek-v4-pro`
- `nvidia/nemotron-3-ultra-550b-a55b`
- `qwen/qwen3.6-35b-a3b`

## Cost estimate

You can estimate run cost without launching the eval:

```bash
uv run scripts/estimate-cost
```

Or estimate a custom lineup:

```bash
uv run python -m baseball_bench.cli estimate-cost \
  --model openai/gpt-5.5 \
  --model anthropic/claude-opus-4.8 \
  --games 4
```

The estimator writes [results/cost-estimate.json](/Users/austin/code/dxd/baseball-bench/results/cost-estimate.json) and uses live OpenRouter pricing when available, falling back to a pinned local snapshot when offline. Use `--offline` to force the fallback pricing table.
