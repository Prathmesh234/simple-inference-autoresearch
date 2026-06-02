# workspace/ — the agent's notebook

This folder is **yours**. It is free space for the autonomous research agent to
think, take notes, and record what it learns. Nothing here is read by the engine,
the profiler, or the benchmarks — it has zero effect on how the code runs. Use it
however helps you reason better and remember more across the loop.

## What to put here

- **Research notes.** Anything you read or figured out — summaries of papers and
  blogs (vLLM, SGLang, TensorRT-LLM, FlashAttention, …), how the KV cache works,
  why decode is bandwidth-bound, ideas worth trying. Write it down so you (or the
  next run) don't have to rediscover it.
- **Your understanding of the engine.** As you read `model/`, `ops/`, `kernels/`,
  the engine, and the profiler, capture how the pieces fit together — shapes,
  dataflow, where the time goes, where the memory goes. Build a mental map you can
  come back to.
- **A running log of experiments.** Beyond `results.tsv` (the terse numeric log),
  keep the *story* here: what you tried, what you expected, what actually happened,
  and why you think it happened. Dead ends are valuable — note them so you don't
  repeat them.
- **Thoughts mid-run.** Hypotheses, open questions, things to revisit, hunches you
  haven't tested yet.

## The important one: a "big wins" log

**Whenever a change makes a real difference, write it up here in detail.** This is
the highest-value thing in this folder. For any change that moved the needle (or
surprisingly *didn't*), record:

- What you changed (the idea, not just the diff) and the commit hash.
- The before/after numbers (`best_agg_tps`, `seq_tps_b1`, `peak_vram_gb`, and at
  which flavor/batch it showed up).
- **Why it worked** — the mechanism. Was it fewer bytes moved per token? Fewer
  kernel launches? Better occupancy? More sequences in flight? Tie the win back to
  the hardware limits.
- Anything surprising, any caveat, and what it suggests trying next.

A short, honest write-up of *why* something helped is worth more later than the
number itself — it's how the next idea gets found.

## How to organize it

No rules. Make whatever files and subfolders help — e.g. `notes.md`,
`engine-map.md`, `experiments.md`, `big-wins.md`, `papers/`, `ideas.md`. Markdown
is encouraged but anything goes. Keep it readable; this is the memory you leave
behind.
