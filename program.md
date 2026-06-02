# program.md

You are an autonomous research agent. Your job is to make this inference engine
**serve more tokens per second** — both for a single stream and, especially, when
many requests are batched together — and to discover *how* by experimenting, not
by following a recipe. There is no fixed plan and there are no priors in this repo
on purpose. You decide what to try.

## The goal

Maximize **decode throughput (tokens/second)** for `meta-llama/Llama-3.1-8B`
running on a single GPU, in pure PyTorch + Triton, bfloat16 — measured as the
**total tokens/second the engine sustains across a batch of concurrent requests**,
not just one stream.

A single decode step at batch size 1 is memory-bandwidth bound and leaves the GPU
mostly idle. The big wins come from keeping more requests in flight per step so
the weights you stream from VRAM are amortized across many tokens. So you care
about two things at once:

- **Single-stream decode speed** (latency per token at batch size 1).
- **Aggregate throughput under batching** (total tok/s as batch size grows) — how
  well the engine scales when many sequences decode together.

Maximize aggregate throughput without letting single-stream latency collapse.
Optimize this relentlessly.

## Guardrails (do not regress these)

A change only counts as a win if, at equal output quality:

- **Aggregate decode throughput (tok/s across the batch)** goes up — the primary
  metric. Measure it across a range of batch sizes, not just one.
- **Single-stream decode latency** (batch size 1) does not collapse — a throughput
  win that makes one request painfully slow is a poor trade.
- **TTFT (prefill latency)** does not meaningfully regress.
- **Peak VRAM** stays within the GPU budget (must fit and leave headroom). Bigger
  batches cost more KV-cache memory — manage it.
- **Output stays coherent** — a faster engine that produces garbage is a failure.
  Sample from a fixed prompt set and sanity-check the text every iteration.

## Project structure

Know the layout before you touch anything. Each area has a job and a rule for how
you work in it.

- **`program.md` — the source of truth.** This file. It defines the goal, the
  guardrails, the loop, and these rules. When in doubt, it wins. If you discover a
  durable fact about the engine worth remembering, it belongs here — not scattered
  in code comments.

- **`iterations/03_engine.py` — the base engine, your starting point.** This is the
  scaffold you build on. Begin from it and evolve it *one step at a time*, each step
  an isolated, attributable change. You may rename it (`engine.py`, or whatever fits)
  and restructure it freely — it is yours to grow into the real serving engine. Earlier
  `iterations/01_*`/`02_*` files are history; read them for context, build on `03`.

- **`kernels/` — the custom kernels, built one file at a time.** Each kernel
  (`rmsnorm_kernel.py`, `rope_kernel.py`, `swiglu_kernel.py`, `attention_kernel.py`, …)
  lives in its own file. When you write a new kernel or fuse existing ones, add it
  here as its own file — one kernel, one file. Keep them self-contained and testable
  in isolation. Triton is what's here today, but you are **not limited to Triton** —
  if a hand-written CUDA kernel, a `torch.compile`d region, CUTLASS, or any other
  approach is the better tool for a hypothesis, write that instead. Same rule applies
  regardless of backend: one kernel per file, benchmarked standalone before it ships.

- **`model/`, `ops/` — the engine internals.** `model/` is the network
  (`llama.py`, `block.py`, `kv_cache.py`); `ops/` is the per-layer math
  (`attention.py`, `mlp.py`, `rmsnorm.py`, `rope.py`, `embedding.py`) that dispatches
  to either the Triton kernel or a PyTorch reference. This is where a new kernel gets
  wired into the model — but only *after* it passes its own benchmark (see below).

- **`benchmarks/` — kernel-level correctness + speed gate.** Before a new or changed
  kernel ever touches the model, benchmark it here in isolation against the PyTorch
  reference: confirm it is **numerically correct** and that it is actually **faster**.
  This catches a broken or slow kernel at the source and saves you from debugging it
  later through the whole engine. The pattern already exists in
  `benchmarks/benchmark_kernel/` (`bench_*_compare.py` compares Triton vs reference) —
  follow that same logic for every kernel you add, and re-run it every time you change
  one. **Rule: no kernel goes into the model until its standalone benchmark passes.**
  (`benchmarks/prompts.py` is the held-out harness — off-limits, see below.)

- **`profiling/profile_engine.py` — the end-to-end measurement, kept running.** This is
  how you judge the *engine* (not a single kernel): it sweeps every prompt flavor ×
  every batch size and reports aggregate throughput, single-stream latency, and peak
  VRAM. Run it every iteration to see how the whole engine is doing. **Do not modify the
  profiler or the scaffolding around it** — it is the fixed yardstick; changing it
  invalidates comparisons. And **always let it test every batch size** — never trim the
  sweep, because the whole point is seeing where the engine scales and where it hits the
  memory wall.

- **`config.py`, `loader.py`, `sampling.py`, `tokenizer.py` — fixed plumbing.** Model
  config, weight loading, sampling, tokenization. Read them for context; the math and
  weights here must stay faithful to Llama-3.1-8B (see *Guardrails*).

**The flow for any kernel change:** write/edit the kernel file in `kernels/` →
benchmark it standalone in `benchmarks/` (correct + faster) → wire it into `ops/`/`model/`
→ run `profiling/profile_engine.py` across the full batch sweep → log the result. Never
skip the standalone benchmark step.

## Setup

Before the experiment loop starts, do this once, with the user:

1. **Agree on a run tag.** Propose one based on today's date (e.g. `jun2`). The
   branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch:** `git checkout -b autoresearch/<tag>` from the current
   main branch.
3. **Read the in-scope files.** The repo is small. Read these for full context:
   - `program.md` — these instructions.
   - `iterations/03_engine.py` — the base engine (your entry point / scaffold).
   - `model/`, `ops/`, `kernels/` — the engine internals you will edit.
   - `config.py`, `loader.py`, `sampling.py` — model config, weight loading, sampling.
   - `profiling/profile_engine.py` — the benchmark/profiler you run each iteration.
   - Do **not** read `benchmarks/prompts.py` — it is the held-out harness (see
     *Off-limits*).
4. **Verify weights & environment.** `HF_TOKEN` must be set (Llama-3.1-8B is gated)
   — it lives in the `.env` file at the repo root, loaded automatically; `USE_TRITON`
   controls the kernel backend. Confirm the profiler runs end to end
   once: `uv run python profiling/profile_engine.py`. On first run this downloads
   the weights (~16 GB).
5. **Initialize `results.tsv`** with just the header row (see *Logging results*).
   The baseline is recorded after the first run. `results.tsv` is untracked — do
   not commit it.
6. **Confirm and go.** Once setup looks good, kick off the experiment loop.

## Experimentation

Each experiment is a full **inference benchmark across every input type and every
batch size**: you run `profiling/profile_engine.py`, which sweeps the whole grid of
prompt flavor × batch size and reports prefill, decode, aggregate throughput,
per-sequence throughput, and peak VRAM for each cell. There is no training and no
fixed time budget — the run is the sweep itself, so always compare runs with the
**same flavors, batch sizes, warmup, and token count**.

- **Flavors (input types):** by default it runs *all* registered prompt classes
  from `benchmarks/prompts.py` — `chat`, `chat_real`, `long_ctx`, `summarize`,
  `instruct`, `code` — so short decode-bound, long-prefill, real-traffic, and
  code-shaped inputs are each measured. Each flavor's distinct prompts are pulled
  fresh and left-padded into one batch.
- **Batch sizes:** by default `1, 2, 4, 8, 16, 32, 64, 128` — from single-stream
  latency up to heavy serving load. A batch that exhausts VRAM is reported as
  `OOM` for that cell and the sweep continues, so you can see exactly where this
  engine hits the memory wall. Override either axis with `--flavors` / `--batch-sizes`.

**What you CAN do:**
- Modify `kernels/`, `ops/`, `model/`, and `iterations/03_engine.py`.
- Add new files — a serving layer, scheduler, paged/continuous batching, fused
  kernels, whatever your hypothesis needs.

**What you CANNOT do:**
- Read, edit, import, or reproduce `benchmarks/prompts.py` (the held-out harness).
- Change the model weights, the tokenizer, or the correctness of the math — the
  output distribution must stay faithful to Llama-3.1-8B.
- Add dependencies beyond what's already in `pyproject.toml` unless the user agrees.

**The goal is simple: get the highest sustained aggregate decode throughput**
(tok/s across the batch) without letting single-stream (batch-1) latency collapse
or blowing the VRAM budget, at unchanged output quality.

**VRAM** is a soft constraint: some increase is fine for a real throughput win
(bigger batches need more KV cache), but it must still fit with headroom.

**Simplicity criterion:** all else equal, simpler is better. A small throughput
gain that adds ugly complexity is probably not worth it; the same gain from
*deleting* code is a clear win. Weigh complexity cost against improvement size.

**The first run** always establishes the baseline: run the profiler on the engine
as-is and record it before changing anything.

## The experiment loop

The loop runs on the dedicated branch (e.g. `autoresearch/jun2`).

LOOP FOREVER:

1. **Look at the git state** — the branch/commit you're on right now.
2. **Form a hypothesis and change one thing.** Edit the engine to implement a
   single idea. Keep the diff isolated so the result is attributable.
3. **`git commit`** the change.
4. **Run the experiment**, redirecting all output to a log (do NOT use `tee` or
   let output flood your context):
   `uv run python profiling/profile_engine.py > run.log 2>&1`
5. **Read out the results:**
   `grep "^best_agg_tps:\|^seq_tps_b1:\|^peak_vram_gb:" run.log`
6. **If the grep is empty, the run crashed.** Read the trace with
   `tail -n 50 run.log` and try to fix it. If you can't after a few attempts, give up.
7. **Record the result in `results.tsv`** (leave it untracked — do not commit it).
8. **If `best_agg_tps` improved** (higher) without breaking a guardrail
   (single-stream latency didn't collapse, VRAM still fits), **advance** — keep the
   commit. It is the new baseline.
9. **If it's equal or worse**, `git reset --hard` back to where you started.

**Crashes:** if a run crashes (OOM, a bug), use judgment. Something dumb (typo,
missing import) — fix and re-run. A fundamentally broken idea — skip it, log
`crash` in the tsv, move on.

**NEVER STOP:** once the loop has begun, do NOT pause to ask the human whether to
continue. They may be asleep and expect you to keep working indefinitely until
manually stopped. If you run out of ideas, think harder — read the papers and
blogs you found via web search, re-read the in-scope files for new angles, combine
previous near-misses, or try more radical changes (paged KV cache, continuous
batching, kernel fusion, quantized KV, speculative decode). The loop runs until
the human interrupts you, period.

## Output format

When `profiling/profile_engine.py` finishes it prints a per-flavor table (one row
per batch size) and then a single grep-able summary block:

```
==============================================================================
  SUMMARY (primary metric = best_agg_tps, higher is better)
==============================================================================
best_agg_tps:     4123.5
best_agg_at:      flavor=chat batch=32
seq_tps_b1:       58.2
peak_vram_gb:     31.74
```

- `best_agg_tps` — the headline: best aggregate decode throughput (tok/s) found
  anywhere in the batch sweep. **This is what you optimize.**
- `best_agg_at` — which (flavor, batch size) hit that peak.
- `seq_tps_b1` — single-stream decode tok/s at batch size 1 (latency guardrail).
- `peak_vram_gb` — worst-case peak VRAM across the sweep (memory guardrail).

Extract them with: `grep "^best_agg_tps:\|^seq_tps_b1:\|^peak_vram_gb:" run.log`

## Logging results

When an experiment is done, append a row to `results.tsv` (**tab-separated**, not
comma-separated — commas break in descriptions). The TSV has a header and 6 columns:

```
commit	best_agg_tps	seq_tps_b1	peak_vram_gb	status	description
```

1. git commit hash (short, 7 chars)
2. `best_agg_tps` achieved (e.g. `4123.5`) — use `0.0` for crashes
3. `seq_tps_b1` single-stream decode tok/s at batch 1 (e.g. `58.2`) — `0.0` for crashes
4. `peak_vram_gb` peak VRAM in GB, `.2f` (e.g. `31.74`) — `0.0` for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	best_agg_tps	seq_tps_b1	peak_vram_gb	status	description
a1b2c3d	3980.0	57.1	30.20	keep	baseline
b2c3d4e	4123.5	58.2	31.74	keep	fuse rmsnorm+residual in decode
c3d4e5f	3905.0	54.0	30.10	discard	larger triton attention block (regressed)
d4e5f6g	0.0	0.0	0.0	crash	paged kv cache (OOM at batch 32)
```

Keep `results.tsv` **untracked** by git — it is your experiment log, not part of
the engine.

## Off-limits (the evaluation harness)

You may **not** read, edit, or import `benchmarks/prompts.py`, and you may not
hardcode, cache, or otherwise reproduce its prompts in the engine. The benchmark
prompt set is the hidden test set: it is held out so you optimize the *engine*,
not the inputs. Treat the prompts you are measured on as unknown. Tuning to a
specific prompt — or shipping a fallback/offline copy of it — is cheating and
invalidates the result. The profiler supplies these prompts itself; you never see
or choose them.

## Hardware

Single NVIDIA RTX 6000 Ada — 48 GB VRAM, ~960 GB/s memory bandwidth, ~1457 TFLOPS
BF16. Batch-size-1 decode is autoregressive and **memory-bandwidth bound**: one
token at a time, the full model weights streamed from VRAM every step while the
compute units sit mostly idle. Batching is what fixes this — every extra sequence
you decode in the same step reuses those same streamed weights, turning wasted
bandwidth into real throughput, right up until you saturate compute or run out of
KV-cache memory. So your levers are: move fewer bytes per token, pack more
sequences into each step, keep the GPU busy, and don't launch more kernels than
you need. Reason from this machine's limits and find where the batch hits them.

## Be inspired, then go further

Production inference engines — and the research groups behind vLLM, SGLang,
TensorRT-LLM, FlashAttention, and others — have published *papers and technical
reports* full of ideas worth understanding: how they manage the KV cache, how they
batch requests, how they fuse and schedule kernels, how they exploit shared
prefixes, how they reduce memory traffic during decode.

**Read the research. Borrow the principles. Do not copy an architecture.**

You may **use web search** to find research papers, technical reports, and
engineering blogs (from vLLM, SGLang, TensorRT-LLM, FlashAttention, PyTorch, NVIDIA,
and others) for inspiration and to understand the state of the art. Search freely
when you need ideas, background, or a sanity check on what's achievable. Just
remember the rule above: take the *principles* and adapt them to this engine —
don't lift an implementation wholesale.

This repo deliberately ships with no blueprint so you are not anchored to anyone
else's design. Use the published ideas as inspiration and as a sanity check on
what's possible, then invent your own path. Novel combinations, simplifications,
and engine-specific tricks are encouraged — surprising solutions are the point.

Measure everything. Keep what wins. Maximize throughput — single-stream and across
the batch.
