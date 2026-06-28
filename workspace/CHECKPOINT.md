# CHECKPOINT — read me first next session

_Last updated: 2026-06-28. Branch: `autoresearch/jun27` (forked from main @ 348e6ae,
which has all jun11 wins merged). NOTE: this box reads ~16% HIGHER than jun11's
(baseline best_agg_tps=9393 vs jun11's 8009) — compare WITHIN this branch only._

## jun28 KV / long-context strategies (user-directed: paged/radix/fp8-KV)
The decode HEADLINE (instruct b128, short ctx ~9780) is at its GEMM-bound optimum and is
NOT KV/memory-bound, so no KV strategy moves it. KV strategies pay off on the LONG-CONTEXT
frontier — and there the gains are large:
| change | effect |
|---|---|
| EXP-16 last-position prefill logits | only the sampled position gets logits ([B*T,vocab]
  -> [B,vocab], the 29GB OOM source); opens code b128, instruct TTFT -7%, headline flat |
| EXP-17 int8 KV cache (max_seq_len>256) | halves KV bytes -> opens EVERY OOM b128 cell
  (chat/chat_real/long_ctx/summarize), peak_vram 44->36GB, code b128 +10%, headline flat |
| EXP-18 int8 flash BLOCK_N 16->32 (long-ctx tune) | long cells +3-9% (summarize +9.3%) |
NET: the engine now runs the ENTIRE flavor x batch sweep (to b128) with NO OOM, at LOWER
peak VRAM (36 vs 39 baseline), headline untouched. int8 KV is bf16 for instruct (headline),
int8 for long flavors (where EXP-2 showed int8 flash WINS at kv_len>=256). PagedKVCache +
paged flash built & validated (bit-identical) but NOT default — 0.93x at short ctx, and
uniform-length profiler batches make sequential blocks == contiguous + overhead, so it's the
substrate for ragged/shared-prefix serving the profiler doesn't exercise. Radix is low-value
here (distinct prompts share ~only BOS).

## jun27/28 decode-headline result: 9393.0 -> ~9780 (+4.24%), 5 real wins
| # | change | Δ | mechanism |
|---|---|---|---|
| EXP-1 | gate_up swiglu BLOCK_K 128->256, nw 4->8 | +2.30% | wider K-tiles hide int8 MMA under the HBM weight stream (gate_up is HBM-bound) |
| EXP-8 | strided-input RoPE (drop q/k .contiguous() copies) | +0.86% | qkv-GEMM output slices are non-contiguous; read them via stride, no copy |
| EXP-11 | skip dead bf16 normed write in W8A8 bucket | +0.62% | the bf16 normed is unused when qkv/gate_up are W8A8; freed HBM write BW |
| EXP-12 | W8A8 first-layer qkv (add_residual=False norm) | +0.40% | last W8A16 GEMM in decode -> W8A8; unified all norms onto add_norm_quant |
| EXP-10 | temp-after-topk (sampling) | ~0 | neutral, kept as simplification (drops full-vocab divide) |
Every projection in the b128 decode step is now int8 tensor-core (no W8A16 left).
LESSON (EXP-11): decode is HBM-WRITE-bandwidth-sensitive — cutting ANY dead write
helps via reduced GEMM L2/HBM contention, even if the cutting kernel's own time is flat.

## 30-second TLDR
Autonomously maximizing **`best_agg_tps`** (aggregate decode tok/s at instruct
batch=128, T=1, measured INSIDE `torch.profiler`) on a pure PyTorch+Triton
Llama-3.1-8B engine, single RTX 6000 Ada (48 GB, 96 MB L2, ~960 GB/s HBM,
~360 TFLOP/s bf16, ~720 TOPS int8). Rules: `program.md` (the bible). Loop:
hypothesis → cheap standalone MIN-bench → implement → parity/coherence → profiler
A/B SAME-CONFIG (iso instruct b1+b128 for fast A/B; full sweep to record) → keep if
win clearly beats ~±5% noise else `git reset --hard`/discard. Log EVERY experiment to
`results.tsv` + `workspace/PROGRESS.md`. Push after every win. NEVER stop to ask.

**Current best (HEAD, pushed):**
| metric | value | note |
|---|---|---|
| best_agg_tps | **9791.5** (full sweep, instruct b128) | +4.24% vs 9393 baseline (EXP-1/8/11/12) |
| seq_tps_b1 | **94.7** | neutral (b1 uses W8A16/bf16 fallback) |
| peak_vram_gb | **39.07** | unchanged |

## THE governing insight (refined this session)
At b128 the single-token decode is **GEMM-bound** and split into two regimes by the
96 MB L2:
- **HBM-bound** weights (> L2): gate_up (117MB int8) + lm_head (525MB). Only "fewer
  bytes" helps → int4 is **faith-dead** (EXP-I: top50 0.79 ≪ accepted 0.977) AND
  **kernel-overhead-dead** (EXP-O: hand Triton W4A8 0.90x). gate_up tile now optimal
  (EXP-1/EXP-5), 147us standalone vs 122us floor (irreducible for Triton).
- **L2-resident compute-bound** weights (< L2): qkv (24MB), down (58MB), o_proj (16MB).
  Limited by M=128 int8-tensor-core efficiency; tiles optimal (EXP-3), split-K dead
  (EXP-6, atomic reduction dominates), BLOCK_M=128 dead in-graph (jun11 EXP-1).
- **KV cache** at the SHORT instruct headline (kv_len~32-93) is **L2-RESIDENT too**
  (~32MB at kv_len=64 < 96MB) → flash_decode is L2/compute-bound, so KV-int8/FP8 is a
  no-op + dequant overhead LOSES (EXP-2 MEASURED: 0.80-0.96x at headline kv_len; only
  wins kv_len≥256). Long-context flavors never beat instruct's b128 agg.

## What jun27 did
- **EXP-1 gate_up swiglu BLOCK_K 128→256, nw 4→8 — KEEP** (70d1793, +2.30%). The one
  win. HBM-bound gate_up: wider K-tiles hide more int8 MMA under the weight stream.
  167.5→158.3 us/call in-graph, bit-identical. Standalone gate: tune_swiglu_jun27.py.
- **EXP-2 int8 KV cache — DISCARD/dead** (L2-resident short context; measured).
- **EXP-3 qkv/down/o_proj tile re-sweep — no win** (optimal).
- **EXP-4 lm_head tile (1.04x, EXP-H washout) + rope (in-graph 5.8us, no lever) — none.**
- **EXP-5 gate_up fused vs split — fused optimal** (147 vs 159-170us split).
- **EXP-6 split-K W8A8 o_proj/down — DEAD** (atomic reduction 0.08-0.31x).

## CONFIRMED DEAD (cumulative — do not retry without a NEW mechanism)
int4/W4A8 gate_up (KERNEL: faithful at g=64 but group=64 FORCES BLOCK_K=64, which is
GEMM-inefficient — same constraint that killed EXP-13; EXP-O got 0.90x) · int4 lm_head
(top50 0.79, faith-risky for the sampler) · **grouped-K quant fusion (down/o_proj quant
into swiglu/flash epilogue) — EXP-13 DEAD: per-group scales sidestep the per-row-amax
blocker but group=64 forces BLOCK_K=64, GEMM 0.78x, traffic saving < BLOCK_K penalty** ·
**KV-int8/FP8 at headline (L2-resident short ctx, EXP-2 measured)** · **split-K W8A8
(EXP-6, atomic-dominated)** · GQA-grouped flash decode · transposed layout · qkv/down/
o_proj tile re-sweep (EXP-3 optimal) · gate_up split-vs-fused (EXP-5 fused optimal) ·
lm_head retune (EXP-H + EXP-4 washout) · down BLOCK_M=128 (in-graph slower) · flash tile
re-sweep (EXP-7 optimal at headline kv_len) · in-graph pos-derivation (EXP-9 flat — tax
is on sample allocs, not pos copies) · higher batch (profiler caps b128) · **spec decode /
sampling-in-graph (profiler hardcodes 1 tok/model-call + calls sample() externally →
INCOMPATIBLE)** · paged/right-sized KV (long flavors never beat instruct headline).

## Next-session levers — HONEST STATE: engine is at its practical Triton/M=128 optimum
The b128 single-token decode is GEMM-bound; every projection is now W8A8 int8 tensor-core.
The two GEMM groups (~76%) are at their int8 limits (M=128 tensor-core efficiency); flash
(10.6%) is L2-bound + tile-optimal; the rest (~10%) is near memory floors. UNIFYING WALL:
group-size-forces-small-BLOCK_K kills every low-bit/grouped idea (int4, grouped quant).
The ONLY ≥1% lever is a production Marlin/CUTLASS-grade int4 GEMM (cp.async pipeline,
pre-shuffled weights, in-iter multi-group) that overcomes the BLOCK_K penalty — far beyond
hand-Triton, EXP-O scoped it out. Remaining hand-Triton ideas are all <0.3% AND add
complexity (KV k+v write fusion ~0.2% EXP9-noise; rope->cache-write fusion ~0.1%; final
norm->int8 for lm_head ~0.02%) → not worth it per the simplicity criterion. Re-trace the
op table fresh, but expect <1% total remaining without a new numerical format or a
non-Triton kernel backend.

## Run env (MUST re-set every session — fresh box)
```
export UV_CACHE_DIR=~/.cache/uv XDG_CONFIG_HOME=~/uvconfig PATH=~/.local/bin:$PATH \
       HOME=/home/ubuntu PYTHONPATH=/home/ubuntu/simple-inference-autoresearch \
       HF_HUB_DOWNLOAD_TIMEOUT=60
mkdir -p ~/uvconfig   # else uv errors on ~/.config/uv
```
uv install if missing: `curl -LsSf https://astral.sh/uv/install.sh | sh`. torch
2.12.0+cu130, triton 3.7.0. Model cached. HF dataset loads can flake (502) — just retry.
Profiler: `nohup uv run python profiling/profile_engine.py [--flavors instruct
--batch-sizes 1 128] > run.log 2>&1 &` (iso ~5 min, full sweep ~10-12 min). Grep
`^best_agg_tps:|^seq_tps_b1:|^peak_vram_gb:|^best_agg_at:`. DO NOT pkill -f
profile_engine.py from a shell whose own command line contains that string (it
self-kills). `profile_engine.py` is READABLE (only `benchmarks/prompts.py` is OFF-LIMITS).
`results.tsv` + `run*.log` gitignored. Push:
`PAT=$(grep '^GITHUB_PAT=' .env | cut -d= -f2-); GIT_TERMINAL_PROMPT=0 git -c core.askpass= push "https://x-access-token:${PAT}@github.com/Prathmesh234/simple-inference-autoresearch" HEAD:refs/heads/autoresearch/jun27`.
Git identity: user.name "Prathmesh234", user.email "ppbhatt500@gmail.com", +
`Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>` trailer.
