# CHECKPOINT — read me first next session

_Last updated: 2026-06-27. Branch: `autoresearch/jun27` (forked from main @ 348e6ae,
which has all jun11 wins merged). HEAD: swiglu-BK256 win + dead-end docs. NOTE: this
box reads ~16% HIGHER than jun11's (baseline best_agg_tps=9393 vs jun11's 8009) —
compare WITHIN this branch only. jun27 baseline 9393.0 -> 9609.7 after EXP-1._

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
| best_agg_tps | **9609.7** (full sweep, instruct b128) | +2.30% vs 9393 baseline (EXP-1) |
| best_agg_tps (iso b128) | **9819.7** | iso runs fresh, ~2% above full sweep |
| seq_tps_b1 | **94.8** | neutral (b1 uses W8A16/bf16 fallback) |
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
int4 gate_up (faith) · int4 lm_head (top50 0.79, faith) · W4A8 grouped gate_up (kernel
overhead) · **KV-int8/FP8 at headline (L2-resident short ctx, EXP-2 measured)** ·
**split-K W8A8 (EXP-6, atomic-dominated)** · GQA-grouped flash decode · transposed
layout · qkv/down/o_proj tile re-sweep (EXP-3 optimal) · gate_up split-vs-fused (EXP-5)
· lm_head retune (EXP-H + EXP-4 washout) · down BLOCK_M=128 (in-graph slower) · higher
batch (profiler caps b128) · **speculative decode / sampling-in-graph (profiler hardcodes
1 token/model-call + calls sample() externally → INCOMPATIBLE with the fixed harness)** ·
paged/right-sized KV (long flavors never beat instruct headline).

## Next-session levers (ALL <~1.5% or blocked — engine is near-optimal for this metric)
- down/o_proj activation-quant fusion into the producer epilogue: BLOCKED by per-row
  amax across the BLOCK_N tiles (swiglu tiles over I=14336); and the quant is graph-hidden
  (~3us launch-bound) so low value anyway (~1.4%).
- first-layer qkv W8A8 (currently W8A16, residual=None): ~0.2%, sub-noise.
- A production-grade Marlin/CUTLASS low-bit kernel for gate_up/lm_head (out of
  Triton-scope; would need a real cp.async pipeline to beat the int8 path on int4).
- TTFT/prefill W8A8 (secondary metric, not the headline).
- HONEST STATE: the b128 single-token decode is at its practical Triton optimum. The
  two GEMM groups (~75% of decode) are at their int8 limits; the rest (flash 10%,
  overhead 8%, sampling 3%) are near their floors. Re-trace the op table fresh first,
  but expect diminishing returns — the high-value axes are closed.

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
