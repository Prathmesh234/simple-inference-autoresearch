# CHECKPOINT — read me first next session

_Last updated: 2026-06-03. Branch: `autoresearch/jun3`. Pushed HEAD: `223fd97`
(code commit `98f9c9a`)._

## 30-second TLDR
We are autonomously maximizing **`best_agg_tps`** (aggregate decode throughput at
instruct batch=128, T=1, measured INSIDE `torch.profiler`) on a pure PyTorch+Triton
Llama-3.1-8B engine, single RTX 6000 Ada (48 GB, 96 MB L2, ~960 GB/s HBM,
~360 TFLOP/s bf16, ~720 TOPS int8). Rules live in `program.md` (the bible / source
of truth): hypothesis → cheap standalone MIN-bench → implement → parity/coherence →
profiler A/B comparing SAME-CONFIG only → keep if win clearly beats ~±5% noise else
`git reset --hard`/discard. Log EVERY experiment (keep AND discard) to `results.tsv`
+ `workspace/PROGRESS.md`. Push after every win. NEVER stop to ask. Cross-engine
optimization catalog is `workspace/INFERENCE_ENGINE_OPTIMIZATIONS.md`.

**Current best (HEAD `98f9c9a`, pushed):**
| metric | value | note |
|---|---|---|
| best_agg_tps | **7309.8** (full sweep, instruct b128) | +45.5% vs jun3 baseline cd35f29 (5024.6) |
| best_agg_tps (ISO b128) | **~7996** (isolated, thermally fresh) | full sweep runs b128 last → throttled, ~9% lower |
| seq_tps_b1 | **94.8** | neutral (b1 uses W8A16 fallback) |
| peak_vram_gb | **38.53** (full) / 11.96 (iso b128) | |

## THE governing insight (do not relearn)
**At b128, decode is COMPUTE/MMA-bound, NOT bandwidth-bound.** ~28 ms/step moves
~8 GB int8 weights = ~280 GB/s effective ≪ 960 GB/s HBM. Consequences:
- int8 **MMA** (2× tensor-core throughput) wins on LARGE-N GEMMs → the lever.
- **bandwidth tricks (int4, KV-int8) are useless** except where a weight > 96 MB L2,
  and those (gate_up, lm_head) fail faithfulness at int4. All DEAD.
- Ada's 96 MB L2 makes "fewer redundant HBM reads" reorderings (GQA-grouping,
  split-K, transposed layout) no-ops. All DEAD.

## W8A8 decode-GEMM conversion is COMPLETE
**GENERAL PRINCIPLE (established EXP-K):** a decode GEMM can only go W8A8 if its
input comes from a kernel already doing a **per-row pass (a NORM)** that yields the
per-token int8 quant *for free*. The standalone `_quant_per_token` kernel is
occupancy-starved (~21 µs at M=128, K=4096: 128 programs on 142 SMs, serial wide
reduction) and torch-op quant is ~87 µs — either erases the GEMM win. So free
norm-fused quant (`RMSNorm.add_norm_quant`) is MANDATORY.

| GEMM | N | input source | status |
|---|---|---|---|
| gate_up | 28672 | post-attn RMSNorm | **W8A8** (EXP-E) ✅ |
| qkv | 6144 | attn RMSNorm | **W8A8** (EXP-J) ✅ |
| lm_head | 128256 | final RMSNorm | **W8A8** ✅ |
| down | 4096 | SwiGLU output | **W8A16** — BLOCKED (post-SwiGLU outliers, rel_err~26 at int8) |
| o_proj | 4096 | attention output | **bf16 cuBLAS** — BLOCKED (per-token scale spans all heads; flash_decode per-(B,head) can't emit it) |
First block always W8A16 (residual=None → no free quant). Layers 1–31 W8A8.

## What THIS session did
- **EXP-J qkv W8A8 — KEEP** (`85d5499`, pushed). Retuned int8 tile BLOCK_N=64
  (st3,nw8) = 2.24× over the W8A16 tile (prior "dead" used wrong (128,128,128)
  tile — same lesson as EXP-G). Free act-quant via `attn_norm.add_norm_quant`.
  ISO +4.0% perfect separation; full sweep +5.2%. rel_err 0.0045, greedy match 0.828.
- **EXP-K o_proj W8A8 — DEAD** (no commit). int8 GEMM 1.30× but quant
  occupancy-starved + free-quant structurally blocked (cross-head scale). Documented.
- **EXP-L down_proj W8A16 BLOCK_K=256 — KEEP** (`98f9c9a` kernel, `223fd97` docs,
  pushed separately). down is the dominant decode op (42.5% CUDA self-time). Tuner
  grid never tried BK=256: MIN-bench 94 µs vs 109 µs (~1.1×). ISO +1.9% perfect
  separation (NEW ~7996 vs BASE ~7845). Same math (fp32 acc, tiling only) → identical
  numerics; coherent output.

## Profiler op breakdown at HEAD (isolated b128, CUDA self-time)
`_w8a16_gemm` 42.5% (down + 2 first-block fallbacks) · `_w8a8_swiglu_fwd` 25.4%
(gate_up) · `_w8a8_gemm` 8.75% (qkv+lm_head) · `_flash_decode_fwd` 7.76% ·
o_proj bf16 cuBLAS 7.25% · `_add_rmsnorm_quant_fwd` 1.74% · rope 0.88%.
→ down_proj W8A16 is now the single biggest op (faithfulness-blocked from int8).

## Hard-won lessons (do NOT relearn)
1. **NO `@triton.autotune` in the decode path** — its `do_bench` allocates+syncs →
   illegal under CUDA-graph capture, and can hit a fragile config. Use per-shape
   **hardcoded** tiles found offline.
2. **"WRONG TILE" is the most productive lever** — EXP-G, EXP-J, EXP-L all found a
   GEMM using a default/shared tile mistuned for the real M=128 decode shape. Default
   (128,128,128) is ~2× too slow. Always MIN-bench tile sweeps; BLOCK_N=64 wins for
   N=6144, BLOCK_K=256 wins for down (N=4096,K=14336).
3. **int64 row indexing** in any new kernel (M·N can exceed 2^31 at prefill → illegal
   memory access surfacing inside a later kernel).
4. **MEASUREMENT NOISE ~±5%.** Compare SAME-CONFIG, same thermal window, multiple
   runs. Isolated b128 (~7900–8050) runs ~9% faster than b128 in a full sweep (it
   runs LAST, post-throttle). MIN-of-many for standalone tile compares (back-to-back
   is clock-drift corrupted). Wins >4% surface clearly; sub-2% wash out.

## CONFIRMED DEAD (cumulative — do not retry)
int4 gate_up (faith) · int4 lm_head (sub-noise) · down W8A8 (post-SwiGLU outliers) ·
o_proj W8A8 (quant occupancy + cross-head scale) · qkv/down cuBLAS bf16 (not better) ·
GQA-grouped flash decode · higher batch (compute-bound → linear) · INT4 RTN · KV-int8 ·
transposed layout · split-K · attention wq/wk/wv/wo W8A16 (skinny-N floor).

## Next-session candidate levers (decode GEMMs largely exhausted)
- **Re-trace the op table FIRST** with fresh eyes before picking a lever.
- Per-head-group-scale o_proj W8A8 + int8-emitting flash_decode (complex, ~+2%, risky).
- High-occupancy standalone quant kernel (split-K amax / 2-pass) to unblock o_proj/down.
- RoPE fusion into qkv epilogue (low value at compute-bound, complex).
- W8A8 prefill for TTFT (secondary metric; primary is decode b128).
- SmoothQuant offline calibration to unlock down/qkv outliers (big effort, low priority).

## Run env (MUST re-set every session — fresh box)
```
export UV_CACHE_DIR=~/.cache/uv XDG_CONFIG_HOME=~/uvconfig PATH=~/.local/bin:$PATH \
       HOME=/home/ubuntu PYTHONPATH=/home/ubuntu/simple-inference-autoresearch
```
uv may need install: `curl -LsSf https://astral.sh/uv/install.sh | sh`. torch
2.12.0+cu130, triton 3.7.0. Model cached. Profiler:
`nohup uv run python profiling/profile_engine.py [--flavors instruct --batch-sizes 1 128] > run.log 2>&1 &`
(isolated ~5 min, full sweep ~10–12 min). Grep `^best_agg_tps:|^seq_tps_b1:|^peak_vram_gb:|^best_agg_at:`.
`profile_engine.py` + `benchmarks/prompts.py` are OFF-LIMITS. `results.tsv` is
gitignored (LOCAL); `workspace/PROGRESS.md` + this file are TRACKED. Push:
`PAT=$(grep '^GITHUB_PAT=' .env | cut -d= -f2-); GIT_TERMINAL_PROMPT=0 git -c core.askpass= push "https://x-access-token:${PAT}@github.com/Prathmesh234/simple-inference-autoresearch" <sha>:autoresearch/jun3`.
Git identity: user.name "Prathmesh234", user.email "ppbhatt500@gmail.com", +
`Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>` trailer.
