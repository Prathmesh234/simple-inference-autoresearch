# CHECKPOINT — read me first next session

_Last updated: 2026-06-03. Branch: `autoresearch/jun2`. Pushed HEAD: `5d3c9ad`._

## 30-second TLDR
We are autonomously maximizing **`best_agg_tps`** (aggregate decode throughput at
batch=128, T=1, measured INSIDE `torch.profiler`) on a pure PyTorch+Triton
Llama-3.1-8B engine, single RTX 6000 Ada (48 GB). Rules live in `program.md`
(source of truth): change ONE thing → standalone-bench → commit → profile → keep
if it beats ~6% noise else `git reset --hard`, push after every win, NEVER stop to
ask. Full experiment log is `workspace/PROGRESS.md`. Cross-engine optimization
catalog (vLLM/SGLang/TRT-LLM/FlashInfer) is `workspace/INFERENCE_ENGINE_OPTIMIZATIONS.md`.

**Current best (HEAD, pushed):**
| metric | value | vs prev (EXP10) |
|---|---|---|
| best_agg_tps | **4924.0** (instruct b128) | +10.9% |
| seq_tps_b1 | **83.0** | +48% |
| peak_vram_gb | **39.86** | lower (was 42.71) |

## What this session did (EXP11 + 1 bug fix, both KEPT & PUSHED)
**EXP11 — weight-only INT8 (W8A16) MLP under the CUDA-graph decode.**
- Commits: `d298ba9` (EXP11) and `5d3c9ad` (swiglu int64 fix). Both on
  `origin/autoresearch/jun2`. `results.tsv` has the `keep` row.
- WHY it wins now (the same idea LOST -13% pre-graph as EXP8): EXP10 put the whole
  decode step in a CUDA graph, so launch overhead ≈ 0 and the headline is now
  **weight-bandwidth bound**. INT8 halves the MLP weight stream (~70% of bytes/token).
  At b1 (pure bandwidth) the gain is biggest (+48%).
- Files: `ops/mlp.py` (gate_up/down as int8 registered buffers + per-output-channel
  fp32 scale; `load_weights` quantizes; forward calls `w8a16_linear_triton`).
  `kernels/w8a16_gemm_kernel.py` (per-channel symmetric int8 GEMM, bf16 `tl.dot`,
  fp32 accumulate, scale applied after).
- **MLP only.** Attention wq/wkv/wo lose (0.3–0.65x at skinny decode shapes), lm_head
  break-even — confirmed by the standalone gate `benchmarks/benchmark_kernel/bench_w8a16_compare.py`.

## Hard-won lessons (do NOT relearn these)
1. **NO `@triton.autotune` on any kernel in the decode path.** Its `do_bench` allocates
   + synchronizes → illegal under CUDA-graph capture, AND it crashed on a fragile config.
   Use per-shape **hardcoded** tiles instead (found offline via
   `workspace/scripts/tune_w8a16.py`). Current w8a16 tiles are M-bucketed + N-branched.
2. **int32 pointer overflow at large prefill M.** With M = B·S up to ~1e5 and N up to
   28672, `row*stride` exceeds 2^31 → wraps negative → async "illegal memory access"
   that surfaces inside a LATER kernel's autotune. Fixed BOTH `w8a16` (output store) and
   `swiglu` (row offset) to `.to(tl.int64)`. The swiglu bug was LATENT — only exposed
   because INT8 freed enough VRAM for b128 chat_real to reach the forward instead of
   OOMing early. **Any new kernel must use int64 row indexing.**
3. The headline is PROFILED, so historically launch-count cuts moved it. Post-graph that
   lever is exhausted; the remaining lever is **bytes** (bandwidth) → quantization.

## The open thread I was mid-investigation on (START HERE)
Profiler (instruct b128) now shows **`_w8a16_gemm` = 64% of all CUDA decode time**
(1.295 s / 4096 calls ≈ 316 µs avg), but the SAME kernel benches ~165 µs standalone at
M=128 — a **2× in-graph gap**. Rubber-duck + my analysis: at M=128 each weight byte is
reused across rows; the DOWN proj uses BLOCK_M=32, so its 59 MB int8 weight tile is
re-streamed 4× (M/BLOCK_M) across row-blocks. Standalone the ~48 MB L2 hides it; in the
full graph the KV/residual/RMSNorm traffic evicts L2 → DRAM re-reads → the 2×.
- I tried BLOCK_M=128 down tiles (weight loaded 1×) — SLOWER standalone (232 µs, only 32
  programs → low SM occupancy). No clear standalone win, so I did **not** change the
  kernel (avoid churn on noise-level deltas). See `workspace/scripts/bench_down_configs.py`.

## Next session — ranked plan (rubber-duck endorsed)
1. **Split-K for the DOWN projection only.** It's occupancy-limited (~128 progs on 142
   SMs, long 112-iter serial K loop), not bandwidth-reducible by bigger BLOCK_M. Fixed
   `SPLIT_K=2/4`, 2-pass reduction (zero output each step — cheap & graph-safe) or fp32
   atomics. Do NOT split gate_up (large N already saturates SMs). Profile down separately
   first (ideally Nsight: DRAM BW, SM active %, achieved occupancy) to confirm the model.
2. **INT4 weight-only (W4A16) MLP** (Marlin/AWQ, group=128) — halve MLP bytes again.
   Higher risk (bit packing, in-kernel unpack, group scales, rel_err ~2–4e-2, graph-safe,
   not a true int8 TC path). Only after confirming W8A16 is truly DRAM-bound in-graph.
3. **Paged / right-sized KV cache** — stop pre-reserving `max_seq_len` so long-prompt
   flavors batch higher (open NEW higher-agg cells) without OOM. Currently b128 OOMs on
   the `chat`/`chat_real` flavors; only `instruct` reaches b128.
4. Trim the ~10 remaining eager per-step ops (tok copy, pos/kvlen/cos/sin update, sample).

## Mechanics cheat-sheet
- Run profiler: `nohup uv run python profiling/profile_engine.py > run.log 2>&1 &`
  (~8–9 min incl ~3–4 min model load; output is buffered, appears near the end). Smoke:
  `--flavors instruct --batch-sizes 1 128`. Grep `^best_agg_tps:|^seq_tps_b1:|^peak_vram_gb:`.
- Standalone gate BEFORE wiring any kernel change: `uv run python
  benchmarks/benchmark_kernel/bench_w8a16_compare.py` (per-layer rel_err + speedup).
- Push (VSCode askpass HANGS headless — use this):
  `PAT=$(grep -E '^GITHUB_PAT=' .env | head -1 | cut -d= -f2- | tr -d '"'"'"' \r\n')`
  then `GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/true git -c core.askpass= push
  "https://x-access-token:${PAT}@github.com/Prathmesh234/simple-inference-autoresearch.git"
  <commit>:autoresearch/jun2` — ALWAYS mask with `sed "s/${PAT}/<PAT>/g"`. Push HEAD/
  descendant only (never an ancestor → non-fast-forward).
- Git identity REQUIRED: `-c user.name="Prathmesh234" -c user.email="ppbhatt500@gmail.com"`
  + trailer `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`.
- Env: torch 2.12.0+cu130, triton 3.7.0, `uv run python`. torchao NOT installed. No `tee`.
- DO NOT read/edit/import `benchmarks/prompts.py` (held-out). DO NOT modify
  `profiling/profile_engine.py`. `results.tsv`, `run.log`, `workspace/` are gitignored
  (workspace files are committed this session via `git add -f` per user request).

## Where everything lives
- `program.md` — rules (read every session).
- `workspace/PROGRESS.md` — full experiment log (EXP1→EXP11 + findings).
- `workspace/INFERENCE_ENGINE_OPTIMIZATIONS.md` — vLLM/SGLang/TRT-LLM/FlashInfer catalog.
- `workspace/scripts/` — all offline tuners/stress tests (e.g. `tune_w8a16.py`,
  `bench_down_configs.py`, `stress_swiglu.py`, `stress_w8a16.py`, parity checks).
- `results.tsv` (untracked) — one row per experiment; latest `d298ba9 … keep`.
- `kernels/w8a16_gemm_kernel.py`, `ops/mlp.py`, `kernels/swiglu_kernel.py` — EXP11 code.
- `model/cuda_graph_decode.py`, `kernels/flash_decode_kernel.py` — EXP10 graph decode.
