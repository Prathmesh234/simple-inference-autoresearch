# CHECKPOINT — read me first next session

_Last updated: 2026-06-11. Branch: `autoresearch/jun11`. HEAD: `f24b12a`
(o_proj-W8A8, EXP-N). NOTE: best_agg_tps numbers on this box read ~16% LOWER than
jun3's box (different/shared hardware) — compare WITHIN this branch only. Baseline
here: ec04c90=5852.1, flash_decode ed11df9=6123.5, down-W8A8 64adb25=7445.0._

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

**Current best (HEAD `f24b12a`, pushed):**
| metric | value | note |
|---|---|---|
| best_agg_tps | **8009.5** (full sweep, instruct b128) | +7.6% vs down-W8A8 7445.0; confirm run 8044.7 |
| best_agg_tps (ISO b128) | **~8663+** (isolated, thermally fresh) | full sweep runs b128 last → throttled, ~16% lower |
| seq_tps_b1 | **94.4** | neutral (b1 uses W8A16/bf16 fallback) |
| peak_vram_gb | **39.07** | +0.54 GB for the int8 o_proj buffer (EXP-N); in budget |

## THE governing insight (do not relearn)
**At b128, decode is COMPUTE/MMA-bound, NOT bandwidth-bound.** ~28 ms/step moves
~8 GB int8 weights = ~280 GB/s effective ≪ 960 GB/s HBM. Consequences:
- int8 **MMA** (2× tensor-core throughput) wins on LARGE-N GEMMs → the lever.
- **bandwidth tricks (int4, KV-int8) are useless** except where a weight > 96 MB L2,
  and those (gate_up, lm_head) fail faithfulness at int4. All DEAD.
- Ada's 96 MB L2 makes "fewer redundant HBM reads" reorderings (GQA-grouping,
  split-K, transposed layout) no-ops. All DEAD.

## W8A8 decode-GEMM conversion (down NOW INCLUDED — EXP-M overturned the old "free-quant" principle)
**OLD principle (EXP-K, NOW PARTIALLY FALSIFIED):** a decode GEMM can only go W8A8 if
its input comes from a NORM that yields per-token int8 quant *for free*; else the
standalone `_quant_per_token` erases the GEMM win.
**EXP-M correction:** that holds only when the int8-MMA GEMM win is small. **down**'s
58.7 MB int8 weight FITS Ada's 96 MB L2 → the GEMM is compute/L2-bound, so int8 MMA
gives ~1.9× (50 µs vs 95 µs). That win is big enough to pay the standalone per-token
quant (~15 µs at K=14336) and still net +21.6% in-graph. **The standalone-quant tax is
worth it when the L2-resident weight makes the int8-MMA win large.** Also: the old
"down int8 rel_err~26" was PER-TENSOR; PER-TOKEN is ~0.07 naive / 0.038 SmoothQuant.

| GEMM | N | input source | status |
|---|---|---|---|
| gate_up | 28672 | post-attn RMSNorm | **W8A8** (EXP-E) ✅ |
| qkv | 6144 | attn RMSNorm | **W8A8** (EXP-J) ✅ |
| lm_head | 128256 | final RMSNorm | **W8A8** ✅ |
| down | 4096 | SwiGLU output (self-quant) | **W8A8** (EXP-M) ✅ tile 64/64/256/nw4/ns4, bucket 64<M≤256 |
| o_proj | 4096 | attention output (self-quant) | **W8A8** (EXP-N) ✅ tile 32/64/256/nw4/ns2, bucket 64<M≤256 |
First block always W8A16 (residual=None → no free quant). Layers 1–31 W8A8.

## What THIS session did (jun11 branch)
- **EXP-L flash_decode hardcode config — KEEP** (`ed11df9`, pushed). Removed
  `@triton.autotune` (key=["D"], D const → cached a bad first-shape config), hardcoded
  BLOCK_N=16/nw=1/ns=2. Op 138→58.7 µs (2.35×). +4.6% (5852.1→6123.5).
- **EXP-M down_proj W8A8 — KEEP** (`64adb25`, pushed). THE big one: +21.6%
  (6123.5→7445.0). See the W8A8 section above. Tuned tile (64,64,256,nw4,ns4) int8
  GEMM 50 µs vs 95 µs W8A16; self-quant `w8a8_down_linear`; dispatch 64<M≤256.
  Standalone gate `benchmarks/benchmark_kernel/bench_w8a8_down_compare.py`. Coherent
  greedy output verified (`workspace/scripts/check_down_w8a8_coherence.py`).
- **EXP-1 down tile BLOCK_M=128 — DISCARD** (reset). Slower in-graph (378 vs 313 µs).
- **EXP-N o_proj W8A8 — KEEP** (`f24b12a`, pushed). +7.6% (7445.0→8009.5, confirm
  8044.7). Overturns EXP-K "o_proj dead": GEMM is 2.11× (24 vs 50.7 µs bf16), NOT the
  "1.3×" wrong-tile claim; 16.7 MB int8 weight is L2-resident. Self-quant
  `w8a8_oproj_linear` (tile 32/64/256/nw4/ns2); dispatch 64<M≤256 in
  `_decode_graph_forward`; b1/prefill keep bf16. Coherent greedy output verified.
  KEY: the ~14 µs per-token quant is LAUNCH-bound (const across K) → hidden in the
  CUDA graph, so the in-graph win (+7.6%) dwarfs the standalone net.

## Profiler op breakdown at HEAD (isolated b128, CUDA self-time)
After EXP-N every major decode GEMM is int8: `_w8a8_swiglu_fwd` (gate_up, the largest
decode op) · `_w8a8_gemm` (qkv + down + o_proj) · `_flash_decode_fwd` · the
`_quant_per_token` launches (down + o_proj, launch-bound, hidden in-graph) ·
`_add_rmsnorm_quant_fwd` · rope · topk. `_w8a16_gemm` is now only lm_head/PREFILL/b1
fallbacks. There is NO remaining bf16/W8A16 GEMM in the b128 decode step.
→ Next lever is NOT another GEMM int8 conversion (done). gate_up swiglu is the biggest
  but retune-resistant; flash_decode is tuned. Look at fusion / launch-count next.

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
int4 gate_up (faith) · int4 lm_head (sub-noise) · qkv/down cuBLAS bf16 (not better) ·
GQA-grouped flash decode · higher batch (profiler caps sweep at b128 → irrelevant to
headline) · INT4 RTN · KV-int8 · transposed layout · split-K · attention wq/wk/wv W8A16
(skinny-N floor) · down tile BLOCK_M=128 (EXP-1, slower in-graph) · down W8A8 tile
re-sweep (64/64/256 optimal) · gate_up swiglu retune (sub-noise, BLOCK_N pinned at 64) ·
**W4A8 grouped-int4 gate_up (EXP-O)** — faithfulness PASSES (g=64 coherent) but the
hand-rolled Triton kernel is overhead-bound (0.90× best): gate_up W8A8 is only ~1.3×
above the HBM floor at M=128, so it is NOT HBM-bound enough to pay for int4
unpack + per-group fp32 rescale.
NOTE: BOTH "down W8A8 DEAD" (jun3, per-tensor artifact) AND "o_proj W8A8 DEAD" (EXP-K,
"1.3× GEMM") were WRONG — now the +21.6% (EXP-M) and +7.6% (EXP-N) wins. The o_proj
"dead" verdict used a generic tile (real is 2.11×) AND under-counted the graph-hidden
quant. LESSON: re-test any "dead" verdict that hinges on a WRONG TILE, PER-TENSOR
quant, or a STANDALONE quant cost (launch overhead vanishes in the CUDA graph).

## Next-session candidate levers (every major decode GEMM is now int8; W4 gate_up dead)
- **Re-trace the op table FIRST** with fresh eyes before picking a lever.
- **gate_up swiglu** (the biggest decode op): W8A8-fused, BLOCK_N=64 pinned; retune
  ~1.5% (sub-noise). W4A8 (EXP-O) is faithful but kernel-overhead-bound (0.90×). The op
  is only ~1.3× above the HBM floor → not enough headroom for any weight-bit trick to
  pay for the unpack overhead. Treat gate_up as DONE.
- **Reduce decode kernel launch count / fusion** — GEMMs are int8; next class of win is
  fewer launches. But the quant launches are already graph-hidden, so low value.
- **SmoothQuant α=0.7 folding** for down/o_proj — NO speed gain, only faithfulness
  margin; needs load-time calibration plumbing. Naive already coherent. Low priority.
- **W8A8 prefill** for TTFT (secondary metric; primary is decode b128).
- KEY LESSON THIS SESSION: an op being "the biggest" + "weight > L2" does NOT mean it is
  HBM-bound. Measure µs-above-HBM-floor first (down/o_proj were ~2× above → L2/compute
  bound → int8 MMA won; gate_up is only ~1.3× above → already near floor → no lever).

## Run env (MUST re-set every session — fresh box)
```
export UV_CACHE_DIR=~/.cache/uv XDG_CONFIG_HOME=~/uvconfig PATH=~/.local/bin:$PATH \
       HOME=/home/ubuntu PYTHONPATH=/home/ubuntu/simple-inference-autoresearch
```
uv may need install: `curl -LsSf https://astral.sh/uv/install.sh | sh`. torch
2.12.0+cu130, triton 3.7.0. Model cached. Profiler:
`nohup uv run python profiling/profile_engine.py [--flavors instruct --batch-sizes 1 128] > run.log 2>&1 &`
(isolated ~5 min, full sweep ~10–12 min). Grep `^best_agg_tps:|^seq_tps_b1:|^peak_vram_gb:|^best_agg_at:`.
`profile_engine.py` + `benchmarks/prompts.py` are OFF-LIMITS. `results.tsv` + `run.log`
are gitignored (LOCAL); the rest of `workspace/` (PROGRESS.md, this file, scripts) is
TRACKED. Push:
`PAT=$(grep '^GITHUB_PAT=' .env | cut -d= -f2-); GIT_TERMINAL_PROMPT=0 git -c core.askpass= push "https://x-access-token:${PAT}@github.com/Prathmesh234/simple-inference-autoresearch" HEAD:refs/heads/autoresearch/jun11`.
Git identity: user.name "Prathmesh234", user.email "ppbhatt500@gmail.com", +
`Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>` trailer.
