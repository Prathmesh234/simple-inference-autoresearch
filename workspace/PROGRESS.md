# Autoresearch progress log — Llama-3.1-8B decode throughput

Branch: `autoresearch/jun2`  (run tag: jun2)
Goal: maximize `best_agg_tps` (aggregate decode tok/s) without collapsing
seq_tps_b1 (batch-1 latency) or blowing VRAM (<~48GB). Output must stay coherent.
Hardware: RTX 6000 Ada, 48GB, ~960 GB/s BW, ~1457 TFLOPS bf16.

(Commits also append the
 `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>` trailer.)

## Current state
- HEAD = `fc9586a` (current best / new baseline). KEEP this.
- Best so far: **best_agg_tps = 2905.1** (flavor=instruct, batch=128),
  seq_tps_b1 = 38.3, peak_vram = 45.20 GB.
- GPU is single-tenant; only ONE profiler run fits VRAM at a time (~42GB each).
  Always confirm GPU is free before launching (`nvidia-smi`).

## The loop (from program.md) — follow exactly
1. hypothesis -> change ONE thing
2. commit (with correct identity above)
3. `nohup uv run python profiling/profile_engine.py > run.log 2>&1 &`  (~6-7 min)
4. `grep "^best_agg_tps:\|^best_agg_at:\|^seq_tps_b1:\|^peak_vram_gb:" run.log`
5. append row to results.tsv (TAB-separated, UNTRACKED — never commit it)
6. if best_agg_tps improved AND guardrails ok -> keep; else `git reset --hard <prev>`
- The `uv run` wrapper spawns a child worker; find the real PID with
  `pgrep -f profiling/profile_engine.py`. Model load alone ~3-4 min.

## results.tsv so far (untracked)
    cff7d14  2589.3  35.3  44.73  keep     baseline
    b101634  2728.3  37.3  44.73  keep     skip per-step .contiguous() on K/V cache views
    8cb4c9f  2711.6  38.7  44.73  discard  fuse QKV into one GEMM (b1 helped, b128 flat)
    9732209  2863.8  37.4  44.73  keep     top-k-first sampling (skip full-vocab sort)
    fc9586a  2905.1  38.3  45.20  keep     fused add+RMSNorm (vLLM residual threading)

## Experiments done
### EXP1 — KEEP (b101634): skip `.contiguous()` on K/V cache views in attention
- `ops/attention.py`: pass `assume_contiguous=True` to `attention_flash_triton`.
- Flash kernel indexes q/k/v with explicit strides; only needs contiguous head_dim.
  Removing per-step clone of strided K/V prefix: +5.4% agg (2589->2728).

### EXP2 — DISCARD (8cb4c9f, reset away): fuse Q/K/V into one GEMM
- b128 headline flat (bandwidth-bound, not launch-bound), seq_tps_b1 37.3->38.7.
  Primary metric didn't move -> discarded. Could be revived for low-batch latency.

### EXP3 — KEEP (9732209): top-k-first sampling (sampling.py)
- A decode-step probe (`workspace/probe_decode.py`) showed model-only forward is
  FLAT ~28ms from B=1..128; only the sampling step scales with batch
  (0.65ms@B1 -> 2.96ms@B128, ~10% of decode at b128). So sampling was the only
  batch-scaling cost in decode.
- Old `sample()`: temperature -> filter_top_k (mask all but 50 to -inf) ->
  filter_top_p which **SORTS the full 128k-vocab row** (profiler: aten::sort
  9.79GB scratch, ~4% CUDA) -> softmax+multinomial over full 128k.
- New fast path (when 0<top_k<vocab): `torch.topk(logits,50)` to get the k
  candidates (sorted desc) + their ids, do top-p + softmax + multinomial in the
  k=50 dimension, then `idx.gather` the choice back to the real token id.
- Nucleus decision moved bf16 -> **float32** (canonical top-p). Standalone gate
  `benchmarks/benchmark_kernel/bench_sampling_compare.py` shows the fast path
  matches the float32 top-k/top-p reference **bit-for-bit (Δ=0)**. Difference vs
  the OLD path is only the old bf16 cumsum's coarseness at the 0.9 boundary
  (old was the less-accurate one). Sampling **~8x faster at b128** (2.82->0.35ms).
- Result: best_agg_tps 2728.3 -> **2863.8 (+5.0%)**, seq_tps_b1 37.3->37.4,
  VRAM unchanged. aten::sort gone from op table, replaced by tiny aten::topk.

### EXP4 — KEEP (fc9586a): fused add+RMSNorm (vLLM residual threading)
- Threaded the residual stream through every block so each sublayer's residual
  add folds into the *next* RMSNorm — removes 64 explicit elementwise add kernels
  per decode step (32 layers x 2). New kernel `kernels/add_rmsnorm_kernel.py`:
  reads hidden+residual once, computes `new_residual = residual + hidden`
  (f32-accumulated, rounded once -> bit-exact vs torch bf16 add) and normalises in
  the same pass. Wired via `RMSNorm.add_norm` (ops/rmsnorm.py), threaded in
  `model/block.py` (forward now returns `(hidden, residual)`) and `model/llama.py`
  (residual=None seed; final `norm.add_norm` folds the last pending add).
- Numerics: the residual *stream* is **bit-identical** to the unfused path
  (verified per-block in workspace/layerdiff.py, max|d|=0 across all 32 layers).
  Normed matches the float32 reference within bf16/reduction noise (rmsnorm_triton
  is itself autotune-non-deterministic at the f32-reduction level, so the bench
  gate requires new_residual bit-exact + normed within tolerance).
- GOTCHA: the residual add MUST accumulate in float32 then round once. An in-kernel
  bf16 `h+r` used a hardware bf16 add that drifted ~1 ULP on rare elements,
  compounding to 0.5 logit drift by layer 31. f32 add -> exact.
- Result: best_agg_tps 2863.8 -> **2905.1 (+1.4%)**, seq_tps_b1 37.4->38.3
  (latency improved too), VRAM 44.73->45.20 (still well under 48GB).

### EXP5 — KEEP (a3884ae): cache RoPE cos/sin dtype cast
- `attention.forward` was calling `cos.to(x.dtype)` / `sin.to(x.dtype)` every layer
  every step (64 tiny copy kernels/step). The rope kernel upcasts cos/sin to f32
  internally anyway, so the per-layer cast was pure launch overhead.
- Added a lazy per-dtype cast cache in `RopeFrequencies` (`_cast_cache` dict);
  `.get()` now takes a `dtype` arg and returns the cached cast. `attention.py`
  calls `rope_freqs.get(..., dtype=x.dtype)` and drops the per-call `.to()`.
- Numerics: bit-identical (Δ=0) — same values, just cached.
- Result: best_agg_tps 2905.1 -> **2968.5 (+2.2%)**, seq_tps_b1 38.3->40.4,
  VRAM 45.27.

### EXP6 — KEEP (43a7158): fuse Q+K RoPE into one kernel launch
- RoPE was launched twice per layer (once for Q, once for K). Rewrote
  `kernels/rope_kernel.py` into a single fused `_rope_qk_fwd`/`_apply_qk` with grid
  `(n_tokens, n_heads_q + n_heads_kv)`, branching on `col_pid < n_heads_q` to pick
  the Q or K tensor — halves rope launches (2->1 per layer).
- GOTCHA: assigning a dtype object (`out_dtype = ptr.dtype.element_ty`) INSIDE an
  if/else branch breaks Triton branch-variable tracking
  (`'dtype' object has no attribute 'type'`). Fix: hoist the dtype assignment ABOVE
  the branch. Head counts made constexpr (auto-specialize; removed from autotune key).
- Numerics: RoPE is elementwise (no reduction) so it's bit-identical regardless of
  autotune num_warps. Verified Δ=0.
- Result: best_agg_tps 2968.5 -> **3085.6 (+3.9%)**, seq_tps_b1 40.4->42.1,
  VRAM 45.27.

### EXP7 — KEEP (c0ae8ae): token-major flash-attention output (drop transpose+contiguous)
- Each layer did `out.transpose(1,2).contiguous()` to go from the flash kernel's
  (B, Hq, T, D) layout to (B, T, Hq, D) for the output projection — a full copy
  kernel + HBM round-trip per layer per step. Added `out_token_major` to
  `attention_flash_triton`: it allocates a (B, T, Hq, D) buffer and feeds the kernel
  the matching output strides, so the kernel writes token-major directly. The
  attention forward then `.view(B, T, hidden)` with NO transpose/contiguous.
- The SDPA fallback keeps its transpose+contiguous (its output is head-major).
- Numerics: bit-identical — verified max|Δ|=0 vs the transpose path for decode
  (Tq=1), causal prefill (Tq=7), and Tq=Tk=1 edge (workspace/attn_tm.py); coherence
  ok (gen_sanity.py).
- Result: best_agg_tps 3085.6 -> **3222.4 (+4.4%)**, seq_tps_b1 42.1->42.3,
  VRAM 45.27.

### EXP8 — DISCARD (27af4d2, reset to c0ae8ae): weight-only int8 (W8A16) MLP
- Quantized the MLP gate_up + down weights to per-output-channel symmetric int8
  (1 byte) + fp32 scale, dequantized inside a tensor-core Triton GEMM
  (kernels/w8a16_gemm_kernel.py: cast int8->bf16 in-register, tl.dot bf16 with
  fp32 accumulate, apply scale once). MLP weights are ~70% of streamed bytes, so
  the hope was to halve the dominant weight stream.
- STANDALONE the int8 GEMM beat cuBLAS bf16 1.8-2.1x at M=1 (decode b1, GEMV/
  memory-bound) but only TIED (~0.9-1.2x) at M=128 (the headline decode batch).
- Two crashes first: (a) autotune key included M -> prefill's varied M (B*prompt_len)
  triggered a 72-config re-tune storm (hung >13min). Fixed: short curated config
  list + key on bucketed M (capped 128). (b) int32 pointer overflow for long-prompt
  prefill (M~1e5, M*N>2^31) -> illegal memory access in the kernel AND in the
  pre-existing swiglu kernel. Fixed kernel with int64 offsets; routed large-M
  (>8192) prefill through a pure-torch dequant fallback (decode keeps int8).
- Numerics fine: greedy gen byte-identical to bf16 baseline; per-tensor rel-err ~1e-2.
- RESULT: best_agg_tps **2789.9 (-13% vs 3222.4)**, seq_tps_b1 38.2 (worse),
  VRAM 39.63 (-5.6GB). The Triton int8-upcast GEMM is less compute-efficient than
  cuBLAS bf16, and at decode batch the MLP GEMM is NOT purely memory-bound, so
  halving weight bytes didn't speed it up — it slowed both throughput AND latency.
  VRAM dropped but didn't open new higher-agg cells (instruct b128 still owns the
  headline, just slower). DISCARDED.
- LESSON: naive W8A16 (dequant->bf16) can't beat cuBLAS bf16 here. A real
  quantization win needs int8 tensor-core MMA (W8A8, int32 accumulate) or a GEMM
  that actually beats cuBLAS — not upcast-to-bf16.
- Decode is **weight-bandwidth bound**: `aten::mm` dominates (streaming 16GB
  weights/token). Model-only decode_ms is ~FLAT ~28ms B=1..128 (weights amortize
  across batch — why batching lifts agg throughput).
- The profiler reports instruct b128 decode ~44.7ms vs a clean ~31ms in the probe
  because the **largest batch of each flavor is measured UNDER torch.profiler**
  (capture_trace=True) — instrumentation overhead inflates the b128 cell. This is
  identical across runs, so run-to-run comparisons stay valid, but the "b128 jump"
  is partly a measurement artifact, NOT a real model-level cliff.
- Long-prompt flavors (long_ctx/summarize/code) OOM at b64/b128 — KV cache is
  statically pre-allocated for full max_seq_len. Only instruct (tiny prompts)
  reaches b128 and owns the headline.

## NEXT IDEAS (ranked)
1. **CUDA graphs / torch.compile(mode="reduce-overhead") for the decode step** —
   32-layer per-step launch overhead. Helps low batch (seq_tps_b1) and possibly
   b128. The decode region is static-shape per step except Tk grows by 1; capture
   per-Tk or pad KV reads. Biggest remaining lever on the flat ~28ms.
2. **Revive QKV fusion (EXP2)** bundled with #1 — clean latency win at low batch.
3. **fp8/int8 KV cache** — fewer bytes/token in the flash KV read AND shrinks the
   KV cache so long_ctx/summarize/code can reach higher batch without OOM, opening
   NEW higher-batch cells = potentially higher agg. Bigger change; needs a kernel.
4. **Paged / right-sized KV cache** — stop pre-allocating full max_seq_len; lets
   long-prompt flavors batch higher (new cells) without OOM.
5. **Fuse SiLU(gate)*up in MLP** / QKV fusion — more launch-count reduction (the
   profiled headline is op-count sensitive; EXP4 confirmed cutting launches helps).

## Guardrails / rules reminders
- DO NOT read/edit/import `benchmarks/prompts.py` (held-out test set).
- DO NOT modify `profiling/profile_engine.py` or trim its sweep.
- One kernel per file in `kernels/`; benchmark standalone (correct+faster) BEFORE
  wiring into ops/model: `benchmarks/benchmark_kernel/bench_*_compare.py`.
- Model math must stay faithful to Llama-3.1-8B (output distribution faithful;
  float32 top-p is the canonical reference).
- Keep results.tsv untracked.
- NEVER STOP the loop to ask permission to continue — keep iterating.

## Scratch files in workspace/ (safe, ignored by engine)
- `probe_decode.py` — decode component timing probe (full vs model-only vs sample).
- `bench_sampling_compare.py` is under benchmarks/benchmark_kernel/ (the gate).
- `PROGRESS.md` — this file. `plan.md` lives in the session-state folder.

## ============ SESSION UPDATE (EXP8-EXP10) ============

### Dead ends (DISCARDED)
- EXP8 W8A16 int8 weight-only MLP (Triton dequant->bf16 tl.dot): -13% headline
  (2789.9). Same FLOPs as bf16, less compute-efficient than cuBLAS at M=128. VRAM -5.6GB.
- EXP9 fuse K+V cache writes (one Triton launch): ~noise (3117.7). aten::copy_ is only
  0.46% of CUDA time, nothing to gain.
- W8A8 via torch._int_mm: standalone SLOWER than cuBLAS bf16 at M=128 (0.2-0.96x).
- bf16 reduced-precision reduction flag: already default True. No gain.
  => Quantization is a DEAD END at decode M=128 (int8 GEMM not optimized for skinny shapes).

### CRITICAL methodology findings
- ~6% run-to-run NOISE (Triton autotune nondeterminism). Need >6% effect or medians.
- Profiler op table: aten::mm = 82.5% of CUDA time. But...
- *** The b128 HEADLINE decode_ms is timed INSIDE the torch.profiler block ***
  (profile_engine captures a full trace for the max batch of each flavor).
  MEASURED at b128: CLEAN decode 28.4ms (4501 tps) vs PROFILED 39.9ms (3205 tps)
  = +40% (11.5ms/step) recoverable CPU-dispatch + profiler per-op overhead from
  ~193 op launches/step. THIS is why every EXP1-7 (launch-count cuts) moved the headline.

### EXP10 (KEEP) — CUDA graph the single-token decode step  [HEAD f18fd95]
best_agg_tps 3222 -> 4440.7 (+37.8%); seq_tps_b1 42.3 -> 56.0 (+32%);
peak_vram 45.27 -> 42.71 GB (lower). Coherent (batched parity 100%; single-stream
greedy a 3-token paraphrase deep in gen from bf16 non-determinism, still coherent).
- Collapse ~193 op launches/step into ONE graph replay -> removes the 11.5ms overhead.
- Per-step position lives in DEVICE buffers updated before replay (graph stays static):
  pos_index (KV write via index_copy_ dim=2), kv_len (int32 scalar), cos/sin (1,head_dim).
- NEW kernels/flash_decode_kernel.py: FlashAttention for Tq=1 reading kv_len from a
  device scalar (grid B*Hq static, loop bound + tail mask from device). Needed because
  SDPA-over-full-cache + bool mask DISABLES the fused flash backend -> math/gemv fallback
  -> 2x SLOWER at b128 (first graph attempt gave 1646 tps; kernel fixed it to 4685).
- ops/attention.py: _decode_graph_forward (decode_ctx path). model/cuda_graph_decode.py:
  CUDAGraphDecoder (lazy capture in warmup, keyed by kv_cache identity + B). model/llama.py
  routes eligible T==1 decode (start_pos>0, cuda, no-grad). USE_CUDA_GRAPH env (default on).
- Prefill / T>1 untouched. index_copy_ + device-kvlen flash are both CUDA-graph safe (validated).

### Next levers to consider (headline is launch/overhead sensitive under profiler)
- The 4440 b128 is now near the CLEAN GPU-bound floor; further headline gains are small.
- Per-step still has ~10 eager ops (tok copy, pos/kvlen/cos/sin updates, sample). Could
  shave but small. mm (82% CUDA) resists quantization at M=128.
- Paged/right-sized KV cache would let long flavors batch higher (new agg cells) w/o OOM.

### EXP11 (KEEP) — weight-only int8 (W8A16) MLP under CUDA-graph  [HEAD d298ba9 + swiglu fix 5d3c9ad]
best_agg_tps 4440.7 -> 4924.0 (+10.9%, >> 6% noise); seq_tps_b1 56.0 -> 83.0 (+48%);
peak_vram 42.71 -> 39.86 GB (lower). Coherent (per-channel int8 rel_err ~1e-2; EXP8
greedy was byte-identical). PUSHED both commits one-by-one.
- WHY IT WINS NOW (EXP8 W8A16 was -13% pre-graph): under the CUDA graph decode replay,
  launch overhead is gone, so the headline is WEIGHT-BANDWIDTH bound. int8 halves the MLP
  weight stream (~70% of bytes/token). At b1 (pure bandwidth) gain is largest (+48%).
- ops/mlp.py: gate_up/down stored as int8 registered buffers + per-output-channel fp32
  scale; load_weights quantizes via quantize_int8_per_channel; forward calls w8a16_linear_triton.
- kernels/w8a16_gemm_kernel.py: per-output-channel symmetric int8 GEMM, bf16 tl.dot,
  fp32 accumulate, scale applied after. NO triton.autotune (its do_bench alloc+sync is
  illegal under graph capture AND crashed on a fragile config). Per-M-bucket + per-N
  hardcoded tiles found offline (workspace/tune_w8a16.py): M<=16 (16,128,128,4,8);
  M<=256 gate_up N>=8192 (128,128,128,2,4) / down (32,128,128,3,8); prefill (128,256,64,4,8).
  Standalone gate at M=128: gate_up 1.59x, down 1.31x vs cuBLAS bf16.
- TWO int32-overflow bugs fixed (large prefill M = B*S up to ~1e5):
  (1) w8a16 output store offs_ym*stride_ym (M*N ~3e9 > 2^31) -> int64.
  (2) swiglu row_pid*gu_row_stride (~2.9e9) -> int64 (latent bug, exposed because int8
      frees VRAM so b128 chat_real now reaches the forward instead of OOMing early).
  Both surfaced as async "illegal memory access" inside a later kernel's autotune do_bench.
- Apply int8 to MLP ONLY: attention wq/wkv/wo LOSE (0.3-0.65x, skinny shapes), lm_head break-even.

### NEW BOTTLENECK (profiler, instruct b128): _w8a16_gemm = 64.07% of CUDA time (1.295s/4096 calls).
The int8 MLP GEMM is now THE dominant cost. Next levers target it:
- INT4 weight-only (W4A16) MLP: halve bytes again (Marlin/AWQ style). Biggest remaining lever.
- Split-K for the down proj (M=128 small, K=14336 huge -> SMs idle on long serial K-loop).
- Better int8 tiles / cp.async-style weight prefetch (Marlin 4-stage pipeline).

### Post-EXP11 investigation (no commit) — why is _w8a16_gemm 2x slower in-graph (316us) than standalone (165us)?
Rubber-duck flagged: at M=128 each weight byte is reused across rows; the DOWN proj uses
BLOCK_M=32 so its 59MB int8 weight tile is re-streamed M/BLOCK_M = 4x across row-blocks.
Standalone the ~48MB L2 partially hides this; in the full decode graph the KV-cache +
residual + RMSNorm traffic evicts it -> DRAM re-reads -> the 2x gap.
- Tuned down tiles offline (workspace/scripts/bench_down_configs.py) trying BLOCK_M=128
  (weightloads=1x): they are SLOWER standalone (232us, only 32 progs -> low SM occupancy).
  Best standalone stays (32,128,128)=139us; (64,64,128)=137us is within noise. NO clear
  standalone win, so the kernel was NOT changed (avoid churn on a noise-level delta).
- CONCLUSION: the in-graph 2x gap is an L2-contention / occupancy tradeoff, NOT a wrong
  tile. Fixing it needs either split-K for down (parallelize K reduction to fill SMs w/o
  growing BLOCK_M) OR a true int8 tensor-core path. Deferred to next session.

### Scripts relocated: all workspace/*.py moved to workspace/scripts/ (this session).

### NEXT SESSION — ranked levers (rubber-duck endorsed order):
1. Split-K for the DOWN projection only (M=128,N=4096,K=14336 -> ~128 progs on 142 SMs,
   long 112-iter serial K loop = occupancy-limited). Try fixed SPLIT_K=2/4, 2-pass
   reduction (graph-safe: zero output each step is cheap) or fp32 atomics. Do NOT split
   gate_up (large N already gives enough progs). Profile down separately first.
2. INT4 weight-only (W4A16) MLP (Marlin/AWQ, group=128) — halve MLP bytes again. Higher
   risk (packing, in-kernel unpack, group scales, rel_err ~2-4e-2, must stay graph-safe).
   Only after confirming W8A16 is truly DRAM-bound in-graph (Nsight: DRAM BW, SM active%).
3. Paged / right-sized KV cache: stop pre-reserving max_seq_len so long-prompt flavors
   batch higher (open NEW higher-agg cells) without OOM.
4. Reduce the remaining ~10 eager per-step ops (tok copy, pos/kvlen/cos/sin update, sample).

---

## Session jun3-cont (EXP-B): fused-qkv int8 + lm_head int8  — KEEP (frontier win)

### What shipped (commit 83fc4ff)
- **Fused QKV projection → one W8A16 int8 GEMM** (N=q+2kv=6144) replacing 3 bf16
  cuBLAS linears. ops/attention.py: wq/wk/wv → w_qkv_int8 buffer + scale (mirrors
  SwiGLUMLP int8 pattern); o_proj stays bf16 (clean square cuBLAS, int8 loses 0.56x).
- **lm_head (untied) → W8A16 int8** (ops/embedding.py OutputProjection). 525MB int8
  weight (>96MB L2 → genuinely HBM-bound), per-channel scale keeps top-50 ordering.

### Measurement (the important methodology lesson)
- Isolated `--flavors instruct --batch-sizes 1 128` A/B (same config, 2 samples each):
  - best_agg_tps: baseline [5583,5662] vs EXP-B [5724,5776] → **+2.5%** (no sample overlap)
  - seq_tps_b1:   baseline [82.3,83.0] vs EXP-B [94.7,95.9] → **+14.6%**
  - prefill_ms b1: 37.46 → 30.61 (**-18% TTFT**); b128 neutral (373.98→371.46)
  - peak_vram:    13.30 → 11.97 (**-1.33GB**, matches int8 weight estimate)
- Full default sweep: agg 5024.6→5055.3 (+0.6% only) — instruct b128 runs LAST so the
  GPU is thermally throttled; the kernel win is masked. seq_tps_b1 still +14.6% (94.8),
  vram -1.33GB (38.53). **LESSON: compare same-config; an isolated-run number vs a
  full-sweep baseline is apples-to-oranges (gave a false +14% before correction).**
- KEEP rationale: strict Pareto improvement (every guardrail better, agg positive),
  pushes the frontier outward on the interactivity (b1) + VRAM axes via "fewer bytes
  per token" (program.md's endorsed strategy). Faithful: per-channel int8, top50=0.98.

### DEAD ENDS confirmed this session (cheap standalone, no wasted profiler runs)
- **GQA-grouped flash decode** (1 program per (b,kv-head), stream K/V once across the 4
  sibling q-heads): standalone EXACTLY 1.00x at kv_len 93..512. Short-context KV fits the
  Ada 96MB L2 so the "4x redundant" reads were already free; long context loses to
  tl.dot M=4→16 padding. DEAD.
- **W8A16 b128 tile retune**: at reliable min-clock, gate_up current config optimal; down
  only ~4.5% (sub-noise). down weight (58MB) fits L2 so BLOCK_M=32's 4x re-read is free.
  (Earlier 9-12% readings were GPU-clock/throttle noise — always use min-of-N.)
- **Transposed (K,N) int8 weight layout** for the W8A16 GEMM: WORSE for gate_up
  (0.70-0.79x). The current (N,K) + tl.dot(.,.trans()) is already efficient. DEAD.
- **o_proj int8**: 0.56x vs cuBLAS bf16 (clean 4096² square). Stays bf16.

### UNIFYING INSIGHT
Ada's **96MB L2** makes every "read fewer *redundant* HBM bytes" trick useless at
short-context b128 (per-layer weights/KV are largely L2-resident). The only genuinely
HBM-bound ops are the ones whose single weight tensor exceeds 96MB: gate_up (117MB,
already a 1-stream BLOCK_M=128 optimum) and lm_head (525MB, now int8). Real wins now
require **fewer weight BYTES** (int4 numerics dead via RTN) or packing more sequences.

### NEXT SESSION — ranked levers
1. **Higher batch / right-sized KV cache**: with -1.33GB headroom + lower per-step VRAM,
   try b192/b256 at instruct (if decode_ms grows sub-linearly, agg climbs). Needs the KV
   cache to not pre-reserve max_seq_len. This is the clearest remaining aggregate lever.
2. INT4 W4A16 *with calibration* (AWQ/GPTQ) for gate_up only — the one >96MB HBM-bound op.
   High effort + faithfulness risk; only if #1 is exhausted.
3. Reduce ~10 eager per-step ops outside the graph (small, profiler-CPU-overhead only).

## EXP-H — retune W8A8 gate_up + lm_head tiles (commit b138eea) — DISCARD
Min-of-many standalone sweeps found faster configs (numerically identical):
gate_up _w8a8_swiglu_fwd BLOCK_K 128->256 nw 4->8 (157->149us, 1.05x); lm_head
_w8a8_gemm (128,128,128,2,4)->(128,256,256,2,8) (684->656us, 1.04x).
BUT the full-engine A/B did NOT confirm: isolated EXP-H [7446,7091,7472] vs EXP-G
[7340,7389,7552] overlap heavily (EXP-G mean higher); full sweep 6718.8 < EXP-G
6804.7. The ~1.5% standalone micro-win is below this window's profiler noise and
the wider tiles may add register/SMEM pressure in the full captured graph.
-> DISCARD (git reset --hard ed884c8). LESSON: standalone min-bench wins on
non-dominant kernels can vanish in the full engine; always confirm with the
profiler A/B and discard if it doesn't separate.

## EXP-I — INT4 weight on gate_up / lm_head (investigation) — DEAD (no commit)
Tested 4-bit (grouped RTN, group=32/64/128) on the two genuinely HBM-bound decode
weights to see if cutting weight bytes below the 96MB L2 helps.
- **Key realization first**: b128 decode is **compute/MMA-bound, NOT bandwidth-bound**.
  ~28ms/step moves ~8GB int8 weights = ~280 GB/s effective, far below 960 GB/s HBM.
  So "fewer weight bytes" (int4) cannot help at b128 except where a weight exceeds L2
  and forces HBM traffic — only gate_up (117MB int8) and lm_head (525MB) qualify.
- **int4 gate_up (117->58MB, would fit L2)**: faithfulness DEAD. Through 32 layers of
  nonlinear SwiGLU the RTN error compounds: final-logit rel_err ~0.22-0.26, argmax only
  0.83, top5 ~0.70 (g=32..128). Confirms the prior INT4-RTN "coherence dead" for
  transformer-block weights. Would need AWQ/GPTQ calibration (huge effort, still risky).
- **int4 lm_head (525->263MB)**: faithfulness OK for greedy (single final projection, no
  compounding): argmax 1.000, top1 1.000, top5 0.867, top50 0.79-0.86. BUT lm_head is only
  ~3% of decode (already W8A8) -> halving its bytes is ~1.2% headline = SUB-NOISE (EXP-H
  lesson: sub-2% washes out in the full engine). Plus top5 0.867 adds sampling-faithfulness
  risk and needs a new W4A8 (b128) + W4A16 (b1) packed/unpack kernel. Not worth it.
  (Could revisit ONLY as a b1-latency/VRAM play, which is a secondary frontier metric.)
-> Both DEAD for the b128 aggregate headline. The decode path's bandwidth tricks are
   exhausted; remaining levers must be COMPUTE/MMA-side, and W8A8 is already on every
   GEMM where it's faithful + faster (gate_up, lm_head). qkv/down/o_proj W8A8 stay dead.

## EXP-J — W8A8 int8-tensor-core qkv decode GEMM (free quant via attn_norm) — KEEP
The decode qkv projection (fused q+k+v, N=6144, K=4096) was the last big GEMM still
on weight-only W8A16. Prior sessions had marked "qkv W8A8 dead (0.976x GEMM-only,
0.465x self-quant)" — but that used the DEFAULT (128,128,128) tile. Re-tuning (same
"wrong shared tile" lesson as EXP-G) found BLOCK_N=64 (BM128,BK128,st3,nw8) runs the
int8 qkv GEMM at 22us vs 49us for the EXP-G-tuned W8A16 = **2.24x**. The standalone
_quant_per_token launch is the killer (self-quant e2e 0.655x), so — like EXP-D for
gate_up — the per-token int8 quant is fused for FREE into the preceding attn_norm
(add_norm_quant), and the W8A8 qkv GEMM consumes it directly.
Wiring: model/block.py attn_norm later-blocks -> add_norm_quant (emits int8+scale);
first block (residual=None) stays W8A16 (1/32 layers). ops/attention.py threads
x_int8/x_scale through forward -> _decode_graph_forward -> _project_qkv, which
dispatches W8A8 for 16<M<=256 (b128 decode) and W8A16 for M<=16 (b1) / M>256 (prefill).
New kernels/w8a8_gemm_kernel.py:w8a8_qkv_prequant (hardcoded tuned tile, graph-safe,
shape asserts). Reuses the existing per-channel int8 qkv weight buffers (no extra VRAM).
Faithfulness (clean isolation, same bf16 normed act): decode-shape rel_err 0.0045 /
cos 0.99996 (10x better than the accepted gate_up W8A8 ~0.06); 32-tok greedy match
0.828 (> accepted gate_up EXP-C 0.79); coherent output; 256-tok divergences are
base-model greedy-degeneracy (repetition attractors), not quant garbage.
Speed: ISO b128 A/B (same thermal window) NEW [7746.6,7706.4] vs BASE-ed884c8
[7453.0,7405.0] = +4.0% PERFECT SEPARATION (min>max). Full sweep 7008.7 vs same-window
ed884c8 confirmation 6660.2 = +5.2% (vs recorded 6804.7 = +3.0%). b1 neutral (94.8),
VRAM unchanged (38.53). KEEP. Cumulative vs baseline cd35f29 (5024.6): +39.5%.

## EXP-K — o_proj W8A8 (investigation) — DEAD (no commit)
o_proj (N=4096,K=4096) is bf16 cuBLAS today; prior "int8 0.56x dead" used the default
tile. Re-tuned int8 tile (BLOCK_N=32,BK=256,st3,nw4) GEMM-only = 22.25us vs cuBLAS
28.83us = 1.30x, and faithful (rel_err 0.0070, attention-output is bounded). BUT the
win needs FREE activation quant, which o_proj cannot get:
- Standalone _quant_per_token at M=128,K=4096 = 21us (occupancy-starved: 128 programs
  on 142 SMs, each a serial 4096-wide reduction) -> W8A8 e2e 45.8us = 0.63x (LOSES).
- torch-op quant (amax+div+round+cast) = 87us (many launches) = 0.21x. Worse.
- Free quant is STRUCTURALLY BLOCKED: o_proj's per-token scale spans all Hq*D=4096
  dims, but flash_decode's per-(B,head) programs each only see D=128 -> cannot compute
  the per-row amax. A per-head-group-scale W8A8 GEMM + int8-emitting flash_decode could
  work but is a large, risky redesign for a 6.75% op.
-> DEAD. GENERAL PRINCIPLE established: a decode GEMM can only go W8A8 if its input
   comes from a kernel already doing a per-row pass (a NORM) that yields the int8 quant
   for free. gate_up/qkv/lm_head (norm inputs) are all W8A8 now; down (swiglu-output
   outliers, faithfulness rel_err~26) and o_proj (attention output, cross-head scale)
   are blocked. The decode GEMM W8A8 conversion is COMPLETE.

## EXP-L — flash_decode: hardcode config (BLOCK_N=16, num_warps=1), remove autotune — KEEP
The decode attention kernel (kernels/flash_decode_kernel.py) used
`@triton.autotune(key=["D"])`. D=128 is constant, so the autotuner runs do_bench
ONCE on the first call's shape and caches that config for every later shape. In a
full sweep the first decode call is chat b1 (32 programs), so the cached config is
tuned for B*Hq=32 and then reused for the instruct b128 headline (B*Hq=4096) — a
slow fit: baseline full-sweep flash_decode = 138us/call (15.06% of decode). It also
varied run-to-run (iso 57-95us) because do_bench is noisy, polluting every A/B, and
a do_bench inside the decode path is unsafe under CUDA-graph capture (alloc+sync).
A direct config sweep of the headline shape (workspace/scripts/bench_flash_decode_cfg.py,
calling `_flash_decode_fwd.fn` to bypass the autotuner) found num_warps=1 is fastest
at every length tested (kv93 24.3us, kv512 314us, kv2048 1601us) — each program is a
tiny single-query attention, so minimal warps = maximal occupancy. The old autotune
list only offered num_warps>=2, so it could NEVER reach the best config. BLOCK_N=16
is best at the short headline kv_len AND at long contexts (kv2048), ~5% off only at
the mid kv~512 range. Hardcoded BLOCK_N=16, num_warps=1, num_stages=2; removed the
autotune entirely (aligns with the CHECKPOINT rule "no autotune in the decode path").
Numerics vs SDPA reference: rel_err ~3e-3 (bf16-level) across kv 40..2000. b1 decode
is config-insensitive (~22.7us flat) so single-stream is unaffected.
Result (full sweep): _flash_decode_fwd 138us -> 58.7us = 2.35x (15.06% -> 6.55%).
best_agg_tps 5852.1 -> 6123.5 = +4.6% headline. seq_tps_b1 94.9 -> 94.5 (neutral).
peak_vram 38.53 unchanged. KEEP. New dominant op: down _w8a16_gemm 48.17%.

## EXP-M — down_proj W8A8 (b128 decode bucket) — KEEP (+21.6%, the big one)
This OVERTURNS the EXP-K "down W8A8 is DEAD / decode GEMM W8A8 is COMPLETE" verdict.
Two prior blockers, both re-examined:

(1) FAITHFULNESS — prior "RTN rel_err~26" was PER-TENSOR quant. With PER-TOKEN
    (per-row) int8 act quant (what the _quant_per_token kernel actually does), the
    down GEMM rel_err on REAL post-SwiGLU activations (8 generic calib prompts, 32
    layers, workspace/scripts/probe_down_smoothquant.py) is only ~0.070 naive /
    ~0.038 SmoothQuant a=0.7 — at/below the accepted gate_up W8A8 band (~0.06).
    End-to-end teacher-forced next-token agreement vs bf16 (probe_down_w8a8_e2e.py):
    W8A16 100% | naive W8A8 93.8% | SQ a=0.7 97.2% (gate_up bf16). Greedy generation
    with the REAL engine (naive W8A8 down, B=128) is fully coherent and factually
    correct (Paris; Rayleigh scattering; Newton's laws) — program.md coherence gate
    PASSED with naive, so SmoothQuant was NOT needed to ship.

(2) "FREE QUANT REQUIRED" PRINCIPLE — falsified for down. The EXP-K principle said a
    decode GEMM can only go W8A8 if a preceding norm yields the int8 quant for free.
    down's input is the SwiGLU output (no per-row pass), so it pays a standalone
    _quant_per_token (~15us at K=14336). That principle held for o_proj because its
    GEMM win was small (1.30x). down is different: its 58.7 MB int8 weight FITS Ada's
    96 MB L2, so the GEMM is compute/L2-bound (not HBM-bound) and int8 tensor cores
    give a ~1.9x GEMM win that DWARFS the quant cost. Lesson: the standalone-quant tax
    is worth paying when the L2-resident weight makes the int8-MMA win big.

MECHANISM / TUNING:
- The generic W8A8 tile (128,128,128 — what made earlier "down W8A8 dead" benches show
  0.65-0.77x) is wrong for down's huge K=14336. A tile sweep at M=128,K=14336,N=4096
  found BM=64,BN=64,BK=256,nw=4,ns=4,GROUP_M=8: int8 GEMM 50.2us vs W8A16 95.5us =
  1.90x (below the 61us HBM floor -> confirms L2-resident, not HBM-bound). Same
  "wrong tile hid the win" lesson as EXP-G/qkv.
- Net standalone incl. the per-token quant (15.5us): M=128 -> 1.19x, M=256 -> 1.45x;
  M=32/64 LOSE (quant under-occupied), so dispatch is 64<M<=256 only. b1/prefill keep
  W8A16. New kernel: kernels/w8a8_gemm_kernel.py::w8a8_down_linear. Standalone gate:
  benchmarks/benchmark_kernel/bench_w8a8_down_compare.py.

RESULT (full sweep): instruct b128 decode 20.9ms -> 17.19ms. best_agg_tps
6123.5 -> 7445.0 = +21.6% headline. seq_tps_b1 94.5 -> 94.4 (neutral, b1 = W8A16).
peak_vram 38.53 unchanged. In-graph op table: _w8a8_gemm calls 2016 -> 4032 (qkv+down),
_w8a16_gemm calls collapse to lm_head/b1/prefill. KEEP.
NEXT: down is no longer the elephant; gate_up _w8a8_swiglu_fwd (24%) and the remaining
_w8a8_gemm are now the largest. Consider fusing the down quant into the swiglu epilogue
(atomic per-row amax) to reclaim the ~15us standalone quant, and/or SmoothQuant a=0.7
folding for extra faithfulness margin.

## EXP-N — o_proj W8A8 (b128 decode bucket) — KEEP (+7.6%)
OVERTURNS the EXP-K "o_proj W8A8 is DEAD" verdict, exactly like EXP-M did for down.
EXP-K killed o_proj on two grounds — both wrong:

(1) "GEMM only 1.30x" — a WRONG-TILE artifact (the same lesson as EXP-G/qkv and
    EXP-M/down). A tile sweep at the real decode shape (M=128, K=N=4096) found
    BM=32,BN=64,BK=256,nw=4,ns=2,GROUP_M=8: int8 GEMM 24.0us vs bf16 cuBLAS 50.7us =
    2.11x. o_proj's 16.7 MB int8 weight is L2-resident on Ada (96 MB L2), so the GEMM
    is compute/L2-bound and the int8 tensor cores (~2x bf16) pay off — even more so
    than down (smaller weight, more L2-resident).

(2) "free quant required / quant occupancy-starved" — the per-token _quant_per_token
    of the attention output (no preceding per-row norm) is ~14us standalone. KEY
    FINDING: that ~14us is CONSTANT across K=4096 and K=14336 (3.5x the data) and
    across num_warps -> it is LAUNCH/occupancy-bound, NOT execution-bound. Inside the
    captured CUDA decode graph (replay skips launch overhead) that cost largely
    vanishes, which is why the in-graph win (+7.6%) DWARFS the standalone net the
    wall-clock bench shows (the standalone bench compares a 2-launch Triton path
    against single-launch cuBLAS, so it under-reports — use CUDA-event device time).

(3) "cross-head scale" faithfulness — the per-token scale spans all Hq*D=4096 (all
    heads share one scale). FALSIFIED qualitatively: greedy generation with the REAL
    engine (B=128, both down + o_proj W8A8) is fully coherent and factually correct
    (Paris; Rayleigh scattering; cookie recipe; Newton's laws). rel_err 1.2e-2 vs bf16.

MECHANISM / TUNING:
- New kernel kernels/w8a8_gemm_kernel.py::w8a8_oproj_linear (self-quant + tuned tile
  32/64/256/nw4/ns2). Dispatch in ops/attention.MultiHeadAttention._decode_graph_forward
  for 64<M<=256; b1 and prefill keep bf16 cuBLAS (self.wo Parameter retained). Weight
  quantized to int8 buffers (w_o_int8 / w_o_scale) at load_weights.
- Standalone gate: benchmarks/benchmark_kernel/bench_w8a8_oproj_compare.py (faithfulness
  rel_err 1.2e-2; standalone wall-clock under-reports speed — see note above).

RESULT (full sweep): best_agg_tps 7445.0 -> 8009.5 = +7.6% headline; confirmation run
8044.7 (stable, ~0.4% apart). seq_tps_b1 94.4 (neutral, b1 = bf16). peak_vram
38.53 -> 39.07 (+0.54 GB int8 o_proj buffer, in budget). KEEP.
NEXT: with qkv, gate_up, down, lm_head AND o_proj all int8, every major decode GEMM is
now W8A8. flash_decode (already tuned) and the per-token quant launches are the only
remaining non-int8 decode work. The launch-bound quant insight (~14us, hidden in-graph)
explains why both down and o_proj won bigger in-graph than standalone — future quant
fusion (e.g. into the flash_decode epilogue) is low-value since the cost is graph-hidden.

## EXP-O — W4A8 grouped-int4 gate_up (the HBM-bound elephant) — DISCARD (standalone 0.90x)
HYPOTHESIS: gate_up is the biggest decode op (~25%) and its 117 MB int8 weight exceeds
Ada's 96 MB L2, so the fused W8A8 swiglu should be HBM-bound; packing the weight to
4-bit (59 MB, also L2-resident) would halve the weight read and approach ~2x.

FAITHFULNESS — PASSES (overturns the old "int4 gate_up dead (faith)" verdict, which
used per-tensor/per-channel int4). Per-GROUP int4 (group=64 along K):
- faith_gateup_int4.py: final-token argmax agreement g=64/g=32 = 1.000 (g=128 = 0.833),
  though logit_rel ~0.24-0.28 and top5 overlap ~0.6-0.7 (looser than down/o_proj).
- coherence_gateup_int4.py: REAL-engine greedy generation at g=64 is fully coherent and
  factually correct (Paris/Washington/London/Ottawa/Moscow/Rome capitals; Rayleigh
  scattering; Newton's laws). So faithfulness is NOT the blocker.

KERNEL PERF — FAILS (the real blocker). Built kernels/w4a8_swiglu_kernel.py: split-half
int4 packing (byte holds K-elem j low-nibble + j+32 high-nibble within each 64-group),
grouped fused GEMM+SwiGLU with int8 activation, int32 dot per group scaled by the
per-group weight scale into an fp32 accumulator. Numerically correct (rel 1.4e-2 vs the
grouped-int4 dequant reference) BUT slower: best tuned config (BM128/BN64/nw4/ns2) =
175us vs 158us W8A8 = 0.90x; default 197us = 0.80x.
WHY: gate_up W8A8 (158us) is only ~1.3x above the 122us HBM floor (117MB/960GB/s) — it
is NOT strongly HBM-bound at M=128, so halving the weight read saves little, and the
int4 unpack (mask/shift/sign-extend) + per-group fp32 rescale (64 groups x (128x64)
FMA x2) + small per-group dots OUTWEIGH the HBM savings. A production low-bit kernel
(Marlin/AWQ-style layout + pipelining) might win, but that is far beyond scope and the
hand-rolled Triton path clearly loses here.
VERDICT: DISCARD. Removed the un-wired kernel + bench. Kept the faith/coherence scripts
as evidence. NEW CONFIRMED-DEAD entry: W4A8 grouped gate_up (kernel overhead-bound; the
op is only ~1.3x above HBM floor, not HBM-bound enough to pay for int4 unpack+regroup).

---

# ============ SESSION jun27 (branch autoresearch/jun27, fresh box) ============
Box reads HIGHER than jun11: baseline full-sweep best_agg_tps=9393.0 (vs jun11's
8009). Compare WITHIN jun27 only. Re-traced the op table fresh (instruct b128,
per 13.6ms decode step): gate_up swiglu 5.36ms (39%), qkv+down+o_proj _w8a8_gemm
4.99ms (37%), flash_decode 1.39ms (10%), norms/quant/rope/KV-write ~1.1ms (8%),
sampling ~0.35ms (3%). (The _w8a16_gemm 365ms in the raw table is PREFILL — 96
GEMMs @ M=3712 + the 63 first-layer-qkv W8A8-ineligible decode calls; NOT decode.)
Call-count decode breakdown of _w8a8_gemm (6048 calls): qkv 1953 (layer0 is W8A16)
+ down 2016 + o_proj 2016 + lm_head 63. So lm_head (~600us/call, 525MB HBM) is
~4.5% of decode hidden inside _w8a8_gemm; down is the biggest single GEMM (~11%).

## EXP-1 — gate_up swiglu BLOCK_K 128->256, num_warps 4->8 — KEEP (+2.3%)
gate_up is the biggest decode op; 117MB int8 weight > 96MB L2 => HBM-bound (122us
floor) and sat at 167.5us in-graph (~45us un-hidden int8 MMA). Widening BLOCK_K
128->256 (half the K-loop trips, wider weight loads) + num_warps 4->8 (spreads the
two int32 accumulators over more threads, eases the register pressure that pins
BLOCK_N=64) hides more MMA under the weight stream. Standalone (tune_swiglu_jun27.py):
147.1us vs 156.0us = 1.06x, bit-identical (rel_err=0). ns>=3 fails (>128KB smem at
BK=256). Only w8a8_swiglu_prequant's launch config changed. Iso A/B 9565.6->9819.7
(+2.66%); full sweep 9393.0->9609.7 (+2.30%); swiglu op 167.5->158.3us/call.
seq_tps_b1 neutral (94.8). Commit 70d1793, pushed. NEW BASELINE.

## EXP-2 — int8 KV cache — DISCARD for the headline (MEASURED, overturns nothing)
Built kernels/flash_decode_int8_kernel.py (int8 K/V + per-(b,head,pos) fp32 scale,
dequant folded into online softmax) + standalone gate (bench_flash_int8_jun27.py).
The jun11 "KV-int8 dead" was an unmeasured analogy to the L2-resident weights; I
hypothesised flash_decode is KV-bandwidth-bound and re-tested. RESULT: faithfulness
EXCELLENT (rel_err ~0.017, cos 0.99998 — attention is a weighted avg, robust to KV
quant), BUT speed at the SHORT instruct headline LOSES: kv_len 32:0.96x 64:0.88x
128:0.80x | only wins at 256:2.40x 512:1.62x 1024:1.40x. MECHANISM: at b128 short
context the K+V cache (~32MB at kv_len=64) FITS the 96MB L2 -> flash_decode is
L2/compute-bound, NOT HBM-bound -> halving bytes is a no-op and the extra scale
loads + dequant FMA make it slower. So jun11 was RIGHT for the headline (now with
the correct mechanism: L2-resident short-context KV, same L2 insight as weights).
int8 KV helps long-context + halves KV VRAM, but long flavors never beat instruct's
b128 agg (longer KV = slower). NOT wired (simplicity). Kernel+bench kept as evidence.
CONFIRMED-DEAD for the b128 headline.

## EXP-3 — re-sweep qkv/down/o_proj _w8a8_gemm tiles — no win (tiles optimal)
Applied the EXP-1 "wrong-tile/un-swept-dim" lever to the other GEMMs
(tune_w8a8_gemm_jun27.py, MIN-of-many at real decode shapes). qkv (128,64,128,8,3)
optimal (best alt 0.996x); down (64,64,256,4,4) optimal (all alts slower); o_proj
best alt (64,64,256,4,4)=1.036x — SUB-NOISE (o_proj ~7% of decode -> 0.26% headline;
EXP-H lesson: sub-2% standalone washes out). The BLOCK_K=256 win is SPECIFIC to the
HBM-bound gate_up (hides compute under the weight stream); the L2-resident
compute-bound GEMMs (qkv/down/o_proj) don't have that lever. GEMM-tile axis exhausted.

## EXP-4 — lm_head tile + rope config probes — no win
(a) lm_head W8A8 GEMM (HBM-bound like gate_up): best (128,256,256,8,2)=655us vs
current (128,128,128,4,2)=683us = 1.042x (== the EXP-H config already DISCARDED for
washing out). 4.5%-of-decode op -> 0.19% headline. SUB-NOISE.
(b) rope: standalone "autotuned" 51us vs hardcoded nw1 24us looked like 2.16x, but
that gap is ALLOCATION+wrapper overhead, not the kernel — the in-graph rope SELF-time
is only 5.79us/call (1.4% of decode, ~2x above its 2.6us BW floor). rope autotune
keys on (HALF,INTERLEAVED) and per-program work is shape-independent (64-wide rotate),
so the cached config is fine for decode (unlike flash_decode EXP-L where config
depended on B*Hq). No real lever.

## EXP-5 — gate_up FUSED vs 2 single-acc GEMMs + swiglu — fused confirmed optimal
Tested whether the 2-accumulator fusion (BLOCK_N pinned 64) loses to 2 single-acc
GEMMs (BLOCK_N free to 128) + separate swiglu (bench_gateup_split_jun27.py). Fused
(shipped BK256/nw8) = 147.4us beats ALL split variants (159-170us, 0.87-0.92x): the
~18MB extra intermediate traffic of the split outweighs any tile-efficiency gain.
EXP-E's fusion decision holds; gate_up is DONE at 147us (25us above the 122us HBM
floor, irreducible for a Triton kernel).

## SESSION VERDICT
One solid win (swiglu BK=256, +2.3%). The b128 single-token decode is GEMM-bound and
every GEMM is at its int8 limit: HBM-bound gate_up/lm_head (int4 faith-dead per EXP-I,
kernel-overhead-dead per EXP-O) and L2-resident compute-bound qkv/down/o_proj (M=128
tensor-core efficiency limited, tiles optimal). Structural levers the FIXED profiler
can't reward: speculative decode (profiler hardcodes 1 token/model-call), paged/
right-sized KV + higher batch (sweep caps at b128; long flavors never beat instruct's
agg). KV-int8 / FP8-KV dead at short context (L2-resident). Remaining ideas are all
<1-2% or faith/profiler-blocked: sampling-in-graph (~1%, graph-safe-RNG risk),
down/o_proj quant fusion (blocked by per-row-amax-across-tiles), first-layer-qkv W8A8
(~0.2%). Engine is near-optimal for this metric.

## EXP-6 — split-K for the W8A8 o_proj/down GEMMs — DEAD (atomic reduction dominates)
o_proj (33us vs 6us int8-compute floor) and down (47us vs 20us) sit far above their
floors -> hypothesised occupancy/latency-bound (~128 programs on 142 SMs = ~1 wave,
serial K-loop latency unhidden), and int8 MMA (2x faster than the old W8A16 these were
split-K-tested on) makes K-loop latency a bigger fraction. Built a W8A8 split-K kernel
(SPLIT_K programs each reduce a K-chunk to int32, atomic-add the SCALED fp32 partial;
correct, rel=0). RESULT: 0.08-0.31x (catastrophically SLOWER) — the fp32 atomic_add
reduction over the M*N output (1M+ atomics) dominates any occupancy gain. Confirms the
prior "split-K DEAD" for W8A8 too. Also: EXP-3 already showed BLOCK_N=32 (more programs,
2 waves) doesn't beat BLOCK_N=64 for o_proj/down -> they are NOT fixably occupancy-bound;
33/47us are the genuine Triton int8-GEMM floors at M=128 for these shapes. GEMM axis
fully closed.

## EXP-7 — re-sweep flash_decode config at headline kv_len — confirmed optimal
flash_decode is 10% of decode and ~6.7x above its L2-read floor (scalar GEMV-style
online softmax, nw=1). Re-swept BLOCK_N{8,16,32,64} x nw{1,2,4} x ns{1,2,3} at the
headline shape. At kv_len 64/93 (instruct prompt~29 -> 29+64) the current EXP-L config
(BLOCK_N=16,nw=1,ns=2) is within 1.005-1.011x of the best = optimal. (At kv_len=128,
ns=1 wins ~1.10x, but instruct decode never reaches kv_len 128, so irrelevant.) No
actionable win. flash_decode is done.

## ============ jun27 FINAL STATE ============
Banked: EXP-1 swiglu BLOCK_K=256/nw8, +2.30% (9393.0 -> 9609.7), pushed. Every other
decode component re-verified optimal or measured-dead this session (EXP-2..7). The b128
single-token decode is at its practical Triton optimum: gate_up/lm_head HBM-bound (int4
faith+kernel dead), qkv/down/o_proj/flash L2-resident compute-bound (tiles optimal,
split-K dead), KV-int8 L2-dead at short ctx, sampling optimal. Structural levers
(spec-decode, paging, higher batch, sampling-in-graph) are incompatible with the fixed
profiler (1 token/model-call, external sample(), b128 cap). Remaining ideas all <~1.5%
and/or blocked (see CHECKPOINT "Next-session levers").

## EXP-8 — strided-input RoPE: eliminate the q/k .contiguous() copies — KEEP (+0.86%)
The clean-vs-profiled probe (clean_vs_profiled_b128.py) showed clean b128 decode=12.62ms
(agg 10141) vs profiled 13.17ms — a 0.54ms (4.3%) profiler tax on eager ops, AND that the
op table's 4096-call `elementwise<128,4>` is the q/k `.contiguous()` copies the RoPE
wrapper forced: the fused-qkv GEMM emits one contiguous (B,T,6144) buffer; q=[:4096],
k=[4096:5120] are NON-contiguous slices (row stride 6144), and _apply_qk copied them. The
RoPE kernel now reads Q/K with an explicit input row stride (no copy), writing contiguous
output. BIT-IDENTICAL (dq=dk=0, parity_rope_strided_jun27.py).
GOTCHA (cost a wasted run): the first cut gated on stride(0)==T*stride(1) — FALSE for the
T==1 decode slice (q.stride()==(6144,4096,128,1): the real batch stride is stride(0)=6144,
not stride(1)=4096 which is the singleton T-dim's natural n_heads*head_dim). So it fell
back to .contiguous() (measured no-op, 9575~=baseline). Fix: for T==1 the row stride is
stride(0). Then the plain `elementwise<128,4>` (4096 calls) VANISHES from the op table.
RESULT: full sweep 9609.7 -> 9692.3 (+0.86%); decode_ms 13.32->13.21; seq_tps_b1 94.7
neutral; vram unchanged. Above both prior with-copies runs (9609.7, 9575.0). KEEP.
Cumulative jun27: 9393 -> 9692 (+3.2%). NEXT: the 0.54ms eager-op profiler tax — move the
cos/sin/kv_len derivation from pos_index INTO the graph (cut _update_pos 4 eager ops -> 1).

## EXP-9 — derive cos/sin/kv_len from pos_index inside the graph — DISCARD (flat)
Hypothesis: the 0.54ms profiler tax (clean b128 12.62ms vs profiled 13.17ms) is eager-op
launch overhead; _update_pos did 4 eager writes/step (pos/kv_len/cos/sin). Moved kv_len
(=pos+1) + cos/sin (index_select) derivation INTO the captured graph, leaving only
pos_index.fill_ eager. Correct (coherent greedy: "first three primes are 2,3,5"). RESULT:
9692.3 -> 9681.5 (FLAT, within noise). The tax is NOT on the tiny pos/cos/sin copies; it's
on the SAMPLE ops (topk/multinomial allocations, which profile_memory=True taxes) — and
sample() is called EXTERNALLY by the profiler so it can't move into the graph. git reset.
LESSON: eager-op COUNT reduction only helps when the eager ops are expensive/allocating;
the per-step pos buffers are too cheap to matter.
