# Session jun30 — continue decode-throughput autoresearch

Branch: `autoresearch/jun30` (forked from `main` @ 876a470, which has all jun27/28
wins merged). Fresh box — numbers compare WITHIN this branch only.

## Inherited state
Every major decode GEMM (qkv, gate_up+swiglu, down, o_proj, lm_head) is W8A8 int8
tensor-core under a captured CUDA graph. int8 KV opens all long-context cells at
lower VRAM. The jun27/28 verdict: the instruct-b128 headline is GEMM-bound at its
int8/M=128 limits; remaining levers are <~1.5% or structural/profiler-blocked. WIP
at HEAD: EXP-20 GQA-grouped int8 flash kernel (built, UNWIRED/unvalidated).

## Baseline (876a470, full sweep)
- **best_agg_tps = 9997.7** (instruct b128), seq_tps_b1 = 94.7, peak_vram = 36.06.
- This box reads ~2% higher than jun27's 9791.
- Fresh op table (instruct b128, profiled decode step ~12.80ms):
  - `_w8a8_swiglu_fwd` (gate_up): 153.7us/call × 32 = ~4.9ms/step (38%)  — HBM-bound (117MB int8 > 96MB L2; floor ~122us, 1.26x)
  - `_w8a8_gemm` (qkv+down+o_proj+lmhead): ~3.8ms/step (30%)
  - `_flash_decode_fwd`: 43.2us × 32 = ~1.38ms/step (11%)  — L2-resident, tile-optimal
  - `_add_rmsnorm_quant_fwd` + `_quant_per_token` + `_rope_qk_fwd` + KV index_copy: ~0.9ms/step (7%)
  - sample(): `aten::topk` ~170us CUDA / 191us CPU + `aten::multinomial` 27us CUDA / 309us CPU per step
  - (`_w8a16_gemm` 312ms in the raw table is PREFILL — 96 GEMMs @ M=3712, not decode.)

## EXP-21 — fused Triton top-k/top-p/sample kernel — KEEP (small, +0.73% controlled A/B)
HYPOTHESIS (from EXP-9's unfinished thread): sample() is the only EAGER work left
on the decode critical path (the model forward is a CUDA-graph replay). Its top-k
fast path runs ~9 small ops AFTER torch.topk — temperature, softmax, cumsum, the
top-p nucleus compare/shift/masked_fill, a 2nd softmax, torch.multinomial, gather.
EXP-9 saw a ~0.54ms "profiler tax" on these and assumed it was un-removable
(sample is called externally, can't go in the graph). But it was never tried to
REDUCE sample's op count. New kernel `kernels/sampling_kernel.py::fused_topk_sample`
keeps torch.topk (the necessary full-vocab reduction) and collapses the entire
post-topk tail into ONE Triton launch: temperature, the float32 top-p nucleus,
renorm softmax, and an inverse-CDF categorical draw from a `torch.rand` uniform.

FAITHFUL: distribution is bit-identical to the canonical float32 top-k/top-p
reference (gate `bench_fused_sample_compare.py`: max|Δprob|=0 over 200×128 trials,
0/25600 selection mismatches vs an independent inverse-CDF reference, Monte-Carlo
empirical freqs match within 0.008). Inverse-CDF on torch.rand is a different RNG
stream than torch.multinomial but the SAME distribution (program.md requires the
distribution stay faithful), and reproducible under torch.manual_seed.

STANDALONE: 2.1-3.0x faster than the torch tail (b128 0.318→0.152ms).

IN-ENGINE (the important part):
- Full sweep: 9997.7 → 9989.9 = FLAT (within ~5% sweep noise; sub-1% can't be seen
  in the full sweep — the EXP-H lesson).
- Same-process A/B (load model once, alternate fused/base, min-of-4, b128 instruct-
  shape): **clean +0.42%** (11.640 vs 11.689ms), **profiled +0.73%** (12.013 vs
  12.100ms). Real and consistent across both timing modes.

MECHANISM / KEY INSIGHT (why the win is only ~0.5%, not the ~4% EXP-9 implied):
In the decode loop the CPU dispatches sample()'s ops WHILE the GPU is still running
the ~12ms graph replay (CUDA is async; the CPU runs ahead). So the sample's CPU
dispatch + profiler bookkeeping is HIDDEN under the replay's GPU execution — cutting
op COUNT does NOT reduce decode_ms. What IS saved is only the sample's TAIL GPU time
(the ~9 small ops' kernels run serially after the replay): ~40us GPU, plus a small
profiler-tax reduction. topk's GPU time (~150-170us, the dominant sample GPU cost)
is unchanged because it's kept. => EXP-9's "sample tax on the headline" was real but
small; the lever is the tail-GPU-time, not the op-count/CPU-dispatch.

KEEP rationale: +0.73% profiled is real (controlled A/B), same magnitude as the
banked EXP-11 (+0.62%) / EXP-12 (+0.40%), faithful (Δ=0), and the sampler is a
strictly better/standard production pattern (2-3x standalone — matters for any
deployment where sample isn't hidden behind a 12ms graph replay, e.g. real low-
latency single-stream serving). Self-contained kernel + USE_FUSED_SAMPLE toggle.

CONSEQUENCE for future sessions: further sampler fusion (e.g. fusing topk too) can
only recover the topk GPU time (~1.2% of decode) — and beating torch's mbtopk radix
select for top-50-over-128k is hard. Sampling is now ~tapped for the headline.

## State after EXP-21: headline ~converged
The b128 instruct decode is GEMM-bound; HBM floor (weights only) ~7.8ms vs measured
~12.8ms (1.64x) — the gap is int8-MMA-at-M=128 inefficiency on the L2-resident GEMMs
+ the un-hidden gate_up MMA, both at their established limits. best_agg_tps is
structurally instruct-bound (shortest kv_len → fastest decode → highest b128 agg);
no long-context cell can beat it (code b128 7090 would need +41%). So headline gains
require speeding instruct b128, which is walled without a Marlin-grade int4 GEMM.
Remaining real progress is on the FRONTIER (long-context decode, VRAM, b1 latency):
next is EXP-20 (GQA-grouped int8 flash) for the long cells.
