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

## EXP-20 — GQA-grouped int8 flash (the jun28 WIP) — DEAD for the profiler's range
Validated the unwired kernel. Faithful (cos 0.99998) but speed: 1.00x at kv 455/768,
1.06-1.08x at kv 1012, crossing to 1.22x@2048 / 1.32x@4096. The profiler's longest
flavor caps at kv ~1068 (summarize/long_ctx b128), where it's only ~1.08x. MECHANISM:
Ada's 96MB L2 absorbs the redundant sibling KV reads — the CONCURRENT working set
(programs resident × per-program KV) fits L2 up to ~kv 1500, so the "4x redundancy"
isn't 4x of HBM until kv>=2048 (which the profiler never reaches). Extends jun3's
short-ctx 1.00x finding. NOT wired (would help only true >=2k-token serving). The
jun28 premise (~4x read reduction -> big win) was wrong: it's L2-bound, not HBM-bound,
in the profiler's kv range.

## int4 KV (grouped) — DEAD (faithfulness), extends EXP-19
EXP-19 killed per-vector int4 KV (rel 0.25). Probed grouped int4 KV (finer scales):
g=32 rel 0.18-0.21, g=16 rel 0.16-0.19, g=8 rel 0.13-0.15 — all >>10x the int8 bar
(0.014), while real byte savings SHRINK with finer grouping (g=8 only 1.35x vs int8;
per-vector 1.97x). int4 KV cannot be both faithful and meaningfully smaller than int8.
DEAD at all group sizes.

## EXP-22 — weight-only int8 o_proj for b1-b64 — KEEP (+5.9% seq_tps_b1, the big frontier win)
The b1 op table showed _w8a16_gemm = 82.5% of b1 decode, all GEMMs int8 EXCEPT o_proj,
which used bf16 cuBLAS for M<=64 (the dispatch capped W8A8 o_proj at 64<M<=256 and fell
back to F.linear bf16 below). o_proj was the LAST bf16 weight in low-batch decode.
INSIGHT (a "wrong dead verdict", like EXP-M/N): "b1 keeps bf16 cuBLAS" missed that
cuBLAS GEMV at M=1 is inefficient AND in the full decode each layer's o_proj weight is
evicted from L2 between layers -> HBM-bound, so bf16 reads 33MB/layer. Dispatch M<=64
decode o_proj to the weight-only int8 W8A16 path (reuses the existing w_o_int8 buffer):
halves the read (16.7MB), no act quant (loses at M<=64, EXP-N), bit-faithful (W8A16 is
the accepted weight-only int8). ONE-LINE change in ops/attention._decode_graph_forward.
RESULT (full sweep): seq_tps_b1 94.7->100.3 (+5.9%); instruct b1 91.3->96.2, b32 +6.3%,
b64 +5.1%; chat b1 94.7->100.3, chat_real 75.9->80.6 — EVERY flavor's b1-b64 cells lift
~5-6%. best_agg_tps 9997.7->9958.4 (FLAT, b128 o_proj is W8A8 unchanged; noise). VRAM
flat (35.82). Focused (cooler) run showed b1 +11.3%; full-sweep (instruct 5th, warmer)
+5.9% — the true gain is in that range. Lifts the entire interactivity/low-batch
frontier. KEEP. The surprise: cuBLAS GEMV at M=1 was much worse than its byte count
(+11% vs the ~3% bytes would predict), so this was a kernel-inefficiency win, not just
a bandwidth one.

## int4 gate_up at b1 (AWQ faithfulness probe) — DEAD
At b1 the gate_up GEMM is a bandwidth-bound GEMV, so the K=64 kernel wall that killed int4
at M=128 (EXP-14/O) is HIDDEN — int4 (58.7MB, fits L2) would halve the read. So the only
blocker at b1 is faithfulness. Probe (faith_awq_int4_gateup.py): naive int4 g64 gate_up ->
argmax 1.0, top5 0.80, top50 0.98, logit_rel 0.11. AWQ (activation-aware scaling, foldable
into mlp_norm for free) a=0.5 -> top5 1.0 but top50 DROPS to 0.90 and rel WORSENS to 0.14;
a=1.0 worse. AWQ trades top5 for top50/rel — no clean win, and rel stays ~2x the int8 bar.
Too lossy for faithful sampling. int4 gate_up is now DEAD at ALL batch sizes (kernel wall
at b128, faithfulness wall at b1).

## EXP-23 — W8A8 int8 qkv+gate_up at PREFILL (M>16) — KEEP (TTFT -30%, peak_vram -7.1GB)
Prefill GEMMs were weight-only W8A16 (int8 weight, bf16 MMA). The per-token int8 activation
is ALREADY emitted by add_norm_quant at prefill (was unused — the W8A16 path read the bf16
normed). Route qkv + the fused gate_up+SwiGLU through the int8 tensor-core W8A8 path for M>16
(covers the b128 decode bucket AND prefill M>256). Standalone at prefill M: qkv 2.3x, gate_up
1.9x vs W8A16 (int8 MMA ~2x bf16 at large M; bench prefill_w8a8_check.py). KEY VRAM win: the
fused W8A8 swiglu never materialises the (M, 2*intermediate) bf16 combined tensor the W8A16
path does (~7GB at b128 x 1004-token prefill); also stop emitting the now-dead bf16 normed
at prefill (emit_bf16 = M<=16). RESULT (full sweep): TTFT instruct b128 365.7->249.8ms (-32%),
code 5342->3707, summarize 14074->9818, long_ctx ->9368 (all ~-30%); peak_vram 35.82->28.72
(-7.1GB!); best_agg_tps 10018 (flat, decode untouched), seq_tps_b1 100.3 (flat). Coherent
(left-padded greedy: Paris/299792458 m/s/1789/Everest 29000ft — all correct). GOTCHA: added
int64 output offsets to _w8a8_gemm + _w8a8_swiglu_fwd (prefill M*N nears int32 at long ctx).
The earlier "batch garbage" in the gate was a RIGHT-padding test bug, not W8A8. KEEP.

## EXP-24 — W8A8 int8 down at PREFILL (M>64) — KEEP (TTFT another -13%, cumulative -40%)
Extend down's W8A8 bucket from 64<M<=256 to all M>64 (decode bucket + prefill). down was the
last bf16-MMA prefill GEMM (~85ms at instruct b128). Its self-quant int8 act (M x 14336,
~1.8GB transient at b128 x 1k-tok) fits the headroom EXP-23 freed. RESULT: TTFT instruct b128
249.8->216.4ms, summarize 9818->8516, long_ctx 9368->8102 (~-13% more; CUMULATIVE EXP-23+24
= -40% vs baseline: instruct 366->216ms, summarize 14074->8516ms). peak_vram 28.72->30.30
(+1.6GB down self-quant, still -5.5GB vs baseline). best_agg_tps 9876 / seq_tps_b1 100.8
(both flat, decode/b1 untouched — the dip is thermal noise). Coherent. KEEP.

## jun30 SESSION FINAL — 4 wins, best_agg_tps walled, frontier expanded
EXP-21 sampler (+0.73% headline), EXP-22 o_proj b1-b64 (+5.9% seq_tps_b1), EXP-23 prefill
qkv+gate_up W8A8 (TTFT -30%, VRAM -7GB), EXP-24 prefill down W8A8 (TTFT cumulative -40%).
Net frontier: interactivity (seq_tps_b1) +5.9%, TTFT -40%, peak_vram -5.5GB, decode headline
flat (~10000, int8 GEMM wall). Comprehensive DEAD confirmations: int4 GEMM (b128 kernel),
int4 gate_up b1 (AWQ faithfulness), int4 KV (grouped, faithfulness), GQA flash (L2-absorbed),
gate_up + b1 W8A16 tiles (optimal). The decode headline needs a Marlin-grade int4 GEMM
(beyond Triton + borderline faithfulness) — not a loop iteration. See CHECKPOINT.md.
