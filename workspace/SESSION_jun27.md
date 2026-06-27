# Session jun27 — continue decode-throughput autoresearch

Branch: `autoresearch/jun27` (forked from main @ 348e6ae, which has all jun11 wins merged).
Box note: numbers compare WITHIN this branch only (hardware/thermal varies ~16% across boxes).

## Inherited state (from jun11 CHECKPOINT/PROGRESS)
Every major decode GEMM is now W8A8 int8 tensor-core: qkv, gate_up(+swiglu fused),
down, o_proj, lm_head. flash_decode tuned (BLOCK_N=16, nw=1). CUDA-graph decode.
Governing insight (jun11): "at b128 decode is COMPUTE/MMA-bound; bandwidth tricks
(int4, KV-int8) dead except weights > 96MB L2".

## My re-analysis of the governing insight — a GAP
The jun11 "bandwidth tricks dead" verdict reasoned about **static weights** (L2-resident
→ no HBM traffic). But it lumped **KV-cache int8** into the same bucket. The KV cache is
NOT a static weight: flash_decode reads the K/V prefix FRESH from HBM every step, and it
GROWS. At b128 the per-layer KV read is ~2·B·Hkv·kv_len·D·2B; for kv_len~90 that's ~47MB
/layer × 32 = ~1.5 GB/step ≈ a real HBM cost. EXP-L measured flash_decode = 58.7µs/call
at the instruct b128 headline, and the KV-read floor at kv_len~90 is ~49µs → flash_decode
is ~80% KV-bandwidth-bound, NOT compute-bound. So **KV-int8 was declared dead by faulty
analogy to weights and never actually measured** (grep confirms: no KV-quant experiment
script exists; only qkv-projection int8). This is exactly the "re-test a dead verdict that
hinges on wrong reasoning" pattern that overturned down/o_proj (EXP-M/N).

KV-int8 estimated win: flash_decode KV read 49→~25µs → ~35µs/call → save ~24µs/call × 32
≈ 0.76 ms/step on a ~17ms step ≈ **~4.5% headline** (more at longer context). Bonus: halves
KV VRAM (guardrail headroom; lets long flavors batch higher, though that won't beat the
short-context instruct headline).

Risk: faithfulness (attention is robust to KV quant — vLLM/TRT ship fp8 KV at negligible
loss); CUDA-graph safety (quant-on-write is static-shape); prefill path isolation (prefill
computes attention from the bf16 k,v it just made, and SEPARATELY writes int8 to cache →
generic prefill attention kernel untouched).

## Plan
1. Baseline full sweep → record results.tsv + fresh op table (re-trace with fresh eyes).
2. Confirm flash_decode share + kv_len at instruct b128 from the op table / trace.
3. If flash_decode is a meaningful share → implement int8 KV cache (per-token-per-head
   symmetric scale): KVCache stores int8 k/v + fp scale; decode write quantizes 1 token
   (graph-safe); flash_decode_kernel reads int8 + dequant; prefill writes int8 to cache
   but keeps bf16 attention. Standalone bench (correctness + speed) BEFORE wiring. Then
   profiler A/B (iso instruct b1+b128, same thermal window). Coherence check.
4. Other candidate levers if KV-int8 underwhelms: fuse down/o_proj quant into producer
   epilogue (marginal); lm_head W4A8 (faith risk ~1.5%); revisit gate_up pipelining.

## Log
- **Baseline (348e6ae, full sweep): best_agg_tps=9393.0, seq_tps_b1=94.7, vram=39.07.**
  Op table (instruct b128, per decode step of 13.627ms): gate_up swiglu 5.36ms (39%),
  qkv+down+o_proj _w8a8_gemm 4.99ms (37%), flash_decode 1.39ms (10%), norms/quant/rope/
  KV-write ~1.1ms (8%), sampling ~0.35ms (3%). (_w8a16_gemm 365ms in the table is PREFILL,
  not decode — 96 prefill GEMMs @ M=3712 + the 63 first-layer-qkv decode calls.)

- **EXP-1 gate_up swiglu BLOCK_K=256/nw=8 — KEEP (70d1793, +2.3%).**
  gate_up is the biggest decode op; its 117MB int8 weight > 96MB L2 → HBM-bound (122µs
  floor) and sat at 167.5µs (~45µs un-hidden int8 MMA). Widened BLOCK_K 128→256 (half the
  K-loop trips, wider weight loads) + num_warps 4→8 (spreads the two int32 accumulators,
  eases the register pressure pinning BLOCK_N=64). Standalone sweep (tune_swiglu_jun27.py):
  147.1µs vs 156.0µs = 1.06x, bit-identical (rel_err=0); ns≥3 fails (>128KB smem at BK=256).
  Iso A/B: 9565.6→9819.7 (+2.66%); full sweep 9393.0→9609.7 (+2.30%); swiglu op 167.5→158.3
  µs/call. seq_tps_b1 neutral (94.8, b1 uses W8A16 fallback). Mechanism deterministic
  (op self-time dropped exactly the standalone delta × 32). NEW BASELINE.

- **EXP-2 int8 KV cache — DISCARD for the headline (measured, not assumed).**
  Built kernels/flash_decode_int8_kernel.py (int8 K/V + per-(b,head,pos) fp32 scale,
  dequant folded into online softmax) and a standalone gate (bench_flash_int8_jun27.py).
  Faithfulness EXCELLENT: rel_err ~0.017, cos 0.99998 (attention is a weighted avg → robust
  to KV quant). BUT speed at the SHORT-context instruct headline LOSES:
    kv_len  32:0.96x  64:0.88x  128:0.80x | 256:2.40x  512:1.62x  1024:1.40x
  Mechanism: at b128 short context the K+V cache (~32MB at kv_len=64) FITS the 96MB L2, so
  flash_decode is L2/compute-bound, NOT HBM-bound → halving bytes is a no-op and the extra
  scale loads + dequant FMA make it slower. int8 KV only wins once kv_len≥256 (KV exceeds L2).
  The instruct headline runs kv_len~32-93, so this can't move it; long-context flavors win
  from it but never beat instruct's agg at b128 (longer KV = slower decode). The jun11
  "KV-int8 dead for the headline" verdict was RIGHT (now measured; mechanism = L2-resident
  short-context KV, the same L2 insight as for weights). NOT wired (simplicity). Kept the
  kernel+bench as evidence. CONFIRMED-DEAD for the b128 headline.

- EXP-3 (in progress): re-sweep qkv/down/o_proj _w8a8_gemm tiles (BLOCK_K/ns/nw) — the
  same "wrong-tile / un-swept dim" lever that just won EXP-1 on swiglu.
