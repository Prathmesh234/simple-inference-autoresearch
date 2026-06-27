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
- (pending baseline)
