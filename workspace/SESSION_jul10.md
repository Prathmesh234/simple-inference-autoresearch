# July 10 research run

Branch: `autoresearch/jul10`

## Baselines

| Model | Mode | Decode tok/s | TTFT ms | Peak VRAM |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | full fixed profiler, best aggregate | 8,837.7 | mixed sweep | 30.30 GB |
| Llama-3.1-8B | full fixed profiler, batch 1 | 85.3 | mixed sweep | 30.30 GB |
| Qwen3.5-9B | warmed greedy, batch 1 | 20.9 | 223.2 | 17.96 GB |
| Qwen3.5-9B | retained path, 3-run median | 26.9 | 208.2 | 17.97 GB |

The Qwen baseline uses the official text backbone behind the engine's model
plugin contract. Its checkpoint has 32 layers: 24 Gated DeltaNet and eight
full-attention layers. All 427 text tensors map exactly.

`benchmarks/bench_qwen.py` is the repeatable Qwen yardstick. It loads once,
warms once, measures three 96-token greedy runs, reports medians, and fails if
cache reuse changes generated output. The retained path is 28.7% faster than
the original Qwen decode baseline.

## Experiments

| Experiment | Result | Decision |
|---|---|---|
| Selective W8A16 for Qwen projections with output width >=4096 | 19.4 -> 15.0 tok/s; VRAM 17.96 -> 10.10 GB | DISCARD: standalone GEMV wins did not survive full-model dispatch |
| `torch.compile(mode="reduce-overhead")` on the Qwen backbone | CUDAGraph output alias crash from mutable recurrent/KV cache | CRASH: dynamic cache needs explicit static-address integration |
| Fused Triton single-token Gated DeltaNet recurrence | kernel 147.3 -> 13.2 us (11.15x); warmed model 20.9 -> 24.1 tok/s (+15.3%) | KEEP |
| Skip recurrent cache self-copy after in-place fused update | 23.2 -> 24.3 tok/s (+4.7%); identical 96-token greedy output | KEEP |
| Static full-attention KV cache, sized to request maximum | 22.9 -> 23.0 tok/s (+0.4%); identical 96-token greedy output | DISCARD: fixed-length attention offsets avoided concatenation |
| Fused Triton RMSNorm for Qwen hidden and strided Q/K norms | 23.5 -> 25.7 tok/s (+9.4%); standalone 3.5-8.6x; identical 96-token greedy output | KEEP |
| Reuse fused Triton SwiGLU in Qwen MLP | 26.6 -> 24.0 tok/s (-9.8%); identical 96-token greedy output | DISCARD: Triton tiled dispatch costs more than two eager batch-1 elementwise kernels |
| Fused single-token causal depthwise-convolution state update | initially 25.6 -> 27.4 tok/s (+7.0%), but six identical 48-token cache reuses produced two output hashes | DISCARD: integrated generation was nondeterministic |
| Fused DeltaNet RMSNorm and SiLU gate | kernel 66.9 -> 4.4 us (15.35x); model 25.1 -> 28.2 tok/s (+12.4%); exact bf16 parity and identical 96-token greedy output | KEEP |
| PyTorch SDPA for eight full-attention layers | 28.0 -> 27.9 tok/s (-0.4%); TTFT 217.9 -> 235.7 ms | DISCARD: short batch-1 attention does not amortize backend overhead |

### Why the fused recurrence wins

The reference recurrent DeltaNet step expands state decay, two reductions,
delta update, outer-product state update, Q/K normalization, and output into
many eager PyTorch kernels for each of 24 linear-attention layers. The fused
kernel tiles the `(128, 128)` recurrent state by value dimension and performs
the complete update in one launch while each state tile is resident. It removes
launch overhead and intermediate traffic without changing the greedy output.

The fused kernel returns the same recurrent-state storage that it updates.
Transformers' generic cache path copied that tensor back onto itself after
every linear-attention layer. Detecting the storage alias avoids 24 redundant
state update calls per decoded token while preserving the original cache path
for prefill and non-aliased updates.

The Qwen RMSNorm path also stores scale as an offset from one and applies norms
to strided Q/K projection views. Extending the shared Triton kernel with a
compile-time weight offset and independent contiguous output addressing reduces
each eager multi-kernel norm to one launch. The strided regression case is part
of the RMSNorm benchmark gate.

The attempted DeltaNet convolution kernel fused state shift, depthwise dot
product, and SiLU. It was deterministic across 1,000 isolated resets and its
state matched the reference exactly, but repeated full-model cache reuse
eventually changed greedy output. The production path therefore remains on the
framework convolution fallback until the integrated instability is resolved.

The DeltaNet output gate previously used casts, square, mean, reciprocal root,
normalization, weight scaling, SiLU, multiplication, and a final cast as eager
operations. The fused kernel preserves Qwen's norm-before-gate ordering and
bf16 intermediate rounding while collapsing that sequence to one launch.
