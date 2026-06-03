"""
CUDA-graph decode runner.

Why
---
The profiled headline metric (best_agg_tps at instruct batch=128) times the
decode loop *inside* a ``torch.profiler`` context. A single decode step launches
~193 GPU ops (32 layers x {q,k,v,o,gate_up,down GEMMs + RoPE + attention +
2 fused norms} plus the final norm and lm_head). Under the profiler each launch
carries per-op CPU dispatch + bookkeeping overhead, and that overhead is exposed
on the critical path: measured at b128 the same decode step costs ~28 ms clean
but ~40 ms under the profiler (+40%). Every prior win in this engine was a
launch-count reduction precisely because launches — not CUDA math — set that
profiled number.

A CUDA graph collapses the entire per-step kernel sequence into ONE graph launch
at replay, eliminating ~all per-op dispatch/profiler overhead.

How a single static graph serves every decode position
------------------------------------------------------
A captured graph replays a fixed sequence of ops against fixed tensor addresses,
so anything that changes per step must live in a device buffer that we overwrite
*before* replay (never re-baked into Python ints or new allocations):

  - ``tok_buf``   (B,1) long      — the next input token id.
  - ``pos_index`` (1,)  long      — current absolute position; used as the
                                    index for the cache ``index_copy_`` scatter.
  - ``cos``/``sin`` (1,head_dim)  — RoPE row for the current position (the
                                    existing RoPE kernel sees seq_len==1).
  - ``kv_len`` (1,) int32         — number of valid cached positions (pos+1),
                                    read inside the decode flash kernel to bound
                                    the K/V loop and mask the tail.

The graph reads the model's static weights and the (fixed-address) preallocated
KV cache pool, writes the new K/V into the cache at ``pos_index`` and reads the
masked prefix — so replaying after updating the four buffers above produces the
correct next-token logits for any position, with no recapture.

Scope / safety
--------------
Used ONLY for the hot single-token decode path (T==1). Prefill and any T>1 call
keep the existing eager path untouched. Capture is lazy and happens during the
profiler's warmup (not the timed region); the side-stream warmup settles Triton
autotune and the SDPA backend before capture so no host work / allocation leaks
into the captured region.
"""

from __future__ import annotations

import torch


class _DecodeCtx:
    """Device buffers carrying the per-step position into the captured graph."""

    __slots__ = ("cos", "sin", "pos_index", "kv_len")


class CUDAGraphDecoder:
    """Captures and replays a CUDA graph for the T==1 decode forward.

    Bound to a specific (model, kv_cache, batch size). The owning LlamaModel
    rebuilds a fresh decoder if the kv_cache object or batch size changes.
    """

    def __init__(self, model, kv_cache, batch_size, head_dim, max_seq_len,
                 device, dtype, warmup_iters: int = 3):
        self.model = model
        self.kv_cache = kv_cache
        self.B = batch_size
        self.S = max_seq_len
        self.device = device
        self.dtype = dtype
        self.warmup_iters = warmup_iters

        self.captured = False
        self.graph: torch.cuda.CUDAGraph | None = None
        self.logits: torch.Tensor | None = None

        # Static input buffer for the next token id.
        self.tok_buf = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        # Per-step position buffers consumed inside the captured graph.
        ctx = _DecodeCtx()
        ctx.cos = torch.zeros(1, head_dim, dtype=dtype, device=device)
        ctx.sin = torch.zeros(1, head_dim, dtype=dtype, device=device)
        ctx.pos_index = torch.zeros(1, dtype=torch.long, device=device)
        ctx.kv_len = torch.zeros(1, dtype=torch.int32, device=device)
        self.ctx = ctx

        # Full-width cos/sin tables already cast to the activation dtype. Calling
        # .get() primes RopeFrequencies' per-dtype cast cache; we then index the
        # cached table per step to fill the (1, head_dim) ctx buffers.
        model.rope_freqs.get(seq_len=1, start_pos=0, dtype=dtype)
        self._cos_tbl, self._sin_tbl = model.rope_freqs._cast_cache[dtype]

    # ── per-step device-buffer updates (run eagerly, before replay) ──────────
    def _update_pos(self, pos: int) -> None:
        self.ctx.pos_index.fill_(pos)
        self.ctx.kv_len.fill_(pos + 1)
        self.ctx.cos.copy_(self._cos_tbl[pos:pos + 1])
        self.ctx.sin.copy_(self._sin_tbl[pos:pos + 1])

    # ── the forward that gets captured ───────────────────────────────────────
    def _forward(self) -> torch.Tensor:
        m = self.model
        x = m.embed(self.tok_buf)
        residual = None
        for layer in m.layers:
            x, residual = layer(
                x, residual, start_pos=0,
                kv_cache=self.kv_cache, decode_ctx=self.ctx,
            )
        x, _ = m.norm.add_norm(x, residual)
        return m.head(x)

    # ── capture / replay ─────────────────────────────────────────────────────
    def capture(self, pos: int) -> None:
        """Settle kernels on a side stream, then capture the decode graph."""
        self._update_pos(pos)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup_iters):
                self.logits = self._forward()
        torch.cuda.current_stream().wait_stream(side)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = self._forward()
        self.captured = True

    def decode_step(self, next_tok: torch.Tensor, pos: int) -> torch.Tensor:
        """Copy in the next token + position, replay the graph, return logits.

        Returns the model's static logits buffer (B, 1, vocab_size); the caller
        samples from it immediately (it is overwritten on the next replay).
        """
        self.tok_buf.copy_(next_tok)
        self._update_pos(pos)
        self.graph.replay()
        return self.logits
