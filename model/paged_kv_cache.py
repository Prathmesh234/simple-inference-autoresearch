"""
PagedKVCache — block-paged key/value cache (vLLM PagedAttention style).

Why
---
The contiguous KVCache (model/kv_cache.py) reserves one ``max_seq_len``-long buffer
per request UP FRONT, at construction. For long-context flavors at high batch that
upfront reservation, on top of the model weights and the (large) prefill activations,
exhausts VRAM and the run OOMs before decode even starts. A paged cache stores K/V in
fixed-size BLOCKS drawn from a flat pool and hands a request only the blocks it needs,
allocated as the sequence grows — so prefill runs with a small cache and the pool fills
during decode. It also removes internal fragmentation and is the substrate prefix-reuse
/ RadixAttention build on (shared blocks across requests).

Layout
------
  k_pool / v_pool : (n_layers, num_blocks, n_heads_kv, block_size, head_dim)
  block_tables    : (max_batch, max_blocks_per_seq) int32  — logical block -> physical
  seq_lens        : (max_batch,) int32                     — current length per request

Each request's blocks are allocated from a shared free list. The flat pool is sized to
a VRAM budget (a fixed number of blocks), NOT to max_batch * max_seq_len, so it does not
pre-reserve the worst case per request.

Decode-graph integration
-------------------------
Blocks for a request are fully allocated at prefill (we know prompt_len + n_new), so the
block table is STATIC during decode — the captured graph reads it from device memory.
The per-step write slot is derived on-device from pos_index (see ``slot_for``), so one
captured graph serves every decode step.
"""

from __future__ import annotations

import torch


class PagedKVCache:
    BACKEND = "paged"

    def __init__(self, n_layers, max_batch, max_seq_len, n_heads_kv, head_dim,
                 block_size=16, dtype=torch.bfloat16, device="cuda"):
        self.n_layers = n_layers
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.n_heads_kv = n_heads_kv
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = torch.device(device)

        self.max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        # Pool sized to exactly cover this batch's worst case (one full max_seq_len per
        # request) — same DATA as contiguous, but allocated as ONE flat pool that the
        # block tables index, so prefill sees only the prompt blocks live at first and
        # the +n_new decode blocks fill in lazily (avoids the contiguous upfront OOM).
        self.num_blocks = max_batch * self.max_blocks_per_seq
        shape = (n_layers, self.num_blocks, n_heads_kv, block_size, head_dim)
        self.k_pool = torch.zeros(shape, dtype=dtype, device=self.device)
        self.v_pool = torch.zeros(shape, dtype=dtype, device=self.device)

        self.block_tables = torch.zeros(max_batch, self.max_blocks_per_seq,
                                        dtype=torch.int32, device=self.device)
        self.seq_lens = torch.zeros(max_batch, dtype=torch.int32, device=self.device)
        self._next_free = 0  # simple bump allocator (reset() rewinds it)
        self._allocated_B = 0

    # ── allocation ───────────────────────────────────────────────────────────
    def reset(self) -> None:
        self._next_free = 0
        self._allocated_B = 0
        self.seq_lens.zero_()

    def allocate(self, B: int, n_blocks_per_seq: int) -> None:
        """Assign ``n_blocks_per_seq`` sequential physical blocks to each of the B
        requests (called once after prefill sizing). Sequential ids -> the paged read
        is contiguous within a request, matching the contiguous kernel's access."""
        assert B <= self.max_batch
        assert n_blocks_per_seq <= self.max_blocks_per_seq
        for b in range(B):
            base = b * self.max_blocks_per_seq
            ids = torch.arange(base, base + n_blocks_per_seq, dtype=torch.int32,
                               device=self.device)
            self.block_tables[b, :n_blocks_per_seq] = ids
        self._allocated_B = B
        self._next_free = B * self.max_blocks_per_seq

    # ── writes ───────────────────────────────────────────────────────────────
    def write_prefill(self, layer_idx, B, k, v):
        """Scatter the prompt K/V (B, Hkv, T, D) into the paged pool. Each request's
        blocks are sequential so this is a reshape-copy into the pool slice."""
        T = k.shape[2]
        nblk = (T + self.block_size - 1) // self.block_size
        for b in range(B):
            base = b * self.max_blocks_per_seq
            # write token t -> block base + t//bs, offset t%bs
            full = (T // self.block_size) * self.block_size
            if full:
                kk = k[b, :, :full].reshape(self.n_heads_kv, full // self.block_size,
                                            self.block_size, self.head_dim).transpose(0, 1)
                vv = v[b, :, :full].reshape(self.n_heads_kv, full // self.block_size,
                                            self.block_size, self.head_dim).transpose(0, 1)
                self.k_pool[layer_idx, base:base + full // self.block_size] = kk
                self.v_pool[layer_idx, base:base + full // self.block_size] = vv
            if full < T:  # ragged tail into the last block
                lb = base + full // self.block_size
                tail = T - full
                self.k_pool[layer_idx, lb, :, :tail] = k[b, :, full:].transpose(0, 1)
                self.v_pool[layer_idx, lb, :, :tail] = v[b, :, full:].transpose(0, 1)

    def __repr__(self):
        gb = 2 * self.k_pool.numel() * self.k_pool.element_size() / 1e9
        return (f"PagedKVCache(layers={self.n_layers}, blocks={self.num_blocks}, "
                f"block_size={self.block_size}, max_batch={self.max_batch}, "
                f"max_seq_len={self.max_seq_len}, vram={gb:.2f} GB)")
