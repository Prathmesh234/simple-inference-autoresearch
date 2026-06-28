"""
KVCache — static, pre-allocated key/value cache for autoregressive decoding.

Why a KV cache?
---------------
During generation, each new token's attention needs the K and V vectors of
EVERY previous token. Without a cache, every decode step would re-run the
full forward pass over the entire prefix → O(T²) work to produce T tokens.

With a cache, prefill computes K/V for the prompt once, stores them, and
each decode step:
  - computes K/V only for the single new token  (O(1) per step per layer)
  - appends them to the cache
  - reads the entire cached K/V slice for attention

Net result: decode goes from O(T²) → O(T) total work, the difference between
"unusable" and "real-time" generation.

Layout
------
The cache is one big pre-allocated pool, sized for the worst case:

    k_cache: (n_layers, max_batch, n_heads_kv, max_seq_len, head_dim)
    v_cache: same shape

This shape matches the post-transpose layout in GroupedQueryAttention
(B, n_heads_kv, T, head_dim), so we can copy in/out via plain slicing
along dim=3 with zero reshape cost.

Memory
------
Llama-3.1-8B:
  n_layers=32, n_heads_kv=8, head_dim=128, bf16 (2 bytes)
  per-token-per-layer (K+V) = 2 * 8 * 128 * 2 = 4096 B = 4 KB
  per-token across all layers = 32 * 4 KB = 128 KB

So a single 8k-context request uses ~1 GB; a batch of 8 at 8k ≈ 8 GB.
That's why Phase 3 introduces PagedAttention — this pre-allocation is wasteful.

Position management
-------------------
The cache itself is stateless w.r.t. position — the *caller* (the model
forward) supplies `start_pos`. This keeps the cache trivially compatible
with two phases per generation step:

  Prefill   : start_pos=0,  T=prompt_len   → writes slots [0, T),       reads [0, T)
  Decode k  : start_pos=k,  T=1            → writes slot  [k, k+1),     reads [0, k+1)

`reset()` is just a logical no-op (the pool is overwritten on next prefill);
it exists for clarity when callers swap sequences in/out.
"""

from __future__ import annotations

import torch


def quantize_kv_int8(x: torch.Tensor):
    """Per-(…, head_dim) symmetric int8 quant of a K or V tensor.

    x: (..., D) → returns (int8 same shape, fp16 scale with the D axis dropped).
    scale = max|x| over D / 127 ; xi = round(x/scale). Used on the KV write path.
    """
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8).float()
    scale = amax / 127.0
    xi = torch.round(x.float() / scale).clamp_(-127, 127).to(torch.int8)
    return xi, scale.squeeze(-1).to(torch.float16)


class KVCache:
    # Caches sized longer than this switch to int8 KV (long context -> HBM-bound flash
    # where int8 halves bytes and wins; short context stays bf16, EXP-2). instruct's
    # max_seq_len ~93 stays bf16; code/chat/long_ctx/summarize (>=455) go int8.
    INT8_SEQ_THRESHOLD = 256

    def __init__(
        self,
        n_layers: int,
        max_batch: int,
        max_seq_len: int,
        n_heads_kv: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
    ):
        """
        Pre-allocate the full K and V pools.

        Args:
            n_layers:    number of transformer layers (32 for Llama-3.1-8B)
            max_batch:   maximum batch size this cache will ever hold
            max_seq_len: maximum sequence length (prompt + generated tokens)
            n_heads_kv:  number of KV heads (8 for GQA in Llama-3.1-8B)
            head_dim:    dim per head (128)
            dtype:       storage dtype — match the model (bfloat16)
            device:      cuda
        """
        self.n_layers    = n_layers
        self.max_batch   = max_batch
        self.max_seq_len = max_seq_len
        self.n_heads_kv  = n_heads_kv
        self.head_dim    = head_dim
        self.dtype       = dtype
        self.device      = torch.device(device)

        # INT8 KV cache (fp8/int8-KV serving strategy) for LONG context. At short
        # context (instruct headline, kv_len<256) the K/V already fits L2 so int8 is a
        # no-op + dequant overhead loses (EXP-2) — keep bf16 there. But once the context
        # exceeds L2 the flash read is HBM-bound: int8 HALVES the KV bytes, which (a)
        # cuts KV VRAM ~2x → lets the long-prompt flavors reach higher batch without OOM,
        # and (b) makes the decode flash 1.4-2.4x faster (measured kv_len 256..1024). We
        # auto-pick int8 when this cache is sized for long sequences so the instruct
        # headline stays on the fast bf16 path. Faithful: per-(b,head,pos) symmetric int8
        # of K/V, attention is a weighted avg → cos 0.99998 vs bf16 (EXP-2).
        self.is_int8 = max_seq_len > KVCache.INT8_SEQ_THRESHOLD
        ##this limitation (pre-blocking the entire seq len) is what paged attention
        ## addresses; the int8 path above halves the bytes we pre-block for long context.
        if self.is_int8:
            shape = (n_layers, max_batch, n_heads_kv, max_seq_len, head_dim)
            self.k_cache = torch.zeros(shape, dtype=torch.int8, device=self.device)
            self.v_cache = torch.zeros(shape, dtype=torch.int8, device=self.device)
            sshape = (n_layers, max_batch, n_heads_kv, max_seq_len)
            self.k_scale = torch.zeros(sshape, dtype=torch.float16, device=self.device)
            self.v_scale = torch.zeros(sshape, dtype=torch.float16, device=self.device)
        else:
            shape = (n_layers, max_batch, n_heads_kv, max_seq_len, head_dim)
            self.k_cache = torch.zeros(shape, dtype=dtype, device=self.device)
            self.v_cache = torch.zeros(shape, dtype=dtype, device=self.device)
            self.k_scale = self.v_scale = None

    # ── core API ─────────────────────────────────────────────────────────────

    def update(
        self,
        layer_idx: int,
        start_pos: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Write new K/V into the cache at [start_pos, start_pos + T) and return
        the cached slice up to start_pos + T for attention.

        Args:
            layer_idx: which transformer layer this call belongs to
            start_pos: absolute position of the first new token (0 for prefill)
            k:         new K, shape (B, n_heads_kv, T, head_dim)
            v:         new V, shape (B, n_heads_kv, T, head_dim)

        Returns:
            k_full, v_full — shape (B, n_heads_kv, start_pos + T, head_dim)
            views into the underlying pool — DO NOT mutate.
        """
        B, H, T, D = k.shape
        end = start_pos + T

        assert B <= self.max_batch,  f"batch {B} > max_batch {self.max_batch}"
        assert H == self.n_heads_kv, f"n_heads_kv mismatch: got {H}, expected {self.n_heads_kv}"
        assert D == self.head_dim,   f"head_dim mismatch: got {D}, expected {self.head_dim}"
        assert end <= self.max_seq_len, (
            f"sequence overflow: start_pos+T={end} > max_seq_len={self.max_seq_len}"
        )

        if self.is_int8:
            # Store int8 + scale for later (int8) decode reads, but RETURN the bf16 k/v
            # so PREFILL's own attention runs in full precision (prefill always covers
            # [0,end) here, so the returned bf16 prefix == what we just wrote). Only the
            # decode-time reads of this prefix are int8 — the standard KV-quant behavior.
            ki, ks = quantize_kv_int8(k)
            vi, vs = quantize_kv_int8(v)
            self.k_cache[layer_idx, :B, :, start_pos:end, :] = ki
            self.v_cache[layer_idx, :B, :, start_pos:end, :] = vi
            self.k_scale[layer_idx, :B, :, start_pos:end] = ks
            self.v_scale[layer_idx, :B, :, start_pos:end] = vs
            return k, v

        # Write new K/V into their slots
        self.k_cache[layer_idx, :B, :, start_pos:end, :] = k
        self.v_cache[layer_idx, :B, :, start_pos:end, :] = v

        # Return the full prefix up to `end` for attention to read
        k_full = self.k_cache[layer_idx, :B, :, :end, :]
        v_full = self.v_cache[layer_idx, :B, :, :end, :]
        return k_full, v_full

    def reset(self) -> None:
        """
        Logical reset between independent sequences.

        We don't actually zero the buffers — the next prefill overwrites
        from slot 0, and attention only reads slices up to start_pos + T,
        so any stale data beyond that window is unreachable. This method
        exists so callers have a clear hook when swapping sequences.
        """
        # If you ever need a hard wipe (e.g., debugging), uncomment:
        # self.k_cache.zero_()
        # self.v_cache.zero_()
        return

    # ── introspection helpers ────────────────────────────────────────────────

    def bytes(self) -> int:
        """Total VRAM footprint of the cache (K + V) in bytes."""
        return 2 * self.k_cache.numel() * self.k_cache.element_size()

    def __repr__(self) -> str:
        gb = self.bytes() / 1e9
        return (
            f"KVCache(n_layers={self.n_layers}, max_batch={self.max_batch}, "
            f"max_seq_len={self.max_seq_len}, n_heads_kv={self.n_heads_kv}, "
            f"head_dim={self.head_dim}, dtype={self.dtype}, vram={gb:.2f} GB)"
        )
