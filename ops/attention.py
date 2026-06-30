"""
Grouped Query Attention (GQA).

Standard Multi-Head Attention recap
-------------------------------------
Every token produces Q, K, V vectors. For a sequence of T tokens:
  - Q: (T, n_heads, head_dim)  — what am I looking for?
  - K: (T, n_heads, head_dim)  — what do I contain?
  - V: (T, n_heads, head_dim)  — what do I pass forward if attended to?

Attention score for head h, query position i, key position j:
  score[h,i,j] = Q[h,i] · K[h,j] / sqrt(head_dim)

Apply causal mask (can't attend to future positions), softmax over j,
then weighted sum over V.

What GQA changes
-----------------
In standard MHA, every head has its own K and V projections.
GQA uses fewer KV heads than Q heads — here 8 KV heads vs 32 Q heads.
Each KV head is shared by 4 Q heads (num_kv_groups = 32 / 8 = 4).

Memory saving: KV cache stores K and V for every layer and every position.
  MHA:  n_heads_q  KV heads = 32 heads → 32 × head_dim per token per layer
  GQA:  n_heads_kv KV heads =  8 heads →  8 × head_dim per token per layer
  Saving: 4× less KV cache memory at no meaningful quality loss.

Implementation: repeat each KV head 4 times so the shape matches Q,
then run standard attention. This is done in-memory (no extra parameters).

Weight shapes for Llama-3.1-8B
--------------------------------
  wq: (n_heads_q  * head_dim, hidden) = (4096, 4096)
  wk: (n_heads_kv * head_dim, hidden) = (1024, 4096)   ← 4× smaller than wq
  wv: (n_heads_kv * head_dim, hidden) = (1024, 4096)
  wo: (hidden, n_heads_q * head_dim)  = (4096, 4096)

Forward pass shape flow
------------------------
  x                          (B, T, hidden)
  → q = x @ wq.T             (B, T, n_heads_q  * head_dim)
  → reshape                  (B, T, n_heads_q,  head_dim)
  → apply_rope               (B, T, n_heads_q,  head_dim)
  → transpose                (B, n_heads_q, T,  head_dim)   ← sdpa expects this

  → k,v same but n_heads_kv  (B, n_heads_kv, T, head_dim)
  → repeat kv heads          (B, n_heads_q,  T, head_dim)   ← now matches q

  → scaled_dot_product_attention → (B, n_heads_q, T, head_dim)
  → transpose + reshape      (B, T, n_heads_q * head_dim)
  → out = x @ wo.T           (B, T, hidden)
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ops.rope import RopeFrequencies, apply_rope

# Triton kernel dispatch (Section 14d).
#   Controlled by the USE_TRITON env var (set in .env or shell).
#   Defaults to True — the Triton FlashAttention kernel is the production path.
#   Set USE_TRITON=false to force PyTorch's fused SDPA (useful for parity checks).
USE_TRITON = os.environ.get("USE_TRITON", "true").lower() in ("1", "true", "yes", "on")


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        rope_freqs: RopeFrequencies,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_heads_q  = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.head_dim     = head_dim
        self.num_kv_groups = num_heads_q // num_heads_kv
        self.rope_freqs   = rope_freqs
        # Needed so each block's attention writes into the right slot of the
        # shared KVCache pool. Set at construction time by LlamaModel.
        self.layer_idx    = layer_idx

        # Q/K/V sizes (out_features of each projection).
        self.q_size  = num_heads_q  * head_dim   # 4096
        self.kv_size = num_heads_kv * head_dim   # 1024

        # Weight-only int8 (W8A16) FUSED qkv projection. The three q/k/v linears
        # share the same input x and are streamed together as ONE int8 GEMM of
        # out_features = q_size + 2*kv_size (= 6144 for Llama-3.1-8B). Fusing them
        # both halves the qkv weight bytes (1 byte vs 2) AND amortizes the single
        # GEMM's fixed overhead across all three projections — standalone this is
        # ~1.25x vs three separate cuBLAS bf16 linears at the b128 decode shape,
        # whereas EACH projection in isolation loses to cuBLAS (the small-N kernel
        # floor) and a plain bf16 fusion gave no headline gain post-CUDA-graph.
        # o_proj stays bf16: it's a clean square (4096x4096) cuBLAS GEMM that the
        # Triton int8 kernel cannot beat at M=128.
        #
        # These are buffers, not Parameters, so `module.to(dtype)` skips the int8
        # storage (integer tensors are not cast) while the fp32 scale is cast to
        # the model dtype — exactly the SwiGLUMLP int8 pattern. Rows are laid out
        # [0:q_size)=q, [q_size:q_size+kv_size)=k, [..:..+kv_size)=v.
        qkv_out = self.q_size + 2 * self.kv_size
        self.register_buffer(
            "w_qkv_int8", torch.empty(qkv_out, hidden_size, dtype=torch.int8)
        )
        self.register_buffer(
            "w_qkv_scale", torch.empty(qkv_out, dtype=torch.float32)
        )
        self.wo = nn.Parameter(torch.empty(hidden_size, num_heads_q * head_dim))
        # int8 mirror of wo for the W8A8 decode o_proj (the bf16 wo stays for
        # prefill / b1). Buffers (not Parameters) so module.to(dtype) skips the
        # int8 storage while casting the fp32 scale — same pattern as w_qkv_int8.
        self.register_buffer(
            "w_o_int8", torch.empty(hidden_size, num_heads_q * head_dim, dtype=torch.int8)
        )
        self.register_buffer(
            "w_o_scale", torch.empty(hidden_size, dtype=torch.float32)
        )

    def load_weights(self, wq, wk, wv, wo):
        from kernels.w8a16_gemm_kernel import quantize_int8_per_channel
        with torch.no_grad():
            qkv = torch.cat([wq, wk, wv], dim=0)
            qkv_int8, qkv_scale = quantize_int8_per_channel(qkv)
            self.w_qkv_int8.copy_(qkv_int8)
            self.w_qkv_scale.copy_(qkv_scale)
            self.wo.copy_(wo)
            wo_int8, wo_scale = quantize_int8_per_channel(wo)
            self.w_o_int8.copy_(wo_int8)
            self.w_o_scale.copy_(wo_scale)

    def _project_qkv(self, x, B, T, x_int8=None, x_scale=None):
        """Fused int8 qkv projection → (q, k, v) reshaped into heads.

        Returns q (B,T,Hq,D), k (B,T,Hkv,D), v (B,T,Hkv,D). The last-dim slices of
        the contiguous (B,T,6144) output are split into heads with reshape (a view,
        no copy, since only the contiguous final dim is split).

        When ``x_int8``/``x_scale`` are provided (the per-token int8 quant of the
        normed activation, emitted for free by the preceding fused
        ``attn_norm.add_norm_quant``) and M = B*T is in the W8A8 decode bucket
        (16 < M <= 256), the projection runs as a true int8 tensor-core GEMM
        (~2.2x over W8A16 at the b128 decode shape). Outside that bucket — b1
        decode (M<=16, latency-bound) and prefill (M>256) — it falls back to the
        weight-only W8A16 path on the bf16 activation.
        """
        if USE_TRITON and x.is_cuda:
            M = B * T
            # W8A8 int8 tensor-core qkv for M>16 — both the b128 decode bucket AND
            # prefill (M>256). The int8 activation comes free from the fused
            # attn_norm.add_norm_quant. At prefill (large M) the int8 MMA runs ~2.3x
            # the W8A16 (upcast) path (TTFT win). M<=16 (b1) stays W8A16 (no int8-MMA
            # benefit at GEMV-tiny M). The kernel's output offsets are int64 (prefill
            # M*N safe).
            if x_int8 is not None and M > 16:
                from kernels.w8a8_gemm_kernel import w8a8_qkv_prequant
                qkv = w8a8_qkv_prequant(
                    x_int8, x_scale, self.w_qkv_int8, self.w_qkv_scale, x.shape
                )
            else:
                from kernels.w8a16_gemm_kernel import w8a16_linear_triton
                qkv = w8a16_linear_triton(x, self.w_qkv_int8, self.w_qkv_scale)
        else:
            w_qkv = self.w_qkv_int8.to(x.dtype) * self.w_qkv_scale.to(x.dtype)[:, None]
            qkv = F.linear(x, w_qkv)
        q = qkv[..., :self.q_size].reshape(B, T, self.num_heads_q, self.head_dim)
        k = qkv[..., self.q_size:self.q_size + self.kv_size].reshape(
            B, T, self.num_heads_kv, self.head_dim)
        v = qkv[..., self.q_size + self.kv_size:].reshape(
            B, T, self.num_heads_kv, self.head_dim)
        return q, k, v

    def _decode_graph_forward(self, x, kv_cache, ctx, x_int8=None, x_scale=None):
        """CUDA-graph-capturable single-token (T==1) decode attention.

        All ops have a fixed launch signature; the per-step position lives in
        device buffers carried by ``ctx`` (filled by the decoder before each
        graph replay), so a single captured graph serves every decode position:

          - RoPE reads ctx.cos/ctx.sin — a static (1, head_dim) buffer holding
            the cos/sin row for the current position (the existing RoPE kernel
            sees seq_len==1, so every batch row maps to row 0).
          - The new K/V are scattered into the preallocated cache at the current
            slot via ``index_copy_`` along the sequence dim with ctx.pos_index
            (a device LongTensor) — graph-safe because the slot is read from
            device memory at replay time.
          - Attention is FlashAttention (kernels/flash_decode_kernel) over the
            FULL static cache, with the valid prefix length ctx.kv_len read from
            a device scalar inside the kernel — keeping the fused flash path
            (no SDPA mask fallback) while staying graph-safe.
        """
        B, T, _ = x.shape  # T == 1
        q, k, v = self._project_qkv(x, B, T, x_int8=x_int8, x_scale=x_scale)

        q, k = apply_rope(q, k, ctx.cos, ctx.sin)

        q = q.transpose(1, 2)  # (B, Hq,  1, D)
        k = k.transpose(1, 2)  # (B, Hkv, 1, D)
        v = v.transpose(1, 2)  # (B, Hkv, 1, D)

        kc = kv_cache.k_cache[self.layer_idx, :B]  # (B, Hkv, S, D) view into pool
        vc = kv_cache.v_cache[self.layer_idx, :B]
        if kv_cache.is_int8:
            # int8 KV (long-context flavors): quantize the new token's K/V per (b,head),
            # scatter int8 + scale into the cache, and read it back with the int8 flash
            # (halved bytes -> fits + faster at long kv_len). is_int8 is a constexpr-like
            # Python bool fixed at graph-capture, so the bf16 headline path skips this.
            from model.kv_cache import quantize_kv_int8
            from kernels.flash_decode_int8_kernel import attention_flash_decode_int8
            ki, ks = quantize_kv_int8(k)
            vi, vs = quantize_kv_int8(v)
            ksc = kv_cache.k_scale[self.layer_idx, :B]
            vsc = kv_cache.v_scale[self.layer_idx, :B]
            kc.index_copy_(2, ctx.pos_index, ki)
            vc.index_copy_(2, ctx.pos_index, vi)
            ksc.index_copy_(2, ctx.pos_index, ks)
            vsc.index_copy_(2, ctx.pos_index, vs)
            out = attention_flash_decode_int8(q, kc, vc, ksc, vsc, ctx.kv_len)
        else:
            kc.index_copy_(2, ctx.pos_index, k)
            vc.index_copy_(2, ctx.pos_index, v)
            from kernels.flash_decode_kernel import attention_flash_decode
            out = attention_flash_decode(q, kc, vc, ctx.kv_len)  # (B, 1, Hq, D)
        out = out.reshape(B, T, self.num_heads_q * self.head_dim)
        M = B * T
        if 64 < M <= 256:
            from kernels.w8a8_gemm_kernel import w8a8_oproj_linear
            return w8a8_oproj_linear(out, self.w_o_int8, self.w_o_scale)
        # b1-b64 decode (M<=64): weight-only int8 o_proj. This was the LAST bf16
        # weight in the low-batch decode step — bf16 cuBLAS reads 33MB/layer, but
        # in the full decode each layer's o_proj weight is evicted from L2 between
        # layers, so it is HBM-bound; int8 halves that read (16.7MB). No per-token
        # act quant here (it loses at M<=64, EXP-N), so weight-only W8A16 — the same
        # bit-faithful int8 path used for qkv/down at this M. b1-b64 are bandwidth-
        # bound, so fewer weight bytes is a direct single-stream/low-batch win.
        # (Prefill, M>256, stays on bf16 cuBLAS via the eager forward path.)
        from kernels.w8a16_gemm_kernel import w8a16_linear_triton
        return w8a16_linear_triton(out, self.w_o_int8, self.w_o_scale)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache=None,
        decode_ctx=None,
        x_int8: torch.Tensor | None = None,
        x_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, T, hidden_size)
            start_pos: position offset — 0 during prefill, >0 during decode
            kv_cache:  optional KVCache object (Section 11) — None for now
            decode_ctx: optional CUDA-graph decode context (see
                       model/cuda_graph_decode.py). When provided, take the
                       graph-capturable single-token decode path: position is
                       read from device buffers (decode_ctx.cos/sin,
                       decode_ctx.pos_index, decode_ctx.key_mask) instead of the
                       Python int start_pos, the new K/V are written into the
                       static cache with index_copy_ (a device-indexed scatter),
                       and attention runs as SDPA over the full preallocated
                       cache masked to the valid prefix. Every op here has a
                       fixed launch signature, so the whole forward can be
                       captured once into a CUDA graph and replayed per step —
                       collapsing ~193 kernel launches into one graph launch and
                       removing the per-op CPU/profiler dispatch overhead that
                       dominates the profiled b128 headline.

        Returns:
            (B, T, hidden_size)
        """
        if decode_ctx is not None:
            return self._decode_graph_forward(x, kv_cache, decode_ctx,
                                              x_int8=x_int8, x_scale=x_scale)

        B, T, _ = x.shape

        # --- 1. Project to Q, K, V (fused int8 GEMM) + reshape into heads ---
        q, k, v = self._project_qkv(x, B, T, x_int8=x_int8, x_scale=x_scale)

        # --- 3. Apply RoPE to Q and K ---
        # Fetch cos/sin already in the activation dtype from RopeFrequencies'
        # cached cast tables (avoids a per-layer, per-step .to(dtype) copy).
        cos, sin = self.rope_freqs.get(seq_len=T, start_pos=start_pos, dtype=x.dtype)
        q, k = apply_rope(q, k, cos, sin)

        # --- 4. Transpose to (B, n_heads, T, head_dim) for SDPA ---
        q = q.transpose(1, 2)  # (B, n_heads_q,  T, head_dim)
        k = k.transpose(1, 2)  # (B, n_heads_kv, T, head_dim)
        v = v.transpose(1, 2)  # (B, n_heads_kv, T, head_dim)

        # --- 5. KV cache update (Section 11) ---
        # During prefill (start_pos=0): writes [0, T), returns [0, T) → standard self-attention.
        # During decode (start_pos>0, T=1): writes [pos, pos+1), returns [0, pos+1)
        # → the new query attends to the full cached prefix + itself.
        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, start_pos, k, v)

        # --- 6. Attention ---
        # Triton FlashAttention handles GQA internally (maps each Q head to its
        # KV head), so we pass the un-repeated K/V. The PyTorch SDPA path needs
        # K/V repeated up to the Q head count first.
        #
        # Causal masking: prefill (T>1, Tq==Tk) is lower-triangular; decode
        # (T==1) attends to the whole cached prefix. The flash kernel derives
        # the per-row mask from a Tk-Tq offset, so we hand it the same is_causal
        # flag SDPA uses.
        causal = (T > 1)

        if USE_TRITON and q.is_cuda:
            from kernels.attention_kernel import attention_flash_triton
            # q/k/v are strided views (q is transposed; k/v are KVCache prefix
            # slices) but all have a contiguous head_dim, which is all the flash
            # kernel needs — it indexes every dim with explicit strides. Passing
            # assume_contiguous=True skips a per-step .contiguous() copy of the
            # whole K/V prefix (heavy HBM traffic + peak VRAM for long contexts).
            # out_token_major=True writes the result directly as (B, T, Hq, D),
            # so we can view it straight into (B, T, hidden) for the output
            # projection — no transpose+contiguous copy per layer per step.
            out = attention_flash_triton(
                q, k, v, causal=causal,
                assume_contiguous=True, out_token_major=True,
            )
            # out: (B, T, n_heads_q, head_dim) contiguous → view to (B, T, hidden)
            out = out.view(B, T, self.num_heads_q * self.head_dim)
        else:
            if self.num_kv_groups > 1:
                k = k.repeat_interleave(self.num_kv_groups, dim=1)
                v = v.repeat_interleave(self.num_kv_groups, dim=1)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=causal,
            )
            # out: (B, n_heads_q, T, head_dim) → merge heads
            out = out.transpose(1, 2).contiguous()       # (B, T, n_heads_q, head_dim)
            out = out.view(B, T, self.num_heads_q * self.head_dim)  # (B, T, hidden)

        # --- 8. Output projection ---
        out = F.linear(out, self.wo)                 # (B, T, hidden)

        return out
