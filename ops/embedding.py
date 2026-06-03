"""
TokenEmbedding and OutputProjection.

What embeddings do
------------------
The model operates on continuous vectors, not integers. The embedding table
is a lookup: given a token ID, return the corresponding row of a learned
(vocab_size, hidden_size) weight matrix. That row is the token's initial
representation before any transformer layers run.

    token_id = 279   ("the")
    embedding = weight[279]   # shape: (hidden_size,)

For a sequence of T tokens you get shape (T, hidden_size), which becomes
the input to the first transformer block.

Tied embeddings
---------------
The output projection (the final step that converts hidden states back to
logit scores over the vocabulary) reuses the exact same weight matrix as
the input embedding. This is called weight tying.

    Input:  token_id → weight[token_id]            (lookup, shape: hidden_size)
    Output: hidden   → hidden @ weight.T            (matmul, shape: vocab_size)

Why tie them?
  - The embedding row for a token represents "what this token means as input"
  - The output projection row for a token represents "how likely is this token next"
  - Tying forces these two representations to live in the same space, which
    works well in practice and saves ~1.2 GB of parameters for this model
    (vocab_size × hidden_size × 2 bytes = 128256 × 4096 × 2 = 1050 MB × 2 without tying)

Shape contract
--------------
TokenEmbedding:
  input:  (batch, seq_len)          integer token IDs
  output: (batch, seq_len, hidden_size)

OutputProjection:
  input:  (batch, seq_len, hidden_size)
  output: (batch, seq_len, vocab_size)   these are raw logits, not probabilities
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Same Triton dispatch toggle as the rest of the engine.
USE_TRITON = os.environ.get("USE_TRITON", "true").lower() in ("1", "true", "yes", "on")


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        # Standard nn.Embedding: lookup table of shape (vocab_size, hidden_size)
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, T) integers in [0, vocab_size)
        # returns:   (B, T, hidden_size)
        return F.embedding(token_ids, self.weight)

    def load_weight(self, weight: torch.Tensor):
        ##this is very important. - here is where the model loading happens 
        ## basically without tracking the gradients we launch the model copying the weights 
        ## from the disk to the gpu into the embed token class
        with torch.no_grad():
            self.weight.copy_(weight)


class OutputProjection(nn.Module):
    """
    The final linear layer: hidden states → vocabulary logits.

    Tied case (tied checkpoints): does NOT own its weight — it holds a reference
    to TokenEmbedding's weight. One tensor, two uses.

    Untied case (e.g. Llama-3.1-8B): owns a separate lm_head weight parameter,
    loaded from the checkpoint's `lm_head.weight`.
    """

    def __init__(self, embedding: TokenEmbedding, tied: bool = True):
        super().__init__()
        self._embedding = embedding
        self.tied = tied
        self._weight = None
        if not tied:
            # Weight-only int8 (W8A16) lm_head. Untied lm_head is a big, genuinely
            # HBM-bound GEMM at decode (vocab x hidden = 128256 x 4096 = 525MB int8,
            # larger than the 96MB L2), so halving its bytes is a real bandwidth win
            # (~1.16x standalone at the b128 decode shape) AND saves ~525MB VRAM.
            # Per-output-channel int8 keeps the top-k ordering faithful (top-50
            # overlap ~0.98), which is what the sampler depends on. Stored as
            # buffers so `module.to(dtype)` skips the int8 cast (mirrors SwiGLUMLP).
            # Tied checkpoints keep sharing the bf16 embedding weight (the embedding
            # lookup needs it bf16), so they are NOT quantized here.
            self.register_buffer(
                "w_int8",
                torch.empty(embedding.vocab_size, embedding.hidden_size, dtype=torch.int8),
            )
            self.register_buffer(
                "w_scale", torch.empty(embedding.vocab_size, dtype=torch.float32)
            )

    @property
    def weight(self) -> torch.Tensor:
        return self._embedding.weight if self.tied else self._weight

    def load_weight(self, weight: torch.Tensor):
        """Quantize and load a separate lm_head weight (untied embeddings only)."""
        if self.tied:
            raise RuntimeError("OutputProjection is tied; it has no own weight to load")
        from kernels.w8a16_gemm_kernel import quantize_int8_per_channel
        with torch.no_grad():
            w_int8, scale = quantize_int8_per_channel(weight)
            self.w_int8.copy_(w_int8)
            self.w_scale.copy_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, hidden_size) → (B, T, vocab_size)
        if self.tied:
            return F.linear(x, self.weight)  # shared bf16 embedding weight
        if USE_TRITON and x.is_cuda:
            # W8A8 (int8 tensor cores) for the b128 decode bucket: lm_head is the
            # single biggest per-step GEMM (vocab x hidden = 128256 x 4096) and the
            # 525MB int8 weight is HBM-bound. The W8A16 path upcasts int8->bf16
            # before the dot, leaving ~1.1ms on the table; quantizing the (post-
            # final-norm, well-behaved: outlier ratio ~2.5) activation to int8 and
            # using true int8 MMA drops the GEMM ~1.6x (1109->695us, near the
            # 547us weight-read floor). Faithful: argmax match 1.000 vs W8A16,
            # top50 overlap 0.977 (== the accepted W8A16 0.982). For M<=16 (b1
            # decode) and M>256 (prefill) w8a8_linear_triton falls back to W8A16,
            # so latency/TTFT are never regressed.
            from kernels.w8a8_gemm_kernel import w8a8_linear_triton
            return w8a8_linear_triton(x, self.w_int8, self.w_scale)
        w = self.w_int8.to(x.dtype) * self.w_scale.to(x.dtype)[:, None]
        return F.linear(x, w)
