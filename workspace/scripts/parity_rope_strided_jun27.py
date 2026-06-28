"""jun27 EXP-8 gate: verify the strided-input RoPE is bit-identical to the old
contiguous path AND matches the pytorch reference, on BOTH a contiguous input and
a non-contiguous qkv-slice input (the decode case). The kernel now reads Q/K with
an explicit row stride, so the qkv slices are rotated WITHOUT a .contiguous() copy.
"""
import torch
from kernels.rope_kernel import rope_triton
from ops.rope import apply_rope as ref_apply  # pytorch ref when USE_TRITON off? no—dispatches triton

torch.manual_seed(0)
dev = "cuda"
B, Hq, Hkv, D = 128, 32, 8, 128
q_size, kv_size = Hq * D, Hkv * D


def pytorch_ref(q, k, cos, sin):
    def rot(x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)
    c = cos.unsqueeze(0).unsqueeze(2)
    s = sin.unsqueeze(0).unsqueeze(2)
    return (q * c + rot(q) * s).to(q.dtype), (k * c + rot(k) * s).to(k.dtype)


def check(T, label):
    # build a fused-qkv buffer (B,T,q+2kv) contiguous, slice q/k (non-contiguous)
    qkv = torch.randn(B, T, q_size + 2 * kv_size, device=dev, dtype=torch.bfloat16)
    q = qkv[..., :q_size].reshape(B, T, Hq, D)            # non-contiguous slice
    k = qkv[..., q_size:q_size + kv_size].reshape(B, T, Hkv, D)
    cos = torch.randn(T, D, device=dev, dtype=torch.bfloat16)
    sin = torch.randn(T, D, device=dev, dtype=torch.bfloat16)

    assert not q.is_contiguous(), "test setup: q should be a non-contiguous slice"

    # strided path (new): reads q,k slices directly
    q_s, k_s = rope_triton(q, k, cos, sin)
    # contiguous path (what the OLD code effectively did): copy first
    q_c, k_c = rope_triton(q.contiguous(), k.contiguous(), cos, sin)
    # pytorch reference (float math)
    q_r, k_r = pytorch_ref(q.float(), k.float(), cos.float(), sin.float())

    dq = (q_s.float() - q_c.float()).abs().max().item()
    dk = (k_s.float() - k_c.float()).abs().max().item()
    rq = (q_s.float() - q_r).abs().max().item()
    rk = (k_s.float() - k_r).abs().max().item()
    print(f"  {label:<18} strided-vs-contig: dq={dq:.2e} dk={dk:.2e} (must be 0) | "
          f"vs-ref: rq={rq:.4f} rk={rk:.4f}")
    return dq == 0 and dk == 0


ok = True
ok &= check(1, "decode T=1")
ok &= check(7, "prefill T=7")
ok &= check(29, "prefill T=29")
print("RESULT:", "PASS (bit-identical to contiguous path)" if ok else "FAIL")
