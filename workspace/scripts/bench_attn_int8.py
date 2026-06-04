"""Bench: W8A16 for attention projections vs cuBLAS bf16, at b128 decode (M=128).

Tests two byte-reduction ideas the rubber-duck flagged as distinct from the
known dead ends:
  - FUSED qkv: one int8 GEMM (N=Hq*D + 2*Hkv*D = 6144, K=4096) vs three cuBLAS
    bf16 linears (q N=4096, k N=1024, v N=1024). Fusion amortizes the per-call
    kernel floor AND halves qkv weight bytes.
  - o_proj: int8 (N=4096, K=4096) vs cuBLAS bf16.
Reports correctness (max abs err vs bf16 reference) and speed.
"""
import torch
import torch.nn.functional as F

from kernels.w8a16_gemm_kernel import w8a16_linear_triton, quantize_int8_per_channel

torch.manual_seed(0)
dev = "cuda"; dt = torch.bfloat16
M = 128
HID = 4096
Nq, Nkv = 4096, 1024  # q_proj, k/v_proj out features


def bench(fn, iters=300, warmup=80):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000


x = torch.randn(M, HID, device=dev, dtype=dt)

# ---- FUSED QKV ----
wq = torch.randn(Nq, HID, device=dev, dtype=dt) * 0.02
wk = torch.randn(Nkv, HID, device=dev, dtype=dt) * 0.02
wv = torch.randn(Nkv, HID, device=dev, dtype=dt) * 0.02
w_qkv = torch.cat([wq, wk, wv], dim=0)  # (6144, 4096)
q8, qs = quantize_int8_per_channel(w_qkv)
q8 = q8.to(dev); qs = qs.to(dev)

ref_qkv = torch.cat([F.linear(x, wq), F.linear(x, wk), F.linear(x, wv)], dim=-1)
out_qkv = w8a16_linear_triton(x, q8, qs)
err_qkv = (out_qkv.float() - ref_qkv.float()).abs().max().item()


def bf16_qkv():
    return torch.cat([F.linear(x, wq), F.linear(x, wk), F.linear(x, wv)], dim=-1)


def int8_qkv():
    return w8a16_linear_triton(x, q8, qs)


t_bf16_qkv = min(bench(bf16_qkv) for _ in range(5))
t_int8_qkv = min(bench(int8_qkv) for _ in range(5))
print(f"QKV fused  N=6144 K=4096 | err={err_qkv:.3f} | "
      f"bf16(3x cublas)={t_bf16_qkv:6.1f}us  int8(1x)={t_int8_qkv:6.1f}us  "
      f"speedup={t_bf16_qkv/t_int8_qkv:.2f}x")

# ---- O_PROJ ----
wo = torch.randn(HID, HID, device=dev, dtype=dt) * 0.02
o8, os_ = quantize_int8_per_channel(wo)
o8 = o8.to(dev); os_ = os_.to(dev)
ref_o = F.linear(x, wo)
out_o = w8a16_linear_triton(x, o8, os_)
err_o = (out_o.float() - ref_o.float()).abs().max().item()

t_bf16_o = min(bench(lambda: F.linear(x, wo)) for _ in range(5))
t_int8_o = min(bench(lambda: w8a16_linear_triton(x, o8, os_)) for _ in range(5))
print(f"o_proj     N=4096 K=4096 | err={err_o:.3f} | "
      f"bf16(cublas)={t_bf16_o:6.1f}us  int8={t_int8_o:6.1f}us  "
      f"speedup={t_bf16_o/t_int8_o:.2f}x")
