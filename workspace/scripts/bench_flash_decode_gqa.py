"""Standalone correctness + speed gate for the GQA-grouped flash decode kernel.

Compares against the existing per-query-head kernel (kernels/flash_decode_kernel.py)
on the real Llama-3.1-8B decode shape. Validates output match and measures the
KV-bandwidth win from streaming each K/V slice once across KV_GROUP query heads.
"""
import torch
import triton

from kernels.flash_decode_kernel import attention_flash_decode
from kernels.flash_decode_gqa_kernel import attention_flash_decode_gqa

torch.manual_seed(0)
dev = "cuda"
dt = torch.bfloat16

# Llama-3.1-8B attention dims
Hq, Hkv, D = 32, 8, 128
S = 4096  # preallocated KV cache length


def ref_sdpa(q, k, v, kv_len):
    # q (B,Hq,1,D); k,v (B,Hkv,S,D). Expand kv to Hq, take first kv_len.
    B = q.shape[0]
    G = Hq // Hkv
    kx = k[:, :, :kv_len].repeat_interleave(G, dim=1).float()
    vx = v[:, :, :kv_len].repeat_interleave(G, dim=1).float()
    qf = q.float()
    scale = 1.0 / (D ** 0.5)
    attn = (qf @ kx.transpose(-1, -2)) * scale  # (B,Hq,1,kv_len)
    attn = attn.softmax(dim=-1)
    o = attn @ vx  # (B,Hq,1,D)
    return o.transpose(1, 2)  # (B,1,Hq,D)


def bench(fn, *args, iters=200, warmup=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # us


for B, kv_len in [(128, 93), (128, 256), (128, 512), (64, 93)]:
    q = torch.randn(B, Hq, 1, D, device=dev, dtype=dt)
    k = torch.randn(B, Hkv, S, D, device=dev, dtype=dt)
    v = torch.randn(B, Hkv, S, D, device=dev, dtype=dt)
    kvlen_t = torch.tensor(kv_len, device=dev, dtype=torch.int32)

    o_old = attention_flash_decode(q, k, v, kvlen_t)
    o_new = attention_flash_decode_gqa(q, k, v, kvlen_t)
    o_ref = ref_sdpa(q, k, v, kv_len)

    err_old = (o_old.float() - o_ref).abs().max().item()
    err_new = (o_new.float() - o_ref).abs().max().item()
    err_cross = (o_old.float() - o_new.float()).abs().max().item()

    t_old = bench(attention_flash_decode, q, k, v, kvlen_t)
    t_new = bench(attention_flash_decode_gqa, q, k, v, kvlen_t)

    print(f"B={B} kv_len={kv_len:4d} | err_old={err_old:.4f} err_new={err_new:.4f} "
          f"cross={err_cross:.4f} | old={t_old:7.1f}us new={t_new:7.1f}us "
          f"speedup={t_old/t_new:.2f}x")
