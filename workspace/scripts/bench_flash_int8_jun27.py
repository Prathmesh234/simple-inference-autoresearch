"""jun27: standalone gate for the int8-KV flash-decode kernel.

Compares, at the real instruct-b128 decode shape (B=128, Hq=32, Hkv=8, D=128,
kv_len swept), the int8-KV flash kernel vs the current bf16 flash kernel:
  - correctness: both vs a torch SDPA fp32 reference (GQA-repeated)
  - speed: CUDA-event timed, MIN-of-many

Hypothesis: KV read halves (bf16->int8) so flash_decode drops toward the int8
read floor; dequant (one FMA/elem) is nearly free. Faithfulness: per-vector
symmetric int8 of K/V; attention is a weighted average so it is robust to KV
quant (vLLM/TRT ship fp8/int8 KV at negligible loss).
"""
import torch, time
import torch.nn.functional as F

torch.manual_seed(0)
dev = "cuda"
B, Hq, Hkv, D = 128, 32, 8, 128
KV_GROUP = Hq // Hkv

from kernels.flash_decode_kernel import attention_flash_decode
from kernels.flash_decode_int8_kernel import attention_flash_decode_int8


def quant_per_vec(x):
    # x (B,Hkv,S,D) bf16 -> int8 (B,Hkv,S,D), scale (B,Hkv,S) fp32
    amax = x.abs().amax(dim=-1).clamp_min(1e-8).float()       # (B,Hkv,S)
    scale = amax / 127.0
    qi = torch.round(x.float() / scale[..., None]).clamp(-127, 127).to(torch.int8)
    return qi, scale


def ref_sdpa(q, k_bf16, v_bf16, kv_len):
    # q (B,Hq,1,D); k/v (B,Hkv,S,D); attend over [0,kv_len)
    kk = k_bf16[:, :, :kv_len].repeat_interleave(KV_GROUP, dim=1).float()
    vv = v_bf16[:, :, :kv_len].repeat_interleave(KV_GROUP, dim=1).float()
    qf = q.float()
    out = F.scaled_dot_product_attention(qf, kk, vv)          # (B,Hq,1,D)
    return out.transpose(1, 2)                                 # (B,1,Hq,D)


def cuda_time(fn, iters=300, reps=5):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters * 1000)     # us
    return best


print(f"{'kv_len':>7} {'bf16_us':>8} {'int8_us':>8} {'speedup':>8} "
      f"{'bf16_rel':>9} {'int8_rel':>9} {'int8_cos':>9}")
for kv_len in (32, 64, 128, 256, 512, 1024):
    S = kv_len
    q = torch.randn(B, Hq, 1, D, device=dev, dtype=torch.bfloat16) * 0.5
    k = torch.randn(B, Hkv, S, D, device=dev, dtype=torch.bfloat16) * 0.5
    v = torch.randn(B, Hkv, S, D, device=dev, dtype=torch.bfloat16) * 0.5
    ki, ks = quant_per_vec(k)
    vi, vs = quant_per_vec(v)
    kvlen = torch.tensor([kv_len], dtype=torch.int32, device=dev)

    ref = ref_sdpa(q, k, v, kv_len)
    o_bf = attention_flash_decode(q, k, v, kvlen)
    o_i8 = attention_flash_decode_int8(q, ki, vi, ks, vs, kvlen)

    def rel(o):
        return ((o.float() - ref).abs() / (ref.abs() + 1e-3)).mean().item()
    def cos(o):
        a = o.float().flatten(); b = ref.flatten()
        return F.cosine_similarity(a, b, dim=0).item()

    t_bf = cuda_time(lambda: attention_flash_decode(q, k, v, kvlen))
    t_i8 = cuda_time(lambda: attention_flash_decode_int8(q, ki, vi, ks, vs, kvlen))

    print(f"{kv_len:>7} {t_bf:>8.1f} {t_i8:>8.1f} {t_bf/t_i8:>7.3f}x "
          f"{rel(o_bf):>9.5f} {rel(o_i8):>9.5f} {cos(o_i8):>9.6f}")
