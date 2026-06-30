"""jun30: does GROUPED int4 KV rescue faithfulness vs EXP-19's per-vector int4
(rel 0.25, DEAD)? Grouping D=128 into G groups (each with its own scale) captures
per-region range -> finer int4. Byte cost: int4 (64B) + G fp16 scales. group=32 ->
4 scales (8B) = 72B vs int8 130B = 1.8x savings; group=16 -> 80B = 1.6x.
Synthetic first (matches EXP-19's harness); promising -> test on real K/V."""
import torch, torch.nn.functional as F
torch.manual_seed(0); dev = "cuda"
B, Hq, Hkv, D = 4, 32, 8, 128; KV_GROUP = Hq // Hkv


def qdq_grouped(x, bits, gsize):
    """Per-(b,head,pos) symmetric int<bits>, grouped along D into gsize-wide groups."""
    n = 2**(bits - 1) - 1
    *lead, d = x.shape
    xg = x.float().reshape(*lead, d // gsize, gsize)
    amax = xg.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    scale = amax / n
    xi = torch.round(xg / scale).clamp(-n, n)
    return (xi * scale).reshape(*lead, d).to(x.dtype)


for S in (512, 1024):
    q = torch.randn(B, Hq, 1, D, device=dev, dtype=torch.bfloat16) * 0.5
    k = torch.randn(B, Hkv, S, D, device=dev, dtype=torch.bfloat16) * 0.5
    v = torch.randn(B, Hkv, S, D, device=dev, dtype=torch.bfloat16) * 0.5
    kk = k.repeat_interleave(KV_GROUP, 1); vv = v.repeat_interleave(KV_GROUP, 1)
    ref = F.scaled_dot_product_attention(q.float(), kk.float(), vv.float())
    print(f"\n=== S={S} ===")
    # baselines: int8 per-vector, int4 per-vector (EXP-19)
    for bits, g, tag in [(8, 128, "int8 per-vec"), (4, 128, "int4 per-vec(EXP19)"),
                          (4, 32, "int4 g=32"), (4, 16, "int4 g=16"), (4, 8, "int4 g=8")]:
        kd = qdq_grouped(k, bits, g).repeat_interleave(KV_GROUP, 1)
        vd = qdq_grouped(v, bits, g).repeat_interleave(KV_GROUP, 1)
        o = F.scaled_dot_product_attention(q.float(), kd.float(), vd.float())
        rel = ((o - ref).abs() / (ref.abs() + 1e-3)).mean().item()
        cos = F.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item()
        bytes_per = (D // 8 if bits == 4 else D) + (D // g) * 2  # int4/int8 + fp16 scales
        print(f"  {tag:<22} rel={rel:.4f} cos={cos:.6f}  bytes/vec={bytes_per} "
              f"({130/bytes_per:.2f}x vs int8-pervec)")
