"""jun28 EXP-19 probe: is int4 KV faithful enough for the long-context flavors? The
long cells are flash-dominated (read ~8.4GB/step of int8 KV); int4 would halve that
again (+30-50% on those cells). Test: quantize K/V to int4 (per-(b,head,pos) symmetric,
16 levels), dequant, run attention, compare to bf16 reference. Coherent + high cos ->
worth building an int4-KV flash."""
import torch, torch.nn.functional as F
torch.manual_seed(0); dev="cuda"
B,Hq,Hkv,D = 4,32,8,128; KV_GROUP=Hq//Hkv

def qdq(x, bits):
    n = 2**(bits-1) - 1   # int4: 7, int8: 127
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-8).float()
    scale = amax / n
    xi = torch.round(x.float()/scale).clamp(-n, n)
    return (xi*scale).to(x.dtype)

for S in (256, 512, 1024):
    q = torch.randn(B,Hq,1,D,device=dev,dtype=torch.bfloat16)*0.5
    k = torch.randn(B,Hkv,S,D,device=dev,dtype=torch.bfloat16)*0.5
    v = torch.randn(B,Hkv,S,D,device=dev,dtype=torch.bfloat16)*0.5
    kk=k.repeat_interleave(KV_GROUP,1); vv=v.repeat_interleave(KV_GROUP,1)
    ref = F.scaled_dot_product_attention(q.float(), kk.float(), vv.float())
    for bits in (8,4):
        kd=qdq(k,bits).repeat_interleave(KV_GROUP,1); vd=qdq(v,bits).repeat_interleave(KV_GROUP,1)
        o = F.scaled_dot_product_attention(q.float(), kd.float(), vd.float())
        rel=((o-ref).abs()/(ref.abs()+1e-3)).mean().item()
        cos=F.cosine_similarity(o.flatten(),ref.flatten(),dim=0).item()
        print(f"  S={S:>4} int{bits}: rel_err={rel:.4f} cos={cos:.6f}")
