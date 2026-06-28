"""jun28 EXP-15 gate: paged flash-decode vs the contiguous flash-decode at the
instruct-b128 headline shape. PagedAttention is only worth shipping if the paged
kernel matches the contiguous one's speed at short context (else it regresses the
headline). PAGE_SIZE=16 == the flash BLOCK_N, so each iter reads one contiguous page.
"""
import torch, time
import torch.nn.functional as F
from kernels.flash_decode_kernel import attention_flash_decode
from kernels.paged_flash_decode_kernel import paged_attention_flash_decode

torch.manual_seed(0); dev = "cuda"
B, Hq, Hkv, D = 128, 32, 8, 128
KV_GROUP = Hq // Hkv
PAGE = 16


def build_paged(k, v, kv_len, page):
    # k,v: (B,Hkv,S,D). Pack into a block pool with a per-seq block table.
    Bn, H, S, Dd = k.shape
    nblk_per = (S + page - 1) // page
    total = Bn * nblk_per
    k_pool = torch.zeros(total, H, page, Dd, device=dev, dtype=k.dtype)
    v_pool = torch.zeros(total, H, page, Dd, device=dev, dtype=v.dtype)
    bt = torch.zeros(Bn, nblk_per, dtype=torch.int32, device=dev)
    for b in range(Bn):
        for lb in range(nblk_per):
            phys = b * nblk_per + lb
            bt[b, lb] = phys
            s0 = lb * page; s1 = min(s0 + page, S)
            k_pool[phys, :, :s1 - s0] = k[b, :, s0:s1]
            v_pool[phys, :, :s1 - s0] = v[b, :, s0:s1]
    return k_pool, v_pool, bt


def ref_sdpa(q, k, v, kv_len):
    kk = k[:, :, :kv_len].repeat_interleave(KV_GROUP, dim=1).float()
    vv = v[:, :, :kv_len].repeat_interleave(KV_GROUP, dim=1).float()
    return F.scaled_dot_product_attention(q.float(), kk, vv).transpose(1, 2)


def cuda_time(fn, iters=400, reps=6):
    for _ in range(20): fn()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters * 1000)
    return best


print(f"{'kv_len':>7} {'contig_us':>10} {'paged_us':>9} {'ratio':>7} {'paged_rel':>10} {'paged_cos':>10}")
for kv_len in (32, 64, 93, 128, 256, 512):
    S = ((kv_len + PAGE - 1) // PAGE) * PAGE
    q = (torch.randn(B, Hq, 1, D, device=dev, dtype=torch.bfloat16) * 0.5)
    k = (torch.randn(B, Hkv, S, D, device=dev, dtype=torch.bfloat16) * 0.5)
    v = (torch.randn(B, Hkv, S, D, device=dev, dtype=torch.bfloat16) * 0.5)
    kvlen = torch.tensor([kv_len], dtype=torch.int32, device=dev)
    k_pool, v_pool, bt = build_paged(k, v, kv_len, PAGE)

    ref = ref_sdpa(q, k, v, kv_len)
    o_c = attention_flash_decode(q, k, v, kvlen)
    o_p = paged_attention_flash_decode(q, k_pool, v_pool, bt, kvlen, PAGE)

    rel = ((o_p.float() - ref).abs() / (ref.abs() + 1e-3)).mean().item()
    cos = F.cosine_similarity(o_p.float().flatten(), ref.flatten(), dim=0).item()
    # paged vs contiguous (should be identical math)
    dpc = (o_p.float() - o_c.float()).abs().max().item()
    t_c = cuda_time(lambda: attention_flash_decode(q, k, v, kvlen))
    t_p = cuda_time(lambda: paged_attention_flash_decode(q, k_pool, v_pool, bt, kvlen, PAGE))
    print(f"{kv_len:>7} {t_c:>10.1f} {t_p:>9.1f} {t_c/t_p:>6.2f}x {rel:>10.5f} {cos:>10.6f}  (paged-vs-contig max|d|={dpc:.4f})")
