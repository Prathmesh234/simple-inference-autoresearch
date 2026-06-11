"""Config sweep for the flash_decode kernel at the b128 headline shape, bypassing
the @triton.autotune wrapper (call .fn directly with explicit meta-params).

Headline decode attention: B=128, Hq=32, Hkv=8 (GQA grp 4), D=128, kv_len ~93.
Grid = B*Hq = 4096 programs. Also check b1 (32 programs) so a single hardcoded
config doesn't wreck single-stream. MIN-of-many timing.
"""
import torch, triton
import kernels.flash_decode_kernel as fdk

torch.manual_seed(0)
dev = "cuda"
Hq, Hkv, D = 32, 8, 128
KV_GROUP = Hq // Hkv
raw = fdk._flash_decode_fwd.fn  # underlying JITFunction (bypass autotuner)


def make(B, kv_len, S):
    q = torch.randn(B, Hq, 1, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, S, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, S, D, device=dev, dtype=torch.bfloat16)
    kvlen = torch.tensor([kv_len], device=dev, dtype=torch.int32)
    out = torch.empty(B, 1, Hq, D, device=dev, dtype=torch.bfloat16)
    return q, k, v, kvlen, out


def run(t, BN, nw, ns):
    q, k, v, kvlen, out = t
    B = q.shape[0]
    sm_scale = 1.0 / (D ** 0.5)
    grid = (B * Hq,)
    raw[grid](
        q, k, v, out, kvlen, sm_scale,
        q.stride(0), q.stride(1), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(2), out.stride(3),
        Hq, KV_GROUP,
        BLOCK_N=BN, D=D, num_warps=nw, num_stages=ns,
    )


def bench(B, kv_len, S, BN, nw, ns, iters=300, reps=6):
    t = make(B, kv_len, S)
    try:
        run(t, BN, nw, ns)
    except Exception:
        return None
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            run(t, BN, nw, ns)
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best * 1000


for (B, kv_len, S) in [(128, 93, 128), (128, 512, 520), (1, 93, 128)]:
    print(f"\n=== B={B} kv_len={kv_len} (programs={B*Hq}) ===")
    results = []
    for BN in (16, 32, 64, 128):
        for nw in (1, 2, 4, 8):
            for ns in (2, 3, 4):
                u = bench(B, kv_len, S, BN, nw, ns)
                if u is not None:
                    results.append((u, BN, nw, ns))
    for u, BN, nw, ns in sorted(results)[:6]:
        print(f"  BN={BN:>3} nw={nw} ns={ns}  {u:>7.2f}us")
