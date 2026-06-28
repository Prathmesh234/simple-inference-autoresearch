"""jun27 EXP-7: re-sweep flash_decode config at the EXACT instruct-b128 headline
shape (B=128, Hq=32, Hkv=8, D=128, kv_len 64 and 93 — prompt 29 -> 29+64). EXP-L
hardcoded BLOCK_N=16/nw=1/ns=2 but tuned across a wide kv_len range; verify it's
still optimal at the headline kv_len. flash_decode is 10% of decode and ~6.7x above
its L2-read floor (scalar GEMV-style online softmax), so a better config could help.
Calls _flash_decode_fwd.fn directly to bypass any autotune. MIN-of-many.
"""
import torch, time
import kernels.flash_decode_kernel as FD

torch.manual_seed(0)
dev = "cuda"
B, Hq, Hkv, D = 128, 32, 8, 128
KV_GROUP = Hq // Hkv
kern = FD._flash_decode_fwd


def run(k, v, q, kvlen, S, BLOCK_N, nw, ns):
    out = torch.empty((B, 1, Hq, D), dtype=q.dtype, device=dev)
    grid = (B * Hq,)
    kern[grid](
        q, k, v, out, kvlen, 1.0 / (D ** 0.5),
        q.stride(0), q.stride(1), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(2), out.stride(3),
        Hq, KV_GROUP, BLOCK_N=BLOCK_N, D=D, num_warps=nw, num_stages=ns,
    )
    return out


def bench(fn, iters=400, reps=6):
    try:
        fn()
    except Exception as e:
        return None
    best = 1e9
    for _ in range(reps):
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t) / iters * 1e6)
    return best


for kv_len in (64, 93, 128):
    S = max(kv_len, 128)
    q = torch.randn(B, 1, Hq, D, device=dev, dtype=torch.bfloat16) * 0.5
    q = q.transpose(1, 2)  # (B,Hq,1,D) like the real call
    k = torch.randn(B, Hkv, S, D, device=dev, dtype=torch.bfloat16) * 0.5
    v = torch.randn(B, Hkv, S, D, device=dev, dtype=torch.bfloat16) * 0.5
    kvlen = torch.tensor([kv_len], dtype=torch.int32, device=dev)
    print(f"\n=== kv_len={kv_len} (current BLOCK_N=16/nw1/ns2) ===")
    res = []
    for BN in (8, 16, 32, 64):
        for nw in (1, 2, 4):
            for ns in (1, 2, 3):
                t = bench(lambda: run(k, v, q, kvlen, S, BN, nw, ns))
                if t:
                    res.append((t, (BN, nw, ns)))
    res.sort()
    cur = next(t for t, c in res if c == (16, 1, 2))
    for t, c in res[:6]:
        tag = "  <-- current" if c == (16, 1, 2) else f"  {cur/t:.3f}x"
        print(f"  BLOCK_N={c[0]:>2} nw={c[1]} ns={c[2]}  {t:>6.1f}us{tag}")
