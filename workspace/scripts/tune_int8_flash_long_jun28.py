"""jun28 EXP-18: the int8 flash kernel (now used ONLY for long-context int8 KV) still
uses BLOCK_N=16 (tuned for the SHORT bf16 headline). At long kv_len (the int8 regime,
455..1068) a larger BLOCK_N = fewer loop iterations may win. Sweep at the real long
shapes. Calls _flash_decode_int8_fwd directly. MIN-of-many."""
import torch, triton, time
import kernels.flash_decode_int8_kernel as FD

torch.manual_seed(0); dev = "cuda"
B, Hq, Hkv, D = 128, 32, 8, 128
KV_GROUP = Hq // Hkv
kern = FD._flash_decode_int8_fwd


def run(ki, vi, ks, vs, q, kvlen, BLOCK_N, nw, ns):
    out = torch.empty((B, 1, Hq, D), dtype=q.dtype, device=dev)
    grid = (B * Hq,)
    kern[grid](
        q, ki, vi, out, ks, vs, kvlen, 1.0/(D**0.5),
        q.stride(0), q.stride(2), q.stride(3),
        ki.stride(0), ki.stride(1), ki.stride(2), ki.stride(3),
        vi.stride(0), vi.stride(1), vi.stride(2), vi.stride(3),
        ks.stride(0), ks.stride(1), ks.stride(2),
        vs.stride(0), vs.stride(1), vs.stride(2),
        out.stride(0), out.stride(2), out.stride(3),
        Hq, KV_GROUP, BLOCK_N=BLOCK_N, D=D, num_warps=nw, num_stages=ns,
    )
    return out


def bench(fn, iters=300, reps=6):
    try: fn()
    except Exception as e: return None
    best = 1e9
    for _ in range(reps):
        for _ in range(15): fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize(); best = min(best, (time.perf_counter()-t)/iters*1e6)
    return best


for kv_len in (455, 768, 1012):
    S = ((kv_len + 63)//64)*64
    q = (torch.randn(B,Hq,1,D,device=dev,dtype=torch.bfloat16)*0.5).transpose(1,2)
    ki = torch.randint(-127,128,(B,Hkv,S,D),dtype=torch.int8,device=dev)
    vi = torch.randint(-127,128,(B,Hkv,S,D),dtype=torch.int8,device=dev)
    ks = (torch.rand(B,Hkv,S,device=dev)*0.01+1e-3).half()
    vs = (torch.rand(B,Hkv,S,device=dev)*0.01+1e-3).half()
    kvlen = torch.tensor([kv_len],dtype=torch.int32,device=dev)
    print(f"\n=== kv_len={kv_len} (current BLOCK_N=16/nw1/ns2) ===")
    res = []
    for BN in (16,32,64,128):
        for nw in (1,2,4):
            for ns in (1,2):
                t = bench(lambda: run(ki,vi,ks,vs,q,kvlen,BN,nw,ns))
                if t: res.append((t,(BN,nw,ns)))
    res.sort()
    cur = next((t for t,c in res if c==(16,1,2)), None)
    for t,c in res[:6]:
        tag = " <-- current" if c==(16,1,2) else (f" {cur/t:.2f}x" if cur else "")
        print(f"  BLOCK_N={c[0]:>3} nw={c[1]} ns={c[2]}: {t:>6.1f}us{tag}")
