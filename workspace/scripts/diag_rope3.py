import torch, time
import kernels.rope_kernel as RK
torch.manual_seed(0); dev="cuda"; B,Hq,Hkv,D=128,32,8,128
qkv = torch.randn(B,1,Hq*D+2*Hkv*D, device=dev, dtype=torch.bfloat16)
q = qkv[..., :Hq*D].reshape(B,1,Hq,D); k = qkv[..., Hq*D:Hq*D+Hkv*D].reshape(B,1,Hkv,D)
cos = torch.randn(1,D,device=dev,dtype=torch.bfloat16); sin=torch.randn(1,D,device=dev,dtype=torch.bfloat16)
# contiguous COPIES (separate real tensors, what flash would consume after old path)
qc, kc = q.contiguous(), k.contiguous()
def bench(fn,n=300):
    for _ in range(40): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e6
# strided path (new): operates on the non-contiguous slices directly
ts = min(bench(lambda: RK.rope_triton(q,k,cos,sin)) for _ in range(3))
# old behavior: caller had contiguous q,k already (the .contiguous() cost was paid in rope)
# emulate full old cost = contiguous(q)+contiguous(k)+rope(contig)
def old():
    a=q.contiguous(); b=k.contiguous(); return RK.rope_triton(a,b,cos,sin)
to = min(bench(old) for _ in range(3))
print(f"NEW strided (no copy): {ts:.2f} us/call")
print(f"OLD contig+rope:       {to:.2f} us/call  -> save {to-ts:.2f}us ({100*(to-ts)/to:.1f}%)")
