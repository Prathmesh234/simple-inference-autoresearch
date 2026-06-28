import torch
dev="cuda"; B,Hq,Hkv,D=128,32,8,128
qkv = torch.randn(B,1,Hq*D+2*Hkv*D, device=dev, dtype=torch.bfloat16)
q = qkv[..., :Hq*D].reshape(B,1,Hq,D)
k = qkv[..., Hq*D:Hq*D+Hkv*D].reshape(B,1,Hkv,D)
print("q.shape", tuple(q.shape), "q.stride", q.stride(), "contig", q.is_contiguous())
print("k.shape", tuple(k.shape), "k.stride", k.stride(), "contig", k.is_contiguous())
T=1
for name,x,nh in [("q",q,Hq),("k",k,Hkv)]:
    c1 = x.stride(-1)==1
    c2 = x.stride(2)==D
    c3 = x.stride(0)==T*x.stride(1)
    print(f"{name}: stride(-1)==1 -> {c1}; stride(2)==D({D}) -> {c2} (got {x.stride(2)}); "
          f"stride(0)==T*stride(1) -> {c3} (got {x.stride(0)} vs {T*x.stride(1)})")
