import torch, triton, triton.language as tl
@triton.jit
def k1(q_ptr,k_ptr,qo,ko,NQ:tl.constexpr,NK:tl.constexpr,D:tl.constexpr,BL:tl.constexpr):
    r=tl.program_id(0); c=tl.program_id(1)
    cols=tl.arange(0,BL); m=cols<D
    if c<NQ:
        base=(r*NQ+c)*D
        x=tl.load(q_ptr+base+cols,mask=m,other=0.0).to(tl.float32)
        tl.store(qo+base+cols,(x*2.0).to(q_ptr.dtype.element_ty),mask=m)
    else:
        h=c-NQ; base=(r*NK+h)*D
        x=tl.load(k_ptr+base+cols,mask=m,other=0.0).to(tl.float32)
        tl.store(ko+base+cols,(x*3.0).to(k_ptr.dtype.element_ty),mask=m)
q=torch.randn(4,3,8,device='cuda',dtype=torch.bfloat16)
k=torch.randn(4,2,8,device='cuda',dtype=torch.bfloat16)
qo=torch.empty_like(q); ko=torch.empty_like(k)
k1[(4,5)](q,k,qo,ko,3,2,8,8)
print("ok", torch.allclose(qo.float(),q.float()*2,atol=1e-2), torch.allclose(ko.float(),k.float()*3,atol=1e-2))
