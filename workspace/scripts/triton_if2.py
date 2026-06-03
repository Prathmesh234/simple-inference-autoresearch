import torch, triton, triton.language as tl
@triton.autotune(configs=[triton.Config({},num_warps=2,num_stages=1),triton.Config({},num_warps=4,num_stages=1)],key=["HALF","INTERLEAVED"])
@triton.jit
def k1(q_ptr,k_ptr,cos_ptr,sin_ptr,qo,ko,NQ:tl.constexpr,NK:tl.constexpr,seq_len,cos_row_stride,
       HEAD_DIM:tl.constexpr,HALF:tl.constexpr,BL:tl.constexpr,INTERLEAVED:tl.constexpr):
    r=tl.program_id(0); c=tl.program_id(1)
    sp=r%seq_len; cs=sp*cos_row_stride
    cols=tl.arange(0,BL); m=cols<HALF
    cos=tl.load(cos_ptr+cs+cols,mask=m,other=0.0).to(tl.float32)
    sin=tl.load(sin_ptr+cs+cols,mask=m,other=0.0).to(tl.float32)
    if c<NQ:
        base=(r*NQ+c)*HEAD_DIM
        if INTERLEAVED:
            oa=base+2*cols; ob=oa+1
        else:
            oa=base+cols; ob=oa+HALF
        xa=tl.load(q_ptr+oa,mask=m,other=0.0).to(tl.float32)
        xb=tl.load(q_ptr+ob,mask=m,other=0.0).to(tl.float32)
        tl.store(qo+oa,(xa*cos-xb*sin).to(q_ptr.dtype.element_ty),mask=m)
        tl.store(qo+ob,(xb*cos+xa*sin).to(q_ptr.dtype.element_ty),mask=m)
    else:
        h=c-NQ; base=(r*NK+h)*HEAD_DIM
        if INTERLEAVED:
            oa=base+2*cols; ob=oa+1
        else:
            oa=base+cols; ob=oa+HALF
        xa=tl.load(k_ptr+oa,mask=m,other=0.0).to(tl.float32)
        xb=tl.load(k_ptr+ob,mask=m,other=0.0).to(tl.float32)
        tl.store(ko+oa,(xa*cos-xb*sin).to(k_ptr.dtype.element_ty),mask=m)
        tl.store(ko+ob,(xb*cos+xa*sin).to(k_ptr.dtype.element_ty),mask=m)
B,T,D=2,4,8; HALF=4
q=torch.randn(B*T,3,D,device='cuda',dtype=torch.bfloat16)
k=torch.randn(B*T,2,D,device='cuda',dtype=torch.bfloat16)
cos=torch.randn(T,D,device='cuda',dtype=torch.bfloat16)
sin=torch.randn(T,D,device='cuda',dtype=torch.bfloat16)
qo=torch.empty_like(q); ko=torch.empty_like(k)
k1[(B*T,5)](q,k,cos,sin,qo,ko,3,2,T,cos.stride(0),D,HALF,8,False)
print("compiled+ran ok")
