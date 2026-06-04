import torch, time, triton
from kernels.flash_decode_kernel import attention_flash_decode, _flash_decode_fwd
torch.manual_seed(0); dev="cuda"
B,Hq,Hkv,D = 128,32,8,128
for KV in (93, 160, 256):
    S = KV+4
    q=torch.randn(B,Hq,1,D,device=dev,dtype=torch.bfloat16)
    k=torch.randn(B,Hkv,S,D,device=dev,dtype=torch.bfloat16)
    v=torch.randn(B,Hkv,S,D,device=dev,dtype=torch.bfloat16)
    kvlen=torch.tensor([KV],dtype=torch.int32,device=dev)
    def call(): return attention_flash_decode(q,k,v,kvlen)
    # warm (triggers autotune)
    for _ in range(30): call()
    torch.cuda.synchronize()
    best=1e9
    for _ in range(30):
        t=time.perf_counter()
        for _ in range(200): call()
        torch.cuda.synchronize(); best=min(best,(time.perf_counter()-t)/200*1e6)
    print(f"KV={KV}: autotuned flash_decode = {best:.2f}us  (B*Hq={B*Hq} programs)")
