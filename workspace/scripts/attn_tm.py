import sys,os; sys.path.insert(0,os.getcwd())
import env_loader, torch
from kernels.attention_kernel import attention_flash_triton
DEV="cuda"; DT=torch.bfloat16
torch.manual_seed(0)
for (B,Hq,Hkv,Tq,Tk,D,causal) in [(2,32,8,1,40,128,False),(2,32,8,7,7,128,True),(1,32,8,1,1,128,False)]:
    q=torch.randn(B,Hq,Tq,D,device=DEV,dtype=DT)
    k=torch.randn(B,Hkv,Tk,D,device=DEV,dtype=DT)
    v=torch.randn(B,Hkv,Tk,D,device=DEV,dtype=DT)
    o_def=attention_flash_triton(q,k,v,causal=causal,assume_contiguous=True)            # (B,Hq,Tq,D)
    o_tm =attention_flash_triton(q,k,v,causal=causal,assume_contiguous=True,out_token_major=True) # (B,Tq,Hq,D)
    # token-major viewed back to (B,Hq,Tq,D)
    o_tm_t=o_tm.transpose(1,2)
    d=(o_def.float()-o_tm_t.float()).abs().max().item()
    print(f"B{B} Hq{Hq} Tq{Tq} Tk{Tk} causal={causal}  max|Δ|={d:.3e}  shape_tm={tuple(o_tm.shape)}")
