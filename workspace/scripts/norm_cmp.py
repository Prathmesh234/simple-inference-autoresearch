import sys,os; sys.path.insert(0,os.getcwd())
import env_loader, torch
from kernels.rmsnorm_kernel import rmsnorm_triton
from kernels.add_rmsnorm_kernel import add_rmsnorm_triton
DEV="cuda"; DT=torch.bfloat16
torch.manual_seed(0)
H=4096
for B in (8,128):
    h=torch.randn(B,H,device=DEV,dtype=DT)
    r=torch.randn(B,H,device=DEV,dtype=DT)
    w=torch.randn(H,device=DEV,dtype=DT)
    eps=1e-5
    # reference: bf16 add then rmsnorm_triton
    new_r_ref=r+h
    normed_ref=rmsnorm_triton(new_r_ref,w,eps)
    normed_f,new_r_f=add_rmsnorm_triton(h,r,w,eps)
    print(f"B={B} Δnormed={ (normed_f.float()-normed_ref.float()).abs().max().item():.3e}  Δresid={(new_r_f.float()-new_r_ref.float()).abs().max().item():.3e}")
