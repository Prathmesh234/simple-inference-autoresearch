import torch, time
import torch.nn.functional as F
from kernels.flash_decode_int8_kernel import attention_flash_decode_int8
from kernels.paged_flash_decode_int8_gqa_kernel import gqa_int8_flash_decode
torch.manual_seed(0); dev="cuda"
B,Hq,Hkv,D = 128,32,8,128; KV_GROUP=Hq//Hkv
def cuda_time(fn,iters=300,reps=6):
    for _ in range(20): fn()
    torch.cuda.synchronize(); best=1e9
    for _ in range(reps):
        s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
        for _ in range(iters): fn()
        e.record();torch.cuda.synchronize();best=min(best,s.elapsed_time(e)/iters*1000)
    return best
print(f"{'kv_len':>7} {'pq_us':>8} {'gqa_us':>8} {'speedup':>8} {'gqa_rel':>9} {'gqa_cos':>9}")
for kv_len in (455,768,1012):
    S=((kv_len+31)//32)*32
    q=torch.randn(B,Hq,1,D,device=dev,dtype=torch.bfloat16)*0.5
    k=torch.randn(B,Hkv,S,D,device=dev,dtype=torch.bfloat16)*0.5
    v=torch.randn(B,Hkv,S,D,device=dev,dtype=torch.bfloat16)*0.5
    amaxk=k.abs().amax(-1,keepdim=True).clamp_min(1e-8).float(); ksc=(amaxk/127); ki=torch.round(k.float()/ksc).clamp(-127,127).to(torch.int8); ks=ksc.squeeze(-1).half()
    amaxv=v.abs().amax(-1,keepdim=True).clamp_min(1e-8).float(); vsc=(amaxv/127); vi=torch.round(v.float()/vsc).clamp(-127,127).to(torch.int8); vs=vsc.squeeze(-1).half()
    kvlen=torch.tensor([kv_len],dtype=torch.int32,device=dev)
    kk=k[:,:,:kv_len].repeat_interleave(KV_GROUP,1).float(); vv=v[:,:,:kv_len].repeat_interleave(KV_GROUP,1).float()
    ref=F.scaled_dot_product_attention(q.float(),kk,vv).transpose(1,2)  # (B,1,Hq,D)
    try:
        o_g=gqa_int8_flash_decode(q,ki,vi,ks,vs,kvlen)
        rel=((o_g.float()-ref).abs()/(ref.abs()+1e-3)).mean().item()
        cos=F.cosine_similarity(o_g.float().flatten(),ref.flatten(),dim=0).item()
        t_p=cuda_time(lambda: attention_flash_decode_int8(q,ki,vi,ks,vs,kvlen))
        t_g=cuda_time(lambda: gqa_int8_flash_decode(q,ki,vi,ks,vs,kvlen))
        print(f"{kv_len:>7} {t_p:>8.1f} {t_g:>8.1f} {t_p/t_g:>7.2f}x {rel:>9.5f} {cos:>9.6f}")
    except Exception as ex:
        print(f"{kv_len}: ERR {str(ex)[:120]}")
