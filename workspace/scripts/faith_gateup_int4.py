"""Is int4-grouped gate_up faithful through the nonlinear SwiGLU?
gate_up int8 weight=117MB > 96MB L2 (HBM-bound). int4=58MB FITS L2 -> potential
structural HBM->L2 win on the dominant (24%) decode GEMM. Gate = coherence.
Compares int4-grouped gate_up vs the SHIPPING int8 per-channel path: per-layer
swiglu-output rel_err + final-token logits argmax/top-k agreement."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0,"/home/ubuntu/simple-inference-autoresearch")
import env_loader  # noqa
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
from transformers import AutoTokenizer
import ops.mlp as mlp_mod
DEV,DT="cuda",torch.bfloat16
loader=WeightLoader.from_pretrained("meta-llama/Llama-3.1-8B")
cfg=ModelConfig.from_hf_config(loader.model_dir/"config.json")
model=LlamaModel(cfg,torch.device(DEV)); model.load_weights(loader); model.to(DEV,DT); model.eval()
tok=AutoTokenizer.from_pretrained(loader.model_dir); tok.pad_token=tok.eos_token
prompts=["The theory of relativity states that","In a surprising turn of events,",
 "The best way to learn programming is to","Once upon a time in a distant galaxy,",
 "The capital of France is","Photosynthesis is the process by which"]
ids=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=32)["input_ids"].to(DEV)

def grouped_int4_deq(W,group):  # W: (2I,K) bf16
    N,K=W.shape; Wg=W.view(N,K//group,group)
    amax=Wg.abs().amax(-1,keepdim=True); scale=amax/7.0
    q=torch.clamp(torch.round(Wg/scale),-8,7)
    return (q*scale).view(N,K).to(W.dtype)

MODE={"int4":False,"group":128}
orig_fwd=mlp_mod.SwiGLUMLP.forward
def patched(self,x,x_int8=None,x_scale=None):
    if not MODE["int4"]:
        return orig_fwd(self,x,x_int8,x_scale)
    W=(self.w_gate_up_int8.float()*self.w_gate_up_scale.float()[:,None]).to(x.dtype)
    Wd=grouped_int4_deq(W,MODE["group"])
    gu=F.linear(x,Wd); I=gu.shape[-1]//2
    g,u=gu[...,:I],gu[...,I:]
    h=F.silu(g)*u
    Wd2=(self.w_down_int8.float()*self.w_down_scale.float()[:,None]).to(x.dtype)
    return F.linear(h,Wd2)

mlp_mod.SwiGLUMLP.forward = patched
def run():
    kv=KVCache(n_layers=cfg.num_hidden_layers,max_batch=ids.shape[0],max_seq_len=ids.shape[1]+2,
      n_heads_kv=cfg.num_key_value_heads,head_dim=cfg.head_dim,dtype=DT,device=DEV)
    with torch.no_grad(): out=model(ids,start_pos=0,kv_cache=kv)
    return out[:,-1,:].float()

MODE["int4"]=False; ref=run()
for grp in (128,64,32):
    MODE["int4"]=True; MODE["group"]=grp; l4=run()
    am=(ref.argmax(-1)==l4.argmax(-1)).float().mean().item()
    ov={}
    for k in (1,5,10):
        t0=ref.topk(k,-1).indices; t1=l4.topk(k,-1).indices
        ov[k]=sum(len(set(t0[i].tolist())&set(t1[i].tolist()))/k for i in range(ids.shape[0]))/ids.shape[0]
    rel=((l4-ref).abs()/(ref.abs()+1.0)).mean().item()
    print(f"int4 gate_up g={grp}: final argmax={am:.3f} top5={ov[5]:.3f} top10={ov[10]:.3f} logit_rel={rel:.4f}")
