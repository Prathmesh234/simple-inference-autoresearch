"""End-to-end parity of the WIRED qkv-W8A8 engine vs a forced-W8A16 reference.
Toggles qkv W8A8 by monkeypatching w8a8_qkv_prequant to delegate to W8A16 for the
reference run (so attn_norm still emits int8 but qkv uses bf16 W8A16 math).
Includes a LONG 256-token greedy decode to expose KV-cache error accumulation."""
import sys, torch
sys.path.insert(0,"/home/ubuntu/simple-inference-autoresearch")
import env_loader  # noqa
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
from transformers import AutoTokenizer
import kernels.w8a8_gemm_kernel as w8mod
from kernels.w8a16_gemm_kernel import w8a16_linear_triton
DEV,DT="cuda",torch.bfloat16
loader=WeightLoader.from_pretrained("meta-llama/Llama-3.1-8B")
cfg=ModelConfig.from_hf_config(loader.model_dir/"config.json")
model=LlamaModel(cfg,torch.device(DEV)); model.load_weights(loader); model.to(DEV,DT); model.eval()
tok=AutoTokenizer.from_pretrained(loader.model_dir); tok.pad_token=tok.eos_token
B=64
base=["The theory of relativity states that","In a surprising turn of events, scientists",
 "The best way to learn programming is to","Once upon a time in a distant galaxy,",
 "The capital of France is","Photosynthesis is the process by which plants",
 "Climate change is primarily caused by","The history of the Roman Empire shows"]
prompts=(base*8)[:B]
ids0=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=24)["input_ids"].to(DEV)
real_qkv=w8mod.w8a8_qkv_prequant
def ref_qkv(x_int8,act_scale,w_int8,w_scale,orig_shape):
    # dequant int8 act back to bf16 and run W8A16 (true weight-only reference)
    x=(x_int8.float()*act_scale[:,None].float()).to(DT)
    M,K=x_int8.shape; N=w_int8.shape[0]
    y=w8a16_linear_triton(x,w_int8,w_scale)
    return y.reshape(*orig_shape[:-1],N)
def gen(mode,nnew):
    w8mod.w8a8_qkv_prequant = real_qkv if mode=="w8a8" else ref_qkv
    kv=KVCache(n_layers=cfg.num_hidden_layers,max_batch=B,max_seq_len=ids0.shape[1]+nnew+1,
      n_heads_kv=cfg.num_key_value_heads,head_dim=cfg.head_dim,dtype=DT,device=DEV)
    out=[]
    with torch.no_grad():
        lg=model(ids0,start_pos=0,kv_cache=kv); nt=lg[:,-1,:].argmax(-1,keepdim=True)
        out.append(nt); pos=ids0.shape[1]
        for _ in range(nnew-1):
            lg=model(nt,start_pos=pos,kv_cache=kv); nt=lg[:,-1,:].argmax(-1,keepdim=True)
            out.append(nt); pos+=1
    return torch.cat(out,1)
for nnew in (32,256):
    gr=gen("ref",nnew); g8=gen("w8a8",nnew)
    m=(gr==g8).float().mean().item(); ex=(gr==g8).all(1).float().mean().item()
    # divergence position: first mismatch index per seq
    first=[]
    for i in range(B):
        d=(gr[i]!=g8[i]).nonzero()
        first.append(int(d[0]) if len(d) else nnew)
    import statistics as st
    print(f"nnew={nnew}: token-match={m:.3f} exact-seq={ex:.3f} median-first-divergence={st.median(first)}/{nnew}")
w8mod.w8a8_qkv_prequant=real_qkv
print("SAMPLE w8a8:",repr(tok.decode(gen('w8a8',40)[0])))
