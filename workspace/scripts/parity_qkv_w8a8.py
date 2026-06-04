"""Decode-path coherence: greedy-generate with qkv W8A8 vs qkv W8A16. Decode M=B
(64, in the 16<M<=256 W8A8 bucket). Reports per-sequence greedy token-match."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0,"/home/ubuntu/simple-inference-autoresearch")
import env_loader  # noqa
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
from transformers import AutoTokenizer
import ops.attention as attn_mod
from kernels.w8a8_gemm_kernel import w8a8_linear_triton
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
 "The stock market today","A healthy breakfast should include"]
prompts=(base*8)[:B]
ids0=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=24)["input_ids"].to(DEV)
orig=attn_mod.GroupedQueryAttention._project_qkv
MODE={"m":"ref"}
def patched(self,x,B,T):
    proj = w8a8_linear_triton if MODE["m"]=="w8a8" else (lambda x,wi,s: w8a16_linear_triton(x,wi,s))
    qkv=proj(x,self.w_qkv_int8,self.w_qkv_scale)
    q=qkv[...,:self.q_size].reshape(B,T,self.num_heads_q,self.head_dim)
    k=qkv[...,self.q_size:self.q_size+self.kv_size].reshape(B,T,self.num_heads_kv,self.head_dim)
    v=qkv[...,self.q_size+self.kv_size:].reshape(B,T,self.num_heads_kv,self.head_dim)
    return q,k,v
attn_mod.GroupedQueryAttention._project_qkv=patched
def gen(mode,nnew=32):
    MODE["m"]=mode
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
gr=gen("ref"); g8=gen("w8a8")
match=(gr==g8).float().mean().item()
exact=(gr==g8).all(1).float().mean().item()
print(f"qkv W8A8 decode greedy: token-match={match:.3f}  exact-seq-match={exact:.3f}  (B={B}, 32 new tok)")
print("REF :",repr(tok.decode(gr[0])))
print("W8A8:",repr(tok.decode(g8[0])))
print("REF :",repr(tok.decode(gr[5])))
print("W8A8:",repr(tok.decode(g8[5])))
