import sys,os; sys.path.insert(0,os.getcwd())
import env_loader, torch
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
DEV="cuda"; DT=torch.bfloat16
loader=WeightLoader.from_pretrained("meta-llama/Llama-3.1-8B")
cfg=ModelConfig.from_hf_config(loader.model_dir/"config.json")
model=LlamaModel(cfg,torch.device(DEV)); model.load_weights(loader); model.to(DEV,DT); model.eval()
torch.manual_seed(0)
B,T=8,29
ids=torch.randint(0,128000,(B,T),device=DEV)

@torch.no_grad()
def threaded(ids,kv):
    x=model.embed(ids); residual=None; states=[]
    for layer in model.layers:
        x,residual=layer(x,residual,start_pos=0,kv_cache=kv)
        # reconstruct full residual stream (pending add resolved): x_full = residual + x
        states.append((residual+x).float().clone())
    return states

@torch.no_grad()
def ref(ids,kv):
    x=model.embed(ids); states=[]
    for layer in model.layers:
        h=layer.attn_norm(x); h=layer.attn(h,start_pos=0,kv_cache=kv); x=x+h
        h=layer.mlp_norm(x); h=layer.mlp(h); x=x+h
        states.append(x.float().clone())
    return states

kv1=KVCache(cfg.num_hidden_layers,B,T+70,cfg.num_key_value_heads,cfg.head_dim,DT,DEV)
kv2=KVCache(cfg.num_hidden_layers,B,T+70,cfg.num_key_value_heads,cfg.head_dim,DT,DEV)
st=threaded(ids,kv1); sr=ref(ids,kv2)
for i,(a,b) in enumerate(zip(st,sr)):
    d=(a-b).abs().max().item()
    if i<3 or d>0: print(f"block {i:2d} max|Δresidstream|={d:.3e}")
