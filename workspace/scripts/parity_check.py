import sys, os; sys.path.insert(0, os.getcwd())
import env_loader, torch
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
DEV="cuda"; DT=torch.bfloat16
loader=WeightLoader.from_pretrained("meta-llama/Llama-3.1-8B")
cfg=ModelConfig.from_hf_config(loader.model_dir/"config.json")
model=LlamaModel(cfg,torch.device(DEV)); model.load_weights(loader); model.to(DEV,DT); model.eval()

# Monkeypatch a reference forward using the OLD unfused path for comparison.
import torch.nn.functional as F
from ops.rmsnorm import _pytorch_rmsnorm
@torch.no_grad()
def ref_forward(ids, start_pos, kv):
    x = model.embed(ids)
    for layer in model.layers:
        h = layer.attn_norm(x)
        h = layer.attn(h, start_pos=start_pos, kv_cache=kv)
        x = x + h
        h = layer.mlp_norm(x)
        h = layer.mlp(h)
        x = x + h
    x = model.norm(x)
    return model.head(x)

torch.manual_seed(0)
B,T=8,29
ids=torch.randint(0,128000,(B,T),device=DEV)
kv1=KVCache(cfg.num_hidden_layers,B,T+70,cfg.num_key_value_heads,cfg.head_dim,DT,DEV)
kv2=KVCache(cfg.num_hidden_layers,B,T+70,cfg.num_key_value_heads,cfg.head_dim,DT,DEV)
with torch.no_grad():
    # prefill
    lo_new=model(ids,start_pos=0,kv_cache=kv1)
    lo_ref=ref_forward(ids,0,kv2)
    d_prefill=(lo_new.float()-lo_ref.float()).abs().max().item()
    # argmax token agreement on last position
    am_new=lo_new[:,-1,:].argmax(-1); am_ref=lo_ref[:,-1,:].argmax(-1)
    print(f"prefill max|Δlogits|={d_prefill:.4e}  argmax_agree={(am_new==am_ref).float().mean().item():.3f}")
    # a few decode steps
    tok=am_new.unsqueeze(-1); pos=T
    maxd=0; agree=1.0
    for s in range(8):
        ln=model(tok,start_pos=pos,kv_cache=kv1)
        lr=ref_forward(tok,pos,kv2)
        maxd=max(maxd,(ln.float()-lr.float()).abs().max().item())
        an=ln[:,-1,:].argmax(-1); ar=lr[:,-1,:].argmax(-1)
        agree=min(agree,(an==ar).float().mean().item())
        tok=an.unsqueeze(-1); pos+=1
    print(f"decode  max|Δlogits|={maxd:.4e}  min_argmax_agree={agree:.3f}")
