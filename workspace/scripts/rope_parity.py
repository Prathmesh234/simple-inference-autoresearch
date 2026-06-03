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
rf=model.rope_freqs
# Reference: old behavior = float32 slice then .to(bf16)
def old_get(seq_len,start_pos):
    c=rf.cos[start_pos:start_pos+seq_len].to(DT); s=rf.sin[start_pos:start_pos+seq_len].to(DT); return c,s
maxd=0
for sp in (0,1,5,100,1000):
    cn,sn=rf.get(seq_len=1,start_pos=sp,dtype=DT)
    co,so=old_get(1,sp)
    maxd=max(maxd,(cn.float()-co.float()).abs().max().item(),(sn.float()-so.float()).abs().max().item())
print("cos/sin cached-vs-old max|Δ| =",maxd)
