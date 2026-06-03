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
def run():
    kv=KVCache(cfg.num_hidden_layers,B,T+70,cfg.num_key_value_heads,cfg.head_dim,DT,DEV)
    with torch.no_grad():
        return model(ids,start_pos=0,kv_cache=kv)
a=run(); b=run()
print("same-path run-to-run max|Δ|=",(a.float()-b.float()).abs().max().item())
