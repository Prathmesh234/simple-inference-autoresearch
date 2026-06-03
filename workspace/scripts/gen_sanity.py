import sys,os; sys.path.insert(0,os.getcwd())
import env_loader, torch
from transformers import AutoTokenizer
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
from sampling import sample
DEV="cuda"; DT=torch.bfloat16
loader=WeightLoader.from_pretrained("meta-llama/Llama-3.1-8B")
cfg=ModelConfig.from_hf_config(loader.model_dir/"config.json")
tok=AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model=LlamaModel(cfg,torch.device(DEV)); model.load_weights(loader); model.to(DEV,DT); model.eval()
prompt="The capital of France is"
ids=tok(prompt,return_tensors="pt").input_ids.to(DEV)
B,P=ids.shape; maxlen=P+40
kv=KVCache(cfg.num_hidden_layers,B,maxlen,cfg.num_key_value_heads,cfg.head_dim,DT,DEV)
with torch.no_grad():
    logits=model(ids,start_pos=0,kv_cache=kv)
    out=[]
    nxt=logits[:,-1,:].argmax(-1,keepdim=True)
    for i in range(40):
        out.append(nxt.item())
        logits=model(nxt,start_pos=P+i,kv_cache=kv)
        nxt=logits[:,-1,:].argmax(-1,keepdim=True)
print(repr(prompt+tok.decode(out)))
