import sys, os; sys.path.insert(0, os.getcwd())
import env_loader, time, torch
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
from sampling import sample
DEV="cuda"; DT=torch.bfloat16
loader=WeightLoader.from_pretrained("meta-llama/Llama-3.1-8B")
cfg=ModelConfig.from_hf_config(loader.model_dir/"config.json")
model=LlamaModel(cfg,torch.device(DEV)); model.load_weights(loader); model.to(DEV,DT); model.eval()

def bench(B, Tprompt, ndecode=32):
    kv=KVCache(cfg.num_hidden_layers,B,Tprompt+4+ndecode*3+10,cfg.num_key_value_heads,cfg.head_dim,DT,DEV)
    ids=torch.randint(0,128000,(B,Tprompt),device=DEV)
    with torch.no_grad():
        kv.reset()
        logits=model(ids,start_pos=0,kv_cache=kv)
        tok=sample(logits[:,-1,:],temperature=0.7,top_k=50,top_p=0.9).unsqueeze(-1)
        pos=Tprompt
        # warmup
        for _ in range(4):
            logits=model(tok,start_pos=pos,kv_cache=kv); pos+=1
            tok=sample(logits[:,-1,:],0.7,50,0.9).unsqueeze(-1)
        torch.cuda.synchronize()
        # time full step (model+sample)
        t=[]
        for _ in range(ndecode):
            torch.cuda.synchronize(); t0=time.perf_counter()
            logits=model(tok,start_pos=pos,kv_cache=kv)
            tok=sample(logits[:,-1,:],0.7,50,0.9).unsqueeze(-1)
            torch.cuda.synchronize(); t.append((time.perf_counter()-t0)*1000); pos+=1
        full=sum(t)/len(t)
        # time model-only (no sample)
        t=[]
        for _ in range(ndecode):
            torch.cuda.synchronize(); t0=time.perf_counter()
            logits=model(tok,start_pos=pos,kv_cache=kv)
            torch.cuda.synchronize(); t.append((time.perf_counter()-t0)*1000)
            tok=sample(logits[:,-1,:],0.7,50,0.9).unsqueeze(-1); pos+=1
        model_only=sum(t)/len(t)
        # time sample-only on a fixed logits
        lg=logits[:,-1,:].contiguous()
        t=[]
        for _ in range(ndecode):
            torch.cuda.synchronize(); t0=time.perf_counter()
            _=sample(lg,0.7,50,0.9)
            torch.cuda.synchronize(); t.append((time.perf_counter()-t0)*1000)
        samp=sum(t)/len(t)
    del kv; torch.cuda.empty_cache()
    print(f"B={B:4d} Tprompt={Tprompt:4d} | full={full:6.2f}ms  model={model_only:6.2f}ms  sample={samp:6.2f}ms  (lm_head incl in model)")

for B in (1,32,64,128):
    bench(B,29,32)
