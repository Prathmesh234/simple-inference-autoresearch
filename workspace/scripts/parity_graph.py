import sys, os; sys.path.insert(0, os.getcwd())
import env_loader, torch
import model.llama as llama_mod
from profiling.profile_engine import build_model, DEVICE, DTYPE
from model.kv_cache import KVCache
from transformers import AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
model, cfg = build_model(MODEL_ID)
tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.pad_token = tok.eos_token

prompt = "The capital of France is"
ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
B, T = ids.shape
NEW = 40

@torch.no_grad()
def greedy(use_graph):
    llama_mod.USE_CUDA_GRAPH = use_graph
    model._decoder = None
    kv = KVCache(cfg.num_hidden_layers, B, T+NEW, cfg.num_key_value_heads,
                 cfg.head_dim, DTYPE, DEVICE)
    out = []
    logits = model(ids, start_pos=0, kv_cache=kv)
    nt = logits[:, -1, :].argmax(-1, keepdim=True)
    out.append(nt.item()); pos = T
    for _ in range(NEW-1):
        logits = model(nt, start_pos=pos, kv_cache=kv)
        nt = logits[:, -1, :].argmax(-1, keepdim=True)
        out.append(nt.item()); pos += 1
    return out

eager = greedy(False)
graph = greedy(True)
print("EAGER :", tok.decode(eager))
print("GRAPH :", tok.decode(graph))
match = sum(int(a==b) for a,b in zip(eager,graph))
print(f"\ntoken match: {match}/{len(eager)}")
print("FIRST DIVERGENCE:", next((i for i,(a,b) in enumerate(zip(eager,graph)) if a!=b), None))

# batched parity at B=8 (closer to headline regime) with random prompts
print("\n=== batched B=8 logits parity (eager vs graph), step-by-step ===")
torch.manual_seed(0)
B2 = 8
ids2 = torch.randint(0, cfg.vocab_size, (B2, T), device=DEVICE)
def run_logits(use_graph):
    llama_mod.USE_CUDA_GRAPH = use_graph; model._decoder = None
    kv = KVCache(cfg.num_hidden_layers, B2, T+NEW, cfg.num_key_value_heads, cfg.head_dim, DTYPE, DEVICE)
    logits = model(ids2, start_pos=0, kv_cache=kv)
    nt = logits[:, -1, :].argmax(-1, keepdim=True); pos=T; seq=[nt]
    for _ in range(10):
        logits = model(nt, start_pos=pos, kv_cache=kv)
        nt = logits[:, -1, :].argmax(-1, keepdim=True); seq.append(nt); pos+=1
    return torch.cat(seq, dim=1)
se = run_logits(False); sg = run_logits(True)
print("eager tokens[0]:", se[0].tolist())
print("graph tokens[0]:", sg[0].tolist())
print("batched token match:", (se==sg).float().mean().item())
