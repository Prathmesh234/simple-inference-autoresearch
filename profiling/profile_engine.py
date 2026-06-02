"""
profile_engine.py — the base profiling script for the inference engine.

One script the agent can point at the engine to see where time and memory go.
It runs the real generation path (prefill once, then decode N tokens with KV
cache) under torch.profiler and, for each prompt-length flavor, sweeps a range of
batch sizes so you can see how aggregate decode throughput scales as more
sequences decode together. Per (flavor, batch size) it reports:

    - prefill latency (ms)            -> Time-To-First-Token
    - decode latency (ms/token)       -> per-step cost for the whole batch
    - aggregate throughput (tok/s)    -> the primary metric: B * 1000 / decode_ms
    - per-sequence throughput (tok/s) -> single-stream speed within the batch
    - peak VRAM (GB)                  -> the memory guardrail (grows with batch)
    - op table sorted by CUDA time    -> where the time actually goes

It writes a Chrome/Perfetto trace for the largest batch of each flavor to
profiling/out/ so you can open the timeline in chrome://tracing or
https://ui.perfetto.dev.

Run:
    uv run python profiling/profile_engine.py                 # full mix + batch sweep
    uv run python profiling/profile_engine.py --flavors instruct chat
    uv run python profiling/profile_engine.py --batch-sizes 1 8 32 64 128
    uv run python profiling/profile_engine.py --new-tokens 128

By default it runs EVERY input type (all flavors in benchmarks/prompts.py) and
sweeps batch sizes up to 128, so you get a full classification of how the engine
behaves across the whole prompt-shape x batch-size grid.

Requires gated Llama weights, so set HF_TOKEN (see .env.example).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import env_loader  # noqa: F401,E402  loads .env (USE_TRITON / HF_TOKEN) before imports
import torch  # noqa: E402
from torch.profiler import profile, ProfilerActivity  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from config import ModelConfig  # noqa: E402
from loader import WeightLoader  # noqa: E402
from model.llama import LlamaModel  # noqa: E402
from model.kv_cache import KVCache  # noqa: E402
from sampling import sample  # noqa: E402
import ops.rmsnorm as rmsnorm_mod  # noqa: E402
from benchmarks.prompts import load_prompts, DATASETS  # noqa: E402

DEVICE = "cuda"
DTYPE = torch.bfloat16
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"
WARMUP = 2
N_NEW_TOKENS = 64

# Batch sizes to sweep per flavor. The primary metric is aggregate decode
# throughput (tok/s across the whole batch), so we measure how it scales as more
# sequences decode together in one step — from single-stream (1) all the way up to
# heavy serving load (128). Wider batches may OOM on the KV cache; those rows are
# reported as crashes by the run, which is itself useful signal.
DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]

TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.9

OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Prompt-length flavors drawn from the HF benchmark mix (see benchmarks/prompts.py).
# Default = EVERY registered input type, so a single run classifies the engine
# across all prompt shapes (short instruct, real chat, long-context, summarize,
# code) instead of a hand-picked subset.
DEFAULT_FLAVORS = list(DATASETS.keys())


@torch.no_grad()
def prefill_and_decode(model, kv_cache, prompt_ids, n_new_tokens):
    """Prefill once, then sample n_new_tokens with the KV cache. Returns timings."""
    kv_cache.reset()
    _, T = prompt_ids.shape

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = model(prompt_ids, start_pos=0, kv_cache=kv_cache)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000

    next_tok = sample(logits[:, -1, :], temperature=TEMPERATURE,
                      top_k=TOP_K, top_p=TOP_P).unsqueeze(-1)

    step_times = []
    pos = T
    for _ in range(n_new_tokens - 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(next_tok, start_pos=pos, kv_cache=kv_cache)
        next_tok = sample(logits[:, -1, :], temperature=TEMPERATURE,
                          top_k=TOP_K, top_p=TOP_P).unsqueeze(-1)
        torch.cuda.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000)
        pos += 1

    decode_ms = sum(step_times) / len(step_times) if step_times else 0.0
    return prefill_ms, decode_ms


def build_model(model_id: str):
    loader = WeightLoader.from_pretrained(model_id)
    cfg = ModelConfig.from_hf_config(loader.model_dir / "config.json")
    model = LlamaModel(cfg, torch.device(DEVICE))
    model.load_weights(loader)
    model.to(DEVICE, DTYPE)
    model.eval()
    return model, cfg


def tokenize_batch(texts, tokenizer):
    """Tokenize DISTINCT prompts into one (B, T) batch with LEFT padding.

    Decoder-only batched generation uses left padding so real content ends at the
    last column: logits[:, -1, :] is then the true final token for every row and
    all rows decode in lockstep at a shared start_pos.
    """
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(texts, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = prev_side
    return enc["input_ids"].to(DEVICE)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    p.add_argument("--flavors", nargs="*", default=DEFAULT_FLAVORS,
                   help="prompt sets from benchmarks/prompts.py")
    p.add_argument("--batch-sizes", nargs="*", type=int, default=DEFAULT_BATCH_SIZES,
                   help="batch sizes to sweep (aggregate throughput vs batch)")
    p.add_argument("--new-tokens", type=int, default=N_NEW_TOKENS)
    p.add_argument("--max-chars", type=int, default=4000,
                   help="truncate prompts to bound prefill")
    args = p.parse_args()

    backend = "triton (fused kernels)" if rmsnorm_mod.USE_TRITON else "pytorch (reference)"
    print(f"Backend: {backend}   Model: {args.model}")

    model, cfg = build_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    max_batch = max(args.batch_sizes)

    # Track the headline numbers across every (flavor, batch) run so we can print
    # a single grep-able summary at the end (consumed by the experiment loop).
    best_agg_tps = 0.0          # best aggregate decode throughput anywhere in the sweep
    best_agg_at = (None, None)  # (flavor, batch) where best_agg_tps was hit
    seq_tps_b1 = 0.0            # single-stream decode tok/s at batch size 1 (latency guardrail)
    peak_gb_overall = 0.0       # worst-case peak VRAM across the whole sweep

    for flavor in args.flavors:
        # Pull enough DISTINCT prompts to fill the widest batch we'll sweep.
        prompts = load_prompts(flavor, n=max_batch, max_chars=args.max_chars)
        purpose = DATASETS[flavor].purpose if flavor in DATASETS else ""

        print(f"\n{'='*78}")
        print(f"  flavor={flavor}   new_tokens={args.new_tokens}")
        if purpose:
            print(f"  class: {purpose}")
        print(f"{'-'*78}")
        print(f"  {'batch':>5}  {'prompt_tok':>10}  {'prefill_ms':>10}  "
              f"{'decode_ms':>9}  {'tok/s(agg)':>10}  {'tok/s/seq':>9}  {'VRAM_GB':>7}")

        for B in args.batch_sizes:
            if B > len(prompts):
                print(f"  (skipping batch {B}: only {len(prompts)} distinct prompts available)")
                continue
            ids = tokenize_batch(prompts[:B], tokenizer)

            # Wider batches grow the KV cache linearly and can exhaust VRAM. An OOM
            # at one batch size should NOT kill the whole sweep: catch it, report
            # the batch as OOM, free the memory, and keep going. The point of going
            # up to 128 is exactly to find where this engine hits the memory wall.
            try:
                kv_cache = KVCache(
                    n_layers=cfg.num_hidden_layers,
                    max_batch=B,
                    max_seq_len=ids.shape[1] + args.new_tokens,
                    n_heads_kv=cfg.num_key_value_heads,
                    head_dim=cfg.head_dim,
                    dtype=DTYPE,
                    device=DEVICE,
                )

                for _ in range(WARMUP):
                    prefill_and_decode(model, kv_cache, ids, min(args.new_tokens, 4))
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(DEVICE)

                # Only capture a full profiler trace for the largest batch of each
                # flavor (the others are quick throughput-only timing runs).
                capture_trace = (B == max(b for b in args.batch_sizes if b <= len(prompts)))
                if capture_trace:
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                 record_shapes=True, profile_memory=True) as prof:
                        prefill_ms, decode_ms = prefill_and_decode(
                            model, kv_cache, ids, args.new_tokens)
                else:
                    prefill_ms, decode_ms = prefill_and_decode(
                        model, kv_cache, ids, args.new_tokens)
                    prof = None
            except torch.cuda.OutOfMemoryError:
                print(f"  {B:>5}  {ids.shape[1]:>10}  {'OOM':>10}  "
                      f"{'OOM':>9}  {'OOM':>10}  {'OOM':>9}  {'OOM':>7}")
                del ids
                if "kv_cache" in locals():
                    del kv_cache
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(DEVICE)
                continue

            peak_gb = torch.cuda.max_memory_allocated(DEVICE) / 1e9
            # Aggregate throughput = whole batch's tokens per second of decode.
            agg_tps = (B * 1000.0 / decode_ms) if decode_ms > 0 else 0.0
            per_seq_tps = (1000.0 / decode_ms) if decode_ms > 0 else 0.0

            # Update headline trackers for the final summary.
            if agg_tps > best_agg_tps:
                best_agg_tps = agg_tps
                best_agg_at = (flavor, B)
            if B == 1:
                seq_tps_b1 = max(seq_tps_b1, per_seq_tps)
            peak_gb_overall = max(peak_gb_overall, peak_gb)

            print(f"  {B:>5}  {ids.shape[1]:>10}  {prefill_ms:>10.2f}  "
                  f"{decode_ms:>9.3f}  {agg_tps:>10.1f}  {per_seq_tps:>9.1f}  {peak_gb:>7.2f}")

            if prof is not None:
                print(f"{'-'*78}")
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
                trace_path = OUT_DIR / f"engine_{flavor}_b{B}_trace.json"
                prof.export_chrome_trace(str(trace_path))
                print(f"  trace -> {trace_path}")

    # ── grep-able summary (the experiment loop reads these lines) ────────────
    # The primary metric is best_agg_tps (higher is better). seq_tps_b1 and
    # peak_vram_gb are guardrails. Keep these prefixes stable so they can be
    # parsed with a simple grep, e.g.  grep "^best_agg_tps:" run.log
    bf, bb = best_agg_at
    print(f"\n{'='*78}")
    print("  SUMMARY (primary metric = best_agg_tps, higher is better)")
    print(f"{'='*78}")
    print(f"best_agg_tps:     {best_agg_tps:.1f}")
    print(f"best_agg_at:      flavor={bf} batch={bb}")
    print(f"seq_tps_b1:       {seq_tps_b1:.1f}")
    print(f"peak_vram_gb:     {peak_gb_overall:.2f}")


if __name__ == "__main__":
    main()
