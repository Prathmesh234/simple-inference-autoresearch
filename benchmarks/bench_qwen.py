"""Repeatable warmed benchmark for the Qwen3.5 model plugin."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

import env_loader  # noqa: F401
from generate import generate_with_stats
from model.registry import build_cache, load_model
from tokenizer import Tokenizer


DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_PROMPT = (
    "Explain why efficient inference matters in one concise paragraph."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1 or args.warmup_runs < 0:
        parser.error("runs must be positive and warmup-runs cannot be negative")

    loaded = load_model(args.model, device="cuda", dtype=torch.bfloat16)
    tokenizer = Tokenizer.from_pretrained(args.model)
    prompt_tokens = len(tokenizer.encode(args.prompt, add_bos=True))
    cache = build_cache(
        loaded,
        max_batch=1,
        max_seq_len=prompt_tokens + args.max_new_tokens,
        dtype=torch.bfloat16,
        device="cuda",
    )
    generation_args = {
        "prompt": args.prompt,
        "model": loaded.model,
        "tokenizer": tokenizer,
        "kv_cache": cache,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0,
        "top_k": 0,
        "top_p": 1,
    }

    for _ in range(args.warmup_runs):
        generate_with_stats(**generation_args)

    samples = []
    outputs = []
    for _ in range(args.runs):
        text, stats = generate_with_stats(**generation_args)
        outputs.append(text)
        samples.append(stats)
    if any(text != outputs[0] for text in outputs[1:]):
        raise RuntimeError("greedy benchmark output changed between runs")

    fields = ("decode_tok_s", "ttft_ms", "prefill_ms", "peak_vram_gb")
    medians = {
        field: round(statistics.median(sample[field] for sample in samples), 2)
        for field in fields
    }
    print(
        json.dumps(
            {
                "model": args.model,
                "warmup_runs": args.warmup_runs,
                "measured_runs": args.runs,
                "prompt_tokens": prompt_tokens,
                "new_tokens": samples[0]["new_tokens"],
                "median": medians,
                "samples": samples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
