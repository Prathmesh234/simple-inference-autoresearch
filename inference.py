"""Run text generation through any registered model plugin."""

from __future__ import annotations

import argparse

import torch

import env_loader  # noqa: F401
from generate import generate_with_stats
from model.registry import build_cache, load_model
from tokenizer import Tokenizer
from utilities import print_metrics_table


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

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

    generated, stats = generate_with_stats(
        args.prompt,
        loaded.model,
        tokenizer,
        cache,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        on_token=lambda text: print(text, end="", flush=True),
    )
    if generated:
        print()
    print_metrics_table(stats, loaded.plugin.family)


if __name__ == "__main__":
    main()
