"""
Benchmark prompts sourced from open-source Hugging Face datasets.

The engine's primary metric is decode throughput (tok/s), with TTFT and peak
VRAM as guardrails. Those numbers only mean something against a realistic prompt
mix, so instead of hand-writing a few strings we draw prompts from public HF
datasets that each stress a different part of the engine:

    DATASETS (curated mix)
    ----------------------
    chat        HuggingFaceH4/ultrachat_200k
                Cleaned multi-turn instruction conversations. Ungated and
                streamable, with a realistic heavy-tailed prompt-length
                distribution and natural shared prefixes across turns.

    chat_real   lmsys/lmsys-chat-1m
                One million real user<->LLM conversations from LMSYS Chatbot Arena.
                Production-like traffic distribution (short questions, long pastes,
                code, multilingual). Good for mixed-batch / continuous-batching.

    long_ctx    abisee/cnn_dailymail (3.0.0)
                News articles for summarization. Long prompts (1-2k+ tokens) =
                prefill-heavy / long-context regime. Stresses the attention kernel
                and KV-cache sizing more than decode.

    summarize   EdinburghNLP/xsum
                BBC articles + one-line summaries. Another long-prefill source with
                a different length profile than cnn_dailymail.

    instruct    tatsu-lab/alpaca
                Short instruction-following prompts. Low prefill, decode-bound =
                isolates per-token decode throughput, the metric we optimize.

    code        openai/openai_humaneval
                Python function-completion prompts. Code-shaped tokens and a
                distinct vocabulary distribution from natural-language chat.

Usage
-----
    from benchmarks.prompts import load_prompts, DATASETS

    prompts = load_prompts("chat", n=64, max_chars=4000)   # list[str]
    prompts = load_prompts("long_ctx", n=16)

`load_prompts` streams from the HF hub (no full download). Set HF_TOKEN for
gated/large datasets. If streaming fails (network down, dataset missing), it
raises — there is no offline substitute, so benchmark numbers are always real.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    hf_id: str          # Hugging Face dataset id
    config: str | None  # dataset config / subset name, if any
    split: str          # which split to stream
    field: str          # which text column holds the prompt
    purpose: str        # what part of the engine this stresses


# Curated registry: name -> how to pull prompts from that HF dataset.
DATASETS: dict[str, DatasetSpec] = {
    "chat": DatasetSpec(
        hf_id="HuggingFaceH4/ultrachat_200k",
        config=None,
        split="train_sft",
        field="messages",  # list of {role, content}; first user turn extracted
        purpose="realistic mixed-length chat / shared prefixes (standard serving bench)",
    ),
    "chat_real": DatasetSpec(
        hf_id="lmsys/lmsys-chat-1m",
        config=None,
        split="train",
        field="conversation",  # list of {role, content}; first user turn extracted
        purpose="production-like real traffic distribution (continuous batching)",
    ),
    "long_ctx": DatasetSpec(
        hf_id="abisee/cnn_dailymail",
        config="3.0.0",
        split="validation",
        field="article",
        purpose="long prefill / long-context (attention + KV-cache pressure)",
    ),
    "summarize": DatasetSpec(
        hf_id="EdinburghNLP/xsum",
        config=None,
        split="validation",
        field="document",
        purpose="long prefill, different length profile",
    ),
    "instruct": DatasetSpec(
        hf_id="tatsu-lab/alpaca",
        config=None,
        split="train",
        field="instruction",
        purpose="short prompt, decode-bound (isolates decode tok/s)",
    ),
    "code": DatasetSpec(
        hf_id="openai/openai_humaneval",
        config=None,
        split="test",
        field="prompt",
        purpose="code-shaped tokens, distinct vocab distribution",
    ),
}


def _extract_text(spec: DatasetSpec, row: dict) -> str | None:
    """Pull a single prompt string out of one dataset row."""
    val = row.get(spec.field)
    if val is None:
        return None
    # Chat datasets store a list of turns; grab the first human/user message.
    if isinstance(val, list):
        for turn in val:
            if isinstance(turn, dict):
                role = turn.get("from") or turn.get("role")
                content = turn.get("value") or turn.get("content")
                if content and role in (None, "human", "user"):
                    return str(content)
        # fall back to the first turn's text
        first = val[0]
        if isinstance(first, dict):
            return str(first.get("value") or first.get("content") or "")
        return str(first)
    return str(val)


def load_prompts(
    name: str,
    n: int = 64,
    *,
    min_chars: int = 1,
    max_chars: int | None = None,
    seed: int = 0,
) -> list[str]:
    """Return up to `n` prompt strings from the named HF dataset.

    Streams from the hub (no full download). Raises if streaming fails so that
    benchmark numbers are never silently computed against substitute prompts.

    Args:
        name:      key in DATASETS (e.g. "chat", "long_ctx", "instruct").
        n:         number of prompts to return.
        min_chars: skip prompts shorter than this (drops empty rows).
        max_chars: truncate prompts longer than this (keep prefill bounded).
        seed:      shuffle seed for reproducible sampling.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown prompt set '{name}'. choices: {', '.join(DATASETS)}")
    spec = DATASETS[name]

    from datasets import load_dataset

    ds = load_dataset(
        spec.hf_id,
        spec.config,
        split=spec.split,
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=1000)

    out: list[str] = []
    for row in ds:
        text = _extract_text(spec, row)
        if not text:
            continue
        text = text.strip()
        if len(text) < min_chars:
            continue
        if max_chars is not None:
            text = text[:max_chars]
        out.append(text)
        if len(out) >= n:
            break

    if not out:
        raise RuntimeError(f"no prompts extracted from '{spec.hf_id}' (field '{spec.field}')")
    return out


if __name__ == "__main__":
    # Quick smoke test: print a couple of prompts from each set.
    for key, spec in DATASETS.items():
        print(f"\n=== {key}  ({spec.hf_id})  — {spec.purpose} ===")
        for p in load_prompts(key, n=2, max_chars=200):
            preview = p.replace("\n", " ")
            print(f"  - {preview[:160]}")
