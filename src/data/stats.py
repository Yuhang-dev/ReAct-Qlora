"""Token-length distribution analysis for ReAct trajectories.

Uses direct tokenization (not Qwen's chat template which expects native tool_call format).
"""

import json
from pathlib import Path

import numpy as np


def _messages_to_text(messages: list[dict]) -> str:
    """Convert ChatML messages to a plain text string for tokenization."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(content)
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        elif role == "tool":
            parts.append(f"Tool Response: {content}")
    return "\n\n".join(parts)


def _count_tokens(messages: list[dict], tokenizer) -> int:
    """Count tokens for a message list using direct tokenization."""
    text = _messages_to_text(messages)
    return len(tokenizer.encode(text))


def compute_token_stats(
    data_path: str | Path,
    tokenizer,
    percentiles: list[float] | None = None,
) -> dict:
    if percentiles is None:
        percentiles = [50, 90, 95, 99, 99.9]

    lengths = []
    data_path = Path(data_path)

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            messages = sample.get("messages", sample)
            lengths.append(_count_tokens(messages, tokenizer))

    lengths = np.array(lengths)
    stats = {
        "count": int(len(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
    }

    for p in percentiles:
        stats[f"p_{int(p)}"] = float(np.percentile(lengths, p))

    for cutoff in [2048, 4096, 8192]:
        stats[f"truncation_rate_at_{cutoff}"] = float(np.mean(lengths > cutoff))

    return stats


def filter_by_length(
    data_path: str | Path,
    tokenizer,
    max_length: int,
    output_path: str | Path,
) -> tuple[int, int]:
    data_path = Path(data_path)
    output_path = Path(output_path)
    kept = 0
    total = 0

    with open(data_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            sample = json.loads(line)
            messages = sample.get("messages", sample)
            if _count_tokens(messages, tokenizer) <= max_length:
                fout.write(line)
                kept += 1

    return kept, total


def generate_length_report(
    data_path: str | Path,
    tokenizer,
    output_path: str | Path,
) -> None:
    stats = compute_token_stats(data_path, tokenizer)
    if stats["count"] == 0:
        print("WARNING: No samples found.")
        return

    report = (
        f"Token Length Statistics\n{'=' * 50}\n"
        f"Samples: {stats['count']}\n"
        f"Mean:    {stats['mean']:.0f} tokens\n"
        f"Median:  {stats['median']:.0f} tokens\n"
        f"Min:     {stats['min']} tokens\n"
        f"Max:     {stats['max']} tokens\n\n"
        f"Percentiles:\n"
    )
    for k, v in stats.items():
        if k.startswith("p_"):
            report += f"  {k}: {v:.0f}\n"
    report += "\nTruncation rates:\n"
    for k, v in stats.items():
        if k.startswith("truncation"):
            report += f"  {k}: {v:.1%}\n"

    Path(output_path).write_text(report, encoding="utf-8")
    print(report)
