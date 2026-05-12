"""Negative sample construction for ReAct error recovery training."""

import copy
import json
import random
from typing import Any


ERROR_TEMPLATES = [
    {"error": "timeout", "message": "Request timed out after 30 seconds."},
    {"error": "connection_error", "message": "Failed to connect to the service."},
    {"error": "rate_limited", "message": "Too many requests. Please retry later."},
    {"error": "invalid_parameter", "message": "One or more parameters are invalid."},
    {"error": "internal_error", "message": "Internal server error. Please try again."},
]

EMPTY_RESULT_TEMPLATES = [
    {"result": [], "count": 0, "message": "No results found."},
    {"result": {}, "count": 0, "message": "Query returned empty."},
    {"data": None, "message": "No matching records."},
]


def inject_tool_error(messages: list[dict], error_index: int | None = None) -> list[dict]:
    """Replace one tool response with an error to create a negative sample."""
    messages = copy.deepcopy(messages)
    tool_indices = [i for i, m in enumerate(messages) if m["role"] == "tool"]

    if not tool_indices:
        return messages

    if error_index is None:
        error_index = random.choice(tool_indices)

    error_template = random.choice(ERROR_TEMPLATES)
    messages[error_index]["content"] = json.dumps(error_template, ensure_ascii=False)
    return messages


def inject_empty_result(messages: list[dict], tool_index: int | None = None) -> list[dict]:
    """Replace one tool response with an empty result."""
    messages = copy.deepcopy(messages)
    tool_indices = [i for i, m in enumerate(messages) if m["role"] == "tool"]

    if not tool_indices:
        return messages

    if tool_index is None:
        tool_index = random.choice(tool_indices)

    empty_template = random.choice(EMPTY_RESULT_TEMPLATES)
    messages[tool_index]["content"] = json.dumps(empty_template, ensure_ascii=False)
    return messages


def generate_negative_samples(
    messages: list[dict],
    count: int = 2,
) -> list[dict]:
    """Generate negative variants of a single ReAct trajectory."""
    variants = []
    tool_indices = [i for i, m in enumerate(messages) if m["role"] == "tool"]

    if not tool_indices:
        return variants

    for _ in range(min(count, len(tool_indices) * 2)):
        variant_type = random.choice(["error", "empty"])
        if variant_type == "error":
            variant = inject_tool_error(messages)
        else:
            variant = inject_empty_result(messages)

        # Mark as negative sample
        variant_meta = copy.deepcopy(variant)
        variants.append(variant_meta)

    return variants


def build_dataset_with_negatives(
    samples: list[dict],
    negative_ratio: float = 0.1,
) -> list[dict]:
    """Take a list of normal samples and return a mixed dataset with negatives.

    Args:
        samples: List of {"messages": [...]} dicts.
        negative_ratio: Fraction of negative samples in the final dataset.

    Returns:
        Mixed list with both positive and negative samples.
    """
    n_negative = int(len(samples) * negative_ratio)
    indices = random.sample(range(len(samples)), min(n_negative, len(samples)))

    negative_samples = []
    for idx in indices:
        variants = generate_negative_samples(samples[idx]["messages"], count=1)
        for v in variants:
            negative_samples.append({"messages": v, "is_negative": True})

    # Mark positives
    result = [{"messages": s["messages"], "is_negative": False} for s in samples]
    result.extend(negative_samples)
    random.shuffle(result)
    return result
