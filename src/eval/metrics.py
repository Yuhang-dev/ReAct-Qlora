"""Automatic evaluation metrics for ReAct trajectories."""

import json
import re
from dataclasses import dataclass, field

from src.data.react_format import parse_action, parse_final_answer


@dataclass
class EvalResult:
    success: bool
    tool_name_correct: bool
    tool_args_correct: bool
    format_valid: bool
    has_error_recovery: bool = False
    details: dict = field(default_factory=dict)


def check_format_compliance(pred_messages: list[dict]) -> tuple[bool, str]:
    """Check if predicted messages follow ReAct format rules."""
    if not pred_messages:
        return False, "Empty prediction"

    assistant_msgs = [m for m in pred_messages if m["role"] == "assistant"]
    if not assistant_msgs:
        return False, "No assistant messages"

    thought_found = False
    for msg in assistant_msgs:
        content = msg["content"]
        if "Thought:" not in content:
            return False, f"Missing Thought in: {content[:60]}..."
        thought_found = True

        if "Action:" in content:
            action = parse_action(content)
            if action is None:
                return False, f"Malformed Action in: {content[:80]}..."
        elif "Final Answer:" not in content:
            return False, f"Neither Action nor Final Answer: {content[:60]}..."

    if not thought_found:
        return False, "No Thought found in any assistant message"

    return True, ""


def compute_tool_accuracy(
    pred_messages: list[dict],
    ref_messages: list[dict],
) -> tuple[bool, bool]:
    """Compute tool name and argument accuracy by comparing with reference."""
    pred_actions = []
    for m in pred_messages:
        if m["role"] == "assistant" and "Action:" in m["content"]:
            action = parse_action(m["content"])
            if action:
                pred_actions.append(action)

    ref_actions = []
    for m in ref_messages:
        if m["role"] == "assistant" and "Action:" in m["content"]:
            action = parse_action(m["content"])
            if action:
                ref_actions.append(action)

    if not ref_actions:
        return True, True  # no tool calls expected

    if not pred_actions:
        return False, False

    name_hits = 0
    arg_hits = 0
    for pa, ra in zip(pred_actions, ref_actions):
        if pa[0] == ra[0]:
            name_hits += 1
            if pa[1] == ra[1]:
                arg_hits += 1

    name_acc = name_hits / max(len(pred_actions), len(ref_actions))
    arg_acc = arg_hits / max(len(pred_actions), len(ref_actions))
    return name_acc >= 0.8, arg_acc >= 0.8


def detect_error_recovery(pred_messages: list[dict]) -> bool:
    """Check if the model recovered from a tool error."""
    has_error = False
    for m in pred_messages:
        if m["role"] == "tool":
            try:
                resp = json.loads(m["content"])
                if "error" in resp:
                    has_error = True
            except json.JSONDecodeError:
                pass

    has_final = any(
        "Final Answer:" in m["content"]
        for m in pred_messages
        if m["role"] == "assistant"
    )

    return has_error and has_final


def evaluate_trajectory(
    pred_messages: list[dict],
    ref_messages: list[dict] | None = None,
) -> EvalResult:
    """Full evaluation of a single predicted trajectory."""
    format_valid, fmt_error = check_format_compliance(pred_messages)

    tool_name_ok = True
    tool_args_ok = True
    if ref_messages is not None:
        tool_name_ok, tool_args_ok = compute_tool_accuracy(pred_messages, ref_messages)

    error_recovery = detect_error_recovery(pred_messages)

    return EvalResult(
        success=False,  # set by LLM judge or external logic
        tool_name_correct=tool_name_ok,
        tool_args_correct=tool_args_ok,
        format_valid=format_valid,
        has_error_recovery=error_recovery,
        details={"fmt_error": fmt_error if not format_valid else ""},
    )


def compute_dataset_metrics(results: list[EvalResult]) -> dict:
    """Aggregate EvalResult list into summary metrics."""
    n = len(results)
    if n == 0:
        return {}

    return {
        "total_samples": n,
        "success_rate": sum(1 for r in results if r.success) / n,
        "format_compliance": sum(1 for r in results if r.format_valid) / n,
        "tool_name_accuracy": sum(1 for r in results if r.tool_name_correct) / n,
        "tool_args_accuracy": sum(1 for r in results if r.tool_args_correct) / n,
        "error_recovery_rate": sum(1 for r in results if r.has_error_recovery) / n,
    }
