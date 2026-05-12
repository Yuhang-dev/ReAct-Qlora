"""Evaluate fine-tuned ReAct model on test set: 5 metrics + LLM-Judge."""

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import torch

from src.data.react_format import parse_action, parse_final_answer

# ── Load model with LoRA ──────────────────────────────────

def load_lora_model(lora_path: str):
    import unsloth
    from unsloth import FastLanguageModel

    print(f"Loading base model + LoRA: {lora_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, lora_path)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def run_inference(model, tokenizer, messages: list[dict], max_new: int = 512) -> str:
    """Generate model response from ChatML messages."""
    # Truncate: use user message + system as prompt, stop before any Final Answer
    prompt_msgs = []
    for m in messages:
        prompt_msgs.append(m)
        if m["role"] == "user":
            break  # stop after first user message (single-turn eval)

    prompt = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


# ── Metrics ───────────────────────────────────────────────

def check_observation_hallucination(generated: str) -> bool:
    """True if model generated a fake Observation (hallucination)."""
    # Model should output Action: ... and STOP, waiting for real tool response.
    # If it generates "Observation:" on its own, that's hallucination.
    return "Observation:" in generated


def check_tool_accuracy(generated: str, expected_tools: list[str]) -> dict:
    """Check if the right tools were called."""
    actions = re.findall(r'Action:\s*(\w+)\(', generated)
    got = set(actions)
    expected = set(expected_tools)
    return {
        "expected": expected,
        "got": got,
        "name_match": len(expected & got) / len(expected) if expected else 1.0,
        "extra_calls": got - expected,
        "missing_calls": expected - got,
    }


def check_param_accuracy(generated: str) -> float:
    """Check if action parameters are well-formed (not empty, has key=value)."""
    actions = re.findall(r'Action:\s*(\w+)\(([^)]*)\)', generated)
    if not actions:
        return 1.0  # no tool calls needed

    well_formed = 0
    for name, params_str in actions:
        params_str = params_str.strip()
        if params_str and re.search(r'\w+\s*=\s*"[^"]*"', params_str):
            well_formed += 1

    return well_formed / len(actions)


def check_format_compliance(generated: str) -> dict:
    """Check ReAct format: Thought -> Action -> Final Answer."""
    has_thought = "Thought:" in generated
    has_action = "Action:" in generated
    has_final = "Final Answer:" in generated
    return {
        "valid": has_thought and (has_action or has_final),
        "has_thought": has_thought,
        "has_action": has_action,
        "has_final": has_final,
    }


def check_error_recovery(messages: list[dict], generated: str) -> bool:
    """Check if this is an error sample and model recovered."""
    # Check if any tool response in the reference contains an error
    has_error = False
    for m in messages:
        if m["role"] == "tool":
            try:
                resp = json.loads(m["content"])
                if "error" in resp:
                    has_error = True
            except Exception:
                pass

    if not has_error:
        return None  # not an error sample

    # Recovery: model doesn't hallucinate Observation AND produces Final Answer
    no_hallucination = "Observation:" not in generated
    has_final = "Final Answer:" in generated
    return no_hallucination and has_final


# ── Main eval ────────────────────────────────────────────

def main():
    lora_path = "outputs/react-qwen2.5-7b-qlora_20260508_1810/checkpoint-1000"
    test_file = "data/processed/test.jsonl"

    # Load test set
    with open(test_file, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line) for line in f if line.strip()]
    print(f"Test samples: {len(test_samples)}")

    # Load model
    model, tokenizer = load_lora_model(lora_path)

    # ── Inference ─────────────────────────────────────────
    results = []
    metrics = defaultdict(list)
    t_start = time.time()

    for i, sample in enumerate(test_samples):
        messages = sample["messages"]
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")

        generated = run_inference(model, tokenizer, messages)

        # Compute metrics
        obs_hall = check_observation_hallucination(generated)
        fmt = check_format_compliance(generated)
        param_acc = check_param_accuracy(generated)

        # Extract expected tools from reference
        ref_tools = set()
        for m in messages:
            if m["role"] == "assistant":
                actions = re.findall(r'Action:\s*(\w+)\(', m["content"])
                ref_tools.update(actions)
        tool_acc = check_tool_accuracy(generated, list(ref_tools))
        err_rec = check_error_recovery(messages, generated)

        results.append({
            "id": i,
            "user_query": user_msg[:80],
            "generated": generated[:300],
            "obs_hallucination": obs_hall,
            "format_valid": fmt["valid"],
            "tool_name_match": tool_acc["name_match"],
            "param_accuracy": param_acc,
            "error_recovery": err_rec,
        })

        metrics["obs_hall"].append(obs_hall)
        metrics["format"].append(fmt["valid"])
        metrics["tool_name"].append(tool_acc["name_match"])
        metrics["param"].append(param_acc)
        if err_rec is not None:
            metrics["error_recovery"].append(err_rec)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  {i+1}/{len(test_samples)} ({elapsed:.0f}s)")

    total_time = time.time() - t_start
    print(f"\nInference done: {len(test_samples)} samples in {total_time:.0f}s")

    # ── Aggregate metrics ─────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    obs_hall_rate = sum(metrics["obs_hall"]) / len(metrics["obs_hall"])
    fmt_rate = sum(metrics["format"]) / len(metrics["format"])
    tool_rate = sum(metrics["tool_name"]) / len(metrics["tool_name"])
    param_rate = sum(metrics["param"]) / len(metrics["param"])
    err_rate = sum(metrics["error_recovery"]) / len(metrics["error_recovery"]) if metrics["error_recovery"] else 0

    print(f"\n{'Metric':<35} {'Value':>10}")
    print("-" * 47)
    print(f"{'Observation Hallucination Rate ↓':<35} {obs_hall_rate:>9.1%}")
    print(f"{'Format Compliance ↑':<35} {fmt_rate:>9.1%}")
    print(f"{'Tool Name Accuracy ↑':<35} {tool_rate:>9.1%}")
    print(f"{'Tool Parameter Accuracy ↑':<35} {param_rate:>9.1%}")
    print(f"{'Error Recovery Rate ↑':<35} {err_rate:>9.1%}")

    # ── Show examples ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUTS")
    print("=" * 60)
    for r in results[:3]:
        print(f"\n[{r['id']}] Q: {r['user_query']}")
        print(f"    Generated: {r['generated'][:200]}")
        print(f"    Obs Hall: {r['obs_hallucination']} | Format: {r['format_valid']} | Tool: {r['tool_name_match']:.0%}")

    # ── Save detailed results ─────────────────────────────
    out_path = Path("outputs/eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": lora_path,
        "test_samples": len(test_samples),
        "inference_time_s": total_time,
        "metrics": {
            "obs_hallucination_rate": obs_hall_rate,
            "format_compliance": fmt_rate,
            "tool_name_accuracy": tool_rate,
            "tool_param_accuracy": param_rate,
            "error_recovery_rate": err_rate,
        },
        "per_sample": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
