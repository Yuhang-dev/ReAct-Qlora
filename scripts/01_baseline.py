"""Run baseline evaluation on base Qwen2.5-7B-Instruct (zero-shot & few-shot)."""

import json
import time
from pathlib import Path

import torch

from src.data.react_format import (
    build_system_prompt,
    parse_action,
    parse_final_answer,
    validate_trajectory,
)
from src.eval.metrics import compute_dataset_metrics, evaluate_trajectory

# ── Tool definitions ──────────────────────────────────────
TOOL_DEFS = """
- get_weather(city: str, date: str = "today") -> dict: Get current weather for a city. Returns temperature, condition, humidity.
- search_web(query: str, num_results: int = 5) -> list: Search the web for information. Returns list of {title, snippet, url}.
- get_stock_price(symbol: str) -> dict: Get current stock price and change. Returns {symbol, price, change_pct}.
- calculate(expression: str) -> float: Evaluate a mathematical expression.
- get_time(city: str) -> str: Get current local time for a city.
"""

FEW_SHOT_EXAMPLES = """
Example 1:
User: 北京今天天气怎么样？
Assistant: Thought: 我需要查询北京当前的天气数据。
Action: get_weather(city="北京")

Observation: {"city": "北京", "temperature": 22, "condition": "晴", "humidity": 45}
Assistant: Thought: 天气数据已获取，北京今天晴朗，温度22°C。可以回答用户了。
Final Answer: 北京今天的天气是晴天，温度22°C，湿度45%，非常适合户外活动。

Example 2:
User: 帮我查一下苹果公司的股价。
Assistant: Thought: 用户想查询苹果公司的股价，苹果的股票代码是AAPL。
Action: get_stock_price(symbol="AAPL")

Observation: {"symbol": "AAPL", "price": 185.50, "change_pct": 2.3}
Assistant: Thought: 已获取苹果股价数据，可以回复用户了。
Final Answer: 苹果(AAPL)当前股价为 $185.50，较前一日上涨 2.3%。

Now answer the following user question using the same Thought-Action-Observation format:
"""


def load_model():
    import unsloth
    from unsloth import FastLanguageModel

    print("Loading Qwen2.5-7B-Instruct (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    user_query: str,
    mode: str = "zero-shot",
    temperature: float = 0.1,
    max_new_tokens: int = 1024,
) -> str:
    """Run inference on the model, return generated text."""
    system_prompt = build_system_prompt(TOOL_DEFS)

    if mode == "zero-shot":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]
    elif mode == "few-shot":
        # Inject few-shot examples into system prompt
        augmented_system = system_prompt + "\n\n" + FEW_SHOT_EXAMPLES
        messages = [
            {"role": "system", "content": augmented_system},
            {"role": "user", "content": user_query},
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    )
    return generated.strip()


def build_pred_messages(user_query: str, generated: str, mode: str) -> list[dict]:
    """Construct full ChatML trajectory from user query + generated text."""
    system_prompt = build_system_prompt(TOOL_DEFS)
    if mode == "few-shot":
        system_prompt += "\n\n" + FEW_SHOT_EXAMPLES

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": generated},
    ]


def main():
    model, tokenizer = load_model()

    # Load test cases
    test_data = json.loads(Path("data/baseline_test.json").read_text(encoding="utf-8"))
    print(f"Loaded {len(test_data)} test cases\n")

    results = {}
    for mode in ["zero-shot", "few-shot"]:
        print(f"{'=' * 60}")
        print(f"Mode: {mode}")
        print(f"{'=' * 60}")

        mode_results = []
        total_time = 0

        for i, test in enumerate(test_data):
            t_start = time.time()

            generated = run_inference(model, tokenizer, test["user_query"], mode=mode)
            elapsed = time.time() - t_start
            total_time += elapsed

            pred_messages = build_pred_messages(test["user_query"], generated, mode)

            # Evaluate
            eval_result = evaluate_trajectory(pred_messages)

            # Check if expected tools were called
            tools_called = []
            for msg in pred_messages:
                if msg["role"] == "assistant" and "Action:" in msg["content"]:
                    action = parse_action(msg["content"])
                    if action:
                        tools_called.append(action[0])

            expected = set(test["expected_tools"])
            called = set(tools_called)
            tool_hit = len(expected & called) / len(expected) if expected else 1.0

            mode_results.append(eval_result)

            # Print per-sample
            status = "OK" if eval_result.format_valid else "FAIL"
            print(f"[{status}] {test['id']} ({test['category']})")
            print(f"    Expected tools: {expected}, Got: {called}")
            print(f"    Format valid: {eval_result.format_valid}")
            if not eval_result.format_valid:
                detail = eval_result.details.get("fmt_error", "")
                print(f"    Format error: {detail[:80]}...")
            print(f"    Generated: {generated[:120]}...")
            print()

        # Summarize mode
        metrics = compute_dataset_metrics(mode_results)
        metrics["avg_time_s"] = total_time / len(test_data)
        metrics["mode"] = mode

        print(f"\n--- {mode} Summary ---")
        for k, v in metrics.items():
            if isinstance(v, float) and k != "mode":
                print(f"  {k}: {v:.2%}" if v <= 1 else f"  {k}: {v:.2f}")

        results[mode] = metrics

        # Free memory between modes
        torch.cuda.empty_cache()

    # ── Save results ──────────────────────────────────────
    output_path = Path("outputs/baseline_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {output_path}")

    # ── Comparison table ──────────────────────────────────
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<30} {'Zero-Shot':<15} {'Few-Shot':<15}")
    print("-" * 60)
    keys = ["format_compliance", "tool_name_accuracy", "tool_args_accuracy",
            "error_recovery_rate", "success_rate"]
    for key in keys:
        zs = results.get("zero-shot", {}).get(key, 0)
        fs = results.get("few-shot", {}).get(key, 0)
        print(f"{key:<30} {zs:>14.1%} {fs:>14.1%}")


if __name__ == "__main__":
    main()
