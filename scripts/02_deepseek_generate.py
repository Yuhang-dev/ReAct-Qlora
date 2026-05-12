"""Generate ReAct training data using DeepSeek V4 Flash as teacher model."""

import argparse
import json
import os
import random
import time
from pathlib import Path

from openai import OpenAI

# ── DeepSeek client ──────────────────────────────────────
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", "sk-placeholder"),
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

# ── System prompt for the teacher ────────────────────────
TEACHER_SYSTEM = """You are a data generation assistant. Generate diverse, high-quality ReAct (Reasoning-Action) training trajectories for an AI agent that uses tools.

## Available Tools
1. get_weather(city, date="today") -> {city, temperature, condition, humidity}
2. search_web(query, num_results=5) -> {results: [{title, snippet, url}]}
3. get_stock_price(symbol) -> {symbol, price, change_pct}
4. calculate(expression) -> {result}
5. get_time(city) -> {city, time, timezone}

## Output Format (STRICT JSON)
You must output exactly ONE JSON object per response, with this structure:

{
  "messages": [
    {"role": "system", "content": "<system prompt describing tools and ReAct format>"},
    {"role": "user", "content": "<user's question>"},
    {"role": "assistant", "content": "Thought: <reasoning>\\nAction: <tool_call>\\n"},
    {"role": "tool", "content": "<JSON tool response>"},
    {"role": "assistant", "content": "Thought: <reasoning>\\nAction: <next_tool>\\n"},
    {"role": "tool", "content": "<JSON tool response>"},
    {"role": "assistant", "content": "Thought: <final reasoning>\\nFinal Answer: <complete answer>\\n"}
  ]
}

## Rules
- Every assistant message MUST start with "Thought:" explaining the reasoning.
- Action format: Action: tool_name(param1="value1", param2="value2")
- Include at least 1-3 tool calls per trajectory.
- For ~15% of trajectories, include a tool error ({"error": "..."}) and show recovery.
- Tool responses must be valid JSON strings.
- Make questions diverse: weather, stocks, search, math, time, multi-step combinations.
- Vary difficulty: simple single-tool queries to complex multi-tool chains.
- Questions can be in Chinese or English.
- The Final Answer must fully address the user's question.
- Do NOT include the system prompt in messages for every sample — only when specified."""

# ── Prompt templates for diversity ───────────────────────
PROMPT_TEMPLATES = [
    # Weather
    "Generate a ReAct trajectory where the user asks about weather in a city. Include 1 tool call.",
    "Generate a ReAct trajectory where the user compares weather between two cities. Include 2 get_weather calls.",
    "Generate a ReAct trajectory where the user asks about weather for outdoor activity planning.",
    # Stock
    "Generate a ReAct trajectory where the user asks for a stock price. Use get_stock_price.",
    "Generate a ReAct trajectory where the user compares two stocks. Include 2 get_stock_price calls.",
    "Generate a ReAct trajectory where the user asks about their stock portfolio value. Use get_stock_price then calculate.",
    # Search
    "Generate a ReAct trajectory where the user searches for technical information. Use search_web.",
    "Generate a ReAct trajectory where the user searches for recent news about a topic.",
    "Generate a ReAct trajectory where the user wants to learn about a historical event or person.",
    # Calculate
    "Generate a ReAct trajectory where the user asks a math calculation. Use calculate.",
    "Generate a ReAct trajectory where the user asks a word problem requiring math.",
    # Time
    "Generate a ReAct trajectory where the user asks for current time in a city. Use get_time.",
    "Generate a ReAct trajectory where the user asks about time difference between two cities.",
    # Multi-tool
    "Generate a ReAct trajectory using get_stock_price AND search_web together.",
    "Generate a ReAct trajectory using get_weather AND get_time together.",
    "Generate a ReAct trajectory using calculate AND search_web together.",
    "Generate a ReAct trajectory using get_stock_price AND get_weather together.",
    "Generate a ReAct trajectory with 3 different tool calls in sequence.",
    # Error recovery
    "Generate a ReAct trajectory where the first tool call returns an error or empty result, and the agent recovers by trying a different approach.",
    "Generate a ReAct trajectory where a search returns no results, and the agent adjusts the query.",
    "Generate a ReAct trajectory where get_weather fails for an invalid city, and the agent handles it gracefully.",
]


def generate_one(max_retries: int = 3) -> dict | None:
    """Generate a single ReAct trajectory via DeepSeek."""
    prompt = random.choice(PROMPT_TEMPLATES)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": TEACHER_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )

            text = response.choices[0].message.content.strip()
            data = json.loads(text)

            messages = data.get("messages", [])
            if not messages:
                continue

            # Validate basic structure
            has_user = any(m["role"] == "user" for m in messages)
            has_tool = any(m["role"] == "tool" for m in messages)
            has_final = any("Final Answer:" in m.get("content", "") for m in messages if m["role"] == "assistant")

            if has_user and has_tool and has_final:
                return {"messages": messages}
            else:
                if attempt < max_retries - 1:
                    time.sleep(0.5)

        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            err = str(e)
            if "rate" in err.lower():
                time.sleep(3 * (attempt + 1))
            elif attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"  FAIL after {max_retries} retries: {err[:100]}")

    return None


def main():
    parser = argparse.ArgumentParser(description="Generate ReAct data with DeepSeek V4 Flash")
    parser.add_argument("--count", type=int, default=2000, help="Number of trajectories to generate")
    parser.add_argument("--output_dir", type=str, default="data/synthetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=20)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} ReAct trajectories with DeepSeek V4 Flash...")
    print(f"Prompt templates: {len(PROMPT_TEMPLATES)}")
    print()

    all_samples = []
    batch_num = 0
    generated = 0

    while generated < args.count:
        batch = []
        batch_target = min(args.batch_size, args.count - generated)
        print(f"Batch {batch_num + 1}: ", end="", flush=True)

        for i in range(batch_target):
            sample = generate_one()
            if sample:
                batch.append(sample)
            if (i + 1) % 5 == 0:
                print(".", end="", flush=True)

        if batch:
            batch_path = output_dir / f"deepseek_batch_{batch_num:03d}.jsonl"
            with open(batch_path, "w", encoding="utf-8") as f:
                for s in batch:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            all_samples.extend(batch)
            print(f" {len(batch)} ok -> {batch_path.name}")
        else:
            print(" 0 samples (all failed)")

        generated += batch_target
        batch_num += 1

        if generated < args.count:
            time.sleep(1)  # Rate limit spacing

    print(f"\nTotal: {len(all_samples)}/{args.count} generated")

    # Save combined
    combined_path = output_dir / "deepseek_combined.jsonl"
    with open(combined_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Combined: {combined_path}")
    print("Done.")


if __name__ == "__main__":
    main()
