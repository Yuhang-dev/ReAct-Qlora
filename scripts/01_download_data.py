"""Download and extract ReAct training data from open-source datasets.

Sources:
  - ToolBench: Real API-calling trajectories (REST, SQL, Bash)
  - HotpotQA: Multi-hop reasoning chains
"""

import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def download_toolbench(output_dir: str, max_samples: int = 4000) -> list[dict]:
    """Download and format ToolBench data into ReAct ChatML format."""
    print("=" * 50)
    print("Downloading ToolBench...")
    print("=" * 50)

    samples = []
    try:
        # Try to load ToolBench subsets
        from datasets import get_dataset_config_names

        configs = get_dataset_config_names("ToolBench/ToolBench")
        print(f"Available configs: {configs}")

        for config in configs[:3]:
            try:
                ds = load_dataset("ToolBench/ToolBench", config, split="train", streaming=True)
                for i, item in enumerate(ds):
                    if i >= max_samples // len(configs[:3]):
                        break
                    formatted = format_toolbench_item(item)
                    if formatted:
                        samples.append(formatted)
                print(f"  {config}: loaded {min(i + 1, max_samples // len(configs[:3]))}")
            except Exception as e:
                print(f"  {config}: SKIP — {e}")
    except Exception as e:
        print(f"ToolBench download failed: {e}")
        print("Will rely on synthetic data instead.")

    print(f"ToolBench: {len(samples)} samples")
    return samples


def format_toolbench_item(item: dict) -> dict | None:
    """Convert a ToolBench item to ReAct ChatML format."""
    try:
        messages = []
        # Extract conversation turns
        conversation = item.get("conversation", item.get("conversations", []))

        for turn in conversation:
            role = turn.get("role", turn.get("from", ""))
            content = turn.get("content", turn.get("value", ""))

            if role in ("user", "human"):
                messages.append({"role": "user", "content": content})
            elif role in ("assistant", "gpt", "model"):
                # Parse thought/action from toolbench format
                if "Thought:" not in content and "Action:" not in content:
                    # Wrap missing format
                    thought = turn.get("thought", turn.get("thinking", ""))
                    action = turn.get("action", turn.get("function_call", ""))
                    if thought or action:
                        content = f"Thought: {thought}\nAction: {action}\n" if action else f"Thought: {thought}\n"
                messages.append({"role": "assistant", "content": content})
            elif role in ("tool", "observation", "function"):
                obs_content = turn.get("content", turn.get("value", ""))
                if isinstance(obs_content, dict):
                    obs_content = json.dumps(obs_content, ensure_ascii=False)
                messages.append({"role": "tool", "content": obs_content})

        if len(messages) >= 2 and any("Action:" in m.get("content", "") for m in messages):
            return {"messages": messages}
    except Exception:
        pass
    return None


def download_hotpotqa(output_dir: str, max_samples: int = 2000) -> list[dict]:
    """Download HotpotQA and convert multi-hop QA into ReAct format."""
    print("\n" + "=" * 50)
    print("Downloading HotpotQA...")
    print("=" * 50)

    samples = []
    try:
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", streaming=True)

        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            formatted = format_hotpotqa_item(item)
            if formatted:
                samples.append(formatted)

            if (i + 1) % 200 == 0:
                print(f"  Processed {i + 1}/{max_samples}...")

    except Exception as e:
        print(f"HotpotQA download failed: {e}")
        print("Will rely on synthetic data instead.")

    print(f"HotpotQA: {len(samples)} samples")
    return samples


def format_hotpotqa_item(item: dict) -> dict | None:
    """Convert HotpotQA to ReAct format with simulated search steps."""
    try:
        question = item["question"]
        answer = item["answer"]
        supporting_facts = item.get("supporting_facts", {})

        # Build context from supporting facts
        titles = supporting_facts.get("title", [])
        sentences = supporting_facts.get("sentences", []) if isinstance(supporting_facts, dict) else []

        # Create a ReAct trajectory with search steps
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use search_web to find information before answering.",
            },
            {"role": "user", "content": question},
        ]

        # Add search steps for each supporting fact
        if isinstance(titles, list):
            for title in titles:
                messages.append({
                    "role": "assistant",
                    "content": f'Thought: 我需要搜索关于 "{title}" 的信息。\nAction: search_web(query="{title}")\n'
                })
                if isinstance(sentences, list) and sentences:
                    snippet = " ".join(sentences[:2]) if len(sentences) >= 2 else sentences[0] if sentences else ""
                    messages.append({
                        "role": "tool",
                        "content": json.dumps({
                            "results": [{"title": title, "snippet": snippet}]
                        }, ensure_ascii=False)
                    })

        messages.append({
            "role": "assistant",
            "content": f"Thought: 我已经收集了足够的信息，可以回答用户了。\nFinal Answer: {answer}\n"
        })

        return {"messages": messages}
    except Exception:
        return None


def save_jsonl(samples: list[dict], path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download ReAct training data")
    parser.add_argument("--output_dir", type=str, default="data/raw")
    parser.add_argument("--toolbench_samples", type=int, default=4000)
    parser.add_argument("--hotpotqa_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # 1. ToolBench
    toolbench_data = download_toolbench(args.output_dir, args.toolbench_samples)
    if toolbench_data:
        save_jsonl(toolbench_data, output_dir / "toolbench.jsonl")
        all_samples.extend(toolbench_data)

    # 2. HotpotQA
    hotpotqa_data = download_hotpotqa(args.output_dir, args.hotpotqa_samples)
    if hotpotqa_data:
        save_jsonl(hotpotqa_data, output_dir / "hotpotqa.jsonl")
        all_samples.extend(hotpotqa_data)

    # Save combined raw
    if all_samples:
        save_jsonl(all_samples, output_dir / "combined_raw.jsonl")

    print("\n" + "=" * 50)
    print(f"DONE: {len(all_samples)} total samples in {output_dir}")
    print("=" * 50)
    print("\nNext: Run 02_generate_synthetic.py to add GPT-4o synthetic data")


if __name__ == "__main__":
    main()
