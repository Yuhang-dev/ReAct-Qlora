"""Prepare ALFWorld data with proper role separation for masked SFT training.

Key changes from previous version:
1. assistant messages -> compute loss (THOUGHT+ACTION+stop token)
2. system/user/observation messages -> masked (label=-100)
3. Chunk long trajectories into segments
4. Add negative feedback samples
"""

import json
import random
from pathlib import Path


def format_training_sample(messages: list[dict]) -> str:
    """Format conversation as training text with explicit role markers.

    The model will learn:
    - system: context (masked)
    - user: observations/task (masked)
    - assistant: THOUGHT+ACTION+<|im_end|> (loss computed)
    """
    parts = []
    for m in messages:
        role = m["role"]
        content = m["content"].strip()
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    return "\n".join(parts)


def chunk_conversation(convs: list[dict], max_turns: int = 6) -> list[list[dict]]:
    """Split long conversation into overlapping chunks.

    Each chunk: system_prompt + task + [N turns of assistant/user pairs]
    """
    chunks = []
    # Skip system prompt (convs[0]) and initial OK (convs[1])
    # convs[2] is task description

    if len(convs) <= 5:
        return [convs]

    # Take the system prompt (convs[0]) and task desc (convs[2])
    base = [convs[0], convs[1], convs[2]]

    # Process turns: convs[3:] are alternating assistant/user pairs
    turns = convs[3:]
    for start in range(0, len(turns), max_turns * 2):
        chunk = list(base)
        end = min(start + max_turns * 2, len(turns))
        chunk.extend(turns[start:end])
        if len(chunk) >= 5:  # At least 1 assistant turn
            chunks.append(chunk)

    return chunks if chunks else [convs]


def add_negative_sample(convs: list[dict]) -> list[dict] | None:
    """Create a negative sample by modifying one successful trajectory.

    Inject an error observation and an assistant recovery response.
    """
    # Find a "take" action and replace its observation with an error
    for i, turn in enumerate(convs):
        if turn["role"] == "assistant" and "ACTION: take" in turn["content"]:
            if i + 1 < len(convs) and convs[i + 1]["role"] == "user":
                # Original observation
                orig_obs = convs[i + 1]["content"]

                # Create error observation
                error_obs = "Nothing happens. You can't reach that from here."
                recovery_action = turn["content"].replace("take", "go to")
                # Find what was being taken, extract location
                import re
                m = re.search(r"take (.+?) from (.+)", turn["content"])
                if m:
                    obj, loc = m.group(1).strip(), m.group(2).strip()
                    error_obs = f"You are not at the {loc}. You need to go there first."
                    recovery_action = f"THOUGHT: I need to go to the {loc} first before taking the {obj}.\nACTION: go to {loc}"

                # Build negative sample
                neg = list(convs[:i])
                neg.append(turn)  # original take action
                neg.append({"role": "user", "content": error_obs})  # error observation
                neg.append({"role": "assistant", "content": recovery_action})  # recovery
                # Continue with rest of trajectory (starting from go to)
                neg.extend(convs[i + 2 :])
                return neg

    return None


def main():
    random.seed(42)

    # Load all data sources
    all_convs = []
    for src in [
        "data/alfworld/train.jsonl",
        "data/alfworld/val.jsonl",
        "data/alfworld/test.jsonl",
        "data/alfworld_synthetic/combined.jsonl",
    ]:
        p = Path(src)
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_convs.append(json.loads(line))

    print(f"Total raw trajectories: {len(all_convs)}")

    # Chunk long trajectories
    all_chunks = []
    for s in all_convs:
        chunks = chunk_conversation(s["conversations"], max_turns=6)
        for c in chunks:
            all_chunks.append(c)

    print(f"After chunking: {len(all_chunks)} training samples")

    # Add negative samples (~10%)
    neg_samples = []
    for s in all_convs[: len(all_convs) // 3]:  # Use subset for negatives
        neg = add_negative_sample(s["conversations"])
        if neg:
            neg_samples.append(neg)
            if len(neg_samples) >= len(all_chunks) // 10:
                break

    print(f"Negative samples: {len(neg_samples)}")

    # Combine and shuffle
    all_data = all_chunks + neg_samples
    random.shuffle(all_data)

    # Format as training text
    train_texts = [format_training_sample(c) for c in all_data]

    # Split
    n = len(train_texts)
    train = train_texts[: int(n * 0.85)]
    val = train_texts[int(n * 0.85) : int(n * 0.92)]
    test = train_texts[int(n * 0.92) :]

    out_dir = Path("data/alfworld_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for text in data:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        print(f"{name}: {len(data)} -> {path}")

    # Show a sample
    print(f"\n=== Sample training text ===")
    print(train[0][:600])
    print(f"... ({len(train[0])} chars)")


if __name__ == "__main__":
    main()
