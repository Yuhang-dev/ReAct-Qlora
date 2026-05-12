"""Preprocess raw ReAct data into training-ready ChatML JSONL."""

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from src.data.react_format import validate_trajectory
from src.data.negative_sampler import build_dataset_with_negatives
from src.data.stats import compute_token_stats, filter_by_length, generate_length_report


def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def save_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Preprocess ReAct training data")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file or directory")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--tokenizer", type=str, default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--min_tokens", type=int, default=100)
    parser.add_argument("--negative_ratio", type=float, default=0.1)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Load raw data
    input_path = Path(args.input)
    if input_path.is_dir():
        samples = []
        for f in sorted(input_path.glob("*.jsonl")):
            print(f"  Loading {f}")
            samples.extend(load_jsonl(f))
    else:
        samples = load_jsonl(input_path)

    print(f"Loaded {len(samples)} raw samples")

    # Validate format
    valid = []
    invalid = 0
    for s in samples:
        messages = s.get("messages", s)
        ok, err = validate_trajectory(messages)
        if ok:
            valid.append(s)
        else:
            invalid += 1
    print(f"Format validation: {len(valid)} valid, {invalid} rejected")

    # Build negative samples
    if args.negative_ratio > 0:
        print(f"Generating negative samples (ratio={args.negative_ratio})")
        valid = build_dataset_with_negatives(valid, negative_ratio=args.negative_ratio)
        print(f"After negative sampling: {len(valid)} total")

    # Filter by token length (use separate read/write files)
    from src.data.stats import _count_tokens

    temp_in = output_dir / "temp_all.jsonl"
    temp_out = output_dir / "temp_filtered.jsonl"
    save_jsonl(valid, temp_in)

    kept, total = filter_by_length(temp_in, tokenizer, args.max_tokens, temp_out)
    print(f"Length filter (max={args.max_tokens}): {kept}/{total} kept")

    valid = load_jsonl(temp_out)
    # Also filter by min tokens
    final = []
    for s in valid:
        messages = s.get("messages", s)
        if _count_tokens(messages, tokenizer) >= args.min_tokens:
            final.append(s)
    print(f"Min-length filter (min={args.min_tokens}): {len(final)} kept")

    # Split and strip internal fields
    random.shuffle(final)
    # Keep only messages
    final_clean = [{"messages": s["messages"]} for s in final]
    n = len(final_clean)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train = final_clean[:n_train]
    val = final_clean[n_train : n_train + n_val]
    test = final_clean[n_train + n_val :]

    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(val, output_dir / "val.jsonl")
    save_jsonl(test, output_dir / "test.jsonl")

    print(f"\nFinal splits: train={len(train)}, val={len(val)}, test={len(test)}")

    # Generate length report
    generate_length_report(output_dir / "train.jsonl", tokenizer, output_dir / "length_report.txt")

    # Cleanup
    temp_in.unlink(missing_ok=True)
    temp_out.unlink(missing_ok=True)

    print("Done.")


if __name__ == "__main__":
    main()
