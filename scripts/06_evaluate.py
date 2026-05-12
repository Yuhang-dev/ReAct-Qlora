"""Evaluate fine-tuned ReAct model against baselines."""

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.data.react_format import parse_final_answer
from src.eval.metrics import compute_dataset_metrics, evaluate_trajectory


def load_test_set(path: str) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def run_inference_vllm(model_path: str, samples: list[dict], max_tokens: int = 1024) -> list:
    """Run batch inference with vLLM."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    tokenizer = llm.get_tokenizer()

    # Build prompts: strip the last assistant message (the answer we want to predict)
    prompts = []
    for s in samples:
        messages = s["messages"]
        # Use all messages except the last assistant turn as prompt
        prompt_msgs = []
        for m in messages:
            if m["role"] == "assistant" and "Final Answer:" in m.get("content", ""):
                break  # Stop before final answer
            prompt_msgs.append(m)
        prompt = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=max_tokens,
    )

    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]


def run_inference_transformers(model, tokenizer, samples: list[dict], max_tokens: int = 1024) -> list:
    """Run inference with HuggingFace model (slower, for validation)."""
    from transformers import GenerationConfig

    results = []
    for s in tqdm(samples, desc="Inference"):
        messages = s["messages"]
        prompt_msgs = []
        for m in messages:
            if m["role"] == "assistant" and "Final Answer:" in m.get("content", ""):
                break
            prompt_msgs.append(m)

        prompt = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        gen_config = GenerationConfig(
            temperature=0.1,
            top_p=0.9,
            max_new_tokens=max_tokens,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        with torch.no_grad():
            output_ids = model.generate(**inputs, generation_config=gen_config)
            output_text = tokenizer.decode(
                output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            results.append(output_text)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ReAct model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model (merged or quantized)")
    parser.add_argument("--test_file", type=str, required=True, help="Test JSONL file")
    parser.add_argument("--output_file", type=str, default="outputs/eval_results.json")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for faster inference")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--llm_judge", action="store_true", help="Use GPT-4o as judge")
    parser.add_argument("--judge_model", type=str, default="gpt-4o")
    args = parser.parse_args()

    samples = load_test_set(args.test_file)
    print(f"Loaded {len(samples)} test samples")

    # ── Inference ───────────────────────────────────────
    print(f"\nRunning inference with model: {args.model_path}")
    t_start = time.time()

    if args.use_vllm:
        predictions = run_inference_vllm(args.model_path, samples, args.max_new_tokens)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        predictions = run_inference_transformers(model, tokenizer, samples, args.max_new_tokens)

    t_end = time.time()
    elapsed = t_end - t_start
    print(f"Inference complete: {len(predictions)} outputs in {elapsed:.1f}s")
    print(f"Avg: {len(predictions) / elapsed:.1f} samples/s")

    # ── Metric evaluation ───────────────────────────────
    results = []
    for i, (sample, pred_text) in enumerate(zip(samples, predictions)):
        # Parse the predicted trajectory
        pred_messages = sample["messages"][:]
        # Replace/append the final assistant message with the prediction
        pred_messages.append({"role": "assistant", "content": pred_text})

        ref_messages = sample.get("messages", sample.get("reference_messages"))

        eval_result = evaluate_trajectory(pred_messages, ref_messages)
        results.append(eval_result)

    metrics = compute_dataset_metrics(results)
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2%}")
        else:
            print(f"  {k}: {v}")

    # ── LLM Judge (optional) ────────────────────────────
    if args.llm_judge:
        from src.eval.llm_judge import LLMJudge

        judge = LLMJudge(model=args.judge_model)
        judge_samples = []
        for sample, pred_text in zip(samples, predictions):
            user_msg = next((m["content"] for m in sample["messages"] if m["role"] == "user"), "")
            true_final = ""
            for m in sample["messages"]:
                if m["role"] == "assistant" and "Final Answer:" in m.get("content", ""):
                    true_final = parse_final_answer(m["content"]) or ""
            pred_final = parse_final_answer(pred_text) or pred_text

            judge_samples.append({
                "question": user_msg,
                "model_answer": pred_final,
                "ground_truth": true_final,
            })

        print(f"\nRunning LLM judge ({args.judge_model}) on {len(judge_samples)} samples...")
        judge_results = judge.judge_batch(judge_samples)
        sr = sum(1 for r in judge_results if r["pass"]) / len(judge_results)
        print(f"LLM Judge Success Rate: {sr:.2%}")

        # Merge judge results into metrics
        for i, jr in enumerate(judge_results):
            results[i].success = jr["pass"]
            results[i].details["judge_reason"] = jr["reason"]

        metrics = compute_dataset_metrics(results)
        metrics["llm_judge_success_rate"] = sr

    # ── Save ─────────────────────────────────────────────
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "metrics": metrics,
        "model_path": args.model_path,
        "num_samples": len(samples),
        "inference_time_s": elapsed,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
