"""Deploy fine-tuned ReAct model with vLLM for local inference."""

import argparse
import json
import time

from vllm import LLM, SamplingParams


def load_model(model_path: str, max_model_len: int = 4096):
    print(f"Loading model: {model_path}")
    llm = LLM(
        model=model_path,
        quantization="awq" if "quantized" in model_path or "awq" in model_path else None,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    return llm


def build_react_prompt(
    user_query: str,
    tool_definitions: dict | None = None,
    conversation_history: list[dict] | None = None,
    tokenizer=None,
    add_generation_prompt: bool = True,
) -> str:
    """Build ReAct prompt with tool definitions and conversation history."""
    from src.data.react_format import build_system_prompt

    tools_str = ""
    if tool_definitions:
        for name, spec in tool_definitions.items():
            params = ", ".join(f'{k}: {v}' for k, v in spec.get("parameters", {}).items())
            tools_str += f"- {name}({params}): {spec.get('description', '')}\n"

    messages = [{"role": "system", "content": build_system_prompt(tools_str)}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": user_query})

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def benchmark(
    llm: LLM,
    tokenizer,
    prompts: list[str],
    sampling_params: SamplingParams,
    num_runs: int = 3,
) -> dict:
    """Benchmark inference speed."""
    print(f"\nBenchmarking with {len(prompts)} prompts, {num_runs} runs each...")

    all_ttft = []
    all_tps = []

    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}")
        t_start = time.perf_counter()

        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            # vLLM returns timing in the output object
            if hasattr(output, "metrics"):
                ttft = output.metrics.first_token_time - output.metrics.arrival_time
                all_ttft.append(ttft)
                tps = len(output.outputs[0].token_ids) / (output.metrics.last_token_time - output.metrics.first_token_time)
                all_tps.append(tps)

        t_end = time.perf_counter()

    metrics = {
        "total_time_s": t_end - t_start,
        "avg_ttft_ms": (sum(all_ttft) / len(all_ttft)) * 1000 if all_ttft else None,
        "avg_tokens_per_sec": sum(all_tps) / len(all_tps) if all_tps else None,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Serve ReAct model with vLLM")
    parser.add_argument("--model_path", type=str, default="outputs/quantized")
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--test_prompts", type=str, default=None, help="JSONL file with test prompts")
    args = parser.parse_args()

    llm = load_model(args.model_path, args.max_model_len)
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=1024,
    )

    # ── Benchmark mode ──────────────────────────────────
    if args.benchmark:
        test_prompts = [
            "北京今天天气如何？请详细说明。",
            "帮我查询一下Python中如何实现异步并发请求。",
            "What is the population of Tokyo and how has it changed over the last decade?",
        ]
        if args.test_prompts:
            # Load custom test prompts
            pass

        formatted_prompts = [
            build_react_prompt(p, tokenizer=tokenizer) for p in test_prompts
        ]
        results = benchmark(llm, tokenizer, formatted_prompts, sampling_params)
        print("\nBenchmark Results:")
        print(f"  Avg TTFT: {results['avg_ttft_ms']:.0f} ms")
        print(f"  Avg throughput: {results['avg_tokens_per_sec']:.1f} tokens/s")
        return

    # ── Interactive mode ────────────────────────────────
    if args.interactive:
        print("\nReAct Agent — Interactive Mode")
        print("Type 'quit' to exit.\n")

        history = []
        while True:
            user_input = input("User: ")
            if user_input.lower() in ("quit", "exit", "q"):
                break

            prompt = build_react_prompt(
                user_input, tokenizer=tokenizer, conversation_history=history
            )
            output = llm.generate([prompt], sampling_params)
            response = output[0].outputs[0].text

            print(f"\nAssistant:\n{response}\n")
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})

    # ── Default: verify model loads and can generate ─────
    else:
        prompt = build_react_prompt("Hello, what tools can you use?", tokenizer=tokenizer)
        output = llm.generate([prompt], sampling_params)
        print("Model output:", output[0].outputs[0].text[:500])


if __name__ == "__main__":
    main()
