"""Merge LoRA weights into base model, then AWQ quantize for deployment."""

import argparse
import gc
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora(
    base_model_name: str,
    lora_path: str,
    output_path: str,
    push_to_hub: bool = False,
    hub_repo: str | None = None,
) -> None:
    """Merge LoRA adapters into base model and save in FP16."""
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {lora_path}")
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, lora_path)
    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    if push_to_hub and hub_repo:
        print(f"Pushing to hub: {hub_repo}")
        model.push_to_hub(hub_repo)
        tokenizer.push_to_hub(hub_repo)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Merge complete.")


def quantize_awq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
    zero_point: bool = True,
) -> None:
    """Quantize merged model with AWQ."""
    from awq import AutoAWQForCausalLM

    print(f"Loading model for AWQ quantization: {model_path}")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # AWQ calibration needs a small dataset
    quant_config = {
        "zero_point": zero_point,
        "q_group_size": group_size,
        "w_bit": bits,
        "version": "GEMM",
    }

    print(f"Quantizing (bits={bits}, group_size={group_size})...")
    model.quantize(tokenizer, quant_config=quant_config)

    print(f"Saving quantized model to: {output_path}")
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Quantization complete.")


def quantize_gguf(
    model_path: str,
    output_path: str,
    quant_type: str = "Q4_K_M",
) -> None:
    """Convert to GGUF format via llama.cpp converter.

    This requires llama.cpp to be installed. Alternatively, use the
    convert-hf-to-gguf.py script from llama.cpp repository.
    """
    print("GGUF conversion requires llama.cpp.")
    print(f"  llama.cpp convert: python convert-hf-to-gguf.py {model_path} --outtype {quant_type} --outfile {output_path}")
    print("Skipping GGUF conversion (manual step).")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA and quantize for deployment")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["merge", "quantize", "all"])
    parser.add_argument("--base_model", type=str,
                        default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    parser.add_argument("--lora_path", type=str, default="outputs/react-qwen2.5-7b-qlora/final_lora")
    parser.add_argument("--merged_path", type=str, default="outputs/merged")
    parser.add_argument("--quantized_path", type=str, default="outputs/quantized")
    parser.add_argument("--quant_method", type=str, default="awq",
                        choices=["awq", "gguf"])
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_repo", type=str, default=None)
    args = parser.parse_args()

    if args.mode in ("merge", "all"):
        merge_lora(
            base_model_name=args.base_model,
            lora_path=args.lora_path,
            output_path=args.merged_path,
            push_to_hub=args.push_to_hub,
            hub_repo=args.hub_repo,
        )

    if args.mode in ("quantize", "all"):
        quant_src = args.merged_path if args.mode == "quantize" else args.merged_path
        if args.quant_method == "awq":
            quantize_awq(
                model_path=quant_src,
                output_path=args.quantized_path,
                bits=args.bits,
                group_size=args.group_size,
            )
        else:
            quantize_gguf(
                model_path=quant_src,
                output_path=args.quantized_path,
            )


if __name__ == "__main__":
    main()
