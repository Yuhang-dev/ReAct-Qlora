"""Verify GPU environment and print a diagnostic report."""

import sys


def check_cuda() -> dict:
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            return {"status": "FAIL", "error": "CUDA not available"}

        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_vram = (
            torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        ) / 1e9
        compute_capability = torch.cuda.get_device_capability(0)

        return {
            "status": "OK",
            "gpu": gpu_name,
            "vram_total_gb": round(total_vram, 1),
            "vram_free_gb": round(free_vram, 1),
            "compute_capability": f"{compute_capability[0]}.{compute_capability[1]}",
            "cuda_version": torch.version.cuda,
        }
    except ImportError:
        return {"status": "FAIL", "error": "PyTorch not installed"}
    except Exception as e:
        return {"status": "FAIL", "error": str(e)}


def check_flash_attention() -> dict:
    try:
        import flash_attn

        return {"status": "OK", "version": getattr(flash_attn, "__version__", "unknown")}
    except ImportError:
        return {"status": "MISSING", "note": "Flash Attention not installed; VRAM usage will be higher"}
    except Exception as e:
        return {"status": "FAIL", "error": str(e)}


def check_packages() -> dict:
    required = [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "peft",
        "trl",
        "unsloth",
        "vllm",
        "autoawq",
        "gradio",
        "openai",
        "wandb",
        "sentencepiece",
        "protobuf",
    ]
    results = {}
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
            results[pkg] = "OK"
        except ImportError:
            results[pkg] = "MISSING"
    return results


def check_hf_token() -> dict:
    import os

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return {"status": "OK", "token_prefix": token[:8] + "..."}
    return {"status": "MISSING", "note": "HF_TOKEN not set; gated models will fail to download"}


def main():
    print("=" * 60)
    print("ReAct Environment Setup — Diagnostic Report")
    print("=" * 60)

    checks = {
        "CUDA / GPU": check_cuda(),
        "Flash Attention": check_flash_attention(),
        "HF Token": check_hf_token(),
        "Packages": check_packages(),
    }

    for name, result in checks.items():
        if name == "Packages":
            pkg_results = result
            print(f"\n[{name}]")
            missing = [p for p, s in pkg_results.items() if s != "OK"]
            if missing:
                print(f"  Missing: {', '.join(missing)}")
            else:
                print(f"  All {len(pkg_results)} packages OK")
        else:
            status = result.get("status", "?")
            icon = "PASS" if status == "OK" else ("WARN" if status == "MISSING" else "FAIL")
            print(f"\n[{name}] {icon}")
            for k, v in result.items():
                if k != "status":
                    print(f"  {k}: {v}")

    print("\n" + "=" * 60)

    # Recommendations
    cuda_info = checks["CUDA / GPU"]
    if cuda_info.get("status") == "OK":
        vram = cuda_info.get("vram_total_gb", 0)
        if vram >= 12:
            print("VRAM OK for QLoRA fine-tuning of 7B model.")
        else:
            print(f"WARNING: {vram}GB VRAM may be tight. Consider a smaller model.")
        print(f"Recommended max_seq_length: {'4096' if vram >= 12 else '2048'}")
    else:
        print("FATAL: CUDA not available. Cannot proceed with training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
