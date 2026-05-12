"""QLoRA fine-tune Qwen2.5-7B-Instruct on ALFWorld ReAct trajectories."""

import argparse, json, random
from datetime import datetime
from pathlib import Path

import unsloth  # noqa: F401
from unsloth import FastLanguageModel, is_bfloat16_supported

import swanlab, torch, yaml
from datasets import Dataset
from transformers import TrainerCallback, TrainingArguments
from trl import SFTTrainer


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_alfworld(file_path: str) -> Dataset:
    """Load ALFWorld conversations and convert to training text."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            convs = sample["conversations"]
            # Convert role names: user->user, assistant->assistant (already correct)
            messages = [{"role": c["role"], "content": c["content"]} for c in convs]
            data.append({"messages": messages})
    return Dataset.from_list(data)


def messages_to_text(messages: list[dict]) -> str:
    """Convert to <|im_start|>...<|im_end|> format."""
    parts = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    return "\n".join(parts)


def format_text(examples: dict) -> dict:
    texts = []
    for messages in examples["messages"]:
        texts.append(messages_to_text(messages))
    return {"text": texts}


class SwanLabCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            if metrics:
                swanlab.log(metrics, step=state.global_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_alfworld.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["output"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = cfg["output"]["run_name"]
    full_name = f"{run_name}_{timestamp}"
    run_dir = Path(cfg["output"]["dir"]) / full_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # SwanLab
    swanlab.init(
        project="alfworld-react",
        experiment_name=run_name,
        config={
            "model": cfg["model"]["name"],
            "lora_r": cfg["lora"]["r"],
            "learning_rate": cfg["training"]["learning_rate"],
            "epochs": cfg["training"]["num_train_epochs"],
        },
        logdir=str(Path(cfg["output"]["dir"]) / "swanlogs"),
    )

    # Model
    print(f"Loading {cfg['model']['name']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=None, load_in_4bit=True,
    )

    lora = cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model, r=lora["r"], target_modules=lora["target_modules"],
        lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        bias="none", use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        random_state=seed, use_rslora=False,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Data
    train_ds = load_alfworld(cfg["data"]["train_file"])
    train_ds = train_ds.map(format_text, batched=True)
    val_ds = load_alfworld(cfg["data"]["val_file"])
    val_ds = val_ds.map(format_text, batched=True)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Training args
    tc = cfg["training"]
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler_type"],
        warmup_ratio=tc["warmup_ratio"],
        num_train_epochs=tc["num_train_epochs"],
        optim=tc["optim"], weight_decay=tc["weight_decay"],
        max_grad_norm=tc["max_grad_norm"],
        fp16=tc["fp16"], bf16=tc["bf16"] and is_bfloat16_supported(),
        logging_steps=tc["logging_steps"], save_steps=tc["save_steps"],
        eval_steps=tc["eval_steps"], save_total_limit=tc["save_total_limit"],
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        dataloader_num_workers=0, remove_unused_columns=False,
        report_to="none", run_name=full_name, seed=seed,
        eval_strategy="steps", save_strategy="steps",
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        dataset_text_field="text", max_seq_length=cfg["model"]["max_seq_length"],
        packing=False,
        callbacks=[SwanLabCallback()],
    )

    print(f"\n{'='*50}")
    print(f"ALFWorld ReAct Training: {full_name}")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"  Epochs: {tc['num_train_epochs']} | LR: {tc['learning_rate']}")
    print(f"  SwanLab: alfworld-react")
    print(f"{'='*50}\n")

    torch.cuda.empty_cache()
    trainer.train()

    # Save
    final = run_dir / "final_lora"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"\nSaved to {final}")

    with open(run_dir / "train_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    swanlab.finish()


if __name__ == "__main__":
    main()
