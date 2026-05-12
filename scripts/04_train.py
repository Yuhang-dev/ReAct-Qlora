"""ReAct QLoRA fine-tuning with Unsloth. Supports single-GPU and DeepSpeed multi-GPU."""

import argparse, json, os, random
from datetime import datetime
from pathlib import Path

import unsloth  # noqa: F401 - must be first
from unsloth import FastLanguageModel, is_bfloat16_supported

import swanlab, torch, yaml
from datasets import Dataset
from transformers import TrainerCallback, TrainingArguments
from trl import SFTTrainer


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl_dataset(path: str) -> Dataset:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            data.append({"messages": sample["messages"]})
    return Dataset.from_list(data)


def messages_to_text(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        elif role == "tool":
            parts.append(f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>")
    return "\n".join(parts)


def format_chatml(examples: dict) -> dict:
    texts = [messages_to_text(m) for m in examples["messages"]]
    return {"text": texts}


class SwanLabCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            if metrics:
                swanlab.log(metrics, step=state.global_step)


tokenizer = None


def main():
    global tokenizer

    parser = argparse.ArgumentParser(description="Train ReAct model with Unsloth QLoRA")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config path (e.g. configs/deepspeed_zero2.json)")
    parser.add_argument("--experiment_name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.train_file:
        cfg["data"]["train_file"] = args.train_file
    if args.val_file:
        cfg["data"]["val_file"] = args.val_file
    if args.output_dir:
        cfg["output"]["dir"] = args.output_dir

    seed = cfg["output"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = args.experiment_name or cfg["output"]["run_name"]
    full_run_name = f"{exp_name}_{timestamp}"

    output_root = Path(cfg["output"]["dir"])
    run_dir = output_root / full_run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    swanlab.init(
        project="react-finetune",
        experiment_name=exp_name,
        config={
            "model": cfg["model"]["name"],
            "lora_r": cfg["lora"]["r"],
            "lora_alpha": cfg["lora"]["alpha"],
            "learning_rate": cfg["training"]["learning_rate"],
            "batch_size": cfg["training"]["per_device_train_batch_size"] * cfg["training"]["gradient_accumulation_steps"],
            "epochs": cfg["training"]["num_train_epochs"],
            "max_seq_length": cfg["model"]["max_seq_length"],
            "deepspeed": args.deepspeed is not None,
            "seed": seed,
        },
        logdir=str(output_root / "swanlogs"),
    )

    print(f"Loading model: {cfg['model']['name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=None,
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )

    lora_cfg = cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        random_state=seed,
        use_rslora=False,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    print(f"Loading training data: {cfg['data']['train_file']}")
    train_ds = load_jsonl_dataset(cfg["data"]["train_file"])
    train_ds = train_ds.map(format_chatml, batched=True)

    val_ds = None
    if cfg["data"]["val_file"] and Path(cfg["data"]["val_file"]).exists():
        print(f"Loading validation data: {cfg['data']['val_file']}")
        val_ds = load_jsonl_dataset(cfg["data"]["val_file"])
        val_ds = val_ds.map(format_chatml, batched=True)

    train_cfg = cfg["training"]

    deepspeed_config = args.deepspeed
    if deepspeed_config:
        print(f"DeepSpeed enabled: {deepspeed_config}")
    else:
        print("Single-GPU mode (no DeepSpeed)")

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        num_train_epochs=train_cfg["num_train_epochs"],
        optim=train_cfg["optim"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"] and is_bfloat16_supported(),
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"] if val_ds else None,
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"] and val_ds is not None,
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        remove_unused_columns=train_cfg["remove_unused_columns"],
        report_to="none",
        run_name=full_run_name,
        seed=seed,
        eval_strategy="steps" if val_ds else "no",
        save_strategy="steps",
        deepspeed=deepspeed_config,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=cfg["model"]["max_seq_length"],
        packing=False,
        callbacks=[SwanLabCallback()],
    )

    print(f"\n{'='*50}")
    print(f"Training: {full_run_name}")
    print(f"  Model:      {cfg['model']['name']}")
    print(f"  Train:      {len(train_ds)} samples")
    if val_ds:
        print(f"  Val:        {len(val_ds)} samples")
    print(f"  Max seq:    {cfg['model']['max_seq_length']}")
    print(f"  LoRA:       r={lora_cfg['r']} alpha={lora_cfg['alpha']}")
    print(f"  Batch:      {train_cfg['per_device_train_batch_size']} x {train_cfg['gradient_accumulation_steps']} = {train_cfg['per_device_train_batch_size'] * train_cfg['gradient_accumulation_steps']}")
    print(f"  LR:         {train_cfg['learning_rate']}")
    print(f"  Epochs:     {train_cfg['num_train_epochs']}")
    if deepspeed_config:
        print(f"  DeepSpeed:  {deepspeed_config}")
    print(f"  SwanLab:    {output_root / 'swanlogs'}")
    print(f"  Output:     {run_dir}")
    print(f"{'='*50}\n")

    torch.cuda.empty_cache()
    trainer.train()

    final_path = run_dir / "final_lora"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nModel saved to {final_path}")

    with open(run_dir / "train_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    swanlab.finish()
    print(f"Experiment logs: {output_root / 'swanlogs'}")


if __name__ == "__main__":
    main()
