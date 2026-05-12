"""ALFWorld QLoRA fine-tuning v2 — with proper Observation masking.

Key changes:
- DataCollatorForCompletionOnlyLM: only assistant messages compute loss
- Observation/system/user tokens masked (label=-100)
- <|im_end|> included in loss to teach model to stop
"""

import argparse, json, random, os
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


def load_dataset_jsonl(path: str) -> Dataset:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return Dataset.from_list(data)


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
    run_name = "alfworld-v2_" + timestamp
    run_dir = Path(cfg["output"]["dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    swanlab.init(
        project="alfworld-react",
        experiment_name="alfworld-v2",
        config={
            "model": cfg["model"]["name"],
            "lora_r": cfg["lora"]["r"],
            "learning_rate": cfg["training"]["learning_rate"],
            "epochs": cfg["training"]["num_train_epochs"],
            "masked_loss": True,
            "collator": "CompletionOnlyLM",
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
        bias="none",
        use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        random_state=seed, use_rslora=False,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Data
    train_ds = load_dataset_jsonl("data/alfworld_v2/train.jsonl")
    val_ds = load_dataset_jsonl("data/alfworld_v2/val.jsonl")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ── KEY FIX: Manual label masking via string splitting ──
    MARKER = "<|im_start|>"

    def tokenize_with_mask(examples):
        all_input_ids = []
        all_labels = []

        for text in examples["text"]:
            # Split by <|im_start|>, each segment is a role block
            blocks = text.split(MARKER)
            tok_blocks = []
            is_assistant = []

            for block in blocks:
                if not block:
                    continue
                block_text = MARKER + block
                tok_ids = tokenizer.encode(block_text, add_special_tokens=False)
                tok_blocks.append(tok_ids)
                # Mark as assistant if block starts with "assistant\n"
                is_assistant.append(block.startswith("assistant\n"))

            # Build flat input_ids and labels
            input_ids = []
            labels = []
            for tok_ids, is_asst in zip(tok_blocks, is_assistant):
                input_ids.extend(tok_ids)
                if is_asst:
                    labels.extend(tok_ids)  # compute loss
                else:
                    labels.extend([-100] * len(tok_ids))  # masked

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        return {"input_ids": all_input_ids, "labels": all_labels}

    train_ds = train_ds.map(tokenize_with_mask, batched=True, batch_size=100)
    val_ds = val_ds.map(tokenize_with_mask, batched=True, batch_size=100)
    train_ds = train_ds.remove_columns(["text"])
    val_ds = val_ds.remove_columns(["text"])
    print(f"Tokenized train: {len(train_ds)}, val: {len(val_ds)}")

    # Verify mask ratio
    sl = train_ds[0]["labels"]
    m = sum(1 for l in sl if l == -100)
    print(f"Sample: {len(sl)-m} loss / {len(sl)} total ({m/len(sl):.0%} masked)")

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
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0, remove_unused_columns=False,
        report_to="none", run_name=run_name, seed=seed,
        eval_strategy="steps", save_strategy="steps",
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        max_seq_length=cfg["model"]["max_seq_length"],
        packing=False,
        callbacks=[SwanLabCallback()],
    )

    print(f"\n{'='*50}")
    print(f"ALFWorld v2 Training: {run_name}")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"  Epochs: {tc['num_train_epochs']} | LR: {tc['learning_rate']}")
    print(f"  Masked loss: YES (assistant only)")
    print(f"  SwanLab: alfworld-react")
    print(f"{'='*50}\n")

    torch.cuda.empty_cache()
    trainer.train()

    final = run_dir / "final_lora"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"\nSaved to {final}")

    swanlab.finish()


if __name__ == "__main__":
    main()
