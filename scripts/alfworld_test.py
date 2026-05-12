"""Test Qwen2.5-7B-Instruct on ALFWorld tasks (zero-shot)."""

import json, re
import torch, unsloth
from unsloth import FastLanguageModel
from datasets import load_dataset


SYSTEM_PROMPT = """Interact with a household to solve a task. You are an intelligent agent in a household environment.
Your target is to perform actions to complete the task goal.

Available actions:
- go to LOCATION
- take OBJECT from LOCATION
- put OBJECT in/on LOCATION
- open OBJECT
- close OBJECT
- toggle OBJECT on/off
- clean OBJECT with LOCATION
- heat OBJECT with LOCATION
- cool OBJECT with LOCATION
- look

Format:
THOUGHT: <your reasoning about the current situation and next step>
ACTION: <your action>

After each action, you will receive an observation of the result."""


def run_alfworld_test(num_tasks=5):
    # Load dataset
    ds = load_dataset("neulab/agent-data-collection", "agenttuning_alfworld",
                      split="raw", streaming=True)

    # Load model
    print("Loading Qwen2.5-7B-Instruct...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit", max_seq_length=4096,
        dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    results = []
    for i, sample in enumerate(ds):
        if i >= num_tasks:
            break

        task_id = sample["id"]
        convs = sample["conversations"]

        # Extract initial task info
        task_desc = convs[2]["content"]  # "Here is your task..."
        initial_obs = convs[3]["content"] if len(convs) > 3 else ""

        print(f"\n{'='*60}")
        print(f"Task {i+1}: {task_id}")
        print(f"{'='*60}")
        print(f"Task desc: {task_desc[:200]}...")
        print(f"Init obs: {initial_obs[:200]}...")
        print()

        # Run multi-turn: follow reference trajectory for up to 6 turns
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_desc},
        ]

        # Add initial observation
        if initial_obs:
            conversation.append({"role": "user", "content": initial_obs})

        ref_actions = []
        model_actions = []

        turn_idx = 3  # Start from first assistant message
        for turn_num in range(6):
            # Model generates
            prompt = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=200, temperature=0.1,
                    top_p=0.9, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            model_output = tokenizer.decode(
                out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True,
            ).strip()

            conversation.append({"role": "assistant", "content": model_output})

            # Extract model action
            action_match = re.search(r'ACTION:\s*(.+)', model_output, re.IGNORECASE)
            model_act = action_match.group(1).strip() if action_match else "NONE"
            model_actions.append(model_act)

            # Get reference action from dataset
            ref_act = None
            for t in range(turn_idx, len(convs)):
                if convs[t]["role"] == "assistant":
                    ref_act_match = re.search(r'ACTION:\s*(.+)', convs[t]["content"], re.IGNORECASE)
                    if ref_act_match:
                        ref_act = ref_act_match.group(1).strip()
                        turn_idx = t + 1
                        break

            if ref_act:
                ref_actions.append(ref_act)

                # Get next observation
                if turn_idx < len(convs) and convs[turn_idx]["role"] == "user":
                    next_obs = convs[turn_idx]["content"]
                    conversation.append({"role": "user", "content": next_obs})
                    turn_idx += 1

            # Check if task complete
            if "THOUGHT: task complete" in model_output.lower() or \
               "thought: task complete" in model_output.lower():
                break

        # Compare
        matches = sum(1 for m, r in zip(model_actions, ref_actions) if m == r)
        print(f"Model actions:      {model_actions}")
        print(f"Reference actions:  {ref_actions}")
        print(f"Action matches: {matches}/{len(ref_actions)}")

        results.append({
            "task_id": task_id,
            "model_actions": model_actions,
            "ref_actions": ref_actions,
            "matches": matches,
            "total_ref": len(ref_actions),
        })

        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_matches = sum(r["matches"] for r in results)
    total_ref = sum(r["total_ref"] for r in results)
    print(f"Overall action accuracy: {total_matches}/{total_ref} ({total_matches/total_ref:.0%})")

    # Save
    out_path = "outputs/alfworld_baseline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    run_alfworld_test()
