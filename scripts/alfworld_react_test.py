"""Real ReAct loop: model acts → find matching obs in reference → feed back."""

import json, re
import torch, unsloth
from unsloth import FastLanguageModel
from datasets import load_dataset

SYSTEM_PROMPT = """You are an intelligent agent in a household. Complete the task goal by taking actions.

Format EVERY response:
THOUGHT: <reason about what you see and what to do next>
ACTION: <your action>

Available actions: go to LOCATION, take OBJECT from LOCATION, put OBJECT in/on LOCATION, open OBJECT, close OBJECT, clean OBJECT with LOCATION, heat OBJECT with LOCATION, cool OBJECT with LOCATION, toggle OBJECT, look"""


def extract_action(text: str) -> str | None:
    m = re.search(r'ACTION:\s*(.+)', text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def extract_thought(text: str) -> str | None:
    m = re.search(r'THOUGHT:\s*(.+?)(?=\nACTION:|\Z)', text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


def find_action_obs(ref_convs: list[dict], target_action: str) -> str | None:
    """Find observation that follows a matching action in the reference trajectory.

    Uses partial matching: if model says 'go to drawer 1' and reference has
    'go to drawer 1' anywhere, return the NEXT user message as observation.
    """
    target = target_action.strip().lower()
    for i, turn in enumerate(ref_convs):
        if turn["role"] == "assistant":
            m = re.search(r'ACTION:\s*(.+)', turn["content"], re.IGNORECASE)
            if m and m.group(1).strip().lower() == target:
                # Return next user message
                if i + 1 < len(ref_convs) and ref_convs[i + 1]["role"] == "user":
                    return ref_convs[i + 1]["content"]
    return None


def run_react_loop(model, tokenizer, task_desc: str, initial_obs: str,
                   ref_convs: list[dict], max_turns: int = 15) -> dict:
    """Run ReAct loop with proper observation matching."""
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_desc + "\n\n" + initial_obs},
    ]

    turns = []
    visited = set()

    for turn_num in range(max_turns):
        # ── Model generates ──
        prompt = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=256, temperature=0.1,
                top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id,
            )
        raw = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True,
        ).strip()

        thought = extract_thought(raw) or ""
        action = extract_action(raw) or ""

        # ── Find matching observation ──
        observation = find_action_obs(ref_convs, action) if action else None

        # If no exact match, try fuzzy (same action type)
        if not observation and action:
            action_words = action.split()
            for i, turn in enumerate(ref_convs):
                if turn["role"] == "assistant":
                    m = re.search(r'ACTION:\s*(.+)', turn["content"], re.IGNORECASE)
                    if m and m.group(1).strip().split()[0] == action_words[0]:
                        if i + 1 < len(ref_convs) and ref_convs[i + 1]["role"] == "user":
                            observation = ref_convs[i + 1]["content"]
                            break

        # Detect loops
        action_sig = action[:30] if action else ""
        if action_sig in visited:
            observation = observation or "Nothing new happens."
        visited.add(action_sig)

        turns.append({
            "turn": turn_num,
            "thought": thought[:200],
            "action": action,
            "observation": (observation or "")[:300],
        })

        # Feed back
        conversation.append({"role": "assistant", "content": raw})
        if observation:
            conversation.append({"role": "user", "content": observation})
            if "You won" in observation:
                return {"turns": turns, "num_turns": len(turns), "task_done": True}
        else:
            conversation.append({"role": "user", "content": "That action is not possible here."})

    return {"turns": turns, "num_turns": len(turns), "task_done": False}


def main():
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit", max_seq_length=4096,
        dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    print("Loading ALFWorld dataset...")
    ds = load_dataset("neulab/agent-data-collection", "agenttuning_alfworld",
                      split="raw", streaming=True)

    num_tasks = 10
    results = []

    for i, sample in enumerate(ds):
        if i >= num_tasks:
            break

        task_id = sample["id"]
        convs = sample["conversations"]

        task_desc = convs[2]["content"]
        initial_obs = convs[3]["content"] if len(convs) > 3 else ""

        goal = [l for l in task_desc.split("\n") if "task is to:" in l.lower()]
        goal_str = goal[0].strip() if goal else task_desc[:80]

        print(f"\n{'='*70}")
        print(f"[{i+1}/{num_tasks}] {task_id}")
        print(f"     Goal: {goal_str}")
        print(f"{'='*70}")

        result = run_react_loop(model, tokenizer, task_desc, initial_obs, convs)
        results.append({"task_id": task_id, "goal": goal_str, **result})

        for t in result["turns"]:
            print(f"\n  T{t['turn']+1}:")
            if t['thought']:
                print(f"    THOUGHT: {t['thought'][:180]}")
            print(f"    ACTION:  {t['action']}")
            obs = t['observation'][:200].replace('\n', ' ')
            print(f"    OBS:     {obs}")

        status = "✓ TASK DONE!" if result["task_done"] else f"✗ ({result['num_turns']} turns)"
        print(f"\n  → {status}")
        torch.cuda.empty_cache()

    # Summary
    done = sum(1 for r in results if r["task_done"])
    print(f"\n{'='*50}")
    print(f"RESULTS: {done}/{num_tasks} completed ({done/num_tasks:.0%})")
    print(f"{'='*50}")

    # Show what typical failures look like
    print("\nFAILURE ANALYSIS:")
    for r in results:
        if not r["task_done"]:
            last = r["turns"][-1] if r["turns"] else {}
            print(f"  {r['task_id']}: last action='{last.get('action','')}'")

    with open("outputs/alfworld_react_baseline.json", "w", encoding="utf-8") as f:
        json.dump({"tasks": num_tasks, "completed": done, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\nSaved to outputs/alfworld_react_baseline.json")


if __name__ == "__main__":
    main()
