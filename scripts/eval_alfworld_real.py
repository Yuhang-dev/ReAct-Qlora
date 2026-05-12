"""Evaluate ReAct model in real ALFWorld environment."""

import json, re, sys, time
from collections import defaultdict

import torch, unsloth
from unsloth import FastLanguageModel
from peft import PeftModel

sys.path.insert(0, ".")
from src.alfworld_env import AlfworldEnv

SYS_PROMPT = """You are a household agent. ONE action per turn. ALWAYS use object NUMBERS (e.g. "handtowel 1").

Actions: go to X N | take X N from Y N | put X N in/on Y N | open X N | close X N | clean X N with Y N | heat X N with Y N | cool X N with Y N | examine X N with Y N

Reply EXACTLY:
THOUGHT: <reasoning>
ACTION: <action>"""


def clean_action(raw: str) -> str:
    m = re.search(r'ACTION:\s*(.+?)(?:\n|$)', raw, re.IGNORECASE)
    if m:
        a = m.group(1).strip()
        if "THOUGHT" in a.upper():
            a = a.split("THOUGHT")[0].strip()
        return a
    return ""


def extract_thought(raw: str) -> str:
    m = re.search(r'THOUGHT:\s*(.+?)(?:\n|ACTION:)', raw, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def run_agent(model, tokenizer, env: AlfworldEnv, max_turns: int = 15) -> dict:
    """Run the ReAct agent in the environment."""
    obs = env.reset()
    msgs = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": obs},
    ]
    trajectory = []

    for turn in range(max_turns):
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=120, temperature=0.1,
                top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id,
            )
        raw = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True,
        ).strip()

        thought = extract_thought(raw)
        action = clean_action(raw)
        msgs.append({"role": "assistant", "content": raw})

        # Execute action in environment
        result = env.step(action) if action else "No action taken."
        msgs.append({"role": "user", "content": result})

        trajectory.append({
            "thought": thought,
            "action": action,
            "observation": result[:300],
        })

        if env.done:
            return {"success": True, "turns": turn + 1, "trajectory": trajectory}

    return {"success": False, "turns": max_turns, "trajectory": trajectory}


def main():
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit", max_seq_length=1024,
        dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    ft_model = PeftModel.from_pretrained(
        model, "outputs_alfworld/alfworld-v2_20260511_2140/checkpoint-80",
    )
    ft_model = ft_model.merge_and_unload()
    FastLanguageModel.for_inference(ft_model)

    # Test each task type × 5 seeds = 30 tasks
    task_types = list(AlfworldEnv.TASKS.keys())
    results = defaultdict(list)
    all_trajs = []

    for task_type in task_types:
        for seed in range(5):
            env = AlfworldEnv(task_type, seed=seed)
            goal = env.reset().split("\n")[-1]

            print(f"\n{'='*60}")
            print(f"[{task_type}] seed={seed} | {goal}")
            print(f"{'='*60}")

            for mname, mdl in [("BASE", model), ("FT", ft_model)]:
                t0 = time.time()
                result = run_agent(mdl, tokenizer, env)
                elapsed = time.time() - t0

                icon = "V" if result["success"] else "X"
                print(f"  {mname}: {icon} ({result['turns']}t, {elapsed:.0f}s)")
                for t in result["trajectory"][:5]:
                    print(f"    -> {t['action'][:60]} | {t['observation'][:60]}")

                results[mname].append(result["success"])
                all_trajs.append({
                    "task_type": task_type,
                    "seed": seed,
                    "goal": goal,
                    "model": mname,
                    **result,
                })

                torch.cuda.empty_cache()
                # Reset env for next model
                env = AlfworldEnv(task_type, seed=seed)

    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")

    for mname in ["BASE", "FT"]:
        sr = sum(results[mname]) / len(results[mname])
        print(f"{mname}: {sum(results[mname])}/{len(results[mname])} = {sr:.0%}")

    # Per task type
    print(f"\n{'Task Type':<25} {'BASE':>8} {'FT':>8}")
    print("-" * 43)
    for tt in task_types:
        base_sr = sum(1 for t in all_trajs if t["task_type"] == tt and t["model"] == "BASE" and t["success"])
        ft_sr = sum(1 for t in all_trajs if t["task_type"] == tt and t["model"] == "FT" and t["success"])
        print(f"{tt:<25} {base_sr:>7}/5 {ft_sr:>7}/5")

    # Save
    out_path = "outputs_alfworld/eval_real.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_trajs, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
