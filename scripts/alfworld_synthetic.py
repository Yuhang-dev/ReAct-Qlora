"""Generate synthetic ALFWorld trajectories using DeepSeek V4 Flash."""

import argparse, json, os, random, time, re
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", "sk-placeholder"),
    base_url="https://api.deepseek.com",
)

# ── Few-shot examples ────────────────────────────────────
FEW_SHOT = """
Example 1:
Task: put some vase in safe.
Initial Observation: You are in the middle of a room. Looking quickly around you, you see a drawer 2, a shelf 5, a drawer 1, a shelf 4, a sidetable 1, a drawer 5, a shelf 6, a shelf 1, a shelf 9, a cabinet 2, a sofa 1, a cabinet 1, a shelf 3, a cabinet 3, a drawer 3, a shelf 11, a shelf 2, a shelf 10, a dresser 1, a shelf 12, a garbagecan 1, a armchair 1, a cabinet 4, a shelf 7, a shelf 8, a safe 1, and a drawer 4.

Trajectory:
THOUGHT: I need to find a vase and put it in the safe. Vases are typically found on shelves. I'll check the shelves.
ACTION: go to shelf 6
Observation: On the shelf 6, you see a vase 2.
THOUGHT: Found a vase. Now I need to take it and go to the safe.
ACTION: take vase 2 from shelf 6
Observation: You pick up the vase 2 from the shelf 6.
THOUGHT: Now I need to go to the safe and put the vase inside. The safe might be closed, so I might need to open it first.
ACTION: go to safe 1
Observation: The safe 1 is closed.
THOUGHT: The safe is closed, I need to open it first.
ACTION: open safe 1
Observation: You open the safe 1. The safe 1 is open. In it, you see a keychain 3.
THOUGHT: The safe is open now. I can put the vase inside.
ACTION: put vase 2 in/on safe 1
Observation: You won!

Example 2:
Task: put two pencils in desk.
Initial Observation: You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 8, a drawer 7, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a shelf 5, a shelf 4, a shelf 3, a shelf 2, a shelf 1, and a sidetable 1.

Trajectory:
THOUGHT: I need to find two pencils and put them in the desk. Pencils are often found on desks or in drawers. I'll start by checking the desk.
ACTION: go to desk 1
Observation: On the desk 1, you see a book 1, a bowl 1, a keychain 1, a laptop 1, and a pencil 2.
THOUGHT: Found one pencil on the desk. I need to also find the second pencil. Let me check the drawers.
ACTION: take pencil 2 from desk 1
Observation: You pick up the pencil 2 from the desk 1.
THOUGHT: I have one pencil. Now checking the drawers for the second one.
ACTION: go to drawer 1
Observation: The drawer 1 is closed.
ACTION: open drawer 1
Observation: You open the drawer 1. The drawer 1 is open. In it, you see a creditcard 1, and a pencil 1.
THOUGHT: Found the second pencil. Taking it.
ACTION: take pencil 1 from drawer 1
Observation: You pick up the pencil 1 from the drawer 1.
THOUGHT: I have both pencils now. Going to the desk to put them there.
ACTION: go to desk 1
Observation: On the desk 1, you see a book 1, a bowl 1, a keychain 1, a laptop 1.
ACTION: put pencil 2 in/on desk 1
Observation: You put the pencil 2 in/on the desk 1.
ACTION: put pencil 1 in/on desk 1
Observation: You won!
"""


GEN_PROMPT = """Generate a new, diverse ALFWorld household task trajectory.

## Task Types (pick one, vary across generations):
1. pick_and_place: take OBJECT and put it in/on LOCATION
2. examine: look at OBJECT under LIGHT
3. clean: clean OBJECT with SINK/BASIN
4. heat: heat OBJECT with MICROWAVE/TOASTER
5. cool: cool OBJECT with FRIDGE
6. pick_two: find two OBJECTs and put them in/on LOCATION

## Format MUST be exactly:
Task: <one sentence describing the goal>
Initial Observation: <describe the room with 15-30 objects, each with a number>

Trajectory:
THOUGHT: <reasoning about what to do next>
ACTION: <action>
Observation: <result of action>
... (continue for 5-15 turns until task complete)
THOUGHT: Task complete.
Observation: You won!

## Rules:
- Objects should have numbers (e.g., "apple 2", "shelf 3")
- Include realistic exploration: agent might check wrong locations first
- Include at least one "closed" container that needs opening
- Actions: go to X, take X from Y, put X in/on Y, open X, close X, clean X with Y, heat X with Y, cool X with Y
- The task should complete successfully with "You won!"
- Room should have diverse objects: furniture (desk, bed, shelf, drawer, cabinet, fridge, countertop, sidetable, dresser, safe, garbagecan, coffeetable, armchair, sofa), small objects (apple, pencil, book, laptop, bowl, mug, plate, vase, creditcard, keychain, cellphone, newspaper, tissue, soap, cloth, sponge, ladle, pan, tomato, lettuce, potato, bread, egg, butter, cup, glass, fork, knife, spoon, remotecontrol, alarmclock, cd)

## Generate one trajectory now. Output ONLY valid JSON:
{"task": "<task description>", "initial_obs": "<room description with objects>", "trajectory": [{"thought": "...", "action": "...", "observation": "..."}]}
"""


def generate_one(max_retries=3) -> dict | None:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": "You are a data generator. Output ONLY valid JSON."},
                    {"role": "user", "content": FEW_SHOT + "\n\n" + GEN_PROMPT},
                ],
                temperature=0.9, max_tokens=2048,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            # Validate structure
            if "task" in data and "initial_obs" in data and "trajectory" in data:
                if all("thought" in t and "action" in t and "observation" in t for t in data["trajectory"]):
                    return data
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def trajectory_to_conversations(task, initial_obs, trajectory):
    """Convert to AgentTuning conversation format."""
    convs = [
        {"role": "user", "content": f"Interact with a household to solve a task...\n\nYour task is to: {task}"},
        {"role": "assistant", "content": "OK. I'll follow your instructions and try my best to solve the task."},
        {"role": "user", "content": f"Here is your task.\n{initial_obs}\nYour task is to: {task}"},
    ]
    for step in trajectory:
        content = f"THOUGHT: {step['thought']}\nACTION: {step['action']}"
        convs.append({"role": "assistant", "content": content})
        convs.append({"role": "user", "content": step["observation"]})
    return convs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="data/alfworld_synthetic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} synthetic ALFWorld trajectories...")
    all_data = []
    batch_size = 20

    for batch_start in range(0, args.count, batch_size):
        batch_end = min(batch_start + batch_size, args.count)
        batch = []
        print(f"  Batch {batch_start//batch_size + 1}: ", end="", flush=True)

        for _ in range(batch_start, batch_end):
            traj = generate_one()
            if traj:
                convs = trajectory_to_conversations(
                    traj["task"], traj["initial_obs"], traj["trajectory"]
                )
                batch.append({"id": f"synth_{len(all_data)}", "conversations": convs})
                all_data.append({"id": f"synth_{len(all_data)}", "conversations": convs})
                print(".", end="", flush=True)
            else:
                print("x", end="", flush=True)

        # Save batch
        if batch:
            batch_path = output_dir / f"batch_{batch_start//batch_size:03d}.jsonl"
            with open(batch_path, "w", encoding="utf-8") as f:
                for item in batch:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f" {len(batch)}/{batch_end - batch_start}")

        if batch_start + batch_size < args.count:
            time.sleep(2)

    # Save combined
    combined = output_dir / "combined.jsonl"
    with open(combined, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nGenerated: {len(all_data)}/{args.count}")
    print(f"Saved to {combined}")


if __name__ == "__main__":
    main()
