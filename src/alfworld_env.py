"""Lightweight ALFWorld-compatible text environment in pure Python.

Supports the subset of ALFWorld needed to evaluate ReAct agents.
No TextWorld/jericho dependencies required.
"""

import json
import random
import re
from pathlib import Path


class AlfworldEnv:
    """Minimal ALFWorld text environment."""

    # Task templates
    TASKS = {
        "pick_and_place": {
            "template": "put some {obj} on {target}.",
            "objects": ["handtowel", "keychain", "newspaper", "cellphone", "cd",
                        "pencil", "book", "bowl", "mug", "plate", "cup", "glassbottle",
                        "spoon", "fork", "knife", "cloth", "sponge", "tissue", "soap",
                        "creditcard", "remotecontrol", "alarmclock", "vase", "laptop",
                        "tomato", "potato", "apple", "lettuce", "bread", "egg"],
            "locations": ["cabinet", "sofa", "bed", "sidetable", "countertop", "diningtable",
                          "shelf", "drawer", "dresser", "desk", "coffeetable", "garbagecan"],
        },
        "pick_two": {
            "template": "find two {obj} and put them in {target}.",
            "objects": ["book", "pencil", "cd", "cellphone", "remotecontrol",
                        "creditcard", "keychain", "newspaper", "tissue", "vase",
                        "laptop", "cup", "bowl", "plate", "mug"],
            "locations": ["cabinet", "sofa", "bed", "sidetable", "countertop", "diningtable",
                          "drawer", "desk", "coffeetable", "dresser"],
        },
        "clean_then_place": {
            "template": "clean some {obj} and put it in {target}.",
            "objects": ["apple", "potato", "lettuce", "tomato", "plate", "bowl", "mug",
                        "cup", "spoon", "fork", "knife", "ladle", "dishsponge"],
            "locations": ["cabinet", "countertop", "diningtable", "sidetable",
                          "shelf", "drawer", "fridge", "microwave"],
        },
        "heat_then_place": {
            "template": "heat some {obj} and put it in {target}.",
            "objects": ["potato", "bread", "mug", "plate", "cup", "egg", "tomato", "apple"],
            "locations": ["countertop", "diningtable", "cabinet", "sidetable",
                          "coffeemachine", "microwave", "fridge"],
        },
        "cool_then_place": {
            "template": "cool some {obj} and put it in {target}.",
            "objects": ["lettuce", "mug", "cup", "plate", "potato", "bread", "apple", "tomato"],
            "locations": ["countertop", "diningtable", "cabinet", "sidetable",
                          "fridge", "microwave", "coffeemachine"],
        },
        "examine": {
            "template": "examine the {obj} with the {target}.",
            "objects": ["bowl", "cellphone", "book", "cd", "newspaper", "keychain",
                        "pencil", "vase", "creditcard"],
            "locations": ["desklamp", "floorlamp"],
        },
    }

    ALL_FURNITURE = [
        "cabinet 1", "cabinet 2", "cabinet 3", "cabinet 4",
        "drawer 1", "drawer 2", "drawer 3", "drawer 4", "drawer 5",
        "shelf 1", "shelf 2", "shelf 3", "shelf 4", "shelf 5",
        "countertop 1", "countertop 2",
        "sidetable 1", "diningtable 1", "coffeetable 1",
        "desk 1", "dresser 1", "sofa 1", "bed 1", "armchair 1",
        "garbagecan 1", "fridge 1", "microwave 1", "toaster 1",
        "sinkbasin 1", "desklamp 1", "floorlamp 1",
        "handtowelholder 1", "towelholder 1", "toilet 1",
        "safe 1", "coffeemachine 1", "stoveburner 1",
    ]

    def __init__(self, task_type="pick_and_place", seed=None):
        if seed is not None:
            random.seed(seed)

        self.task_type = task_type
        task_info = self.TASKS[task_type]
        self.obj = random.choice(task_info["objects"])
        self.target = random.choice(task_info["locations"])

        # Generate room
        self.furniture = random.sample(
            self.ALL_FURNITURE, min(18, len(self.ALL_FURNITURE))
        )
        # Ensure target is in room
        if self.target + " 1" not in self.furniture:
            self.furniture.append(self.target + " 1")

        # Place objects on furniture
        self.objects = {}  # location -> [(obj_name, obj_num), ...]
        obj_counters = {}
        for furn in self.furniture:
            self.objects[furn] = []

        # Place distractors
        distractors = [o for o in task_info["objects"] if o != self.obj]
        for _ in range(random.randint(10, 25)):
            furn = random.choice(self.furniture)
            obj_name = random.choice(distractors)
            obj_counters[obj_name] = obj_counters.get(obj_name, 0) + 1
            obj_num = obj_counters[obj_name]
            self.objects[furn].append((obj_name, obj_num))

        # Place target object
        target_furn = random.choice(self.furniture)
        obj_counters[self.obj] = obj_counters.get(self.obj, 0) + 1
        self.target_obj_num = obj_counters[self.obj]
        self.object_location = target_furn
        self.objects[target_furn].append((self.obj, self.target_obj_num))

        # Place second target for pick_two
        if task_type == "pick_two":
            furn2 = random.choice([f for f in self.furniture if f != target_furn])
            obj_counters[self.obj] = obj_counters.get(self.obj, 0) + 1
            self.target_obj_num2 = obj_counters[self.obj]
            self.object_location2 = furn2
            self.objects[furn2].append((self.obj, self.target_obj_num2))

        # Agent state
        self.inventory = []
        self.current_location = None  # middle of room
        self.closed = set(random.sample(self.furniture, min(5, len(self.furniture))))
        self.obj_states = {}  # obj_name_num -> state (hot, cold, clean)

        # Task complete flag
        self.done = False
        self.max_turns = 50
        self.turn_count = 0

    def reset(self) -> str:
        """Return initial observation."""
        self.done = False
        self.turn_count = 0
        desc = self._room_desc()
        task = self.TASKS[self.task_type]["template"].format(
            obj=self.obj, target=self.target
        )
        return f"You are in the middle of a room. Looking quickly around you{desc}\nYour task is to: {task}"

    def _room_desc(self) -> str:
        if not self.furniture:
            return ", you see nothing."
        items = sorted(self.furniture)
        if len(items) == 1:
            return f", you see a {items[0]}."
        return ", you see a " + ", a ".join(items[:-1]) + f", and a {items[-1]}."

    def step(self, action_str: str) -> str:
        """Execute one action and return observation."""
        self.turn_count += 1
        if self.turn_count > self.max_turns:
            return "Too many turns. Task failed."

        action_str = action_str.strip().rstrip(".")

        # Parse action
        go_match = re.match(r"go to (.+)", action_str, re.IGNORECASE)
        take_match = re.match(r"take (.+) from (.+)", action_str, re.IGNORECASE)
        put_match = re.match(r"put (.+) in/on (.+)", action_str, re.IGNORECASE)
        open_match = re.match(r"open (.+)", action_str, re.IGNORECASE)
        close_match = re.match(r"close (.+)", action_str, re.IGNORECASE)
        clean_match = re.match(r"clean (.+) with (.+)", action_str, re.IGNORECASE)
        heat_match = re.match(r"heat (.+) with (.+)", action_str, re.IGNORECASE)
        cool_match = re.match(r"cool (.+) with (.+)", action_str, re.IGNORECASE)
        examine_match = re.match(r"examine (.+) with (.+)", action_str, re.IGNORECASE)
        look_match = re.match(r"look", action_str, re.IGNORECASE)

        if go_match:
            return self._go(go_match.group(1))
        elif take_match:
            return self._take(take_match.group(1), take_match.group(2))
        elif put_match:
            return self._put(put_match.group(1), put_match.group(2))
        elif open_match:
            return self._open(open_match.group(1))
        elif close_match:
            return self._close(close_match.group(1))
        elif clean_match:
            return self._clean(clean_match.group(1), clean_match.group(2))
        elif heat_match:
            return self._heat(heat_match.group(1), heat_match.group(2))
        elif cool_match:
            return self._cool(cool_match.group(1), cool_match.group(2))
        elif examine_match:
            return self._examine(examine_match.group(1), examine_match.group(2))
        elif look_match:
            return self._look()
        else:
            return f"I don't understand: {action_str}"

    def _go(self, location: str) -> str:
        location = location.strip()
        if location not in self.furniture:
            return f"The {location} is not here."
        self.current_location = location
        obs = f"You arrive at loc {self.furniture.index(location)}. "
        if location in self.closed:
            obs += f"The {location} is closed."
        elif location in self.objects and self.objects[location]:
            items = [f"a {o} {n}" for o, n in self.objects[location]]
            obs += f"On the {location}, you see " + ", ".join(items[:-1])
            if len(items) > 1:
                obs += f", and {items[-1]}."
            else:
                obs += f"{items[0]}."
        else:
            obs += f"On the {location}, you see nothing."
        return obs

    def _take(self, obj_str: str, source: str) -> str:
        source = source.strip()
        obj_match = re.match(r"(\w+)\s+(\d+)", obj_str.strip())
        if not obj_match:
            return f"What is '{obj_str}'?"
        obj_name, obj_num = obj_match.group(1), int(obj_match.group(2))

        if source not in self.furniture:
            return f"The {source} is not here."
        if source in self.closed:
            return f"The {source} is closed. Open it first."

        key = (obj_name, obj_num)
        if source in self.objects and key in self.objects[source]:
            self.objects[source].remove(key)
            self.inventory.append(key)
            return f"You pick up the {obj_name} {obj_num} from the {source}."
        return f"There is no {obj_name} {obj_num} on the {source}."

    def _put(self, obj_str: str, target: str) -> str:
        target = target.strip()
        obj_match = re.match(r"(\w+)\s+(\d+)", obj_str.strip())
        if not obj_match:
            return f"What is '{obj_str}'?"
        obj_name, obj_num = obj_match.group(1), int(obj_match.group(2))

        key = (obj_name, obj_num)
        if key not in self.inventory:
            return f"You don't have the {obj_name} {obj_num}."

        if target not in self.furniture:
            return f"The {target} is not here."
        if target in self.closed:
            return f"The {target} is closed. Open it first."

        self.inventory.remove(key)
        if target not in self.objects:
            self.objects[target] = []
        self.objects[target].append(key)
        obs = f"You put the {obj_name} {obj_num} in/on the {target}."

        # Check win condition
        if self._check_win():
            self.done = True
            obs += " You won!"
        return obs

    def _open(self, obj: str) -> str:
        obj = obj.strip()
        if obj not in self.furniture:
            return f"The {obj} is not here."
        if obj in self.closed:
            self.closed.remove(obj)
            obs = f"You open the {obj}. The {obj} is open."
            if obj in self.objects and self.objects[obj]:
                items = [f"a {o} {n}" for o, n in self.objects[obj]]
                obs += " In it, you see " + ", ".join(items[:-1])
                if len(items) > 1:
                    obs += f", and {items[-1]}."
                else:
                    obs += f"{items[0]}."
            else:
                obs += " In it, you see nothing."
            return obs
        return f"The {obj} is already open."

    def _close(self, obj: str) -> str:
        obj = obj.strip()
        if obj not in self.furniture:
            return f"The {obj} is not here."
        if obj not in self.closed:
            self.closed.add(obj)
            return f"You close the {obj}."
        return f"The {obj} is already closed."

    def _clean(self, obj_str: str, tool: str) -> str:
        return self._process_obj(obj_str, tool, "clean")

    def _heat(self, obj_str: str, tool: str) -> str:
        return self._process_obj(obj_str, tool, "heat")

    def _cool(self, obj_str: str, tool: str) -> str:
        return self._process_obj(obj_str, tool, "cool")

    def _process_obj(self, obj_str: str, tool: str, action: str) -> str:
        obj_match = re.match(r"(\w+)\s+(\d+)", obj_str.strip())
        if not obj_match:
            return f"What is '{obj_str}'?"
        obj_name, obj_num = obj_match.group(1), int(obj_match.group(2))
        key = (obj_name, obj_num)

        if key not in self.inventory:
            return f"You don't have the {obj_name} {obj_num}."

        if tool.strip() not in self.furniture:
            return f"The {tool.strip()} is not here."

        if action == "clean":
            self.obj_states[key] = "clean"
        elif action == "heat":
            self.obj_states[key] = "hot"
        elif action == "cool":
            self.obj_states[key] = "cold"

        return f"You {action} the {obj_name} {obj_num} using the {tool.strip()}."

    def _examine(self, obj_str: str, tool: str) -> str:
        tool = tool.strip()
        if tool not in self.furniture:
            return f"The {tool} is not here."
        return f"You examine the {obj_str.strip()} under the {tool}. It looks normal."

    def _look(self) -> str:
        if self.current_location:
            return self._go(self.current_location)
        return self._room_desc()

    def _check_win(self) -> bool:
        tt = self.task_type
        obj_key = (self.obj, self.target_obj_num)

        if tt == "pick_and_place":
            target = self.target + " 1"
            return target in self.objects and obj_key in self.objects[target]

        if tt == "pick_two":
            target = self.target + " 1"
            obj_key2 = (self.obj, self.target_obj_num2)
            return (target in self.objects and
                    obj_key in self.objects[target] and
                    obj_key2 in self.objects[target])

        if tt == "clean_then_place":
            target = self.target + " 1"
            return (target in self.objects and obj_key in self.objects[target] and
                    obj_key in self.obj_states and self.obj_states[obj_key] == "clean")

        if tt == "heat_then_place":
            target = self.target + " 1"
            return (target in self.objects and obj_key in self.objects[target] and
                    obj_key in self.obj_states and self.obj_states[obj_key] == "hot")

        if tt == "cool_then_place":
            target = self.target + " 1"
            return (target in self.objects and obj_key in self.objects[target] and
                    obj_key in self.obj_states and self.obj_states[obj_key] == "cold")

        if tt == "examine":
            return (obj_key in self.inventory and
                    self.current_location and tool in self.current_location)

        return False


def test_env():
    """Quick test of the environment."""
    for task_type in AlfworldEnv.TASKS:
        env = AlfworldEnv(task_type, seed=42)
        obs = env.reset()
        print(f"[{task_type}]")
        print(f"  {obs[:150]}")
        # Try a few actions
        for action in ["go to cabinet 1", "open cabinet 1", "look"]:
            result = env.step(action)
            print(f"  > {action}")
            print(f"    {result[:100]}")
        print()


if __name__ == "__main__":
    test_env()
