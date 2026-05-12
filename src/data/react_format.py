"""ReAct format definition, validation, and ChatML conversion."""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

# ── Regex patterns ──────────────────────────────────────────────
THOUGHT_PATTERN = re.compile(r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer)|\Z)", re.DOTALL)
ACTION_PATTERN = re.compile(r"Action:\s*(.+?)(?=\n(?:Observation|Thought)|\Z)", re.DOTALL)
FINAL_ANSWER_PATTERN = re.compile(r"Final Answer:\s*(.+?)$", re.DOTALL)
TOOL_CALL_PATTERN = re.compile(r"(\w+)\(([^)]*)\)")

# ── Default system prompt ───────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful assistant with access to the following tools:

{tool_definitions}

You MUST follow this format for every response:
Thought: <your reasoning about what to do next>
Action: <tool_name>(param1="value1", param2="value2")

When you receive a tool response, continue with:
Observation: <tool response content>
Thought: <further reasoning>
Action: <next tool call>

When you have enough information to answer the user's question, respond with:
Thought: <final reasoning>
Final Answer: <your complete answer to the user>

Important rules:
- If a tool returns an error or empty result, acknowledge it and try a different approach.
- Think step by step. Do not skip the Thought step.
- Only call one tool at a time."""


@dataclass
class ReactMessage:
    role: str  # "user" | "assistant" | "tool"
    content: str


@dataclass
class ReactTrajectory:
    messages: list[ReactMessage] = field(default_factory=list)
    has_error_recovery: bool = False
    tool_call_count: int = 0

    def to_chatml(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self.messages]


def build_system_prompt(tool_definitions: str) -> str:
    return SYSTEM_PROMPT.format(tool_definitions=tool_definitions)


def parse_thought(text: str) -> Optional[str]:
    m = THOUGHT_PATTERN.search(text)
    return m.group(1).strip() if m else None


def parse_action(text: str) -> Optional[tuple[str, dict]]:
    """Parse Action into (tool_name, params_dict)."""
    m = ACTION_PATTERN.search(text)
    if not m:
        return None
    call_str = m.group(1).strip()
    cm = TOOL_CALL_PATTERN.match(call_str)
    if not cm:
        return None
    name = cm.group(1)
    params = {}
    for pm in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', cm.group(2)):
        params[pm.group(1)] = pm.group(2)
    return name, params


def parse_final_answer(text: str) -> Optional[str]:
    m = FINAL_ANSWER_PATTERN.search(text)
    return m.group(1).strip() if m else None


def validate_trajectory(messages: list[dict]) -> tuple[bool, str]:
    """Validate a ReAct trajectory in ChatML format.

    Returns (is_valid, error_message).
    """
    if not messages:
        return False, "Empty message list"

    has_user = any(m["role"] == "user" for m in messages)
    if not has_user:
        return False, "No user message found"

    has_final = any("Final Answer:" in m["content"] for m in messages if m["role"] == "assistant")
    if not has_final:
        return False, "No Final Answer found in trajectory"

    for msg in messages:
        if msg["role"] == "assistant":
            content = msg["content"]
            if "Thought:" not in content and "Action:" not in content and "Final Answer:" not in content:
                return False, f"Assistant message missing Thought/Action/Final Answer: {content[:80]}..."
            if "Action:" in content:
                if parse_action(content) is None:
                    return False, f"Malformed Action in: {content[:80]}..."

    return True, ""


def messages_to_trajectory(messages: list[dict]) -> ReactTrajectory:
    """Convert ChatML messages to a ReactTrajectory with metadata."""
    traj = ReactTrajectory(messages=[ReactMessage(role=m["role"], content=m["content"]) for m in messages])
    has_error = False
    for m in messages:
        if m["role"] == "assistant":
            if "Action:" in m["content"]:
                traj.tool_call_count += 1
        if m["role"] == "tool":
            try:
                resp = json.loads(m["content"])
                if "error" in resp:
                    has_error = True
            except json.JSONDecodeError:
                pass
    # error recovery = had an error AND still got to Final Answer
    traj.has_error_recovery = has_error and any(
        "Final Answer:" in m["content"] for m in messages if m["role"] == "assistant"
    )
    return traj


def format_trajectory_for_training(
    user_query: str,
    thoughts_and_actions: list[dict],
    system_prompt: str | None = None,
) -> list[dict]:
    """Build a ChatML trajectory from structured steps.

    thoughts_and_actions: list of dicts like:
      [{"thought": "...", "action": "get_weather(city=\"Beijing\")"},
       {"thought": "...", "final_answer": "Today is sunny."}]

    An optional "observation" key provides the tool response.
    An optional "observation_error" key provides a tool error response.
    """
    messages: list[dict] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_query})

    for step in thoughts_and_actions:
        thought = step["thought"]
        if "action" in step:
            messages.append({"role": "assistant", "content": f"Thought: {thought}\nAction: {step['action']}\n"})
        if "observation" in step:
            messages.append({"role": "tool", "content": step["observation"]})
        if "observation_error" in step:
            messages.append({"role": "tool", "content": json.dumps(step["observation_error"], ensure_ascii=False)})
        if "final_answer" in step:
            messages.append({"role": "assistant", "content": f"Thought: {thought}\nFinal Answer: {step['final_answer']}\n"})

    return messages
