"""Unit tests for ReAct data format validation."""

import json
from src.data.react_format import (
    parse_action,
    parse_final_answer,
    parse_thought,
    validate_trajectory,
    format_trajectory_for_training,
)


def test_parse_thought():
    text = "Thought: 我需要查询天气数据。\nAction: get_weather(city=\"北京\")\n"
    assert parse_thought(text) == "我需要查询天气数据。"


def test_parse_action():
    text = "Thought: Let me check.\nAction: search_web(query=\"Python async\", num_results=\"5\")\n"
    name, params = parse_action(text)
    assert name == "search_web"
    assert params == {"query": "Python async", "num_results": "5"}


def test_parse_final_answer():
    text = "Thought: I have enough info.\nFinal Answer: 今天北京晴天，适合运动。\n"
    assert parse_final_answer(text) == "今天北京晴天，适合运动。"


def test_validate_valid_trajectory():
    msgs = [
        {"role": "user", "content": "天气怎么样？"},
        {"role": "assistant", "content": "Thought: 需要查询。\nAction: get_weather(city=\"北京\")\n"},
        {"role": "tool", "content": json.dumps({"temperature": 22, "condition": "晴"})},
        {"role": "assistant", "content": "Thought: 数据已获取。\nFinal Answer: 北京晴，22°C。\n"},
    ]
    ok, err = validate_trajectory(msgs)
    assert ok, f"Expected valid, got: {err}"


def test_validate_missing_final_answer():
    msgs = [
        {"role": "user", "content": "天气怎么样？"},
        {"role": "assistant", "content": "Thought: 需要查询。\nAction: get_weather(city=\"北京\")\n"},
    ]
    ok, err = validate_trajectory(msgs)
    assert not ok


def test_validate_malformed_action():
    msgs = [
        {"role": "user", "content": "天气怎么样？"},
        {"role": "assistant", "content": "Thought: 需要查询。\nAction: not a valid call\n"},
        {"role": "assistant", "content": "Thought: 完成。\nFinal Answer: 回答了。\n"},
    ]
    ok, err = validate_trajectory(msgs)
    assert not ok


def test_format_trajectory_for_training():
    steps = [
        {"thought": "需要查询天气", "action": "get_weather(city=\"北京\")", "observation": '{"temperature": 22}'},
        {"thought": "数据完整，可以回答了", "final_answer": "北京22°C"},
    ]
    msgs = format_trajectory_for_training("天气?", steps)
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert "Action:" in msgs[1]["content"]
    assert msgs[2]["role"] == "tool"
    assert "Final Answer:" in msgs[3]["content"]


def test_validate_empty_trajectory():
    ok, err = validate_trajectory([])
    assert not ok
