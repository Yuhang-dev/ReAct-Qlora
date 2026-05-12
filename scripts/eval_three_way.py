"""Three-way comparison: Baseline B / Baseline A / FT ReAct.

Baseline B: Instruct + no ReAct prompt → direct tool call
Baseline A: Instruct + strong ReAct prompt + 2 Few-shot → ReAct loop
FT ReAct:   Fine-tuned model + ReAct prompt → ReAct loop
"""

import json, re, time
from collections import defaultdict

import torch, unsloth
from unsloth import FastLanguageModel
from peft import PeftModel

# ── Test cases ────────────────────────────────────────────
# single-tool, multi-tool, error-recovery
TEST_CASES = [
    # === Single tool (10) ===
    {"q": "北京今天天气怎么样？", "category": "single", "expected_tool": "get_weather"},
    {"q": "苹果公司现在的股价是多少？", "category": "single", "expected_tool": "get_stock_price"},
    {"q": "帮我算一下 3.14 * 2.5 等于多少", "category": "single", "expected_tool": "calculate"},
    {"q": "现在北京时间几点了？", "category": "single", "expected_tool": "get_time"},
    {"q": "搜索一下Python异步编程的最新框架", "category": "single", "expected_tool": "search_web"},
    {"q": "上海明天会下雨吗？需要带伞吗？", "category": "single", "expected_tool": "get_weather"},
    {"q": "Tesla现在的股价是多少？", "category": "single", "expected_tool": "get_stock_price"},
    {"q": "帮我算一下 128 * 0.85 等于多少", "category": "single", "expected_tool": "calculate"},
    {"q": "伦敦现在是几点？", "category": "single", "expected_tool": "get_time"},
    {"q": "鲁迅的代表作有哪些？", "category": "single", "expected_tool": "search_web"},

    # === Multi-tool (10) ===
    {"q": "查一下微软的股价，顺便看看西雅图的天气", "category": "multi", "expected_tool": "get_stock_price"},
    {"q": "帮我算一下 250+128 然后搜索这个数字有什么特殊意义", "category": "multi", "expected_tool": "calculate"},
    {"q": "东京的天气和现在的时间分别是什么？", "category": "multi", "expected_tool": "get_weather"},
    {"q": "英伟达和特斯拉哪个涨得更多？", "category": "multi", "expected_tool": "get_stock_price"},
    {"q": "北京、上海、广州三个城市的天气分别是怎样的？", "category": "multi", "expected_tool": "get_weather"},
    {"q": "查一下Meta的股价，顺便搜索一下它最近有什么重大新闻", "category": "multi", "expected_tool": "get_stock_price"},
    {"q": "帮我算一下我的三只股票(S=100股*AAPL + 50股*TSLA)的总价值", "category": "multi", "expected_tool": "get_stock_price"},
    {"q": "搜索深度学习框架对比，顺便看看成都天气", "category": "multi", "expected_tool": "search_web"},
    {"q": "苹果公司总部在哪个城市？那个城市现在天气如何？", "category": "multi", "expected_tool": "search_web"},
    {"q": "计算 (1234+5678)*0.9 的结果，然后查一下这个数字是否在数学中有特殊意义", "category": "multi", "expected_tool": "calculate"},

    # === Error recovery (10) ===
    {"q": "帮我查一下火星上的天气怎么样？", "category": "error", "expected_tool": "get_weather"},
    {"q": "帮我查一下XYZ公司(股票代码不存在)的股价", "category": "error", "expected_tool": "get_stock_price"},
    {"q": "帮我搜一下这个不存在的关键词: zxcvbnm123456", "category": "error", "expected_tool": "search_web"},
    {"q": "帮我算一下 100 / 0 等于多少", "category": "error", "expected_tool": "calculate"},
    {"q": "帮我查一下亚特兰蒂斯的当前时间", "category": "error", "expected_tool": "get_time"},
    {"q": "冥王星表面的天气如何？", "category": "error", "expected_tool": "get_weather"},
    {"q": "帮我查一下PONY公司(不存在)的股价", "category": "error", "expected_tool": "get_stock_price"},
    {"q": "搜索一个不可能存在的词汇: qwertyasdfghzxcvbn", "category": "error", "expected_tool": "search_web"},
    {"q": "帮我算一下 abc / xyz 等于多少", "category": "error", "expected_tool": "calculate"},
    {"q": "帮我查一下亚特兰蒂斯城的时间", "category": "error", "expected_tool": "get_time"},
]


# ── Shared tool executor ──────────────────────────────────
def execute_action(action_text: str) -> str:
    from src.demo.react_demo import TOOLS
    m = re.match(r"(\w+)\((.+)\)", action_text.strip())
    if not m:
        return json.dumps({"error": f"无法解析: {action_text}"}, ensure_ascii=False)
    tool_name, params_str = m.group(1), m.group(2)
    if tool_name not in TOOLS:
        return json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False)
    params = {}
    for pm in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', params_str):
        params[pm.group(1)] = pm.group(2)
    try:
        result = TOOLS[tool_name]["func"](**params)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ── Prompts ───────────────────────────────────────────────
REACT_SYSTEM = """You are a helpful assistant with access to tools.

Tools:
- get_weather(city="城市名"): 查询天气
- get_stock_price(symbol="股票代码"): 查询股价
- search_web(query="关键词"): 搜索网络
- calculate(expression="数学表达式"): 计算
- get_time(city="城市名"): 查询时间

Reply format:
Thought: <reasoning about what to do next>
Action: <tool_name(param="value")>

After receiving a tool response, continue with:
Thought: <further reasoning>
Action: <next tool> or Final Answer: <complete answer>"""

NO_REACT_SYSTEM = """You are a helpful assistant. You can call these tools:
- get_weather(city="城市名"), get_stock_price(symbol="股票代码"), search_web(query="关键词"), calculate(expression="数学表达式"), get_time(city="城市名")
Call tools directly. You will receive results."""

FEW_SHOT_EXAMPLE = """
Example 1:
User: 北京今天天气怎么样？
Assistant: Thought: 需要查询北京天气。
Action: get_weather(city="北京")

Observation: {"city":"北京","temperature":22,"condition":"晴"}
Assistant: Thought: 数据已获取。
Final Answer: 北京今天晴，22°C。

Example 2:
User: 查一下苹果股价
Assistant: Thought: 苹果代码是AAPL。
Action: get_stock_price(symbol="AAPL")

Observation: {"symbol":"AAPL","price":185.5,"change_pct":2.3}
Assistant: Thought: 股价已获取。
Final Answer: 苹果(AAPL)当前$185.5，涨2.3%。
"""


# ── Run one ReAct loop ────────────────────────────────────
def run_react_loop(model, tokenizer, user_query: str, system_prompt: str,
                   max_turns: int = 6) -> dict:
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    obs_hallucination = False
    format_ok = True
    tool_calls = []
    final_answer = None
    error_encountered = False
    error_recovered = False

    for turn in range(max_turns):
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

        # Check for hallucinated Observation
        if "Observation:" in raw:
            obs_hallucination = True

        thought = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|\Z)", raw, re.DOTALL)
        action = re.search(r"Action:\s*(.+?)(?=\n|$)", raw)
        final = re.search(r"Final Answer:\s*(.+?)$", raw, re.DOTALL)

        thought_str = thought.group(1).strip() if thought else ""
        action_str = action.group(1).strip() if action else ""
        final_str = final.group(1).strip() if final else ""

        if not thought_str and not action_str and not final_str:
            format_ok = False
            break

        conversation.append({"role": "assistant", "content": raw})

        if final_str:
            final_answer = final_str
            break

        if action_str:
            tool_calls.append(action_str)
            obs = execute_action(action_str)
            conversation.append({"role": "tool", "content": obs})
            if '"error"' in obs:
                error_encountered = True
            elif error_encountered and not error_recovered:
                # If we got an error before and now have a non-error observation, mark recovered
                error_recovered = True
        else:
            break

    return {
        "obs_hall": obs_hallucination,
        "format_ok": format_ok and "Thought:" in raw,
        "tool_calls": len(tool_calls),
        "final_answer": final_answer,
        "error_encountered": error_encountered,
        "error_recovered": error_recovered,
    }


def run_direct_tool_call(model, tokenizer, user_query: str,
                         max_turns: int = 6) -> dict:
    """Baseline B: model can call tools directly, no ReAct format enforced."""
    conversation = [
        {"role": "system", "content": NO_REACT_SYSTEM},
        {"role": "user", "content": user_query},
    ]
    tool_calls = 0
    final_answer = None

    for turn in range(max_turns):
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

        conversation.append({"role": "assistant", "content": raw})

        # Try to extract function call in various formats
        action = re.search(r"Action:\s*(.+?)(?:\n|$)", raw)
        tool_call = re.search(r'get_weather\(|get_stock_price\(|search_web\(|calculate\(|get_time\(', raw)
        final = re.search(r"Final Answer:\s*(.+?)$", raw, re.DOTALL)

        if action:
            obs = execute_action(action.group(1).strip())
            conversation.append({"role": "tool", "content": obs})
            tool_calls += 1
        elif tool_call and not action:
            # Direct tool call without Action: prefix
            m = re.search(r'(get_\w+\([^)]+\)|search_\w+\([^)]+\)|calculate\(.+\))', raw)
            if m:
                obs = execute_action(m.group(1))
                conversation.append({"role": "tool", "content": obs})
                tool_calls += 1
            elif final:
                final_answer = final.group(1).strip()
                break
            elif tool_calls > 0:
                # Assume the answer is the response
                final_answer = raw[:300]
                break
        elif final:
            final_answer = final.group(1).strip()
            break
        elif not tool_call:
            # Model gave direct answer without tool
            final_answer = raw[:300]
            break

    return {
        "obs_hall": False,
        "format_ok": True,
        "tool_calls": tool_calls,
        "final_answer": final_answer,
        "error_encountered": False,
        "error_recovered": False,
    }


# ── LLM Judge ─────────────────────────────────────────────
def judge(question: str, answer: str | None, category: str) -> bool:
    """Use heuristic + lightweight rules for quick evaluation."""
    if not answer:
        return False

    # Quick heuristic checks per category
    if category == "single":
        # Should have a concrete answer (not just a tool call)
        return len(answer) > 10 and "Action:" not in answer
    elif category == "multi":
        return len(answer) > 20
    elif category == "error":
        # Should acknowledge limitation in some way
        error_keywords = ["无法", "不支持", "抱歉", "错误", "不存在", "sorry", "cannot", "error"]
        return any(kw in answer.lower() for kw in error_keywords)
    return len(answer) > 10


# ── Main ──────────────────────────────────────────────────
def main():
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        max_seq_length=2048, dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Load FT adapter
    ft_model = PeftModel.from_pretrained(
        model, "outputs/react-qwen2.5-7b-qlora_20260508_1810/checkpoint-1000",
    )
    FastLanguageModel.for_inference(ft_model)

    # Prepare Baseline A prompt
    react_system_with_fewshot = REACT_SYSTEM + "\n\nHere are examples:\n" + FEW_SHOT_EXAMPLE

    results = defaultdict(lambda: defaultdict(list))
    all_results = []

    for i, tc in enumerate(TEST_CASES):
        q, cat = tc["q"], tc["category"]
        print(f"\n[{i+1:2d}/30] ({cat}) {q[:50]}")

        for mname, mdl, sys_prompt in [
            ("Baseline_B", model, NO_REACT_SYSTEM),
            ("Baseline_A", model, react_system_with_fewshot),
            ("FT_ReAct", ft_model, REACT_SYSTEM),
        ]:
            t0 = time.time()
            if mname == "Baseline_B":
                r = run_direct_tool_call(mdl, tokenizer, q)
            else:
                r = run_react_loop(mdl, tokenizer, q, sys_prompt)

            elapsed = time.time() - t0
            passed = judge(q, r["final_answer"], cat)

            results[mname]["success"].append(passed)
            results[mname]["obs_hall"].append(r["obs_hall"])
            results[mname]["format_ok"].append(r["format_ok"])
            results[mname]["tool_calls"].append(r["tool_calls"])
            results[mname]["time"].append(elapsed)

            all_results.append({
                "question": q, "category": cat, "model": mname,
                "passed": passed, **r,
            })

            icon = "V" if passed else "X"
            oh = "H" if r["obs_hall"] else "-"
            print(f"  {mname:<12}: {icon} | tool_calls={r['tool_calls']} | obs_hall={oh} | {elapsed:.1f}s")

        torch.cuda.empty_cache()

    # ── Summary ────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    header = f"{'Model':<15} {'Success':>10} {'ObsHall':>10} {'Format':>10} {'Calls':>8}"
    print(header)
    print(sep)
    for mname in ["Baseline_B", "Baseline_A", "FT_ReAct"]:
        sr = sum(results[mname]["success"]) / 30
        oh = sum(results[mname]["obs_hall"]) / 30
        fmt = sum(results[mname]["format_ok"]) / 30
        calls = sum(results[mname]["tool_calls"]) / 30
        print(f"{mname:<15} {sr:>9.0%} {oh:>9.0%} {fmt:>9.0%} {calls:>7.1f}")

    # Per category
    print(f"\n--- By Category ---")
    for cat in ["single", "multi", "error"]:
        print(f"\n[{cat}]")
        for mname in ["Baseline_B", "Baseline_A", "FT_ReAct"]:
            cat_results = [r for r in all_results if r["model"] == mname and r["category"] == cat]
            sr = sum(r["passed"] for r in cat_results) / len(cat_results)
            print(f"  {mname}: {sr:.0%}")

    # Save
    out = {
        "summary": {m: {
            "success_rate": sum(results[m]["success"]) / 30,
            "obs_hall_rate": sum(results[m]["obs_hall"]) / 30,
            "format_rate": sum(results[m]["format_ok"]) / 30,
            "avg_tool_calls": sum(results[m]["tool_calls"]) / 30,
        } for m in ["Baseline_B", "Baseline_A", "FT_ReAct"]},
        "details": all_results,
    }
    with open("outputs/three_way_eval.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to outputs/three_way_eval.json")


if __name__ == "__main__":
    main()
