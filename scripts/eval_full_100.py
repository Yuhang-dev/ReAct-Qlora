"""Comprehensive 100-case three-way evaluation with realistic tool validation."""

import json, re, time, random
from collections import defaultdict
from pathlib import Path

import torch, unsloth
from unsloth import FastLanguageModel
from peft import PeftModel


# ═══════════════════════════════════════════════════════════
# IMPROVED MOCK TOOLS — with real validation
# ═══════════════════════════════════════════════════════════

# Real stock symbols with company names
REAL_STOCKS = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.", "NVDA": "NVIDIA Corp.", "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.", "TSM": "Taiwan Semiconductor", "BABA": "Alibaba Group",
    "0700.HK": "Tencent Holdings", "INTC": "Intel Corp.", "AMD": "Advanced Micro Devices",
    "NFLX": "Netflix Inc.", "DIS": "Walt Disney Co.", "JPM": "JPMorgan Chase",
    "V": "Visa Inc.", "WMT": "Walmart Inc.", "JNJ": "Johnson & Johnson",
    "MA": "Mastercard Inc.", "PG": "Procter & Gamble", "KO": "Coca-Cola Co.",
    "PEP": "PepsiCo Inc.", "ADBE": "Adobe Inc.", "CRM": "Salesforce Inc.",
    "ORCL": "Oracle Corp.", "IBM": "IBM Corp.", "CSCO": "Cisco Systems",
    "BA": "Boeing Co.", "CAT": "Caterpillar Inc.", "XOM": "Exxon Mobil",
}

# Real cities (at least plausibly valid for weather)
VALID_CITIES = {
    "北京", "上海", "广州", "深圳", "杭州", "成都", "南京", "武汉", "重庆", "西安",
    "东京", "大阪", "首尔", "釜山",
    "纽约", "洛杉矶", "芝加哥", "旧金山", "西雅图", "波士顿", "华盛顿", "迈阿密",
    "伦敦", "巴黎", "柏林", "罗马", "马德里", "莫斯科", "阿姆斯特丹", "苏黎世",
    "悉尼", "墨尔本", "新加坡", "迪拜", "多伦多", "温哥华", "孟买", "曼谷",
}

INVALID_CITIES = {"火星", "木星", "亚特兰蒂斯", "Atlantis", "XYZ城", "Moon City", "太阳城"}
INVALID_STOCKS = {"XYZ", "PONY", "FAKE", "ABCDE", "NOTREAL", "ZZZZZ", "PEGASUS", "UNICORN"}


def tool_get_weather(city: str, date: str = "today") -> dict:
    if city in INVALID_CITIES or (city not in VALID_CITIES and not any(c in VALID_CITIES for c in [city])):
        return {"error": "invalid_city", "message": f"'{city}' not found in weather database. Only Earth cities are supported."}
    seed = hash(f"w_{city}") & 0x7FFFFFFF
    rng = random.Random(seed)
    return {
        "city": city, "temperature": rng.randint(-5, 40),
        "condition": rng.choice(["晴天", "多云", "阴天", "小雨", "中雨", "雷阵雨", "雾", "霾", "晴转多云"]),
        "humidity": rng.randint(20, 95),
    }


def tool_get_stock_price(symbol: str) -> dict:
    symbol_upper = symbol.upper()
    if symbol_upper in INVALID_STOCKS:
        return {"error": "invalid_symbol", "message": f"Stock symbol '{symbol}' not found on any exchange."}
    if symbol_upper not in REAL_STOCKS:
        return {"error": "unknown_symbol", "message": f"'{symbol}' is not a recognized stock symbol. Please verify the ticker."}
    seed = hash(f"s_{symbol_upper}") & 0x7FFFFFFF
    rng = random.Random(seed)
    return {
        "symbol": symbol_upper, "company": REAL_STOCKS[symbol_upper],
        "price": round(rng.uniform(20, 600), 2),
        "change_pct": round(rng.uniform(-6, 6), 2),
        "currency": "USD" if symbol_upper != "0700.HK" else "HKD",
    }


def tool_search_web(query: str, num_results: int = 3) -> dict:
    # Return empty for obviously garbage queries
    if len(query) < 3 or re.match(r'^[a-z]{20,}$', query) or re.match(r'^[a-z]+$', query) and len(query) > 15:
        return {"results": [], "count": 0, "message": "No results found for this query."}
    seed = hash(f"sw_{query}") & 0x7FFFFFFF
    rng = random.Random(seed)
    return {"results": [
        {"title": f"{query} - 百科", "snippet": f"关于「{query}」的详细条目。", "url": "https://wiki.example.com"},
        {"title": f"{query} 最新资讯", "snippet": f"「{query}」相关的最新报道。", "url": "https://news.example.com"},
        {"title": f"{query} 官方文档", "snippet": f"「{query}」的教程和文档。", "url": "https://docs.example.com"},
    ][:num_results], "count": min(num_results, 3)}


def tool_calculate(expression: str) -> dict:
    try:
        # Check for variable-like invalid expressions
        if re.search(r'[a-zA-Z]{2,}', expression):
            return {"error": "invalid_expression", "message": f"Cannot evaluate '{expression}': contains non-numeric variables."}
        result = eval(expression, {"__builtins__": {}}, {})
        if isinstance(result, complex):
            return {"error": "complex_result", "message": "Result is a complex number."}
        return {"result": result}
    except ZeroDivisionError:
        return {"error": "division_by_zero", "message": "Division by zero is undefined."}
    except Exception as e:
        return {"error": "eval_error", "message": str(e)}


def tool_get_time(city: str) -> dict:
    if city in INVALID_CITIES or (city not in VALID_CITIES and len(city) > 15):
        return {"error": "invalid_city", "message": f"'{city}' not found in timezone database."}
    seed = hash(f"t_{city}") & 0x7FFFFFFF
    rng = random.Random(seed)
    return {"city": city, "time": f"{rng.randint(0,23):02d}:{rng.randint(0,59):02d}", "timezone": f"UTC{rng.choice(['+8','-5','+0','+1','+9','-8'])}"}


def execute_action(action_text: str) -> str:
    m = re.match(r"(\w+)\((.+)\)", action_text.strip())
    if not m:
        return json.dumps({"error": "parse_error", "message": f"Cannot parse: {action_text}"}, ensure_ascii=False)
    name, params_str = m.group(1), m.group(2)
    tools = {"get_weather": tool_get_weather, "get_stock_price": tool_get_stock_price,
             "search_web": tool_search_web, "calculate": tool_calculate, "get_time": tool_get_time}
    if name not in tools:
        return json.dumps({"error": "unknown_tool", "message": f"Tool '{name}' not found."}, ensure_ascii=False)
    params = {}
    for pm in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', params_str):
        params[pm.group(1)] = pm.group(2)
    try:
        result = tools[name](**params)
        return json.dumps(result, ensure_ascii=False)
    except TypeError as e:
        return json.dumps({"error": "param_error", "message": str(e)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": "exec_error", "message": str(e)}, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════
# 100 TEST CASES
# ═══════════════════════════════════════════════════════════

def generate_100_tests():
    cases = []

    # === Single tool (35) ===
    weather_qs = [
        "北京今天天气怎么样？", "上海会下雨吗？", "深圳现在温度是多少？",
        "杭州天气如何？适合出游吗？", "成都今天冷吗？", "南京天气如何？",
        "东京明天天气怎么样？", "纽约这周末天气如何？", "伦敦今天什么天气？",
        "悉尼现在天气如何？",
    ]
    for q in weather_qs:
        cases.append({"q": q, "cat": "single", "tool": "get_weather", "difficulty": "easy"})

    stock_qs = [
        "苹果公司现在的股价是多少？", "Tesla股价如何？", "微软股价多少钱？",
        "英伟达今天的股价？", "亚马逊股价是多少？", "Meta的股价？",
        "台积电股票多少钱一股？", "腾讯股价多少？", "可口可乐股价？",
        "Netflix的股价？",
    ]
    for q in stock_qs:
        cases.append({"q": q, "cat": "single", "tool": "get_stock_price", "difficulty": "easy"})

    calc_qs = [
        "帮我算一下 3.14 * 2.5", "计算 128*0.85", "156+244等于多少？",
        "1000除以8等于多少？", "7.5*6.2等于多少？",
    ]
    for q in calc_qs:
        cases.append({"q": q, "cat": "single", "tool": "calculate", "difficulty": "easy"})

    time_qs = [
        "北京时间现在几点？", "伦敦现在几点了？", "东京时间？",
        "纽约现在几点？", "巴黎现在什么时间？",
    ]
    for q in time_qs:
        cases.append({"q": q, "cat": "single", "tool": "get_time", "difficulty": "easy"})

    search_qs = [
        "搜索一下Python异步编程的最佳实践", "鲁迅的代表作有哪些？",
        "2024年诺贝尔物理学奖得主是谁？", "量子计算最新进展",
        "深度学习框架对比",
    ]
    for q in search_qs:
        cases.append({"q": q, "cat": "single", "tool": "search_web", "difficulty": "easy"})

    # === Multi-tool (35) ===
    multi_qs = [
        ("查苹果股价，顺便看纽约天气", ["get_stock_price", "get_weather"]),
        ("搜索深度学习框架，顺便看成都天气", ["search_web", "get_weather"]),
        ("微软股价和西雅图天气", ["get_stock_price", "get_weather"]),
        ("算一下 (350+650)*1.2 然后搜索结果的意义", ["calculate", "search_web"]),
        ("东京天气和时间分别是什么？", ["get_weather", "get_time"]),
        ("特斯拉和英伟达哪个涨得多？", ["get_stock_price", "get_stock_price"]),
        ("北京、上海、广州三地天气对比", ["get_weather", "get_weather", "get_weather"]),
        ("帮我查Meta股价和最近新闻", ["get_stock_price", "search_web"]),
        ("计算一下我100股AAPL+50股TSLA的总价值", ["get_stock_price", "get_stock_price", "calculate"]),
        ("搜索Kubernetes部署方案并查看旧金山天气", ["search_web", "get_weather"]),
        ("杭州和深圳的天气和时间", ["get_weather", "get_weather", "get_time", "get_time"]),
        ("阿里股价多少？顺便查最近新闻", ["get_stock_price", "search_web"]),
        ("算一下15.8*23.4然后搜索结果在数学中的意义", ["calculate", "search_web"]),
        ("苹果总部在哪？那个城市天气如何？", ["search_web", "get_weather"]),
        ("英伟达股价和硅谷（圣何塞）天气", ["get_stock_price", "get_weather"]),
        ("腾讯和阿里谁股价更高？", ["get_stock_price", "get_stock_price"]),
        ("帮我查一下伦敦、巴黎、柏林三地的天气", ["get_weather", "get_weather", "get_weather"]),
        ("搜索React最新版本 顺便看看首尔天气", ["search_web", "get_weather"]),
        ("AMD和Intel哪个股票涨得更多？", ["get_stock_price", "get_stock_price"]),
        ("计算(1234+5678)*0.9然后搜索结果是否在数学中有特殊意义", ["calculate", "search_web"]),
        ("迪士尼股价和所在地(洛杉矶)天气", ["get_stock_price", "get_weather"]),
        ("可口可乐和百事谁股价更高？", ["get_stock_price", "get_stock_price"]),
        ("查一下阿姆斯特丹时间并搜索当地天气", ["get_time", "get_weather"]),
        ("搜索Golang最新版本并查看柏林天气", ["search_web", "get_weather"]),
        ("算一下我的投资组合: 200股NVDA+100股AMD总价值", ["get_stock_price", "get_stock_price", "calculate"]),
        ("搜索2024年AI最新突破，顺便看旧金山天气", ["search_web", "get_weather"]),
        ("Adobe和Salesforce哪个股价更高？", ["get_stock_price", "get_stock_price"]),
        ("查东京时间，并看东京天气", ["get_time", "get_weather"]),
        ("墨尔本现在是几点？什么天气？", ["get_time", "get_weather"]),
        ("搜索Rust语言特性，看看柏林天气和时间", ["search_web", "get_weather", "get_time"]),
        ("JPMorgan和Visa哪个股价高？", ["get_stock_price", "get_stock_price"]),
        ("查杭州时间并搜索西湖天气", ["get_time", "get_weather"]),
        ("Boeing和Caterpillar哪个股价高？", ["get_stock_price", "get_stock_price"]),
        ("IBM股价多少？顺便看看纽约天气和时间", ["get_stock_price", "get_weather", "get_time"]),
        ("宝洁和强生哪个股价更高？", ["get_stock_price", "get_stock_price"]),
    ]
    for q, tools in multi_qs:
        cases.append({"q": q, "cat": "multi", "tool": tools[0], "tools": tools, "difficulty": "medium"})

    # === Error recovery (30) ===
    error_qs = [
        ("帮我查一下火星上的天气怎么样？", "get_weather", "invalid_city"),
        ("帮我查一下XYZ公司的股价", "get_stock_price", "invalid_stock"),
        ("帮我查一下亚特兰蒂斯的天气", "get_weather", "invalid_city"),
        ("帮我查PONY公司的股票", "get_stock_price", "invalid_stock"),
        ("帮我算一下 abc / xyz 等于多少", "calculate", "invalid_expr"),
        ("帮我查一下木星上的天气", "get_weather", "invalid_city"),
        ("帮我查FAKE公司的股价", "get_stock_price", "invalid_stock"),
        ("帮我算一下 100 / 0", "calculate", "div_by_zero"),
        ("亚特兰蒂斯现在几点了？", "get_time", "invalid_city"),
        ("搜索 qwertyasdfghzxcvbn", "search_web", "garbage_query"),
        ("帮我查PEGASUS股票代码的股价", "get_stock_price", "invalid_stock"),
        ("冥王星表面的天气如何？", "get_weather", "invalid_city"),
        ("帮我查NOTREAL公司股价", "get_stock_price", "invalid_stock"),
        ("帮我算 xyz * 123", "calculate", "invalid_expr"),
        ("太阳城现在是几点？", "get_time", "invalid_city"),
        ("帮我查ZZZZZ公司股价", "get_stock_price", "invalid_stock"),
        ("搜索一个完全不存在的词汇: asdfghjklqwertyuiop", "search_web", "garbage_query"),
        ("帮我查月球上的天气", "get_weather", "invalid_city"),
        ("帮我查UNICORN公司股价", "get_stock_price", "invalid_stock"),
        ("算 5/0 等于多少？", "calculate", "div_by_zero"),
        ("Moon City现在是几点？", "get_time", "invalid_city"),
        ("帮我搜 zzzzzzzzzzzzzzzzzzzzzz", "search_web", "garbage_query"),
        ("帮我查ABCDE的股票", "get_stock_price", "invalid_stock"),
        ("金星上的天气如何？", "get_weather", "invalid_city"),
        ("帮我算 3 + abc", "calculate", "invalid_expr"),
        ("XXX城天气怎么样？", "get_weather", "invalid_city"),
        ("查一下NOTREAL股票", "get_stock_price", "invalid_stock"),
        ("帮我算 5/(2-2)", "calculate", "div_by_zero"),
        ("亚特兰蒂斯城时间", "get_time", "invalid_city"),
        ("搜索 aaaaaaaaaaaaaaaaaaaaaa", "search_web", "garbage_query"),
    ]
    for q, tool, _ in error_qs:
        cases.append({"q": q, "cat": "error", "tool": tool, "difficulty": "hard"})

    return cases


# ═══════════════════════════════════════════════════════════
# MODEL & INFERENCE
# ═══════════════════════════════════════════════════════════

REACT_SYSTEM = """You are a helpful assistant with tools: get_weather(city="name"), get_stock_price(symbol="TICKER"), search_web(query="keywords"), calculate(expression="math"), get_time(city="name").

Reply format:
Thought: <reasoning>
Action: <tool(param="value")>
After tool response: Thought: <reasoning> Final Answer: <answer>"""

NO_REACT_SYSTEM = """You are a helpful assistant with tools: get_weather, get_stock_price, search_web, calculate, get_time. Call tools directly to answer questions."""

FEWSHOT_SYSTEM = REACT_SYSTEM + """

Examples:
User: 北京天气？
Assistant: Thought: 查天气 Action: get_weather(city="北京")
Observation: {"temperature":22} Assistant: Thought: 已获取 Final Answer: 22°C
User: 苹果股价？
Assistant: Thought: 代码AAPL Action: get_stock_price(symbol="AAPL")
Observation: {"price":185.5} Assistant: Thought: 已获取 Final Answer: $185.5"""


def run_agent(model, tokenizer, query: str, system: str, is_direct: bool = False, max_turns: int = 6) -> dict:
    conversation = [{"role": "system", "content": system}, {"role": "user", "content": query}]
    obs_hall_count = 0
    tool_calls = []
    final_answer = None
    error_count = 0
    recovered = False

    for turn in range(max_turns):
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, temperature=0.1, top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id)
        raw = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        if "Observation:" in raw and not is_direct:
            obs_hall_count += 1

        thought = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|\Z)", raw, re.DOTALL)
        action = re.search(r"Action:\s*(.+?)(?:\n|$)", raw)
        final = re.search(r"Final Answer:\s*(.+?)$", raw, re.DOTALL)

        thought_str = thought.group(1).strip() if thought else ""
        action_str = action.group(1).strip() if action else ""
        final_str = final.group(1).strip() if final else ""

        if is_direct and not action_str and not final_str and not thought_str:
            # Try to extract direct tool call
            tool_match = re.search(r'(get_weather|get_stock_price|search_web|calculate|get_time)\s*\([^)]+\)', raw)
            if tool_match:
                action_str = tool_match.group(0)

        conversation.append({"role": "assistant", "content": raw})

        if final_str:
            final_answer = final_str
            break

        if action_str:
            tool_calls.append(action_str)
            obs = execute_action(action_str)
            if '"error"' in obs:
                error_count += 1
            elif error_count > 0:
                recovered = True
            conversation.append({"role": "tool", "content": obs})
        elif is_direct:
            final_answer = raw[:300]
            break
        else:
            break

    return {
        "tool_calls": len(tool_calls),
        "final_answer": final_answer,
        "obs_hall": obs_hall_count,
        "errors": error_count,
        "recovered": recovered,
        "has_answer": final_answer is not None and len(final_answer) > 5,
    }


def judge_answer(query: str, answer: str | None, category: str, error_count: int, recovered: bool) -> bool:
    if not answer:
        return False
    ans = answer.lower()

    if category == "single":
        return len(answer) > 10 and "action:" not in ans and "thought:" not in ans

    elif category == "multi":
        return len(answer) > 15 and "action:" not in ans

    elif category == "error":
        error_signals = ["无法", "不支持", "抱歉", "错误", "不存在", "未定义", "无穷", "sorry", "cannot", "error", "invalid", "not found", "no results", "零", "undefined", "没有意义"]
        acknowledges_error = any(s in ans for s in error_signals)
        return acknowledges_error and len(answer) > 5

    return len(answer) > 10


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit", max_seq_length=2048, dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    ft_model = PeftModel.from_pretrained(
        model, "outputs/react-qwen2.5-7b-qlora_20260508_1810/checkpoint-1000",
    )
    FastLanguageModel.for_inference(ft_model)

    cases = generate_100_tests()
    print(f"Test cases: {len(cases)}")
    print(f"  Single: {sum(1 for c in cases if c['cat']=='single')}")
    print(f"  Multi:  {sum(1 for c in cases if c['cat']=='multi')}")
    print(f"  Error:  {sum(1 for c in cases if c['cat']=='error')}")

    all_results = []
    metrics = defaultdict(lambda: defaultdict(list))

    for i, tc in enumerate(cases):
        q, cat = tc["q"], tc["cat"]
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(cases)}...")

        for mname, mdl, sys_prompt, is_direct in [
            ("Baseline_B", model, NO_REACT_SYSTEM, True),
            ("Baseline_A", model, FEWSHOT_SYSTEM, False),
            ("FT_ReAct", ft_model, REACT_SYSTEM, False),
        ]:
            t0 = time.time()
            r = run_agent(mdl, tokenizer, q, sys_prompt, is_direct)
            r["time"] = time.time() - t0
            r["passed"] = judge_answer(q, r["final_answer"], cat, r["errors"], r["recovered"])
            r["question"] = q
            r["category"] = cat
            r["model"] = mname

            for k in ["passed", "tool_calls", "obs_hall", "errors", "time", "has_answer"]:
                metrics[mname][k].append(r[k])
            if cat == "error":
                metrics[mname]["recovered_count"].append(int(r["recovered"]))

            all_results.append(r)
            torch.cuda.empty_cache()

    # ── Summary ──
    n = len(cases)
    print(f"\n{'='*65}")
    print(f"RESULTS (n={n})")
    print(f"{'='*65}")
    header = f"{'Model':<15} {'Success':>9} {'ObsHall':>9} {'AvgCalls':>9} {'AvgTime':>9}"
    print(header)
    print("-" * 55)

    for mname in ["Baseline_B", "Baseline_A", "FT_ReAct"]:
        sr = sum(metrics[mname]["passed"]) / n
        oh = sum(metrics[mname]["obs_hall"]) / n
        ac = sum(metrics[mname]["tool_calls"]) / n
        at = sum(metrics[mname]["time"]) / n
        print(f"{mname:<15} {sr:>8.1%} {oh:>8.1%} {ac:>8.1f} {at:>8.1f}s")

    # Per category
    for cat_name in ["single", "multi", "error"]:
        cat_n = sum(1 for c in cases if c["cat"] == cat_name)
        print(f"\n--- {cat_name} (n={cat_n}) ---")
        for mname in ["Baseline_B", "Baseline_A", "FT_ReAct"]:
            cat_results = [r for r in all_results if r["model"] == mname and r["category"] == cat_name]
            sr = sum(r["passed"] for r in cat_results) / cat_n
            errs = sum(r["errors"] for r in cat_results) / cat_n
            recs = sum(int(r.get("recovered", False)) for r in cat_results)
            print(f"  {mname:<15}: SR={sr:.0%}  avg_errors={errs:.1f}  recovered={recs}")

    # Error category detail
    if "error" in [c["cat"] for c in cases]:
        print(f"\n--- Error Recovery Detail ---")
        for mname in ["Baseline_B", "Baseline_A", "FT_ReAct"]:
            err_results = [r for r in all_results if r["model"] == mname and r["category"] == "error"]
            has_ans = sum(r["has_answer"] for r in err_results)
            rec = sum(int(r.get("recovered", False)) for r in err_results)
            print(f"  {mname}: answers={has_ans}/{len(err_results)}  recovered={rec}")

    # Save
    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_cases": n,
        "model": "Qwen2.5-7B-Instruct + QLoRA ReAct",
        "summary": {
            mname: {
                "success_rate": sum(metrics[mname]["passed"]) / n,
                "obs_hall_rate": sum(metrics[mname]["obs_hall"]) / n,
                "avg_tool_calls": sum(metrics[mname]["tool_calls"]) / n,
                "avg_time_s": sum(metrics[mname]["time"]) / n,
            }
            for mname in ["Baseline_B", "Baseline_A", "FT_ReAct"]
        },
        "per_category": {},
        "all_results": all_results,
    }

    for cat_name in ["single", "multi", "error"]:
        cat_n = sum(1 for c in cases if c["cat"] == cat_name)
        out["per_category"][cat_name] = {}
        for mname in ["Baseline_B", "Baseline_A", "FT_ReAct"]:
            cat_results = [r for r in all_results if r["model"] == mname and r["category"] == cat_name]
            out["per_category"][cat_name][mname] = {
                "success_rate": sum(r["passed"] for r in cat_results) / cat_n,
                "avg_errors": sum(r["errors"] for r in cat_results) / cat_n,
                "recovered": sum(int(r.get("recovered", False)) for r in cat_results),
            }

    path = "outputs/eval_100_full.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
