"""Programmatic synthetic ReAct trajectory generation.

Generates diverse, high-quality ReAct training data without relying on external
teacher models. Uses template-based generation with parameter randomization for
variety, plus carefully crafted error-recovery scenarios.
"""

import argparse
import json
import random
from pathlib import Path

from src.data.react_format import build_system_prompt, validate_trajectory

# ── Tool definitions ──────────────────────────────────────
TOOL_DEFS = """
- get_weather(city: str, date: str = "today") -> dict: Get current weather for a city. Returns {city, temperature, condition, humidity}.
- search_web(query: str, num_results: int = 5) -> list: Search the web. Returns list of {title, snippet, url}.
- get_stock_price(symbol: str) -> dict: Get current stock price. Returns {symbol, price, change_pct}.
- calculate(expression: str) -> float: Evaluate a math expression.
- get_time(city: str) -> str: Get current local time for a city.
"""

SYSTEM_PROMPT = build_system_prompt(TOOL_DEFS)

# ── City data for realistic queries ───────────────────────
CITIES = [
    "北京", "上海", "广州", "深圳", "杭州", "成都", "南京", "武汉",
    "重庆", "西安", "东京", "纽约", "伦敦", "巴黎", "悉尼", "新加坡",
    "首尔", "柏林", "多伦多", "迪拜",
]

WEATHER_CONDITIONS = ["晴", "多云", "阴天", "小雨", "中雨", "雷阵雨", "雾", "霾", "晴转多云", "阴转晴"]
WEATHER_TEMPS = list(range(-5, 40))

STOCK_SYMBOLS = {
    "苹果": "AAPL", "微软": "MSFT", "谷歌": "GOOGL", "亚马逊": "AMZN",
    "特斯拉": "TSLA", "英伟达": "NVDA", "Meta": "META", "阿里巴巴": "BABA",
    "腾讯": "0700.HK", "台积电": "TSM", "AMD": "AMD", "英特尔": "INTC",
}

SEARCH_TOPICS = [
    "Python 异步编程最佳实践",
    "机器学习最新进展",
    "量子计算原理",
    "深度学习框架对比",
    "Rust 语言特性",
    "数据库索引优化",
    "分布式系统设计模式",
    "前端框架发展趋势",
    "网络安全防护技术",
    "Kubernetes 部署方案",
]


def generate_sample(category: str) -> dict | None:
    """Generate a single ReAct trajectory sample."""
    gen_funcs = {
        "weather_single": gen_weather_single,
        "weather_compare": gen_weather_compare,
        "weather_multi": gen_weather_multi,
        "stock_single": gen_stock_single,
        "stock_compare": gen_stock_compare,
        "stock_calc": gen_stock_calc,
        "search_single": gen_search_single,
        "search_multi": gen_search_multi,
        "calc_simple": gen_calc_simple,
        "calc_word_problem": gen_calc_word_problem,
        "time_single": gen_time_single,
        "time_compare": gen_time_compare,
        "multi_weather_stock": gen_multi_weather_stock,
        "multi_search_weather": gen_multi_search_weather,
        "multi_stock_search": gen_multi_stock_search,
        "multi_calc_search": gen_multi_calc_search,
        "error_empty": gen_error_empty,
        "error_timeout": gen_error_timeout,
        "error_invalid_param": gen_error_invalid_param,
    }

    func = gen_funcs.get(category)
    if func is None:
        return None

    messages = func()
    if messages is None:
        return None

    ok, err = validate_trajectory(messages)
    if not ok:
        return None

    return {"messages": messages, "category": category}


# ── Individual generators ─────────────────────────────────

def gen_weather_single() -> list[dict]:
    city = random.choice(CITIES)
    temp = random.choice(WEATHER_TEMPS)
    cond = random.choice(WEATHER_CONDITIONS)
    humidity = random.randint(20, 95)

    queries = [
        f"{city}今天天气怎么样？",
        f"帮我查一下{city}的天气",
        f"{city}现在是什么天气？",
        f"我想知道{city}今天冷不冷",
        f"{city}今天适合出门吗？查一下天气",
    ]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": random.choice(queries)},
        {"role": "assistant",
         "content": f"Thought: 用户想了解{city}的天气情况。我需要调用 get_weather 工具来获取{city}的天气数据。\nAction: get_weather(city=\"{city}\")\n"},
        {"role": "tool",
         "content": json.dumps({"city": city, "temperature": temp, "condition": cond, "humidity": humidity}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 已获得{city}的完整天气数据：温度{temp}°C，天气{cond}，湿度{humidity}%。可以根据这些信息回答用户了。\nFinal Answer: {city}今天天气{cond}，温度{temp}°C，湿度{humidity}%。{'适合户外活动。' if temp >= 15 and temp <= 32 and cond in ('晴', '多云', '晴转多云') else '建议根据天气情况合理安排出行。'}\n"},
    ]


def gen_weather_compare() -> list[dict]:
    c1, c2 = random.sample(CITIES, 2)
    t1, t2 = random.choice(WEATHER_TEMPS), random.choice(WEATHER_TEMPS)
    c1_cond = random.choice(WEATHER_CONDITIONS)
    c2_cond = random.choice(WEATHER_CONDITIONS)

    queries = [
        f"{c1}和{c2}哪个更暖和？",
        f"对比一下{c1}和{c2}的温度",
        f"{c1}和{c2}今天的温差大吗？",
    ]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": random.choice(queries)},
        {"role": "assistant",
         "content": f"Thought: 需要分别查询{c1}和{c2}的天气数据才能进行比较。先查{c1}。\nAction: get_weather(city=\"{c1}\")\n"},
        {"role": "tool",
         "content": json.dumps({"city": c1, "temperature": t1, "condition": c1_cond}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 已获取{c1}的温度{t1}°C，现在需要查{c2}的数据。\nAction: get_weather(city=\"{c2}\")\n"},
        {"role": "tool",
         "content": json.dumps({"city": c2, "temperature": t2, "condition": c2_cond}, ensure_ascii=False)},
        {"role": "assistant",
         "content": (
             f"Thought: 两个城市的数据都已获得。{c1}温度{t1}°C，{c2}温度{t2}°C。温差{abs(t1 - t2)}°C。"
             f"可以给出比较结果了。\nFinal Answer: {c1}当前温度{t1}°C（{c1_cond}），{c2}当前温度{t2}°C（{c2_cond}）。"
             f"{c1 if t1 > t2 else c2}更{'暖和' if max(t1, t2) > min(t1, t2) else '相似'}，两地温差{abs(t1 - t2)}°C。\n"
         )
         },
    ]


def gen_weather_multi() -> list[dict]:
    cities = random.sample(CITIES, min(3, len(CITIES) // 2))
    temps = [random.choice(WEATHER_TEMPS) for _ in cities]
    conds = [random.choice(WEATHER_CONDITIONS) for _ in cities]

    city_list = "、".join(cities)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"帮我查一下{city_list}这三个地方的天气情况"},
    ]

    weather_data = []
    for i, city in enumerate(cities):
        messages.append({
            "role": "assistant",
            "content": f"Thought: {'开始' if i == 0 else '继续'}查询{city}的天气。\nAction: get_weather(city=\"{city}\")\n"
        })
        data = {"city": city, "temperature": temps[i], "condition": conds[i], "humidity": random.randint(30, 80)}
        messages.append({"role": "tool", "content": json.dumps(data, ensure_ascii=False)})
        weather_data.append(data)

    summary = "\n".join(f"- {d['city']}：温度{d['temperature']}°C，{d['condition']}，湿度{d['humidity']}%" for d in weather_data)
    messages.append({
        "role": "assistant",
        "content": f"Thought: 三个城市的数据都已获取，整理后回答用户。\nFinal Answer: 以下是三个城市的天气情况：\n{summary}\n"
    })

    return messages


def gen_stock_single() -> list[dict]:
    company, symbol = random.choice(list(STOCK_SYMBOLS.items()))
    price = round(random.uniform(50, 500), 2)
    change = round(random.uniform(-5, 5), 2)

    queries = [f"{company}今天的股价是多少？", f"查一下{company}的股票", f"{company}股价怎么样了？"]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": random.choice(queries)},
        {"role": "assistant",
         "content": f"Thought: 用户想查询{company}的股价，{company}的股票代码是 {symbol}。\nAction: get_stock_price(symbol=\"{symbol}\")\n"},
        {"role": "tool",
         "content": json.dumps({"symbol": symbol, "price": price, "change_pct": change}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 已获取{symbol}的股价数据，可以直接回复用户了。\nFinal Answer: {company}（{symbol}）当前股价为 ${price}，较前一日{'上涨' if change > 0 else '下跌'}{abs(change):.2f}%。\n"},
    ]


def gen_stock_compare() -> list[dict]:
    (c1, s1), (c2, s2) = random.sample(list(STOCK_SYMBOLS.items()), 2)
    p1 = round(random.uniform(50, 500), 2)
    p2 = round(random.uniform(50, 500), 2)
    ch1 = round(random.uniform(-5, 5), 2)
    ch2 = round(random.uniform(-5, 5), 2)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{c1}和{c2}最近哪个涨得更多？"},
        {"role": "assistant",
         "content": f"Thought: 需要分别查{c1}和{c2}的股价才能比较。先查{c1}。\nAction: get_stock_price(symbol=\"{s1}\")\n"},
        {"role": "tool", "content": json.dumps({"symbol": s1, "price": p1, "change_pct": ch1}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 已获取{s1}数据，继续查{c2}。\nAction: get_stock_price(symbol=\"{s2}\")\n"},
        {"role": "tool", "content": json.dumps({"symbol": s2, "price": p2, "change_pct": ch2}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 两边数据齐了。{c1}({s1})涨跌幅{ch1:+.2f}%，{c2}({s2})涨跌幅{ch2:+.2f}%。\nFinal Answer: {c1}({s1})当前${p1}，{'上涨' if ch1 > 0 else '下跌'}{abs(ch1):.2f}%。{c2}({s2})当前${p2}，{'上涨' if ch2 > 0 else '下跌'}{abs(ch2):.2f}%。{c1 if abs(ch1) > abs(ch2) else c2}的涨跌幅度更大。\n"},
    ]


def gen_stock_calc() -> list[dict]:
    company, symbol = random.choice(list(STOCK_SYMBOLS.items()))
    price = round(random.uniform(50, 500), 2)
    shares = random.choice([10, 50, 100, 200, 500, 1000])
    total = round(price * shares, 2)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"我买了{shares}股{company}的股票，现在值多少钱？"},
        {"role": "assistant",
         "content": f"Thought: 需要先查{company}的当前股价，然后计算{shares}股的总价值。\nAction: get_stock_price(symbol=\"{symbol}\")\n"},
        {"role": "tool", "content": json.dumps({"symbol": symbol, "price": price, "change_pct": round(random.uniform(-3, 3), 2)}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 当前股价${price}，需要计算{shares} × {price}。\nAction: calculate(expression=\"{shares} * {price}\")\n"},
        {"role": "tool", "content": json.dumps({"result": total}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 计算完成，{shares}股的总价值是${total}。\nFinal Answer: {company}（{symbol}）当前股价${price}/股，你持有的{shares}股总价值为 ${total}。\n"},
    ]


def gen_search_single() -> list[dict]:
    topic = random.choice(SEARCH_TOPICS)
    result_count = random.randint(1, 5)

    snippets = []
    for i in range(result_count):
        snippets.append({
            "title": f"{topic} - 来源{i+1}",
            "snippet": f"关于「{topic}」的第{i+1}条搜索结果。这里包含相关的详细介绍和最新信息。",
            "url": f"https://example.com/{i+1}",
        })

    queries = [f"搜索一下{topic}", f"我想了解{topic}", f"帮我找一些关于{topic}的资料"]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": random.choice(queries)},
        {"role": "assistant",
         "content": f"Thought: 用户想了解「{topic}」，我需要使用搜索工具查找相关信息。\nAction: search_web(query=\"{topic}\", num_results=\"{result_count}\")\n"},
        {"role": "tool",
         "content": json.dumps({"results": snippets}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 搜索返回了{result_count}条相关结果，可以整理后回答用户。\nFinal Answer: 关于「{topic}」的搜索结果如下：\n" +
                    "\n".join(f"- [{s['title']}]({s['url']}): {s['snippet']}" for s in snippets) +
                    "\n\n如需更详细的信息，建议点击相关链接查看原文。\n"},
    ]


def gen_search_multi() -> list[dict]:
    t1, t2 = random.sample(SEARCH_TOPICS, 2)

    s1_snippets = [{"title": f"{t1} - 资料{i+1}", "snippet": f"关于{t1}的内容摘要{i+1}", "url": f"https://ex.com/a{i}"} for i in range(random.randint(1, 3))]
    s2_snippets = [{"title": f"{t2} - 资料{i+1}", "snippet": f"关于{t2}的内容摘要{i+1}", "url": f"https://ex.com/b{i}"} for i in range(random.randint(1, 3))]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"帮我分别查一下「{t1}」和「{t2}」的相关信息"},
        {"role": "assistant",
         "content": f"Thought: 需要分别搜索两个主题。先搜索「{t1}」。\nAction: search_web(query=\"{t1}\")\n"},
        {"role": "tool", "content": json.dumps({"results": s1_snippets}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 第一个主题搜索完成，继续搜索「{t2}」。\nAction: search_web(query=\"{t2}\")\n"},
        {"role": "tool", "content": json.dumps({"results": s2_snippets}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 两个主题的搜索结果都拿到了，整理后一并回复用户。\nFinal Answer: 搜索结果如下：\n\n【{t1}】\n" +
                    "\n".join(f"- {s['snippet']}" for s in s1_snippets) +
                    f"\n\n【{t2}】\n" +
                    "\n".join(f"- {s['snippet']}" for s in s2_snippets) + "\n"},
    ]


def gen_calc_simple() -> list[dict]:
    a = round(random.uniform(1, 100), 2)
    b = round(random.uniform(1, 100), 2)
    ops = ["+", "-", "*", "/"]
    op = random.choice(ops)
    expr = f"{a} {op} {b}"

    if op == "/" and b == 0:
        b = 1.0
        expr = f"{a} / {b}"

    try:
        result = eval(expr)
        result = round(result, 4)
    except Exception:
        result = 0

    queries = [f"帮我算一下 {a} {op} {b} 等于多少？", f"计算 {a}{op}{b}", f"{a} {op} {b} 的结果是多少？"]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": random.choice(queries)},
        {"role": "assistant",
         "content": f"Thought: 用户需要计算一个数学表达式，我可以用 calculate 工具来完成。\nAction: calculate(expression=\"{expr}\")\n"},
        {"role": "tool", "content": json.dumps({"result": result}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 计算已完成，直接返回结果。\nFinal Answer: {expr} = {result}\n"},
    ]


def gen_calc_word_problem() -> list[dict]:
    problems = [
        {"query": "一个长方形长12米宽8米，面积是多少平方米？", "expr": "12 * 8", "answer": "长方形面积 = 12 × 8 = 96平方米"},
        {"query": "圆的半径是5厘米，面积是多少平方厘米（π取3.14）？", "expr": "3.14 * 5 * 5", "answer": "圆的面积 = 3.14 × 5² = 78.5平方厘米"},
        {"query": "一套书原价128元，打8折后多少钱？", "expr": "128 * 0.8", "answer": "128 × 0.8 = 102.4元"},
        {"query": "小明有250元，买了3本书每本42元，还剩多少？", "expr": "250 - 3 * 42", "answer": "250 - 3 × 42 = 250 - 126 = 124元"},
        {"query": "一个梯形的上底6cm下底10cm高5cm，面积是多少？", "expr": "(6 + 10) * 5 / 2", "answer": "梯形面积 = (6 + 10) × 5 ÷ 2 = 40平方厘米"},
    ]

    prob = random.choice(problems)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prob["query"]},
        {"role": "assistant",
         "content": f"Thought: 这是一个数学应用题。我需要用 calculate 工具来计算。\nAction: calculate(expression=\"{prob['expr']}\")\n"},
        {"role": "tool", "content": json.dumps({"result": eval(prob["expr"])}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 计算完成，可以给出答案了。\nFinal Answer: {prob['answer']}\n"},
    ]


def gen_time_single() -> list[dict]:
    city = random.choice(CITIES)
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)

    queries = [f"{city}现在几点了？", f"帮我看看{city}现在什么时间", f"{city}现在是几点？"]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": random.choice(queries)},
        {"role": "assistant",
         "content": f"Thought: 用户想知道{city}当前的时间。我使用 get_time 工具查询。\nAction: get_time(city=\"{city}\")\n"},
        {"role": "tool",
         "content": json.dumps({"city": city, "time": f"{hour:02d}:{minute:02d}", "timezone": "UTC+8"}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 已获取{city}的当前时间，直接回复用户即可。\nFinal Answer: {city}当前时间是 {hour:02d}:{minute:02d}。\n"},
    ]


def gen_time_compare() -> list[dict]:
    c1, c2 = random.sample(CITIES, 2)
    h1, h2 = random.randint(6, 22), random.randint(0, 23)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{c1}和{c2}现在分别是几点？时差多久？"},
        {"role": "assistant",
         "content": f"Thought: 需要分别查询两个城市的时间。先查{c1}。\nAction: get_time(city=\"{c1}\")\n"},
        {"role": "tool",
         "content": json.dumps({"city": c1, "time": f"{h1:02d}:00", "timezone": "UTC+8"}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: {c1}时间是{h1}:00，继续查{c2}。\nAction: get_time(city=\"{c2}\")\n"},
        {"role": "tool",
         "content": json.dumps({"city": c2, "time": f"{h2:02d}:00", "timezone": "UTC+0"}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 两个城市的时间都有了。{c1}是{h1}:00，{c2}是{h2}:00，时差约{abs(h1 - h2)}小时。\nFinal Answer: {c1}当前时间 {h1:02d}:00，{c2}当前时间 {h2:02d}:00，两地时差约{abs(h1 - h2)}小时。\n"},
    ]


def gen_multi_weather_stock() -> list[dict]:
    city = random.choice(CITIES)
    company, symbol = random.choice(list(STOCK_SYMBOLS.items()))
    temp = random.choice(WEATHER_TEMPS)
    cond = random.choice(WEATHER_CONDITIONS)
    price = round(random.uniform(50, 500), 2)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"查一下{company}的股价，顺便看看{city}的天气"},
        {"role": "assistant",
         "content": f"Thought: 用户有两个需求：查{company}股价和{city}天气。先查股价。\nAction: get_stock_price(symbol=\"{symbol}\")\n"},
        {"role": "tool",
         "content": json.dumps({"symbol": symbol, "price": price, "change_pct": round(random.uniform(-3, 3), 2)}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 股价拿到了，现在查{city}天气。\nAction: get_weather(city=\"{city}\")\n"},
        {"role": "tool",
         "content": json.dumps({"city": city, "temperature": temp, "condition": cond, "humidity": random.randint(30, 80)}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 两个信息都有了。\nFinal Answer: {company}({symbol})当前股价${price}。{city}天气{cond}，温度{temp}°C。\n"},
    ]
    return messages


def gen_multi_search_weather() -> list[dict]:
    topic = random.choice(SEARCH_TOPICS)
    city = random.choice(CITIES)
    temp = random.choice(WEATHER_TEMPS)
    cond = random.choice(WEATHER_CONDITIONS)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"搜索{topic}的最新资料，另外帮我看看{city}天气"},
        {"role": "assistant",
         "content": f"Thought: 先搜索{topic}。\nAction: search_web(query=\"{topic}\", num_results=\"3\")\n"},
        {"role": "tool",
         "content": json.dumps({"results": [{"title": f"{topic}概览", "snippet": f"关于{topic}的详细介绍", "url": "https://ex.com"}]}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 搜索完成，接下来查{city}天气。\nAction: get_weather(city=\"{city}\")\n"},
        {"role": "tool",
         "content": json.dumps({"city": city, "temperature": temp, "condition": cond, "humidity": random.randint(30, 80)}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 信息齐全。\nFinal Answer: 【{topic}搜索】找到相关文章若干篇。\n【{city}天气】{cond}，温度{temp}°C。\n"},
    ]


def gen_multi_stock_search() -> list[dict]:
    company, symbol = random.choice(list(STOCK_SYMBOLS.items()))
    price = round(random.uniform(50, 500), 2)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"查一下{company}的股价，顺便搜索一下{company}最近有什么重大新闻"},
        {"role": "assistant",
         "content": f"Thought: 先查股价。\nAction: get_stock_price(symbol=\"{symbol}\")\n"},
        {"role": "tool",
         "content": json.dumps({"symbol": symbol, "price": price, "change_pct": round(random.uniform(-3, 3), 2)}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 股价已获取，现在搜索{company}新闻。\nAction: search_web(query=\"{company} 重大新闻\")\n"},
        {"role": "tool",
         "content": json.dumps({"results": [{"title": f"{company}最新动态", "snippet": f"{company}近期发布了新产品，市场反应积极。", "url": "https://ex.com"}]}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 股价和新闻都拿到了。\nFinal Answer: {company}({symbol})当前股价${price}。根据搜索，{company}近期发布了新产品，市场反应积极。\n"},
    ]


def gen_multi_calc_search() -> list[dict]:
    a, b = random.randint(10, 500), random.randint(10, 500)
    expr = f"{a} * {b}"
    result = a * b

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"帮我算一下{a}乘以{b}等于多少，然后搜索一下这个数字有什么特殊意义"},
        {"role": "assistant",
         "content": f"Thought: 先计算{a} × {b}。\nAction: calculate(expression=\"{expr}\")\n"},
        {"role": "tool", "content": json.dumps({"result": result}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 结果是{result}。现在搜索这个数字的意义。\nAction: search_web(query=\"{result} 数学意义\")\n"},
        {"role": "tool",
         "content": json.dumps({"results": [{"title": f"数字{result}", "snippet": f"{result}是一个合数", "url": "https://ex.com"}]}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 计算和搜索都完成了。\nFinal Answer: {a} × {b} = {result}。根据搜索，{result}是一个普通的合数，没有特殊的数学意义。\n"},
    ]


# ── Error recovery generators ────────────────────────────

def gen_error_empty() -> list[dict]:
    topic = random.choice(SEARCH_TOPICS)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"搜索一下关于{topic}的最新资料"},
        {"role": "assistant",
         "content": f"Thought: 用户想搜索{topic}的资料。\nAction: search_web(query=\"{topic}\", num_results=\"5\")\n"},
        {"role": "tool",
         "content": json.dumps({"results": [], "count": 0, "message": "No results found."}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 搜索返回了空结果，可能是关键词太宽泛。我尝试用更精确的关键词重新搜索。\nAction: search_web(query=\"{topic} 教程 入门\")\n"},
        {"role": "tool",
         "content": json.dumps({"results": [{"title": f"{topic}入门教程", "snippet": f"关于{topic}的详细讲解", "url": "https://ex.com"}], "count": 1}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 调整搜索词后找到了结果。\nFinal Answer: 第一次搜索「{topic}」未找到结果。调整关键词后找到一篇相关文章：《{topic}入门教程》，建议从基础开始了解。\n"},
    ]


def gen_error_timeout() -> list[dict]:
    city = random.choice(CITIES)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{city}现在天气怎么样？"},
        {"role": "assistant",
         "content": f"Thought: 需要查询{city}的天气。\nAction: get_weather(city=\"{city}\")\n"},
        {"role": "tool",
         "content": json.dumps({"error": "timeout", "message": "Request timed out after 30 seconds."}, ensure_ascii=False)},
        {"role": "assistant",
         "content": f"Thought: 天气服务超时了，可能是网络问题。我应该告知用户当前无法获取数据，并建议稍后重试。\nFinal Answer: 抱歉，查询{city}天气时服务请求超时。这可能是由于网络波动或服务暂时不可用导致的。建议您稍后重试，或者尝试通过其他渠道（如天气App）获取{city}的最新天气信息。\n"},
    ]


def gen_error_invalid_param() -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "帮我查一下火星上的天气"},
        {"role": "assistant",
         "content": "Thought: 用户想查火星的天气，但 get_weather 工具可能不支持火星。我先尝试一下。\nAction: get_weather(city=\"火星\")\n"},
        {"role": "tool",
         "content": json.dumps({"error": "invalid_parameter", "message": "city '火星' not found in weather database."}, ensure_ascii=False)},
        {"role": "assistant",
         "content": "Thought: get_weather 不支持火星，因为火星不在地球城市数据库中。我应该向用户解释这一点，并建议通过搜索来了解火星的气候知识。\nAction: search_web(query=\"火星 天气 气候\")\n"},
        {"role": "tool",
         "content": json.dumps({"results": [{"title": "火星气候特征", "snippet": "火星表面平均温度约-63°C，大气稀薄...", "url": "https://ex.com"}], "count": 1}, ensure_ascii=False)},
        {"role": "assistant",
         "content": "Thought: 搜索找到了火星气候信息，现在可以给用户一个有用的回答了。\nFinal Answer: 很抱歉，天气查询工具目前只支持地球上的城市，无法查询火星的实时天气。不过根据搜索资料，火星表面平均温度约-63°C，大气非常稀薄，主要由二氧化碳组成。火星偶尔会有沙尘暴等气象现象。\n"},
    ]


# ── Main pipeline ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ReAct training data")
    parser.add_argument("--total", type=int, default=8000, help="Total synthetic samples to generate")
    parser.add_argument("--output_dir", type=str, default="data/synthetic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Distribution: ~20 categories
    categories = (
        ["weather_single"] * 600 +
        ["weather_compare"] * 300 +
        ["weather_multi"] * 200 +
        ["stock_single"] * 500 +
        ["stock_compare"] * 300 +
        ["stock_calc"] * 200 +
        ["search_single"] * 600 +
        ["search_multi"] * 300 +
        ["calc_simple"] * 500 +
        ["calc_word_problem"] * 300 +
        ["time_single"] * 300 +
        ["time_compare"] * 200 +
        ["multi_weather_stock"] * 400 +
        ["multi_search_weather"] * 400 +
        ["multi_stock_search"] * 400 +
        ["multi_calc_search"] * 300 +
        ["error_empty"] * 400 +
        ["error_timeout"] * 300 +
        ["error_invalid_param"] * 300
    )

    random.shuffle(categories)
    categories = categories[:args.total]

    print(f"Generating {len(categories)} synthetic ReAct trajectories...")
    all_samples = []
    cat_counts = {}

    for i, cat in enumerate(categories):
        sample = generate_sample(cat)
        if sample:
            all_samples.append(sample)
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(categories)}...")

    # Save
    output_path = output_dir / "synthetic_react.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nGenerated {len(all_samples)} samples -> {output_path}")

    # Per-category stats
    print("\nCategory distribution:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    # Quick validation
    invalid = 0
    for s in all_samples:
        ok, _ = validate_trajectory(s["messages"])
        if not ok:
            invalid += 1
    print(f"\nFormat validation: {len(all_samples) - invalid}/{len(all_samples)} pass" +
          (f" ({invalid} FAIL)" if invalid else " (all pass)"))


if __name__ == "__main__":
    main()
