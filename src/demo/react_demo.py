"""ReAct Agent Demo: QLoRA fine-tuned model + real tool execution + trajectory viz."""

import json, re, html
import gradio as gr
import torch, unsloth
from unsloth import FastLanguageModel
from peft import PeftModel

# ── Tool Implementations ──────────────────────────────────

def tool_get_weather(city: str, date: str = "today") -> dict:
    """Mock weather API with realistic data."""
    import random, hashlib
    seed = int(hashlib.md5(city.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    conditions = ["晴天", "多云", "阴天", "小雨", "中雨", "雷阵雨", "雾", "霾"]
    return {
        "city": city, "temperature": rng.randint(-5, 38),
        "condition": rng.choice(conditions),
        "humidity": rng.randint(20, 95),
    }

def tool_get_stock_price(symbol: str) -> dict:
    import random, hashlib
    seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    return {
        "symbol": symbol.upper(),
        "price": round(rng.uniform(50, 500), 2),
        "change_pct": round(rng.uniform(-5, 5), 2),
    }

def tool_search_web(query: str, num_results: int = 3) -> dict:
    return {
        "results": [
            {"title": f"{query} - 百科", "snippet": f"关于「{query}」的详细介绍...", "url": "https://wiki.example.com"},
            {"title": f"{query} 最新资讯", "snippet": f"「{query}」相关的最新报道和分析。", "url": "https://news.example.com"},
            {"title": f"{query} 教程", "snippet": f"「{query}」的入门到精通教程。", "url": "https://docs.example.com"},
        ][:num_results]
    }

def tool_calculate(expression: str) -> dict:
    try:
        return {"result": eval(expression, {"__builtins__": {}}, {})}
    except Exception as e:
        return {"error": str(e)}

def tool_get_time(city: str) -> dict:
    import random, hashlib
    seed = int(hashlib.md5(city.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    return {"city": city, "time": f"{rng.randint(0,23):02d}:{rng.randint(0,59):02d}", "timezone": "UTC+8"}

TOOLS = {
    "get_weather": {"func": tool_get_weather, "desc": "查询城市天气。参数: city(城市名), date(日期,可选)"},
    "get_stock_price": {"func": tool_get_stock_price, "desc": "查询股票价格。参数: symbol(股票代码如AAPL)"},
    "search_web": {"func": tool_search_web, "desc": "搜索网络信息。参数: query(搜索关键词)"},
    "calculate": {"func": tool_calculate, "desc": "数学计算。参数: expression(表达式如'3+5*2')"},
    "get_time": {"func": tool_get_time, "desc": "查询城市当前时间。参数: city(城市名)"},
}


# ── Execute action ────────────────────────────────────────

def execute_action(action_text: str) -> str:
    """Parse ACTION text and call the actual tool function."""
    # Match tool_name(...) — greedy .+ to capture up to the LAST )
    m = re.match(r"(\w+)\((.+)\)", action_text.strip())
    if not m:
        return json.dumps({"error": f"无法解析动作: {action_text}"}, ensure_ascii=False)

    tool_name = m.group(1)
    params_str = m.group(2)

    if tool_name not in TOOLS:
        return json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False)

    # Parse parameters
    params = {}
    for pm in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', params_str):
        params[pm.group(1)] = pm.group(2)

    try:
        result = TOOLS[tool_name]["func"](**params)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ── Model ─────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

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

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is not None:
        return
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit", max_seq_length=2048,
        dtype=None, load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(
        model, "outputs/react-qwen2.5-7b-qlora_20260508_1810/checkpoint-1000",
    )
    FastLanguageModel.for_inference(model)
    print("Model ready.")


# ── ReAct Loop ────────────────────────────────────────────

def run_react_agent(user_query: str, max_turns: int = 8):
    """Run the full ReAct loop and return trajectory for display."""
    load_model()

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    trajectory = []

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
        raw = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Parse
        thought_m = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|\Z)", raw, re.DOTALL)
        action_m = re.search(r"Action:\s*(.+?)(?=\n|$)", raw)
        final_m = re.search(r"Final Answer:\s*(.+?)$", raw, re.DOTALL)

        thought = thought_m.group(1).strip() if thought_m else ""
        action = action_m.group(1).strip() if action_m else ""
        final_answer = final_m.group(1).strip() if final_m else ""

        conversation.append({"role": "assistant", "content": raw})

        if final_answer:
            trajectory.append({
                "turn": turn + 1, "thought": thought,
                "action": action, "type": "final",
                "content": final_answer,
            })
            return trajectory, True

        if action:
            observation = execute_action(action)
            conversation.append({"role": "tool", "content": observation})

            trajectory.append({
                "turn": turn + 1, "thought": thought,
                "action": action, "type": "action",
                "content": observation,
            })
        else:
            trajectory.append({
                "turn": turn + 1, "thought": thought,
                "action": "", "type": "error",
                "content": "模型未输出有效 Action",
            })
            return trajectory, False

    return trajectory, False


# ── HTML Rendering ────────────────────────────────────────

def render_trajectory(user_query: str, trajectory: list, success: bool) -> str:
    """Render ReAct trajectory as styled HTML."""
    status_color = "#4caf50" if success else "#f44336"
    status_text = "任务完成" if success else "未完成"

    h = f"""
    <div style="font-family: 'Segoe UI', sans-serif; max-width: 850px;">
    <div style="background: #e3f2fd; padding: 14px; border-radius: 8px; margin-bottom: 16px;">
      <strong>用户</strong><br>{html.escape(user_query)}
    </div>
    """

    for step in trajectory:
        # Thought bubble
        if step["thought"]:
            h += f"""
            <div style="background: #fff3e0; border-left: 4px solid #ff9800; padding: 10px; margin: 6px 0; border-radius: 4px;">
              <strong>T{step['turn']} THOUGHT</strong><br>
              <span style="color: #e65100;">{html.escape(step['thought'])}</span>
            </div>"""

        # Action
        if step["action"]:
            h += f"""
            <div style="background: #e8f5e9; border-left: 4px solid #4caf50; padding: 10px; margin: 6px 0 6px 20px; border-radius: 4px;">
              <strong> ACTION</strong><br>
              <code style="background: #1b5e20; color: #a5d6a7; padding: 2px 6px; border-radius: 3px;">{html.escape(step['action'])}</code>
            </div>"""

        # Observation or Final Answer
        if step["type"] == "action":
            try:
                obs_data = json.loads(step["content"])
                obs_str = json.dumps(obs_data, indent=2, ensure_ascii=False)
                border = "#f44336" if "error" in obs_data else "#2196f3"
                bg = "#ffebee" if "error" in obs_data else "#e8eaf6"
            except json.JSONDecodeError:
                obs_str = step["content"]
                border = "#9e9e9e"
                bg = "#f5f5f5"

            h += f"""
            <div style="background: {bg}; border-left: 4px solid {border}; padding: 10px; margin: 4px 0 4px 40px; border-radius: 4px;">
              <strong> OBSERVATION</strong>
              <pre style="margin: 4px 0 0 0; font-size: 13px; white-space: pre-wrap;">{html.escape(obs_str)}</pre>
            </div>"""

        elif step["type"] == "final":
            h += f"""
            <div style="background: #f3e5f5; border-left: 4px solid #9c27b0; padding: 12px; margin: 6px 0; border-radius: 4px;">
              <strong> FINAL ANSWER</strong><br>
              <span style="color: #6a1b9a;">{html.escape(step['content'])}</span>
            </div>"""

    h += f"""
    <div style="margin-top: 12px; padding: 8px; background: {status_color}22; border-radius: 6px; text-align: center;">
      <span style="color: {status_color}; font-weight: bold;">{status_text}</span>
    </div>
    </div>"""
    return h


# ── Gradio UI ─────────────────────────────────────────────

def run_demo(user_query: str):
    traj, success = run_react_agent(user_query)
    html_out = render_trajectory(user_query, traj, success)
    return html_out


def create_ui():
    with gr.Blocks(title="ReAct Agent Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # ReAct Agent — Thought Action Observation

        **Qwen2.5-7B-Instruct + QLoRA 微调** | RTX 4070s 12GB | SwanLab 追踪

        输入问题，观察 Agent 如何推理、调用工具、处理结果。
        """)

        with gr.Row():
            with gr.Column(scale=2):
                query = gr.Textbox(
                    label="你的问题",
                    placeholder="例如：北京今天天气怎么样？适合户外运动吗？",
                    lines=2,
                )
                btn = gr.Button("开始推理", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("""
                ### 可用工具
                - **get_weather** — 天气查询
                - **get_stock_price** — 股价查询
                - **search_web** — 网络搜索
                - **calculate** — 数学计算
                - **get_time** — 时间查询
                """)

        output = gr.HTML(label="推理轨迹")

        gr.Examples(
            examples=[
                ["北京今天天气怎么样？适合户外运动吗？"],
                ["帮我查一下苹果(AAPL)的当前股价"],
                ["搜索一下Python异步编程的最佳实践"],
                ["上海和北京哪个更热？温差多少？"],
                ["帮我算一下 (15.8+23.4)*1.15 等于多少"],
                ["东京现在几点了？"],
            ],
            inputs=[query],
        )

        btn.click(fn=run_demo, inputs=[query], outputs=[output])

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
