"""ReAct Demo: Baseline A vs Fine-tuned, side-by-side trajectory comparison.

Shares one base model instance. FT = base + LoRA adapter. ~6GB VRAM total.
"""

import json, re, html
import gradio as gr
import torch, unsloth
from unsloth import FastLanguageModel
from peft import PeftModel

from src.demo.react_demo import execute_action, TOOLS

# ── Prompts ────────────────────────────────────────────────
REACT_SYSTEM = """You are a helpful assistant with access to tools.

Tools:
- get_weather(city="城市名"): 查询天气。返回 temperature, condition, humidity。
- get_stock_price(symbol="股票代码"): 查询股价。返回 price, change_pct。
- search_web(query="关键词"): 搜索网络。返回 results 列表。
- calculate(expression="数学表达式"): 数学计算。返回 result。
- get_time(city="城市名"): 查询当前时间。返回 time。

Reply format:
Thought: <reason about what to do>
Action: <tool_name(param="value")>

After tool response, continue with:
Thought: <further reasoning>
Action: <next tool> or Final Answer: <complete answer>"""

FEW_SHOT = """
Example 1:
User: 北京今天天气怎么样？
Assistant: Thought: 需要查询北京天气。
Action: get_weather(city="北京")

Observation: {"city":"北京","temperature":22,"condition":"晴","humidity":45}
Assistant: Thought: 数据已获取，可以回答。
Final Answer: 北京今天晴，温度22°C，湿度45%。

Example 2:
User: 查一下苹果的股价
Assistant: Thought: 苹果的股票代码是AAPL。
Action: get_stock_price(symbol="AAPL")

Observation: {"symbol":"AAPL","price":185.5,"change_pct":2.3}
Assistant: Thought: 股价已获取。
Final Answer: 苹果(AAPL)当前股价$185.5，较前日上涨2.3%。
"""

BASELINE_A_SYSTEM = REACT_SYSTEM + "\n\nHere are examples:\n" + FEW_SHOT


# ── Model (shared base) ───────────────────────────────────
base_model = None
ft_model = None
tokenizer = None

def load_models():
    global base_model, ft_model, tokenizer
    if base_model is not None:
        return
    print("Loading shared base model...")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        max_seq_length=2048, dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(base_model)

    print("Loading LoRA adapter for FT model...")
    ft_model = PeftModel.from_pretrained(
        base_model, "outputs/react-qwen2.5-7b-qlora_20260508_1810/checkpoint-1000",
    )
    FastLanguageModel.for_inference(ft_model)
    print("Both models ready.")


# ── Run one agent ─────────────────────────────────────────
def run_agent(model, user_query: str, system_prompt: str, max_turns: int = 6):
    conversation = [
        {"role": "system", "content": system_prompt},
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
        raw = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True,
        ).strip()

        thought = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|\Z)", raw, re.DOTALL)
        action = re.search(r"Action:\s*(.+?)(?:\n|$)", raw)
        final = re.search(r"Final Answer:\s*(.+?)$", raw, re.DOTALL)

        thought_str = thought.group(1).strip() if thought else ""
        action_str = action.group(1).strip() if action else ""
        final_str = final.group(1).strip() if final else ""

        # Check for hallucinated Observation
        has_obs_hall = "Observation:" in raw and not any(
            c["role"] == "tool" and "Observation:" in c.get("content", "")
            for c in conversation
        )

        conversation.append({"role": "assistant", "content": raw})

        if final_str:
            trajectory.append({
                "turn": turn + 1, "thought": thought_str,
                "action": "", "type": "final",
                "content": final_str, "obs_hall": has_obs_hall,
            })
            return trajectory, True

        if action_str:
            obs = execute_action(action_str)
            conversation.append({"role": "tool", "content": obs})
            trajectory.append({
                "turn": turn + 1, "thought": thought_str,
                "action": action_str, "type": "action",
                "content": obs, "obs_hall": has_obs_hall,
            })
        else:
            trajectory.append({
                "turn": turn + 1, "thought": thought_str,
                "action": "", "type": "error",
                "content": "未输出有效Action", "obs_hall": has_obs_hall,
            })
            return trajectory, False

    return trajectory, False


# ── Render one trajectory column ──────────────────────────
def render_column(trajectory: list, success: bool, model_name: str, accent_color: str) -> str:
    status_icon = "&#10003; 完成" if success else "&#10007; 未完成"
    status_color = "#4caf50" if success else "#f44336"

    h = f"""
    <div style="background: #fff; border: 2px solid {accent_color}; border-radius: 8px; padding: 12px; min-height: 200px;">
      <h3 style="color: {accent_color}; margin: 0 0 8px 0; font-size: 15px;">{model_name}</h3>
    """

    for step in trajectory:
        if step["obs_hall"]:
            h += """
            <div style="background: #ffebee; border-left: 4px solid #f44336; padding: 6px; margin: 4px 0; font-size: 12px;">
              <strong>&#9888; 幻觉:</strong> 模型自己编造了 Observation！
            </div>"""

        if step["thought"]:
            h += f"""
            <div style="background: #fff3e0; border-left: 3px solid #ff9800; padding: 6px; margin: 3px 0; border-radius: 3px; font-size: 13px;">
              <strong>T{step['turn']}.</strong> <span style="color:#e65100;">{html.escape(step['thought'][:150])}</span>
            </div>"""

        if step["action"]:
            h += f"""
            <div style="background: #e8f5e9; border-left: 3px solid #4caf50; padding: 6px; margin: 3px 0 3px 12px; border-radius: 3px; font-size: 12px;">
              <strong>ACTION</strong> <code style="color:#1b5e20;">{html.escape(step['action'][:100])}</code>
            </div>"""

        if step["type"] == "action":
            try:
                od = json.loads(step["content"])
                obs_str = json.dumps(od, indent=2, ensure_ascii=False)[:300]
                is_err = "error" in od
            except Exception:
                obs_str = step["content"][:300]
                is_err = False
            bg = "#ffebee" if is_err else "#e8eaf6"
            border = "#f44336" if is_err else "#9e9e9e"
            h += f"""
            <div style="background:{bg}; border-left:3px solid {border}; padding:6px; margin:2px 0 2px 24px; border-radius:3px; font-size:12px;">
              <pre style="margin:0;white-space:pre-wrap;">{html.escape(obs_str)}</pre>
            </div>"""

        elif step["type"] == "final":
            h += f"""
            <div style="background:#f3e5f5; border-left:3px solid #9c27b0; padding:6px; margin:3px 0; border-radius:3px; font-size:13px;">
              <strong>ANSWER:</strong> {html.escape(step['content'][:200])}
            </div>"""

    h += f"""
      <div style="margin-top: 8px; text-align: center; font-size: 13px; color: {status_color};">
        {status_icon} | {len(trajectory)} turns
      </div>
    </div>"""
    return h


# ── Main UI handler ───────────────────────────────────────
def run_compare(user_query: str):
    load_models()

    # Run both
    traj_a, succ_a = run_agent(base_model, user_query, BASELINE_A_SYSTEM)
    traj_ft, succ_ft = run_agent(ft_model, user_query, REACT_SYSTEM)

    # Count hallucinated observations
    obs_hall_a = sum(1 for t in traj_a if t["obs_hall"])
    obs_hall_ft = sum(1 for t in traj_ft if t["obs_hall"])

    # Render both columns
    col_a = render_column(traj_a, succ_a,
                          f"Baseline A (Instruct + Few-shot) | ObsHall: {obs_hall_a}",
                          "#f44336")
    col_ft = render_column(traj_ft, succ_ft,
                           f"Ours (QLoRA Fine-tuned) | ObsHall: {obs_hall_ft}",
                           "#4caf50")

    # Summary header
    header = f"""
    <div style="background:#e3f2fd; padding:12px; border-radius:8px; margin-bottom:12px;">
      <strong>Q:</strong> {html.escape(user_query)}
    </div>
    <div style="display: flex; gap: 16px;">
      <div style="flex: 1;">{col_a}</div>
      <div style="flex: 1;">{col_ft}</div>
    </div>"""

    return header


def create_ui():
    with gr.Blocks(title="ReAct Dual Demo") as demo:
        gr.Markdown("""
        # ReAct Agent — 微调前后对比

        **左: Baseline A** (Qwen2.5-7B-Instruct + 强 ReAct Prompt + Few-shot)
        **右: Ours** (QLoRA 微调模型 + ReAct Prompt)

        12GB 4070s 单卡同时运行两个模型。实时展示完整推理轨迹。
        """)

        query = gr.Textbox(
            label="你的问题",
            placeholder="例如：帮我查一下苹果的股价，顺便看看纽约的天气",
            lines=2,
        )
        btn = gr.Button("开始对比", variant="primary", size="lg")
        output = gr.HTML()

        gr.Examples(
            examples=[
                ["北京今天天气怎么样？适合户外运动吗？"],
                ["帮我查一下苹果(AAPL)的当前股价"],
                ["帮我算一下 (15.8+23.4)*1.15 等于多少"],
                ["查一下微软股价，顺便看看西雅图天气"],
                ["帮我查一下火星上的天气怎么样？"],
                ["搜索Python异步编程的最佳实践"],
            ],
            inputs=[query],
        )

        btn.click(fn=run_compare, inputs=[query], outputs=[output])

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
