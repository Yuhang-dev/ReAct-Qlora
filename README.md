# ReAct QLoRA Fine-tuning on Qwen2.5-7B-Instruct (RTX 4070s 12GB)

> 在单卡 RTX 4070s (12GB) 上完成 Qwen2.5-7B-Instruct 的 ReAct 指令微调，实现 Thought → Action → Observation 闭环推理。

## 项目概述

让模型学会像人类一样思考：先推理（Thought）当前状态 → 调用工具（Action）→ 观察结果（Observation）→ 继续推理或给出最终答案。重点强化**错误恢复能力**——当工具返回错误时，模型能自动切换策略而不是编造假答案。

## 核心成果

### 三线对比评测 (100 cases)

| Model | Success | ObsHall | Description |
|-------|---------|---------|-------------|
| **FT_ReAct (Ours)** | **92.0%** | 0% | QLoRA fine-tuned |
| Baseline_B | 88.0% | 0% | Instruct + direct tool call |
| Baseline_A | 83.0% | 1% | Instruct + ReAct prompt + Few-shot |

### 分任务类型

| Task Type | Baseline_B | Baseline_A | **FT_ReAct** |
|-----------|-----------|-----------|-------------|
| Single-tool (35) | 94% | 91% | **97%** |
| Multi-tool (35) | 91% | 91% | **97%** |
| Error Recovery (30) | 77% | 63% | **80%** |

### 训练指标

| Metric | Value |
|--------|-------|
| Base Model | Qwen2.5-7B-Instruct |
| Method | QLoRA (4-bit, R=32, Alpha=64) |
| Training Data | 8,640 samples (10% negative) |
| Epochs / Steps | 2.2 epochs / 1,000 steps |
| Best Eval Loss | 0.1766 |
| VRAM Usage | ~9.8 GB / 12 GB |
| Training Time | ~29 hours |
| GPU | NVIDIA RTX 4070 SUPER |

## 项目结构

```
Reason-act/
├── configs/              # 训练超参数配置
│   ├── train_config.yaml        # Toy tools 训练配置
│   └── train_alfworld.yaml      # ALFWorld 训练配置
├── scripts/              # 完整管道脚本
│   ├── 03_preprocess.py         # 数据预处理（含负样本注入）
│   ├── 04_train.py              # Unsloth QLoRA 训练入口
│   ├── eval_full_100.py         # 100 条三线对比评测
│   ├── eval_three_way.py        # 30 条快速评测
│   └── watch_train.py           # 训练监控 + 手机推送
├── src/
│   ├── data/
│   │   ├── react_format.py      # ReAct 格式定义 & 校验
│   │   └── negative_sampler.py  # 负样本构造
│   ├── eval/
│   │   ├── metrics.py           # 评测指标
│   │   └── llm_judge.py         # LLM-as-Judge
│   └── demo/
│       ├── react_demo.py        # 单模型 Gradio Demo
│       └── dual_demo.py         # 双轨对比 Demo (Base vs FT)
└── tests/                # 单元测试
```

## 快速开始

### 环境安装

```bash
# 1. PyTorch with CUDA 13.0 (for RTX 4070s)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2. Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 3. Dependencies
pip install transformers datasets accelerate bitsandbytes peft trl
pip install swanlab gradio openai pyyaml
```

### 加载微调模型

```python
from unsloth import FastLanguageModel
from peft import PeftModel

# Load base model in 4-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=2048, dtype=None, load_in_4bit=True,
)

# Load our ReAct LoRA adapter
model = PeftModel.from_pretrained(
    model, "Yuh4ng/react-qwen2.5-7b-qlora-4070s"
)

FastLanguageModel.for_inference(model)
```

### 启动 Demo

```bash
# 单模型 Demo
python -m src.demo.react_demo

# 双轨对比 Demo (同屏展示 Base vs FT)
python -m src.demo.dual_demo
# 打开 http://localhost:7860
```

## ReAct 推理效果

```
用户: 帮我算一下 (15.8+23.4)*1.15 等于多少

T1 THOUGHT: 用户需要计算数学表达式，用 calculate 工具
   ACTION:  calculate(expression="(15.8+23.4)*1.15")
   OBS:     {"result": 48.53}

T2 THOUGHT: 计算完成，直接给出结果
   ANSWER:  结果是 48.53。
```

```
用户: 帮我查一下火星上的天气怎么样？（错误恢复场景）

T1 THOUGHT: 火星不在地球数据库中，先尝试 get_weather
   ACTION:  get_weather(city="火星")
   OBS:     {"error": "invalid_city", "message": "not found in weather database"}

T2 THOUGHT: 不支持火星，改用搜索获取气候知识
   ACTION:  search_web(query="火星 气候")
   OBS:     搜索结果...

T3 ANSWER:  很抱歉，天气工具只支持地球城市。根据搜索，火星表面平均温度约-63°C...
```

## 技术亮点

1. **12GB 极限压缩**: Unsloth QLoRA (4-bit) + gradient checkpointing + Xformers，显存占用 ~9.8GB
2. **负样本训练**: 10% 训练数据包含工具错误/超时/空结果，训练模型学会错误恢复
3. **Observation 幻觉消除**: 微调后 0% 幻觉率（基座模型 Zero-shot 约 90%）
4. **双模型并排 Demo**: 同一 GPU 同时运行 Base 和 FT 模型，实时对比推理轨迹

## License

MIT
