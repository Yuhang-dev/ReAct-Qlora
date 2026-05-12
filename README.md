# ReAct QLoRA Fine-tuning on Qwen2.5-7B-Instruct

Single-GPU (RTX 4070s 12GB) ReAct instruction fine-tuning with QLoRA. Supports single-GPU and multi-GPU (DeepSpeed ZeRO-2) training.

## Results

### Three-way benchmark (100 test cases)

| Model | Success | ObsHall |
|-------|---------|---------|
| **FT_ReAct (Ours)** | **92.0%** | 0% |
| Baseline_B (Instruct + direct call) | 88.0% | 0% |
| Baseline_A (Instruct + ReAct prompt + FS) | 83.0% | 1% |

| Task Type | Baseline_B | Baseline_A | **Ours** |
|-----------|-----------|-----------|----------|
| Single-tool (35) | 94% | 91% | **97%** |
| Multi-tool (35) | 91% | 91% | **97%** |
| Error Recovery (30) | 77% | 63% | **80%** |

### Training

| Metric | Value |
|--------|-------|
| Base Model | Qwen2.5-7B-Instruct |
| Method | QLoRA (R=32, Alpha=64) |
| Data | 8,640 ReAct trajectories (10% negative samples) |
| Best Eval Loss | 0.1766 (step 1,000) |
| VRAM | ~9.8 GB / 12 GB |
| GPU | RTX 4070 SUPER |

## Structure

```
├── configs/                    # Training & DeepSpeed configs
│   ├── train_config.yaml
│   └── deepspeed_zero2.json
├── scripts/
│   ├── 04_train.py             # QLoRA training (single/multi-GPU)
│   ├── eval_full_100.py        # 100-case three-way evaluation
│   └── watch_train.py          # Training monitor
├── src/
│   ├── data/                   # Data format & preprocessing
│   ├── eval/                   # Metrics & judge
│   └── demo/                   # Gradio dual-track demo
└── tests/
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

## Example ReAct Trajectory

```
Q: 帮我查一下火星上的天气怎么样？

T1 THOUGHT: 火星不在地球数据库中，先尝试 get_weather
   ACTION:  get_weather(city="火星")
   OBS:     {"error": "invalid_city", "message": "not found in database"}

T2 THOUGHT: 工具不支持火星，改用搜索
   ACTION:  search_web(query="火星 气候")
   OBS:     {"results": [...]}

T3 ANSWER: 天气工具只支持地球城市。根据搜索，火星表面平均温度约-63°C...
```

## Features

- QLoRA (4-bit) with Unsloth: ~9.8GB VRAM on RTX 4070s
- DeepSpeed ZeRO-2 for multi-GPU training
- 10% negative samples for error recovery training
- Dual-track Gradio demo (Base vs FT side-by-side)
- SwanLab experiment tracking

## Multi-GPU Training

```bash
deepspeed --num_gpus=4 scripts/04_train.py \
  --config configs/train_config.yaml \
  --deepspeed configs/deepspeed_zero2.json
```

## License

MIT
