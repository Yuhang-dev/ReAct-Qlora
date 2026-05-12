### ReAct 指令微调 — Qwen2.5-7B Agent 推理能力强化

在单卡 RTX 4070s (12G) 上使用 QLoRA 对 Qwen2.5-7B-Instruct 进行 ReAct 微调，使模型具备 Thought → Action → Observation 闭环推理与错误恢复能力。

- 构造 10k 条 ReAct 训练数据，其中 10% 为工具调用失败的负样本，用于训练模型的错误恢复策略
- 使用 Unsloth + QLoRA (4-bit, R=32) 将显存压缩至 9.8G，支持 DeepSpeed ZeRO-2 多卡扩展
- 建立三线 Baseline（直接调用 / ReAct Prompt+Few-shot / 微调模型），100 case 评测：成功率 92%（+4pp），Observation 幻觉率从 90% 降至 0%
- 实现完整工具执行层 + Gradio 双轨对比 Demo + LLM-as-Judge 自动评估

**技术栈**：PyTorch, Transformers, QLoRA/PEFT, DeepSpeed, Unsloth, SwanLab
