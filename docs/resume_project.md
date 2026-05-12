# ReAct 指令微调 — Qwen2.5-7B Agent 推理能力强化

**个人项目 | 2026.05 | 独立完成**

## 项目背景

在单卡 RTX 4070s (12GB) 上对 Qwen2.5-7B-Instruct 进行 ReAct（Reasoning + Acting）指令微调，使模型学会 Thought → Action → Observation 的闭环推理模式。模型在调用工具前先输出推理过程，接收工具返回后继续决策，遇到错误时能自动切换策略。

## 技术方案

- 构造 **10,000 条** ReAct 格式训练数据（HotpotQA + 程序化模板 + DeepSeek 合成），含 10% 错误恢复负样本
- 使用 **Unsloth + QLoRA** (4-bit, R=32) 在 12GB 显存限制下完成微调，显存占用 ~9.8GB
- 训练 2.2 epochs，最佳 eval_loss = 0.1766
- 支持 **DeepSpeed ZeRO-2** 多卡扩展
- 使用 **SwanLab** 进行实验追踪

## 量化成果

| 指标 | 基座模型 | 微调后 | 提升 |
|------|---------|--------|------|
| 总体成功率 (100 cases) | 88% | **92%** | +4pp |
| 错误恢复率 (30 cases) | 77% | **80%** | +3pp |
| Observation 幻觉率 | ~90% | **0%** | 消除 |

## 工程实践

- 编写完整数据管道（生成/清洗/校验/负样本注入）
- 实现 ReAct 推理评估框架，支持 LLM-as-Judge 自动化评分
- 实现工具执行层，支持真实 API 调用（天气/股票/搜索/计算/时间）
- 构建 Gradio 双轨对比 Demo（基座 vs 微调模型同屏推理）
- 编写训练监控脚本（进度推送/手机通知）
- 模型已开源至 HuggingFace，代码托管于 GitHub

## 技术栈

`PyTorch` `Transformers` `Unsloth` `PEFT/QLoRA` `DeepSpeed` `SwanLab` `Gradio` `DeepSeek API`

---

**GitHub**: https://github.com/Yuhang-dev/ReAct-Qlora
**HuggingFace**: https://huggingface.co/Yuh4ng/react-qwen2.5-7b-qlora-4070s
