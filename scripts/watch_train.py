"""Monitor training progress and push key metrics to phone every 30 min.

Usage: python scripts/watch_train.py --pushplus_token YOUR_TOKEN
Get token: https://www.pushplus.plus (微信扫码即可)
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests


def find_latest_run(output_dir: str) -> Path | None:
    """Find the latest training run directory."""
    d = Path(output_dir)
    if not d.exists():
        return None
    runs = sorted(
        [p for p in d.iterdir() if p.is_dir() and not p.name.startswith("swan")],
        key=os.path.getmtime, reverse=True,
    )
    return runs[0] if runs else None


def read_trainer_state(run_dir: Path) -> dict | None:
    """Read trainer_state.json for real metrics."""
    state_file = run_dir / "trainer_state.json"
    if not state_file.exists():
        return None

    try:
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def parse_progress(state: dict, total_steps: int = 1620) -> dict:
    """Extract key metrics from trainer state."""
    logs = state.get("log_history", [])
    if not logs:
        return {"step": 0, "epoch": 0}

    last = logs[-1]
    current_step = last.get("step", 0)
    current_epoch = last.get("epoch", 0)
    loss = last.get("loss", None)
    eval_loss = last.get("eval_loss", None)
    lr = last.get("learning_rate", None)
    grad_norm = last.get("grad_norm", None)

    # Find recent loss trend (last 5 logged steps)
    recent_losses = []
    for entry in logs[-10:]:
        if "loss" in entry:
            recent_losses.append(entry["loss"])

    # Best metrics
    best_metric = state.get("best_metric")
    best_model_checkpoint = state.get("best_model_checkpoint", "")

    progress_pct = (current_step / total_steps) * 100 if total_steps > 0 else 0

    return {
        "step": current_step,
        "total_steps": total_steps,
        "progress_pct": progress_pct,
        "epoch": current_epoch,
        "total_epochs": state.get("num_train_epochs", 3),
        "loss": loss,
        "eval_loss": eval_loss,
        "learning_rate": lr,
        "grad_norm": grad_norm,
        "recent_losses": recent_losses,
        "best_metric": best_metric,
        "best_model_checkpoint": best_model_checkpoint,
        "finished": current_step >= total_steps,
    }


def estimate_time_remaining(
    step: int, total_steps: int, elapsed_seconds: float
) -> str:
    """Estimate remaining training time."""
    if step < 2:
        return "计算中..."

    sec_per_step = elapsed_seconds / step
    remaining_steps = total_steps - step
    remaining_sec = sec_per_step * remaining_steps

    eta = datetime.now() + timedelta(seconds=remaining_sec)
    hours = int(remaining_sec // 3600)
    minutes = int((remaining_sec % 3600) // 60)

    return f"{hours}h {minutes}m (预计 {eta.strftime('%H:%M')} 完成)"


def build_progress_message(info: dict, run_name: str, elapsed: float) -> str:
    """Build a human-readable progress message for push."""
    eta = estimate_time_remaining(
        info["step"], info["total_steps"], elapsed
    )

    lines = [
        f"📊 ReAct 训练进度报告",
        f"",
        f"🏷 Run: {run_name}",
        f"📍 Step: {info['step']}/{info['total_steps']} ({info['progress_pct']:.1f}%)",
        f"📖 Epoch: {info['epoch']:.2f}/{info['total_epochs']}",
        f"⏱ 预计剩余: {eta}",
    ]

    if info["loss"] is not None:
        lines.append(f"📉 Train Loss: {info['loss']:.4f}")

    if info["eval_loss"] is not None:
        lines.append(f"🔍 Eval Loss: {info['eval_loss']:.4f}")

    if info["learning_rate"] is not None:
        lines.append(f"🔧 LR: {info['learning_rate']:.2e}")

    if info["grad_norm"] is not None:
        lines.append(f"📐 Grad Norm: {info['grad_norm']:.2f}")

    if info["recent_losses"]:
        losses = info["recent_losses"]
        if len(losses) >= 3:
            trend = "📈 上升" if losses[-1] > losses[0] else "📉 下降"
            lines.append(f"📊 Loss 趋势: {trend} ({losses[0]:.4f} → {losses[-1]:.4f})")

    if info.get("best_metric"):
        lines.append(f"🏆 Best Eval Loss: {info['best_metric']:.4f}")

    lines.append(f"")
    lines.append(f"📱 SwanLab: https://swanlab.cn/@chengyuhang/react-finetune")

    return "\n".join(lines)


def send_pushplus(token: str, title: str, content: str) -> bool:
    """Send notification via PushPlus."""
    try:
        resp = requests.post(
            "https://www.pushplus.plus/send",
            json={"token": token, "title": title, "content": content},
            timeout=10,
        )
        data = resp.json()
        return data.get("code") == 200
    except Exception as e:
        print(f"Push failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Monitor training with progress pushes")
    parser.add_argument("--pushplus_token", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--interval", type=int, default=1800,
                        help="Check interval in seconds (default: 30 min)")
    parser.add_argument("--total_steps", type=int, default=1620)
    args = parser.parse_args()

    interval = args.interval
    print(f"🔍 监控训练中... (每 {interval // 60} 分钟检查一次)")
    print(f"📱 PushPlus: {'已配置' if args.pushplus_token else '未配置 (仅终端输出)'}")
    print()

    start_time = time.time()
    last_push_step = 0
    finish_notified = False

    while True:
        run_dir = find_latest_run(args.output_dir)

        if run_dir is None:
            print(f"[{time.strftime('%H:%M:%S')}] 等待训练开始...")
            time.sleep(interval)
            continue

        state = read_trainer_state(run_dir)
        if state is None:
            print(f"[{time.strftime('%H:%M:%S')}] 等待 trainer_state.json...")
            time.sleep(interval)
            continue

        info = parse_progress(state, args.total_steps)
        elapsed = time.time() - start_time

        # ── Completion notification ──────────────────
        if info["finished"] and not finish_notified:
            final_msg = (
                f"🎉 ReAct 训练完成！\n\n"
                f"🏷 Run: {run_dir.name}\n"
                f"📖 完成 Epochs: {info['epoch']:.1f}\n"
                f"📉 Final Loss: {info['loss']:.4f}\n"
                f"🏆 Best Eval Loss: {info.get('best_metric', 'N/A')}\n"
                f"💾 模型路径: {run_dir / 'final_lora'}\n\n"
                f"下一步: 运行评测 & 合并部署"
            )
            print(f"\n{'='*60}")
            print(final_msg)
            print(f"{'='*60}")

            if args.pushplus_token:
                ok = send_pushplus(args.pushplus_token, "🎉 ReAct 训练完成！", final_msg)
                print("✅ 手机通知已发送" if ok else "❌ 通知发送失败")

            finish_notified = True
            break

        # ── Periodic progress push ───────────────────
        if info["step"] > last_push_step:
            last_push_step = info["step"]

            msg = build_progress_message(info, run_dir.name, elapsed)

            # Terminal output
            print(f"\n{'─'*50}")
            print(msg)
            print(f"{'─'*50}")

            # Phone push
            if args.pushplus_token:
                title = f"ReAct [{info['progress_pct']:.0f}%] Step {info['step']}/{info['total_steps']}"
                send_pushplus(args.pushplus_token, title, msg)

        time.sleep(interval)


if __name__ == "__main__":
    main()
