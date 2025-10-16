import json, matplotlib.pyplot as plt, os, numpy as np

def total_minutes(state_path):
    """读 train_runtime 字段 → 分钟"""
    with open(state_path, 'r', encoding='utf-8') as f:
        log = json.load(f, strict=False)
    # 官方秒数
    total_sec = log.get("train_runtime", 0)
    return total_sec / 60.

def main():
    root = "../result"
    full_time = total_minutes(os.path.join(root, "full/checkpoint-4689/trainer_state.json"))
    lora_time = total_minutes(os.path.join(root, "LORA/checkpoint-4689/trainer_state.json"))

    methods = ["Full", "LoRA"]
    times   = [full_time, lora_time]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(methods, times, color=['#C8BFD8', '#FFD1DC'])

    for b, t in zip(bars, times):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + max(times)*0.02,
                f'{t:.1f} min', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel("Training Time (min)")
    ax.set_title("Total Training Time (train_runtime)")
    ax.set_ylim(0, max(times) * 1.15)
    plt.tight_layout()
    plt.savefig("../result/time_compare.png", dpi=300)
    plt.close()
    print("训练时间对比图已保存：../result/time_compare.png")

if __name__ == "__main__":
    main()