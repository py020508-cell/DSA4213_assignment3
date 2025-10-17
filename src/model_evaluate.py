import json
import os
import matplotlib.pyplot as plt
import numpy as np


def read_metrics(trainer_state_path):

    if not os.path.exists(trainer_state_path):
        raise FileNotFoundError(f"{trainer_state_path} does not exist！")

    with open(trainer_state_path, "r") as f:
        trainer_state = json.load(f)

    log_history = trainer_state.get("log_history", [])
    epochs = []
    accuracies = []
    f1_scores = []

    for log in log_history:
        if "epoch" in log and "eval_accuracy" in log:
            epochs.append(log["epoch"])
            accuracies.append(log["eval_accuracy"])
            f1_scores.append(log["eval_f1"])

    return epochs, accuracies, f1_scores


def main():
    # trainer_state.json 路径
    full_train_path = "../result/full/checkpoint-4689/trainer_state.json"
    lora_train_path = "../result/LORA/checkpoint-4689/trainer_state.json"

    # get Full 和 LoRA
    epochs_full, acc_full, f1_full = read_metrics(full_train_path)
    epochs_lora, acc_lora, f1_lora = read_metrics(lora_train_path)

    # Separate display curves
    offset = 0.01  
    acc_full_plot = np.array(acc_full) + offset
    f1_full_plot  = np.array(f1_full) - offset
    acc_lora_plot = np.array(acc_lora) + 2*offset
    f1_lora_plot  = np.array(f1_lora) - 2*offset

    plt.figure(figsize=(7, 5))

    # Draw a line chart
    plt.plot(epochs_full, acc_full, marker='o', color='mediumorchid', linestyle='-', label='Full Accuracy')
    plt.plot(epochs_full, f1_full, marker='s', color='mediumorchid', linestyle='--', label='Full F1')
    plt.plot(epochs_lora, acc_lora, marker='o', color='pink', linestyle='-', label='LoRA Accuracy')
    plt.plot(epochs_lora, f1_lora, marker='s', color='pink', linestyle='--', label='LoRA F1')

    plt.xticks(np.array(epochs_full))
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Figure2 Full vs LoRA Evaluation per Epoch")

    
    all_scores = np.concatenate([acc_full_plot, f1_full_plot, acc_lora_plot, f1_lora_plot])
    plt.ylim(all_scores.min() - 0.01, all_scores.max() + 0.01)

    plt.grid(True, linestyle='--', alpha=0.5)

    # Add value labels
    for x, y in zip(epochs_full, acc_full): plt.text(x, y, f'{y:.3f}', ha='center', va='center', fontsize=7,
                                                     color='black')
    for x, y in zip(epochs_full, f1_full):  plt.text(x, y, f'{y:.3f}', ha='center', va='center', fontsize=7,
                                                     color='black')
    for x, y in zip(epochs_lora, acc_lora): plt.text(x, y, f'{y:.3f}', ha='center', va='center', fontsize=7,
                                                     color='black')
    for x, y in zip(epochs_lora, f1_lora):  plt.text(x, y, f'{y:.3f}', ha='center', va='center', fontsize=7,
                                                     color='black')
    plt.legend()
    plt.tight_layout()

    # save the image
    save_path = "../result/full_lora_compare.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("have done")


if __name__ == "__main__":
    main()
