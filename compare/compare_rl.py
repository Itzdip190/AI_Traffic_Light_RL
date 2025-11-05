# compare/compare_rl.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def load_rewards(path):
    # Handle SB3 Monitor CSV files
    df = pd.read_csv(path, comment='#')
    if "r" in df.columns:
        return df["r"].values
    elif "episode_reward" in df.columns:
        return df["episode_reward"].values
    else:
        print(f"⚠️ Columns found in {path}: {df.columns.tolist()}")
        raise ValueError(f"No reward column found in {path}")

def moving_avg(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode="valid")

def find_latest_log(path_pattern):
    files = glob.glob(path_pattern, recursive=True)
    return max(files, key=os.path.getmtime) if files else None

ppo_path = find_latest_log("logs/ppo/**/monitor.csv")
dqn_path = find_latest_log("logs/dqn/**/monitor.csv")

if not ppo_path or not dqn_path:
    print("⚠️ Missing PPO or DQN logs!")
    print(f"PPO path found: {ppo_path}")
    print(f"DQN path found: {dqn_path}")
    raise FileNotFoundError("Please ensure PPO and DQN logs exist before running this comparison.")

print(f"📂 Using PPO log: {ppo_path}")
print(f"📂 Using DQN log: {dqn_path}")

ppo_rewards = load_rewards(ppo_path)
dqn_rewards = load_rewards(dqn_path)

ppo_smooth = moving_avg(ppo_rewards)
dqn_smooth = moving_avg(dqn_rewards)

plt.figure(figsize=(10,6))
plt.plot(ppo_smooth, label="PPO", linewidth=2)
plt.plot(dqn_smooth, label="DQN", linewidth=2, linestyle='--')
plt.title("PPO vs DQN: Learning Curve")
plt.xlabel("Episodes")
plt.ylabel("Smoothed Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
os.makedirs("logs", exist_ok=True)
plt.savefig("logs/compare_ppo_dqn.png")
plt.show()

print("✅ Comparison plot saved at logs/compare_ppo_dqn.png")
