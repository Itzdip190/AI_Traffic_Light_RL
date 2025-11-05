# ============================================================
#  File: analyze_results.py
#  Purpose: Analyze RL vs Fixed Timer performance
#           and visualize early experiment results (Day 7)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# ============================================================
# 1️⃣ File Validation
# ============================================================
project_root = os.getcwd()
logs_dir = os.path.join(project_root, "logs")
charts_dir = os.path.join(project_root, "charts")

fixed_path = os.path.join(logs_dir, "fixed_metrics.csv")
rl_path = os.path.join(logs_dir, "rl_metrics.csv")

print("📂 Current Directory:", project_root)
print("🔍 Checking required files...")

if not os.path.exists(fixed_path):
    sys.exit("❌ Missing 'logs/fixed_metrics.csv'. Run `python train/test_compare.py` first.")
if not os.path.exists(rl_path):
    sys.exit("❌ Missing 'logs/rl_metrics.csv'. Run `python train/test_compare.py` first.")

if os.path.getsize(fixed_path) == 0 or os.path.getsize(rl_path) == 0:
    sys.exit("❌ One or both CSV files are empty. Re-run `test_compare.py` to collect valid data.")

# ============================================================
# 2️⃣ Load and Validate Data
# ============================================================
try:
    fixed = pd.read_csv(fixed_path)
    rl = pd.read_csv(rl_path)
except pd.errors.EmptyDataError:
    sys.exit("❌ CSV files are empty or corrupted. Check `test_compare.py` outputs.")

required_cols = ["episode", "avg_wait_time", "throughput"]
for df_name, df in [("Fixed Timer", fixed), ("RL Agent", rl)]:
    for col in required_cols:
        if col not in df.columns:
            sys.exit(f"❌ Missing column '{col}' in {df_name} data. Re-run `test_compare.py`.")

# ============================================================
# 3️⃣ Show Quick Data Summary
# ============================================================
print("\n✅ Loaded Data Successfully!")
print("\n📊 Fixed Timer Data Sample:")
print(fixed.head())

print("\n🤖 RL Agent Data Sample:")
print(rl.head())

print("\n🧩 Summary Statistics:")
summary = pd.DataFrame({
    "Metric": ["Avg Wait (mean)", "Throughput (mean)", "Reward (mean)"],
    "Fixed Timer": [
        fixed["avg_wait_time"].mean(),
        fixed["throughput"].mean(),
        fixed["reward"].mean() if "reward" in fixed.columns else "-"
    ],
    "RL Agent": [
        rl["avg_wait_time"].mean(),
        rl["throughput"].mean(),
        rl["reward"].mean() if "reward" in rl.columns else "-"
    ]
})
print(summary.to_string(index=False))

# ============================================================
# 4️⃣ Plot and Save Charts
# ============================================================
os.makedirs(charts_dir, exist_ok=True)
plt.style.use('seaborn-v0_8-deep')  # cleaner visuals

# --- Plot 1: Average Waiting Time ---
plt.figure(figsize=(8, 5))
plt.plot(fixed["episode"], fixed["avg_wait_time"], label="Fixed Timer", linestyle='--', marker='o')
plt.plot(rl["episode"], rl["avg_wait_time"], label="RL Agent", linestyle='-', marker='x')
plt.title("🚦 Average Waiting Time per Episode")
plt.xlabel("Episode")
plt.ylabel("Average Waiting Time (s)")
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, "avg_wait_time.png"))
plt.close()

# --- Plot 2: Throughput ---
plt.figure(figsize=(8, 5))
plt.plot(fixed["episode"], fixed["throughput"], label="Fixed Timer", linestyle='--', marker='o')
plt.plot(rl["episode"], rl["throughput"], label="RL Agent", linestyle='-', marker='x')
plt.title("🚘 Throughput (Vehicles Passed per Episode)")
plt.xlabel("Episode")
plt.ylabel("Vehicles Passed")
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, "throughput.png"))
plt.close()

# --- Plot 3: Reward Curve ---
if "reward" in rl.columns:
    plt.figure(figsize=(8, 5))
    plt.plot(rl["episode"], rl["reward"], label="RL Reward", color='green', marker='x')
    plt.title("🎯 Reward Curve (RL Agent)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "reward_curve.png"))
    plt.close()

# ============================================================
# 5️⃣ Completion Message
# ============================================================
print("\n✅ Charts saved successfully in 'charts/' folder:")
print("   • avg_wait_time.png")
print("   • throughput.png")
if "reward" in rl.columns:
    print("   • reward_curve.png")
print("\n📈 Analysis complete! RL agent performance can now be compared visually.")
