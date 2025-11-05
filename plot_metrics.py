import os, pandas as pd, matplotlib.pyplot as plt

LOG_PATH = "logs/metrics.csv"
OUT_DIR = "output/plots"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(LOG_PATH)
x = df["sim_time"] if "sim_time" in df else df["step"]

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axes[0].plot(x, df["avg_waiting_time"], label="Avg waiting (s)")
axes[0].set_title("Average Waiting Time")
axes[0].grid(True)

axes[1].plot(x, df["avg_queue_length"], label="Avg queue", color="orange")
axes[1].set_title("Average Queue Length")
axes[1].grid(True)

axes[2].plot(x, df["rl_reward"], label="RL reward", color="green")
axes[2].set_title("Reward")
axes[2].grid(True)
axes[2].set_xlabel("Simulation Time (s)")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "metrics_plot.png"), dpi=200)
print(f"📈 Plot saved → {OUT_DIR}/metrics_plot.png")

# Rolling average example
df["avg_waiting_ma"] = df["avg_waiting_time"].rolling(30, min_periods=1).mean()
plt.figure(figsize=(10,4))
plt.plot(x, df["avg_waiting_time"], alpha=0.4, label="Raw")
plt.plot(x, df["avg_waiting_ma"], label="30-step MA", linewidth=2)
plt.title("Smoothed Average Waiting Time")
plt.xlabel("Simulation Time (s)")
plt.ylabel("Waiting (s)")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "waiting_smooth.png"), dpi=200)
print(f"📊 Plot saved → {OUT_DIR}/waiting_smooth.png")
