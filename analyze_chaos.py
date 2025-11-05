import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================
#  Chaos Factor Analysis Script
# ==============================

LOG_FILE = "logs/chaos_metrics.csv"

if not os.path.exists(LOG_FILE):
    print("❌ No log file found. Run a few simulation episodes first.")
    exit()

# --- Load CSV
df = pd.read_csv(LOG_FILE)

# --- Basic summary
print("✅ Chaos Metrics Loaded")
print(df.head())
print("\nStatistics Summary:")
print(df.describe())

# --- Handle missing or empty columns
for col in ["avg_wait", "total_collisions", "total_violators", "total_throughput", "total_reward"]:
    if col not in df.columns:
        print(f"⚠️ Missing column '{col}' in log file.")
        exit()

# --- Plot 1: Reward vs Episode
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["total_reward"], marker="o", label="Total Reward")
plt.title("Episode Reward Trend (Chaos Factor)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 2: Average Waiting Time
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["avg_wait"], color="orange", marker="o", label="Average Wait")
plt.title("Average Waiting Time per Episode")
plt.xlabel("Episode")
plt.ylabel("Avg Waiting Time (s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 3: Collisions vs Episode
plt.figure(figsize=(10, 5))
plt.bar(df["episode"], df["total_collisions"], color="red", label="Collisions")
plt.title("Collisions per Episode")
plt.xlabel("Episode")
plt.ylabel("Collision Count")
plt.grid(True, axis="y")
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 4: Rush vs Normal Episodes (Group Comparison)
rush_df = df[df["rush"] == 1]
normal_df = df[df["rush"] == 0]

if not rush_df.empty and not normal_df.empty:
    avg_rush_wait = rush_df["avg_wait"].mean()
    avg_normal_wait = normal_df["avg_wait"].mean()

    print(f"\n🚦 Average Wait (Rush Episodes): {avg_rush_wait:.2f}")
    print(f"🕒 Average Wait (Normal Episodes): {avg_normal_wait:.2f}")

    plt.figure(figsize=(7, 5))
    plt.bar(["Rush Hour", "Normal"], [avg_rush_wait, avg_normal_wait], color=["crimson", "green"])
    plt.title("Average Waiting Time Comparison")
    plt.ylabel("Avg Waiting Time (s)")
    plt.tight_layout()
    plt.show()

# --- Plot 5: Violators vs Collisions
plt.figure(figsize=(10, 5))
plt.scatter(df["total_violators"], df["total_collisions"], color="purple", alpha=0.7)
plt.title("Violators vs Collisions")
plt.xlabel("Number of Violators")
plt.ylabel("Number of Collisions")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n✅ Analysis complete! All plots generated successfully.")

