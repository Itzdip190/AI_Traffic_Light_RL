import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Ensure project root is in path ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.traffic_env import TrafficEnv
from stable_baselines3 import DQN

# ================================================================
# 🚦 Helper: Test Fixed Timer Controller
# ================================================================
def test_fixed_timer(episodes=5):
    print("🚦 Testing Fixed Timer Controller...")
    results = {"episode": [], "avg_wait_time": [], "throughput": [], "rewards": []}

    for ep in range(episodes):
        print(f"  ▶️ Episode {ep + 1}/{episodes}")
        env = TrafficEnv(
            net_file="sumo_project/cross.net.xml",
            route_file="sumo_project/cross.rou.xml",
            cfg_file="sumo_project/cross.sumocfg",
            use_gui=False,
            max_steps=500,
            fixed_timing=True
        )
        obs, _ = env.reset()
        done = False
        total_reward = 0
        total_wait = 0
        step_count = 0

        while not done:
            obs, reward, done, _, _ = env.step(0)
            total_reward += reward
            total_wait += -reward  # since reward = -waiting_time
            step_count += 1

        env.close()
        results["episode"].append(ep + 1)
        results["avg_wait_time"].append(total_wait / step_count)
        results["throughput"].append(step_count)
        results["rewards"].append(total_reward)

        print(f"     ✅ Avg Wait: {total_wait / step_count:.2f} | Total Reward: {total_reward:.2f}")

    return results


# ================================================================
# 🤖 Helper: Test RL (DQN) Controller
# ================================================================
def test_rl_controller(episodes=5):
    print("\n🤖 Testing DQN RL Controller...")

    env = TrafficEnv(
        net_file="sumo_project/cross.net.xml",
        route_file="sumo_project/cross.rou.xml",
        cfg_file="sumo_project/cross.sumocfg",
        use_gui=False,
        max_steps=500,
        fixed_timing=False
    )

    model_path = "train/dqn_model.zip"
    if os.path.exists(model_path):
        print("📂 Loading existing DQN model...")
        model = DQN.load(model_path, env=env)
    else:
        print("🧠 Training new DQN model (short run)...")
        model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001,
                    buffer_size=10000, exploration_fraction=0.2)
        model.learn(total_timesteps=10000)
        model.save(model_path)
        print("✅ Model trained and saved.")

    results = {"episode": [], "avg_wait_time": [], "throughput": [], "reward": []}

    for ep in range(episodes):
        print(f"  ▶️ Episode {ep + 1}/{episodes}")
        obs, _ = env.reset()
        done = False
        total_reward = 0
        total_wait = 0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            total_wait += -reward
            step_count += 1

        results["episode"].append(ep + 1)
        results["avg_wait_time"].append(total_wait / step_count)
        results["throughput"].append(step_count)
        results["reward"].append(total_reward)

        print(f"     ✅ Avg Wait: {total_wait / step_count:.2f} | Total Reward: {total_reward:.2f}")

    env.close()
    return results


# ================================================================
# 💾 Save metrics for analysis
# ================================================================
def save_metrics(fixed_results, rl_results):
    os.makedirs("logs", exist_ok=True)

    pd.DataFrame({
        "episode": fixed_results["episode"],
        "avg_wait_time": fixed_results["avg_wait_time"],
        "throughput": fixed_results["throughput"],
        "reward": fixed_results["rewards"]
    }).to_csv("logs/fixed_metrics.csv", index=False)

    pd.DataFrame({
        "episode": rl_results["episode"],
        "avg_wait_time": rl_results["avg_wait_time"],
        "throughput": rl_results["throughput"],
        "reward": rl_results["reward"]
    }).to_csv("logs/rl_metrics.csv", index=False)

    print("💾 Metrics saved to logs/fixed_metrics.csv and logs/rl_metrics.csv")


# ================================================================
# 🏁 Main Execution
# ================================================================
if __name__ == "__main__":
    print("Day 7: Compare RL vs Fixed Timer Performance\n")

    fixed_results = test_fixed_timer(episodes=5)
    rl_results = test_rl_controller(episodes=5)

    save_metrics(fixed_results, rl_results)

    print("\n✅ All metrics logged successfully. Run `python analyze_results.py` to visualize results.")
