# eval/eval_ppo.py
"""
Evaluate a trained PPO traffic light control model on SUMO.
Compatible with Gymnasium 1.x and Stable-Baselines3 2.x.
Automatically detects the best or final model and logs results.
"""

import os
import sys
import csv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# --- Fix for Windows import issues ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.traffic_env import TrafficEnv  # ✅ Your custom SUMO environment


# ================= CONFIG =================
MODEL_PATH = "logs/ppo/run_seed_42/ppo_final_numpy1.zip"   # Folder containing ppo_final.zip / best_model.zip
EVAL_EPISODES = 10                   # Number of test episodes
OUT_CSV = "logs/ppo/eval_results.csv"
USE_GUI = True                       # 👀 True = visualize SUMO-GUI, False = faster headless eval
MAX_STEPS = 1000                     # Max steps per episode
SEED = 42                            # Random seed for reproducibility
# ==========================================


def find_model(model_dir: str):
    """
    Automatically detect which PPO model to load.
    Prioritizes 'best_model.zip', falls back to 'ppo_final.zip'.
    """
    best_model = os.path.join(model_dir, "best_model", "best_model.zip")
    final_model = os.path.join(model_dir, "ppo_final.zip")

    if os.path.exists(best_model):
        print(f"✅ Found best model: {best_model}")
        return best_model
    elif os.path.exists(final_model):
        print(f"⚠️ Best model not found, using final model: {final_model}")
        return final_model
    else:
        raise FileNotFoundError(
            f"❌ No PPO model found in {model_dir}\n"
            f"Expected:\n  {best_model}\n  or\n  {final_model}"
        )


def make_env():
    """Factory for TrafficEnv (wrapped in DummyVecEnv)."""
    def _init():
        env = TrafficEnv(
            net_file="sumo_project/map.net.xml",
            route_file="sumo_project/routes.rou.xml",
            cfg_file="sumo_project/cross.sumocfg",
            use_gui=USE_GUI,
            max_steps=MAX_STEPS,
        )
        np.random.seed(SEED)
        return env
    return _init


def evaluate_model():
    """Evaluate a PPO model and save results."""
    model_path = MODEL_PATH

    print(f"🚦 Loading PPO model from: {model_path}")

    # Load environment and model
    env = DummyVecEnv([make_env()])
    model = PPO.load(MODEL_PATH, device="cpu")


    records = []
    total_rewards = []

    for ep in range(EVAL_EPISODES):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        print(f"\n🎬 Starting Evaluation Episode {ep + 1}/{EVAL_EPISODES}")
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = term.any() or trunc.any()

            if done:
                metrics = info[0] if isinstance(info, list) else info
                metrics["episode"] = ep + 1
                metrics["steps"] = steps
                metrics["reward_total"] = total_reward
                records.append(metrics)
                total_rewards.append(total_reward)
                print(f"🏁 Episode {ep + 1} finished | Steps: {steps} | Reward: {total_reward:.2f}")
                break

    env.close()

    # ---------- SAVE CSV RESULTS ----------
    keys = sorted(set().union(*(r.keys() for r in records)))
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in keys})

    # ---------- SUMMARY ----------
    avg_reward = np.mean(total_rewards)
    print(f"\n✅ PPO Evaluation Complete.")
    print(f"📂 Results saved at: {OUT_CSV}")
    print(f"📊 Average Reward: {avg_reward:.2f}\n")

    print("📈 Average Environment Metrics:")
    for key in keys:
        if key not in ["episode", "steps", "reward_total"]:
            try:
                avg_val = np.mean([r[key] for r in records if key in r])
                print(f"  • {key}: {avg_val:.3f}")
            except Exception:
                pass


if __name__ == "__main__":
    evaluate_model()
