# train/train_dqn.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from env.traffic_env import TrafficEnv


# --- Configuration ---
RUN_NAME = "run_seed_42"
LOG_DIR = f"logs/dqn/{RUN_NAME}"
MODEL_DIR = "results/models"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Initialize environment ---
env = TrafficEnv(
    net_file="sumo_project/cross.net.xml",
    route_file="sumo_project/cross.rou.xml",
    cfg_file="sumo_project/cross.sumocfg",
    use_gui=False,
    max_steps=1000
)

# --- Wrap environment for logging ---
env = Monitor(env, LOG_DIR)

# --- Create DQN model ---
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=50000,            # larger buffer = more stable learning
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    target_update_interval=1000,
    train_freq=4,
    verbose=1,
    tensorboard_log=LOG_DIR,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    seed=42,
)

# --- Train model ---
print("🚦 Training DQN agent...")
model.learn(total_timesteps=100000, log_interval=10)
print("✅ Training completed!")

# --- Save trained model ---
save_path = os.path.join(MODEL_DIR, "dqn_traffic_model")
model.save(save_path)
print(f"💾 Model saved at: {save_path}.zip")

# --- Evaluate model performance ---
print("📊 Evaluating trained DQN agent...")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
print(f"Average Reward: {mean_reward:.2f} ± {std_reward:.2f}")

# --- Cleanup ---
env.close()
print("🏁 Environment closed successfully.")
