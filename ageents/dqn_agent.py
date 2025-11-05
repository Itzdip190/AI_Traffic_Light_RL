import os
import gymnasium as gym
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from env.traffic_env import TrafficEnv  # your custom environment

# ---------- 1. Create Environment ----------
env = TrafficEnv(
    net_file="sumo_project/cross.net.xml",
    route_file="sumo_project/cross.rou.xml",
    cfg_file="sumo_project/cross.sumocfg",
    max_steps=300,
)

# ---------- 2. Logging Setup ----------
log_dir = "results/logs/"
os.makedirs(log_dir, exist_ok=True)
new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

# ---------- 3. Create DQN Agent ----------
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0005,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    tau=0.1,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    verbose=1,
    tensorboard_log=log_dir,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

model.set_logger(new_logger)

# ---------- 4. Train the Model ----------
print("🚦 Training DQN Agent...")
model.learn(total_timesteps=10000)  # ~10k steps for first test

# ---------- 5. Save the Model ----------
save_path = "results/models/dqn_traffic_model"
model.save(save_path)
print(f"✅ Model saved at: {save_path}")

env.close()
