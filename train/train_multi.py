import os
import sys
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.multi_traffic_env import MultiTrafficEnv


# === Paths ===
NET_FILE = "sumo_project/cross.net.xml"
ROUTE_FILE = "sumo_project/cross.rou.xml"
CFG_FILE = "sumo_project/cross.sumocfg"
LOG_DIR = "logs/multi"
os.makedirs(LOG_DIR, exist_ok=True)


class SingleTLEnvWrapper(gym.Env):
    """
    Wraps one traffic light for independent PPO training.
    Makes it a valid Gymnasium environment.
    """
    metadata = {"render.modes": []}

    def __init__(self, multi_env, tl_id):
        super().__init__()
        self.multi_env = multi_env
        self.tl_id = tl_id
        self.action_space = multi_env.action_space[tl_id]
        self.observation_space = multi_env.observation_space[tl_id]

    def reset(self, seed=None, options=None):
        obs, _ = self.multi_env.reset()
        return obs[self.tl_id], {}

    def step(self, action):
        # Each step executes random actions for others + selected action for this TL
        actions = {tl: self.multi_env.action_space[tl].sample() for tl in self.multi_env.tls_ids}
        actions[self.tl_id] = action
        obs, rewards, dones, done_all, infos = self.multi_env.step(actions)
        done = done_all or dones[self.tl_id]
        return obs[self.tl_id], rewards[self.tl_id], done, False, infos[self.tl_id]

    def close(self):
        self.multi_env.close()


def train_multi():
    env = MultiTrafficEnv(NET_FILE, ROUTE_FILE, CFG_FILE, use_gui=False)
    agents = {}

    for tl in env.tls_ids:
        print(f"🟢 Training PPO agent for traffic light: {tl}")
        single_env = SingleTLEnvWrapper(env, tl)

        # Train PPO
        model = PPO("MlpPolicy", single_env, verbose=1, learning_rate=3e-4, n_steps=1024)
        model.learn(total_timesteps=20000)
        model.save(f"{LOG_DIR}/ppo_{tl}.zip")
        agents[tl] = model

    env.close()
    print("✅ Multi-agent PPO training complete.")


if __name__ == "__main__":
    train_multi()
