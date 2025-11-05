# train/train_ppo.py
import os
import sys
import random
import numpy as np
from multiprocessing import freeze_support
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

# --- Ensure root path is importable ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.traffic_env import TrafficEnv  # ✅ your SUMO environment

# ================== CONFIG ==================
NET_FILE = "sumo_project/cross.net.xml"
ROUTE_FILE = "sumo_project/cross.rou.xml"
CFG_FILE = "sumo_project/cross.sumocfg"

USE_GUI = False           # turn True if you want SUMO-GUI visualization
MAX_STEPS = 1000          # steps per episode
N_ENVS = 1                # ✅ single SUMO instance for stability on Windows
TOTAL_TIMESTEPS = 50_000  # ✅ start small, can increase later
SEED = 42

BASE_LOG_DIR = "logs/ppo"
LOG_DIR = os.path.join(BASE_LOG_DIR, f"run_seed_{SEED}")
EVAL_FREQ = 10_000        # evaluate every N timesteps
# ============================================


# ---------------- ENV FACTORY ----------------
def make_env(rank, seed=0):
    """
    Environment creation function.
    Each instance gets a unique seed to ensure independent behavior.
    """
    def _init():
        env = TrafficEnv(
            net_file=NET_FILE,
            route_file=ROUTE_FILE,
            cfg_file=CFG_FILE,
            use_gui=USE_GUI,
            max_steps=MAX_STEPS,
        )
        env.reset(seed=seed + rank)
        np.random.seed(seed + rank)
        random.seed(seed + rank)
        return env
    return _init


if __name__ == "__main__":
    freeze_support()  # ✅ Required for Windows multiprocessing safety
    os.makedirs(LOG_DIR, exist_ok=True)

    # ---------------- ENV SETUP ----------------
    print("🧠 Initializing SUMO-PPO Training Environment...")
    env_fns = [make_env(i, seed=SEED) for i in range(N_ENVS)]

    # ✅ Always use DummyVecEnv for Windows + SUMO stability
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, filename=os.path.join(LOG_DIR, "monitor.csv"))

    # ---------------- LOGGER ----------------
    logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])
    print(f"🚦 Starting PPO training with {N_ENVS} SUMO environment(s)...")
    print(f"📂 Logs will be saved in: {LOG_DIR}")

    # ---------------- MODEL SETUP ----------------
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        seed=SEED,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,         # encourages exploration
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=LOG_DIR,
    )
    model.set_logger(logger)

    # ---------------- EVAL & CHECKPOINTS ----------------
    eval_env = DummyVecEnv([make_env(1000)])
    eval_env = VecMonitor(eval_env)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(LOG_DIR, "best_model"),
        log_path=os.path.join(LOG_DIR, "eval"),
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=EVAL_FREQ,
        save_path=os.path.join(LOG_DIR, "checkpoints"),
        name_prefix="ppo_traffic"
    )

    # ---------------- TRAINING ----------------
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_cb, ckpt_cb])
    except KeyboardInterrupt:
        print("\n🛑 Training manually interrupted. Saving model...")
    except Exception as e:
        print(f"\n⚠️ Training stopped due to error: {e}")
    finally:
        model.save(os.path.join(LOG_DIR, "ppo_final"))
        vec_env.close()
        eval_env.close()
        print("\n✅ PPO training complete.")
        print(f"📊 Logs and models saved in: {LOG_DIR}")
