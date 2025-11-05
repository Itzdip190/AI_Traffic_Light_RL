# compare/compare_multi.py
import os
import sys
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from env.multi_traffic_env import MultiTrafficEnv

NET_FILE = "sumo_project/cross.net.xml"
ROUTE_FILE = "sumo_project/cross.rou.xml"
CFG_FILE = "sumo_project/cross.sumocfg"
LOG_DIR = "logs/multi"
os.makedirs(LOG_DIR, exist_ok=True)


def evaluate_multi():
    # Start the SUMO environment (GUI recommended for visual debug)
    env = MultiTrafficEnv(NET_FILE, ROUTE_FILE, CFG_FILE, use_gui=True)

    # Ensure models exist before loading
    missing = []
    for tl in env.tls_ids:
        model_path = os.path.join(LOG_DIR, f"ppo_{tl}.zip")
        if not os.path.isfile(model_path):
            missing.append(model_path)

    if missing:
        env.close()
        raise FileNotFoundError(
            "Model files not found for some traffic lights:\n" + "\n".join(missing) +
            "\n\nTrain the agents first (train/train_multi.py) or place the model files under logs/multi."
        )

    # Load models
    models = {tl: PPO.load(os.path.join(LOG_DIR, f"ppo_{tl}.zip")) for tl in env.tls_ids}

    log_path = os.path.join(LOG_DIR, "multi_eval.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "tl_id", "reward", "avg_wait"])

        obs, _ = env.reset()
        for step in range(500):
            # Predict per-TL action. .predict returns (action, state) so take [0]
            actions = {tl: models[tl].predict(obs[tl])[0] for tl in env.tls_ids}
            obs, rewards, dones, done_all, infos = env.step(actions)
            for tl, reward in rewards.items():
                writer.writerow([step, tl, reward, -reward])
            if done_all:
                break

    env.close()
    print(f"✅ Evaluation complete. Results saved to {log_path}")


if __name__ == "__main__":
    evaluate_multi()
