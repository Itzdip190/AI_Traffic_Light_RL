# tools/convert_model_numpy2_to_numpy1.py
import os
from stable_baselines3 import PPO

old_path = "logs/ppo/run_seed_42/ppo_final.zip"           # model made under NumPy 2
new_path = "logs/ppo/run_seed_42/ppo_final_numpy1.zip"    # will become NumPy 1-compatible

print("🚦 Loading model built with NumPy 2.x …")
model = PPO.load(old_path, device="cpu")

print("💾 Re-saving model in NumPy 1.x-compatible format …")
os.makedirs(os.path.dirname(new_path), exist_ok=True)
model.save(new_path)
print(f"✅ Conversion complete → {new_path}")
