from env.traffic_env import TrafficEnv

env = TrafficEnv(
    net_file="sumo_project/cross.net.xml",
    route_file="sumo_project/cross.rou.xml",
    cfg_file="sumo_project/cross.sumocfg",
    use_gui=False,
    max_steps=200,
    fixed_timing=True
)

print("🚦 Starting SUMO test...")

obs, _ = env.reset()
done = False
episode_reward = 0

while not done:
    # Fixed-timer mode, so action is ignored
    obs, reward, done, _, _ = env.step(0)
    episode_reward += reward

print("✅ Finished simulation!")
print("Total episode reward:", episode_reward)
