from traffic_env import SumoTrafficEnv

env = SumoTrafficEnv("C:/Users/souha/Documents/AI_based_TraficLight/sumo_project/cross.sumocfg")
obs, info = env.reset()

for step in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step {step}: Action={action}, Reward={reward}, State={obs}")
    if done:
        break

env.close()
