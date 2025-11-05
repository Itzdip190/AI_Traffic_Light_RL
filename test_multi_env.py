from env.multi_traffic_env import MultiTrafficEnv

if __name__ == "__main__":
    net_file = "sumo_project/cross.net.xml"
    route_file = "sumo_project/cross.rou.xml"
    cfg_file = "sumo_project/cross.sumocfg"

    env = MultiTrafficEnv(net_file, route_file, cfg_file, use_gui=True)

    obs, _ = env.reset()
    print("Initial observation keys:", obs.keys())

    for step in range(10):
        # Choose random actions for each traffic light
        actions = {tl: env.action_space[tl].sample() for tl in env.tls_ids}
        next_obs, rewards, dones, done_all, infos = env.step(actions)
        print(f"Step {step}: rewards = {rewards}")

        if done_all:
            break

    env.close()
    print("✅ Simulation completed successfully.")
