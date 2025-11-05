import os
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci


class MultiTrafficEnv(gym.Env):
    """
    Multi-Agent SUMO Traffic Environment
    Each traffic light acts as an independent RL agent.
    """

    def __init__(self, net_file, route_file, cfg_file, use_gui=False, max_steps=1000):
        super().__init__()
        self.net_file = net_file
        self.route_file = route_file
        self.cfg_file = cfg_file
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.current_step = 0
        self.sumo_cmd = ["sumo-gui" if use_gui else "sumo", "-c", cfg_file, "--start"]

        self._start_sumo()
        self.tls_ids = traci.trafficlight.getIDList()

        # Observation and action spaces per TL
        self.observation_space = spaces.Dict({
            tl: spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
            for tl in self.tls_ids
        })
        self.action_space = spaces.Dict({
            tl: spaces.Discrete(len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)[0].phases))
            for tl in self.tls_ids
        })

    def _start_sumo(self):
        if traci.isLoaded():
            traci.close()
        traci.start(self.sumo_cmd)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self._start_sumo()
        obs = {tl: self._get_state(tl) for tl in self.tls_ids}
        return obs, {}

    def step(self, actions):
        rewards = {}
        next_obs = {}
        dones = {}
        infos = {}

        for tl, action in actions.items():
            traci.trafficlight.setPhase(tl, action)

        traci.simulationStep()
        self.current_step += 1

        for tl in self.tls_ids:
            next_obs[tl] = self._get_state(tl)
            rewards[tl] = self._calculate_reward(tl)
            dones[tl] = self.current_step >= self.max_steps
            infos[tl] = {}

        done_all = all(dones.values())
        return next_obs, rewards, dones, done_all, infos

    def _get_state(self, tl_id):
        # Example observation: queue length + waiting time normalized
        lane_list = traci.trafficlight.getControlledLanes(tl_id)
        queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in lane_list]
        wait_times = [traci.lane.getWaitingTime(lane) for lane in lane_list]
        avg_queue = np.mean(queue_lengths) / 50.0
        avg_wait = np.mean(wait_times) / 100.0
        phase = traci.trafficlight.getPhase(tl_id) / 8.0
        return np.array([avg_queue, avg_wait, phase, random.random(), 0, 0], dtype=np.float32)

    def _calculate_reward(self, tl_id):
        waiting = np.mean([
            traci.lane.getWaitingTime(lane)
            for lane in traci.trafficlight.getControlledLanes(tl_id)
        ])
        # Negative waiting time = minimize congestion
        return -waiting

    def close(self):
        traci.close()
