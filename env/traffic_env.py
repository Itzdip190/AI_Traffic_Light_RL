import os
import csv
import random
import numpy as np
from collections import defaultdict
import gymnasium as gym
from gymnasium import spaces
import traci
import sys

# Allow importing project-level utilities
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Optional import of your custom metric logger
try:
    from save_metrics_snippet import log_step
except ImportError:
    def log_step(step, reward, tls_id):
        pass  # fallback if logging module not found


class TrafficEnv(gym.Env):
    """
    SUMO Traffic Environment (Hardened)
    - safe lane-change handling for violators
    - robust checks for vehicle existence
    - graceful TraCI exception handling
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, net_file, route_file, cfg_file,
                 use_gui=False, max_steps=1000, log_dir="logs", debug_spawn=False):
        super().__init__()

        # --- SUMO setup ---
        self.net_file = net_file
        self.route_file = route_file
        self.cfg_file = cfg_file
        self.use_gui = use_gui
        self.max_steps = max_steps

        # --- RL setup ---
        self.action_space = spaces.Discrete(4)  # four signal phases
        self.observation_space = spaces.Box(low=0, high=100, shape=(8,), dtype=np.float32)

        # --- Traffic light ID ---
        self.tls_id = "TL1"  # adjust if your net.xml uses a different ID

        # --- Chaos configuration ---
        self.chaos_enabled = True
        self.base_arrival_rate = {
            "north": 0.05,
            "south": 0.05,
            "east": 0.05,
            "west": 0.05,
        }
        self.rush_hour_prob = 0.25
        self.rush_multiplier_range = (2.0, 5.0)
        self.p_violator = 0.08
        self.violator_behaviors = ["cut", "wrong_lane_turn", "run_red"]
        self.max_vehicles_per_step = 10

        # --- Logging ---
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "chaos_metrics.csv")

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode", "avg_wait", "total_collisions", "total_violators",
                    "total_throughput", "rush", "rush_multiplier", "total_reward"
                ])

        # --- Runtime state ---
        self.current_step = 0
        self.episode_count = 0
        self.total_reward = 0.0
        self.vehicle_meta = {}
        self.chaos_stats = defaultdict(int)

        # --- Route mapping (matches your .rou.xml) ---
        self.ROUTE_MAP = {
            "north": "north_south",
            "south": "south_north",
            "east": "east_west",
            "west": "west_east"
        }

        # Debug flag: print spawn messages if True
        self.debug_spawn = debug_spawn

    # -------------------------------------------------
    # RESET
    # -------------------------------------------------
    def reset(self, **kwargs):
        if traci.isLoaded():
            traci.close()

        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        traci.start([sumo_binary, "-c", self.cfg_file])

        self.current_step = 0
        self.total_reward = 0.0
        self.vehicle_meta.clear()
        self.chaos_stats.clear()

        # --- Random chaos settings ---
        if self.chaos_enabled and random.random() < self.rush_hour_prob:
            self.episode_has_rush = True
            self.rush_multiplier = float(np.random.uniform(*self.rush_multiplier_range))
            start = random.randint(20, self.max_steps // 2)
            duration = random.randint(30, 120)
            self.rush_window = (start, min(self.max_steps, start + duration))
        else:
            self.episode_has_rush = False
            self.rush_multiplier = 1.0
            self.rush_window = None

        # --- Randomize lane arrival rates ---
        self.arrival_rate = {
            lane: base * np.random.uniform(0.8, 1.25)
            for lane, base in self.base_arrival_rate.items()
        }

        self.episode_count += 1
        return self._get_obs(), {}

    # -------------------------------------------------
    # STEP
    # -------------------------------------------------
    def step(self, action):
        try:
            self._apply_action(action)
            self._inject_arrivals()
            self._apply_violator_behaviors()
            traci.simulationStep()
        except traci.TraCIException as e:
            # Graceful termination on TraCI error within a worker:
            print(f"[TraCI Error] step(): {e}")
            # Try to ensure SUMO instance closes cleanly, then return a terminal step
            try:
                if traci.isLoaded():
                    traci.close()
            except Exception:
                pass
            return self._get_obs(), 0.0, True, False, {}

        obs = self._get_obs()
        reward = float(self._compute_reward())  # ensure Python float
        self.total_reward += reward

        try:
            log_step(self.current_step, reward, self.tls_id)
        except Exception:
            pass

        done = self.current_step >= self.max_steps
        info = self._gather_info()
        self.current_step += 1

        if done:
            self._log_episode_metrics()
            try:
                if traci.isLoaded():
                    traci.close()
            except Exception:
                pass

        return obs, reward, done, False, info

    # -------------------------------------------------
    # ARRIVALS (Poisson + Chaos)
    # -------------------------------------------------
    def _inject_arrivals(self):
        multiplier = 1.0
        if self.episode_has_rush and self.rush_window:
            s, e = self.rush_window
            if s <= self.current_step <= e:
                multiplier = self.rush_multiplier

        for lane, lam in self.arrival_rate.items():
            n_new = np.random.poisson(lam * multiplier)
            n_new = min(n_new, self.max_vehicles_per_step)
            for _ in range(n_new):
                is_violator = random.random() < self.p_violator
                behavior = random.choice(self.violator_behaviors) if is_violator else None
                self._spawn_vehicle(lane, is_violator, behavior)

    # -------------------------------------------------
    # VEHICLE SPAWN
    # -------------------------------------------------
    def _spawn_vehicle(self, lane, violator=False, behavior=None):
        vid = f"{lane}_{self.current_step}_{random.randint(0,9999)}"
        try:
            route_id = self.ROUTE_MAP.get(lane, "north_south")
            traci.vehicle.add(vid, routeID=route_id)
            if self.debug_spawn:
                print(f"🚗 Spawned vehicle {vid} on route {route_id}")

            self.vehicle_meta[vid] = {"violator": violator, "behavior": behavior}
            if violator:
                # best-effort safety settings
                try:
                    if behavior == "cut":
                        traci.vehicle.setLaneChangeMode(vid, 1621)
                    elif behavior == "run_red":
                        traci.vehicle.setSpeedMode(vid, 0)
                except traci.TraCIException:
                    # if setting modes fails, ignore — vehicle still exists
                    pass
                self.chaos_stats["violators_spawned"] += 1
        except traci.TraCIException:
            # spawn failed (route mismatch or SUMO busy) — ignore silently
            return

    # -------------------------------------------------
    # VIOLATOR BEHAVIOR (HARDENED)
    # -------------------------------------------------
    def _apply_violator_behaviors(self):
        # Iterate over a copy because we may remove entries
        for vid, meta in list(self.vehicle_meta.items()):
            # skip if vehicle no longer exists in SUMO
            try:
                if vid not in traci.vehicle.getIDList():
                    # vehicle departed / removed — clean up metadata
                    self.vehicle_meta.pop(vid, None)
                    continue
            except traci.TraCIException:
                # if TraCI fails, skip this vehicle
                continue

            if not meta.get("violator"):
                continue

            behavior = meta.get("behavior")
            try:
                # get lane id (e.g., "west_in_0") and derive edge/lane index
                lane_id = traci.vehicle.getLaneID(vid)  # e.g. "west_in_0"
                # lane_id can be something like ":someInternal:..." — guard that
                if lane_id is None or "_" not in lane_id:
                    # fallback: use traci.vehicle.getLaneIndex
                    cur_index = int(traci.vehicle.getLaneIndex(vid))
                    edge_id = None
                else:
                    try:
                        # last token after last underscore is lane index
                        cur_index = int(lane_id.rsplit("_", 1)[-1])
                        edge_id = lane_id.rsplit("_", 1)[0]
                    except Exception:
                        # fallback
                        cur_index = int(traci.vehicle.getLaneIndex(vid))
                        edge_id = None

                # compute number of lanes on this edge (if available)
                lane_count = None
                if edge_id:
                    try:
                        lane_count = traci.edge.getLaneNumber(edge_id)
                    except traci.TraCIException:
                        lane_count = None

                # default safe lane_count = cur_index+1
                if lane_count is None or lane_count <= 0:
                    lane_count = max(cur_index + 1, 1)

                # behavior-specific logic with clamped target index
                if behavior == "cut" and random.random() < 0.2:
                    # pick a random target lane within [0, lane_count-1], not equal to current
                    candidate = random.randint(0, lane_count - 1)
                    if candidate != cur_index:
                        target = candidate
                        # clamp to valid range
                        target = max(0, min(target, lane_count - 1))
                        # duration should be a small float
                        duration = float(1.0)
                        try:
                            traci.vehicle.changeLane(vid, int(target), duration)
                        except traci.TraCIException:
                            pass

                elif behavior == "wrong_lane_turn" and random.random() < 0.15:
                    # try to move to the "other" lane if available, else skip
                    if lane_count > 1:
                        # simple heuristic: move one lane towards center (flip 0<->last)
                        if cur_index == 0:
                            target = min(1, lane_count - 1)
                        else:
                            target = max(0, cur_index - 1)
                        target = int(max(0, min(target, lane_count - 1)))
                        try:
                            traci.vehicle.changeLane(vid, target, float(0.5))
                        except traci.TraCIException:
                            pass

                elif behavior == "run_red" and random.random() < 0.1:
                    # small speed bump to emulate running red
                    try:
                        sp = traci.vehicle.getSpeed(vid)
                        # only set if speed is valid number
                        if sp is None:
                            sp = 0.0
                        traci.vehicle.setSpeed(vid, float(sp + 1.0))
                    except traci.TraCIException:
                        pass

            except traci.TraCIException:
                # if anything goes wrong for this vehicle, remove metadata to avoid repeated errors
                try:
                    self.vehicle_meta.pop(vid, None)
                except Exception:
                    pass
                continue

    # -------------------------------------------------
    # ACTION HANDLER
    # -------------------------------------------------
    def _apply_action(self, action):
        try:
            traci.trafficlight.setPhase(self.tls_id, int(action))
        except traci.TraCIException:
            pass

    # -------------------------------------------------
    # OBSERVATION
    # -------------------------------------------------
    def _get_obs(self):
        obs = np.zeros(8, dtype=np.float32)
        try:
            lanes = traci.trafficlight.getControlledLanes(self.tls_id)
            counts = [traci.lane.getLastStepVehicleNumber(l) for l in lanes[:8]]
            obs[:len(counts)] = counts
        except traci.TraCIException:
            pass
        return obs

    # -------------------------------------------------
    # REWARD
    # -------------------------------------------------
    def _compute_reward(self):
        try:
            collisions = len(traci.simulation.getCollidingVehiclesIDList())
            throughput = len(traci.simulation.getDepartedIDList())
            waits = [traci.vehicle.getWaitingTime(v) for v in traci.vehicle.getIDList()]
            wait_penalty = np.mean(waits) if waits else 0.0
            reward = throughput - 0.02 * wait_penalty - 5 * collisions
        except traci.TraCIException:
            reward = 0.0
        return reward

    # -------------------------------------------------
    # INFO
    # -------------------------------------------------
    def _gather_info(self):
        try:
            collisions = len(traci.simulation.getCollidingVehiclesIDList())
            violators = sum(v["violator"] for v in self.vehicle_meta.values())
        except traci.TraCIException:
            collisions, violators = 0, 0
        return {
            "step": self.current_step,
            "rush": self.episode_has_rush,
            "rush_multiplier": self.rush_multiplier,
            "collisions": collisions,
            "violators": violators,
        }

    # -------------------------------------------------
    # EPISODE LOGGING
    # -------------------------------------------------
    def _log_episode_metrics(self):
        try:
            vehicle_ids = traci.vehicle.getIDList()
            avg_wait = np.mean([traci.vehicle.getWaitingTime(v) for v in vehicle_ids] or [0])
        except traci.TraCIException:
            avg_wait = 0.0

        collisions = len(traci.simulation.getCollidingVehiclesIDList())
        throughput = len(traci.simulation.getDepartedIDList())
        total_violators = self.chaos_stats.get("violators_spawned", 0)

        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.episode_count,
                avg_wait,
                collisions,
                total_violators,
                throughput,
                int(self.episode_has_rush),
                round(self.rush_multiplier, 2),
                round(self.total_reward, 2),
            ])

        print(f"✅ Episode {self.episode_count} logged: "
              f"wait={avg_wait:.2f}, collisions={collisions}, rush={self.episode_has_rush}")

    # -------------------------------------------------
    # CLOSE
    # -------------------------------------------------
    def close(self):
        try:
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass


# ==================================================
# MANUAL TEST
# ==================================================
if __name__ == "__main__":
    env = TrafficEnv(
        net_file="sumo_project/cross.net.xml",
        route_file="sumo_project/cross.rou.xml",
        cfg_file="sumo_project/cross.sumocfg",
        use_gui=True,
        max_steps=200,
        debug_spawn=True
    )

    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward

    print("Simulation finished. Total Reward:", total_reward)
