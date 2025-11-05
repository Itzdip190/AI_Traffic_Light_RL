import os, csv, traci

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
CSV_PATH = os.path.join(LOG_DIR, "metrics.csv")

# create header once
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "sim_time", "avg_waiting_time", "total_waiting_time",
            "avg_queue_length", "vehicles_in_sim", "rl_reward", "light_state"
        ])

def log_step(step, rl_reward, tls_id):
    """Log traffic metrics at each simulation step"""
    sim_time = traci.simulation.getTime()
    veh_ids = traci.vehicle.getIDList()
    veh_count = len(veh_ids)
    total_wait = sum(traci.vehicle.getWaitingTime(v) for v in veh_ids) if veh_count else 0
    avg_wait = total_wait / veh_count if veh_count else 0

    lanes = traci.lane.getIDList()
    avg_queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes) / len(lanes) if lanes else 0

    try:
        state = traci.trafficlight.getRedYellowGreenState(tls_id)
    except Exception:
        state = ""

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            step, sim_time, avg_wait, total_wait, avg_queue, veh_count, rl_reward, state
        ])
