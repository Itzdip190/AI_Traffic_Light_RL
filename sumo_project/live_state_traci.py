import traci
import sumolib
import os
import statistics

# Path setup
sumoBinary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe"
sumoConfig = "C:/Users/souha/Documents/AI_based_TraficLight/sumo_project/cross.sumocfg"

# Start SUMO
traci.start([sumoBinary, "-c", sumoConfig])

waiting_times = []
travel_times = []
depart_times = {}

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    for veh_id in traci.vehicle.getIDList():
        # Track waiting times
        wait_time = traci.vehicle.getWaitingTime(veh_id)
        waiting_times.append(wait_time)

        # Track travel times
        if veh_id not in depart_times:
            depart_times[veh_id] = traci.simulation.getTime()
        else:
            travel_times.append(traci.simulation.getTime() - depart_times[veh_id])

traci.close()

# Compute metrics
avg_wait = statistics.mean(waiting_times) if waiting_times else 0
avg_travel = statistics.mean(travel_times) if travel_times else 0
throughput = len(depart_times)

print(f"Average Waiting Time: {avg_wait:.2f} s")
print(f"Average Travel Time: {avg_travel:.2f} s")
print(f"Throughput (vehicles passed): {throughput}")
