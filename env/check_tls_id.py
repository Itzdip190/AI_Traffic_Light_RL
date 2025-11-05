import os
import traci

# Path to your SUMO config file
sumo_cmd = [
    "sumo",  # or "sumo-gui" if you want to see it
    "-c", "sumo_project/cross.sumocfg",
    "--start",
    "--no-step-log", "true"
]

print("Starting SUMO to check traffic light IDs...")
traci.start(sumo_cmd)

# Get all available traffic light IDs
tls_ids = traci.trafficlight.getIDList()

if len(tls_ids) == 0:
    print("❌ No traffic lights found in your SUMO network.")
    print("Please open 'cross.net.xml' in NetEdit and add a traffic light at the junction.")
else:
    print("✅ Found Traffic Light IDs:", tls_ids)

traci.close()
