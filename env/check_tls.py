# env/check_tls.py
import os
import sys
import traci
import time

# ---> Edit this path if your .sumocfg lives elsewhere
SUMO_CFG = r"C:\Users\souha\Documents\AI_based_TraficLight\sumo_project\cross.sumocfg"

# Choose "sumo" or "sumo-gui" for visible GUI
SUMO_BINARY = "sumo"  # or "sumo-gui"

def main():
    if "SUMO_HOME" not in os.environ:
        print("ERROR: SUMO_HOME not set. Set it to your SUMO install folder.")
        sys.exit(1)

    sumo_cmd = [SUMO_BINARY, "-c", SUMO_CFG, "--start", "--no-step-log", "true"]
    print("Starting SUMO with command:", " ".join(sumo_cmd))
    try:
        traci.start(sumo_cmd)
    except Exception as e:
        print("Failed to start TraCI / SUMO:", type(e).__name__, e)
        sys.exit(1)

    try:
        # 1) List traffic light ids
        try:
            tls_ids = traci.trafficlight.getIDList()
        except Exception as e:
            print("ERROR calling traci.trafficlight.getIDList():", type(e).__name__, e)
            tls_ids = []

        print("Traffic light IDs:", tls_ids)

        # 2) For each TL, print controlled links (and count)
        for tl in tls_ids:
            try:
                links = traci.trafficlight.getControlledLinks(tl)
                print(f"\nTL '{tl}' controlled links (count = {len(links)}):")
                for i, link_group in enumerate(links):
                    print(f"  link {i}: {link_group}")
                # compute minimal state-string length required
                # each link_group is a list of connections; total chars = sum(len(g) for g in links)
                total_conn = sum(len(g) for g in links)
                print(f"  -> total controlled connections = {total_conn}")
            except Exception as e:
                print(f"ERROR getting controlled links for TL '{tl}':", type(e).__name__, e)

        # 3) Print lane list (useful to map lanes)
        try:
            lanes = traci.lane.getIDList()
            print("\nLane IDs (sample up to 50):", lanes[:50])
        except Exception as e:
            print("ERROR calling traci.lane.getIDList():", type(e).__name__, e)

        # keep the simulation running a tiny bit so you can visually confirm (if using GUI)
        print("\nSleeping 3 seconds so you can inspect SUMO GUI (if opened)...")
        time.sleep(3)

    finally:
        try:
            traci.close()
        except Exception:
            pass
        print("Closed TraCI and exiting.")

if __name__ == "__main__":
    main()
