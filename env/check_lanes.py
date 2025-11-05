import os
import traci
import sumolib

net_path = "sumo_project/cross.net.xml"

print("🔍 Checking lanes in:", net_path)

net = sumolib.net.readNet(net_path)
edges = net.getEdges()

print("\nAvailable lanes in this network:")
for edge in edges:
    for lane in edge.getLanes():
        print("  ", lane.getID())
