import yaml
import csv
import math
from pid_controller import PIDController
from acc_system import AdaptiveCruiseControl

import sys
import os

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configs
    config = load_yaml('vehicle_params.yaml')
    gains = load_yaml('tuning_results.yaml')
    # Load sensor data
    with open('sensor_data.csv', 'r') as f:
        reader = csv.DictReader(f)
        sensor_data = list(reader)
    # Init ACC
    acc = AdaptiveCruiseControl(config)
    pid_speed = PIDController(**gains['pid_speed'])
    pid_distance = PIDController(**gains['pid_distance'])
    acc.set_pid_controllers(pid_speed, pid_distance)
    # Simulation state
    results = []
    ego_speed = 0.0
    dt = float(sensor_data[1]['time']) - float(sensor_data[0]['time'])
    for row in sensor_data:
        t = float(row['time'])
        lead_speed = float(row['lead_speed']) if row['lead_speed'] else None
        distance = float(row['distance']) if row['distance'] else float('inf')
        # ACC compute
        accel_cmd, mode, distance_error, dist, ttc = acc.compute(ego_speed, lead_speed, distance, dt)
        # Update ego vehicle
        ego_speed += accel_cmd * dt
        ego_speed = max(0.0, ego_speed)
        # Output row
        results.append({
            'time': f"{t:.1f}",
            'ego_speed': f"{ego_speed:.3f}",
            'acceleration_cmd': f"{accel_cmd:.3f}",
            'mode': mode,
            'distance_error': f"{distance_error:.3f}" if distance_error != '' else '',
            'distance': f"{dist:.3f}" if dist != '' else '',
            'ttc': f"{ttc:.3f}" if ttc != '' else '',
        })
    # Write results
    with open('simulation_results.csv', 'w', newline='') as f:
        fieldnames = ['time','ego_speed','acceleration_cmd','mode','distance_error','distance','ttc']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

if __name__ == '__main__':
    main()
