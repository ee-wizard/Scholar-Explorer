import math
from pid_controller import PIDController

class AdaptiveCruiseControl:
    def __init__(self, config):
        self.config = config
        acc = config['acc_settings']
        self.set_speed = acc['set_speed']
        self.time_headway = acc['time_headway']
        self.min_gap = acc['min_gap']
        self.emergency_ttc = acc['emergency_ttc']
        self.accel_limits = acc['accel_limits']
        # PID controllers will be set externally
        self.pid_speed = None
        self.pid_distance = None

    def set_pid_controllers(self, pid_speed, pid_distance):
        self.pid_speed = pid_speed
        self.pid_distance = pid_distance

    def compute(self, ego_speed, lead_speed, distance, dt):
        # Mode selection
        if lead_speed is None or math.isinf(distance):
            # Cruise mode
            error = self.set_speed - ego_speed
            acc_cmd = self.pid_speed.compute(error, dt)
            mode = 'cruise'
            distance_error = ''
            ttc = ''
        else:
            rel_speed = ego_speed - lead_speed
            ttc = distance / max(-rel_speed, 1e-6) if rel_speed < 0 else float('inf')
            desired_gap = self.min_gap + self.time_headway * ego_speed
            distance_error = distance - desired_gap
            if ttc < self.emergency_ttc:
                # Emergency mode
                acc_cmd = self.accel_limits[0]  # max deceleration
                mode = 'emergency'
            else:
                # Follow mode
                acc_cmd = self.pid_distance.compute(distance_error, dt)
                mode = 'follow'
        # Clamp acceleration
        acc_cmd = max(self.accel_limits[0], min(self.accel_limits[1], acc_cmd))
        return acc_cmd, mode, distance_error if mode != 'cruise' else '', distance if mode != 'cruise' else '', ttc if mode != 'cruise' else ''
