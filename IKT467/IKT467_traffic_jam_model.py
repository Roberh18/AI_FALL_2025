"""
TRAFFIC JAM MODEL - IKT467 Final Project 
Authors: Matin Mohammadi, Robert Hanssen, Sander Gnanavel
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
import json
from datetime import datetime
from collections import deque
import time as time_module
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Traffic Simulation - IKT467", page_icon="T", layout="wide")

PLOT_WIDTH = 10
PLOT_HEIGHT = 5.5
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

# =============================================================================
# UNIFIED VEHICLE AND DRIVER PARAMETERS
# All vehicles have IDENTICAL physical capabilities.
# The ONLY difference between human and AV is DRIVER BEHAVIOR.
# =============================================================================

VEHICLE_PHYSICAL_LIMITS = {
    "max_accel_mps2": 3.0,
    "max_decel_mps2": 6.0,
    "vehicle_length_m": 5.0,
}

# Base IDM parameters
BASE_IDM_PARAMS = {
    "a": 1.5,
    "b": 2.0,
    "delta": 4.0,
    "s0": 2.0,
    "T": 1.2,
}

HUMAN_AGGRESSION_MODIFIERS = {
    "calm": {
        "a_factor": 0.80, "b_factor": 0.80, "s0_add": 1.0,
        "T_factor": 1.33, "v0_factor": 0.95, "rt_factor": 1.10, "noise_std": 0.05,
    },
    "normal": {
        "a_factor": 1.00, "b_factor": 1.00, "s0_add": 0.0,
        "T_factor": 1.00, "v0_factor": 1.00, "rt_factor": 1.00, "noise_std": 0.10,
    },
    "aggressive": {
        "a_factor": 1.20, "b_factor": 1.30, "s0_add":-0.5,
        "T_factor": 0.80, "v0_factor": 1.12, "rt_factor": 0.85, "noise_std": 0.15,
    }
}

AV_BEHAVIOR_PARAMS = {
    "a_factor": 1.00, "b_factor": 1.00, "s0_add":-0.3,
    "T_factor": 0.67, "v0_factor": 1.00, "noise_std": 0.02,
}

NO_LEADER_DISTANCE = 1e9

# Scenario configurations
SCENARIO_CONFIGS = {
    "free_flow": {"description": "Stable low-density traffic", "num_vehicles": 30, "perturbation": None},
    "phantom_jam": {"description": "Emergent jam from perturbation", "num_vehicles": 120,
                   "perturbation": "single_brake", "perturbation_strength": 0.05},
    "driver_variation": {"description": "Mixed aggressive/calm drivers", "num_vehicles": 100,
                        "perturbation": "single_brake", "perturbation_strength": 0.1},
    "bottleneck": {"description": "Speed reduction zone", "num_vehicles": 80, "perturbation": "bottleneck"},
    "correctness_human": {"description": "Stop-and-go waves (track human)", "num_vehicles": 120,
                         "perturbation": "single_brake", "perturbation_strength": 0.05,
                         "sample_car_type": "human", "p_autonomous": 0.3},
    "correctness_autonomous": {"description": "Stop-and-go waves (track AV)", "num_vehicles": 120,
                              "perturbation": "single_brake", "perturbation_strength": 0.05,
                              "sample_car_type": "autonomous", "p_autonomous": 0.3},
}


class DriverType(Enum):
    HUMAN = "human"
    AUTONOMOUS = "autonomous"


@dataclass
class Kinematics:
    pos_m: float = 0.0
    speed_mps: float = 0.0
    accel_mps2: float = 0.0


class PerceptionBuffer:

    def __init__(self, maxlen_steps: int):
        self.ego = deque(maxlen=max(1, maxlen_steps))
        self.leader = deque(maxlen=max(1, maxlen_steps))

    def push(self, ego_pos, ego_speed, leader_pos, leader_speed):
        self.ego.append((ego_pos, ego_speed))
        self.leader.append((leader_pos, leader_speed))

    def read(self, delay_steps):
        if delay_steps <= 0 or len(self.ego) == 0:
            return (self.ego[-1] if self.ego else (0.0, 0.0),
                    self.leader[-1] if self.leader else (NO_LEADER_DISTANCE, 0.0))
        idx = min(delay_steps, len(self.ego))
        return self.ego[-idx], self.leader[-idx]


@dataclass
class Driver:
    reaction_time_s: float = 1.0

    def get_desired_acceleration(self, vehicle, road, dt): raise NotImplementedError


@dataclass
class HumanDriver(Driver):
    aggression: str = "normal"
    
    def __post_init__(self):
        m = HUMAN_AGGRESSION_MODIFIERS.get(self.aggression, HUMAN_AGGRESSION_MODIFIERS["normal"])
        self.idm_params = {
            "a": BASE_IDM_PARAMS["a"] * m["a_factor"],
            "b": BASE_IDM_PARAMS["b"] * m["b_factor"],
            "delta": BASE_IDM_PARAMS["delta"],
            "s0": BASE_IDM_PARAMS["s0"] + m["s0_add"],
            "T": BASE_IDM_PARAMS["T"] * m["T_factor"],
            "v0_factor": m["v0_factor"],
            "noise_std": m["noise_std"],
        }
        self.reaction_time_s *= m["rt_factor"]

    def get_desired_acceleration(self, vehicle, road, dt):
        rt_steps = max(1, int(self.reaction_time_s / max(dt, 1e-6)))
        (_, v), (lead_pos, v_lead) = vehicle.perception.read(rt_steps)
        ego_pos = vehicle.kinematics.pos_m
        
        v = max(v, 0.01)
        v0 = max(road.speedlimit_mps * self.idm_params["v0_factor"], 0.1)
        a, b, delta, s0, T = (self.idm_params[k] for k in ["a", "b", "delta", "s0", "T"])
        
        free_term = a * (1 - (v / v0) ** delta)
        
        if lead_pos > NO_LEADER_DISTANCE * 0.5:
            return free_term + np.random.normal(0, self.idm_params["noise_std"])
        
        s = max(lead_pos - ego_pos - vehicle.vehicle_length_m, 0.5)
        dv = v - v_lead
        s_star = s0 + max(0.0, v * T + (v * dv) / (2 * np.sqrt(max(a * b, 0.01))))
        
        return free_term - a * (min(s_star, 500.0) / s) ** 2 + np.random.normal(0, self.idm_params["noise_std"])


@dataclass
class AutonomousDriver(Driver):
    communication_delay_s: float = 0.05
    
    def __post_init__(self):
        self.idm_params = {
            "a": BASE_IDM_PARAMS["a"] * AV_BEHAVIOR_PARAMS["a_factor"],
            "b": BASE_IDM_PARAMS["b"] * AV_BEHAVIOR_PARAMS["b_factor"],
            "delta": BASE_IDM_PARAMS["delta"],
            "s0": BASE_IDM_PARAMS["s0"] + AV_BEHAVIOR_PARAMS["s0_add"],
            "T": BASE_IDM_PARAMS["T"] * AV_BEHAVIOR_PARAMS["T_factor"],
            "v0_factor": AV_BEHAVIOR_PARAMS["v0_factor"],
            "noise_std": AV_BEHAVIOR_PARAMS["noise_std"],
        }

    def get_desired_acceleration(self, vehicle, road, dt):
        delay = max(1, int(self.reaction_time_s / max(dt, 1e-6))) + int(self.communication_delay_s / max(dt, 1e-6))
        (_, v), (lead_pos, v_lead) = vehicle.perception.read(delay)
        ego_pos = vehicle.kinematics.pos_m
        
        v = max(v, 0.01)
        v0 = max(road.speedlimit_mps * self.idm_params["v0_factor"], 0.1)
        a, b, delta, s0, T = (self.idm_params[k] for k in ["a", "b", "delta", "s0", "T"])
        
        free_term = a * (1 - (v / v0) ** delta)
        
        if lead_pos > NO_LEADER_DISTANCE * 0.5:
            return free_term + np.random.normal(0, self.idm_params["noise_std"])
        
        s = max(lead_pos - ego_pos - vehicle.vehicle_length_m, 0.5)
        dv = v - v_lead
        s_star = s0 + max(0.0, v * T + (v * dv) / (2 * np.sqrt(max(a * b, 0.01))))
        
        return free_term - a * (min(s_star, 500.0) / s) ** 2 + np.random.normal(0, self.idm_params["noise_std"])


@dataclass
class Vehicle:
    vehicle_id: int
    vehicle_length_m: float = VEHICLE_PHYSICAL_LIMITS["vehicle_length_m"]
    max_accel_mps2: float = VEHICLE_PHYSICAL_LIMITS["max_accel_mps2"]
    max_decel_mps2: float = VEHICLE_PHYSICAL_LIMITS["max_decel_mps2"]
    autonomous: bool = False
    kinematics: Kinematics = field(default_factory=Kinematics)
    driver: Optional[Driver] = None
    has_exited: bool = False
    perception: Optional[PerceptionBuffer] = field(default=None, repr=False)
    entry_time: float = 0.0
    
    def __post_init__(self):
        if self.driver is None:
            self.driver = AutonomousDriver() if self.autonomous else HumanDriver()
    
    def get_max_speed(self, speedlimit):
        return speedlimit * self.driver.idm_params.get("v0_factor", 1.0)


@dataclass
class Road:
    road_id: int
    segment_length_m: float = 5000.0
    speedlimit_mps: float = 27.78
    
    def get_max_capacity(self, veh_len=5.0, T=1.5):
        return max(1, int(self.segment_length_m / (veh_len + self.speedlimit_mps * T)))
    
    def get_max_density_vpkm(self, veh_len=5.0, T=1.5):
        return self.get_max_capacity(veh_len, T) / (self.segment_length_m / 1000.0)


@dataclass
class ObservationZone:
    start_m: float
    end_m: float
    
    @property
    def length_m(self): return self.end_m - self.start_m
    
    def contains(self, pos, road_len, periodic):
        return self.start_m <= (pos % road_len if periodic else pos) <= self.end_m
    
    def get_vehicles(self, vehicles, road_len, periodic):
        return [v for v in vehicles if not v.has_exited and self.contains(v.kinematics.pos_m, road_len, periodic)]
    
    def get_density(self, vehicles, road_len, periodic):
        return len(self.get_vehicles(vehicles, road_len, periodic)) / (self.length_m / 1000.0)


def validate_config(cfg):
    warnings = []
    density = cfg.get("num_vehicles", 50) / (cfg.get("road_length_m", 5000) / 1000.0)
    if density > 40: warnings.append(f"High density ({density:.0f} veh/km) - expect congestion.")
    elif density < 15: warnings.append(f"Low density ({density:.0f} veh/km) - free flow expected.")
    return warnings


class TrafficSimulation:

    def __init__(self, config):
        self.config = config
        self.road = Road(101, config["road_length_m"], config["speed_limit_mps"])
        self.vehicles = []
        self.time = 0.0
        self.dt = float(config.get("dt", 0.1))
        self.next_vehicle_id = 0
        self.exited_vehicles = []
        self.use_periodic = config.get("boundary_type", "periodic") == "periodic"
        
        # Automatic warmup
        self.min_warmup_time = 30.0 
        self.stability_threshold = 0.03  
        self.stability_window_s = 15.0  
        self.stability_buffer = deque(maxlen=int(self.stability_window_s / self.dt))
        self.warmup_complete = False
        self.warmup_time_actual = 0.0
        
        if not self.use_periodic:
            demand = int(config.get("demand_vph", 0))
            self.headway_target_s = 3600.0 / max(1, demand) if demand > 0 else None
            self.next_inflow_time = 0.0
        else:
            self.headway_target_s = None
            self.next_inflow_time = None
        
        zone_len = min(1000.0, self.road.segment_length_m * 0.2)
        self.obs_zone = ObservationZone(self.road.segment_length_m * 0.4,
                                        self.road.segment_length_m * 0.4 + zone_len)
        
        self.num_segments = max(1, int(self.road.segment_length_m / 250.0))
        self.segment_len = self.road.segment_length_m / self.num_segments
        
        self.flow_window_s = float(config.get("flow_window_s", 20.0))
        self.zone_crossing_times = []
        self.collision_events = []
        self.near_miss_events = []
        self.perturbation_applied = False
        
        self.max_capacity = self.road.get_max_capacity()
        self.max_safe_density = self.road.get_max_density_vpkm()
        
        self.history = {"time": [], "vehicles": {}, "aggregates": {
            k: [] for k in ["zone_density_vpkm", "zone_density_pct", "zone_speed_mps",
                           "zone_flow_vph", "zone_exit_flow_vph", "local_densities",
                           "local_density_pcts", "total_density_vpkm", "total_density_pct",
                           "avg_gap_m", "avg_time_headway_s", "min_gap_m", "speed_variance",
                           "human_avg_speed", "av_avg_speed", "human_count", "av_count",
                           "all_min_speed_mps", "all_max_speed_mps", "all_mean_speed_mps",
                           "sample_car_speed_mps", "sample_car_accel_mps2"]
        }}
        self.sample_car_id = None
        self._initialize_vehicles()

    def _check_stability(self):
        """
        Check if traffic has reached a stable state.
        
        Stability requires BOTH:
        1. Low coefficient of variation (CV < threshold)
        2. No significant trend (mean not drifting)
        """
        if len(self.stability_buffer) < self.stability_buffer.maxlen:
            return False
        
        speeds = np.array(self.stability_buffer)
        mean_speed = np.mean(speeds)
        
        if mean_speed < 0.1:
            return False
        
        # Check 1: CV must be low
        cv = np.std(speeds) / mean_speed
        if cv >= self.stability_threshold:
            return False
        
        # Check 2: No significant trend (compare first half to second half)
        half = len(speeds) // 2
        first_half_mean = np.mean(speeds[:half])
        second_half_mean = np.mean(speeds[half:])
        
        # Trend check: means shouldn't differ by more than 2%
        if abs(first_half_mean - second_half_mean) / mean_speed > 0.02:
            return False
        
        return True

    def _initialize_vehicles(self):
        num, p_av = self.config["num_vehicles"], self.config["p_autonomous"]
        spacing = self.road.segment_length_m / max(1, num)
        maxlen = max(2, int(1.5 / self.dt) + 5)
        
        # Calculate expected equilibrium speed based on density
        veh_len = VEHICLE_PHYSICAL_LIMITS["vehicle_length_m"]
        gap = spacing - veh_len
        density_vpkm = num / (self.road.segment_length_m / 1000.0)
        
        # Estimate equilibrium speed based on IDM
        # At equilibrium: s* ≈ s, so s0 + v*T = gap
        # Therefore: v_eq = (gap - s0) / T
        avg_T = BASE_IDM_PARAMS["T"]
        avg_s0 = BASE_IDM_PARAMS["s0"]
        
        # Calculate IDM-based equilibrium speed
        v_eq_idm = (gap - avg_s0) / max(avg_T, 0.5)
        
        # Also consider free-flow speed
        v_free = self.road.speedlimit_mps
        
        if density_vpkm < 15:
            # Low density: free flow, but start slightly below to avoid initial spike
            v_equilibrium = v_free * 0.95
        elif density_vpkm < 30:
            # Medium density: blend between free flow and IDM equilibrium
            blend = (density_vpkm - 15) / 15  # 0 at 15, 1 at 30
            v_equilibrium = v_free * (1 - blend) + min(v_eq_idm, v_free) * blend
            v_equilibrium = min(v_equilibrium, v_free * 0.9)
        else:
            # High density: use IDM equilibrium, but ensure reasonable bounds
            v_equilibrium = min(v_eq_idm, v_free * 0.8)
            v_equilibrium = max(v_equilibrium, v_free * 0.2)  # At least 20% of limit
        
        for i in range(num):
            pos = (i * spacing + np.random.uniform(-0.1, 0.1) * spacing) % self.road.segment_length_m
            v = self._create_vehicle(pos, p_av, v_equilibrium)
            v.perception = PerceptionBuffer(maxlen)
            v.perception.push(v.kinematics.pos_m, v.kinematics.speed_mps, NO_LEADER_DISTANCE, 0.0)
        
        sample_type = self.config.get("sample_car_type")
        candidates = ([v for v in self.vehicles if not v.autonomous] if sample_type == "human"
                     else [v for v in self.vehicles if v.autonomous] if sample_type == "autonomous"
                     else self.vehicles)
        self.sample_car_id = candidates[len(candidates) // 2].vehicle_id if candidates else None

    def _create_vehicle(self, pos, p_av, init_speed=None):
        is_av = np.random.random() < p_av
        if is_av:
            driver = AutonomousDriver(self.config.get("av_reaction_time_s", 0.4),
                                     self.config.get("av_comm_delay_s", 0.06))
        else:
            agg = (np.random.choice(["calm", "normal", "aggressive"], p=[0.25, 0.5, 0.25])
                   if self.config.get("scenario") == "driver_variation" 
                   else self.config.get("human_aggression", "normal"))
            driver = HumanDriver(self.config.get("human_reaction_time_s", 1.2), agg)
        
        vehicle = Vehicle(self.next_vehicle_id, autonomous=is_av, driver=driver, entry_time=self.time)
        vehicle.kinematics.pos_m = pos
        # Use equilibrium speed if provided, otherwise use speed limit with small variance
        base_speed = init_speed if init_speed is not None else self.road.speedlimit_mps
        vehicle.kinematics.speed_mps = base_speed * (1 + np.random.uniform(-0.02, 0.02))
        
        self.vehicles.append(vehicle)
        self.history["vehicles"][vehicle.vehicle_id] = {"pos": [], "speed": [], "accel": [], "autonomous": is_av}
        self.next_vehicle_id += 1
        return vehicle

    def _apply_perturbation(self):
        if not self.warmup_complete: return
        scenario_cfg = SCENARIO_CONFIGS.get(self.config.get("scenario", "free_flow"), {})
        perturbation = scenario_cfg.get("perturbation")
        
        if perturbation == "single_brake" and not self.perturbation_applied:
            if self.time >= self.warmup_time_actual + 5.0:
                self.perturbation_applied = True
                strength = scenario_cfg.get("perturbation_strength", 0.05)  # Brake to 5% of speed
                center = self.road.segment_length_m * 0.5
                affected = 0
                # Affect vehicles in a 500m zone, up to 5 vehicles
                for v in sorted(self.vehicles, key=lambda x: abs(x.kinematics.pos_m % self.road.segment_length_m - center)):
                    if v.has_exited: continue
                    p = v.kinematics.pos_m % self.road.segment_length_m
                    if abs(p - center) < 500:
                        v.kinematics.speed_mps *= strength  # Hard brake to 5% speed
                        affected += 1
                        if affected >= 5: break
                        
        elif perturbation == "bottleneck":
            bn_start, bn_end = self.road.segment_length_m * 0.55, self.road.segment_length_m * 0.65
            for v in self.vehicles:
                if v.has_exited: continue
                p = v.kinematics.pos_m % self.road.segment_length_m
                if bn_start <= p <= bn_end and v.kinematics.speed_mps > self.road.speedlimit_mps * 0.4:
                    v.kinematics.accel_mps2 = min(v.kinematics.accel_mps2, -4.0)

    def _get_local_densities(self):
        counts = [0] * self.num_segments
        for v in self.vehicles:
            if v.has_exited: continue
            idx = min(int((v.kinematics.pos_m % self.road.segment_length_m) / self.segment_len), self.num_segments - 1)
            counts[idx] += 1
        seg_km = self.segment_len / 1000.0
        return [c / seg_km for c in counts], [(c / max(self.max_capacity / self.num_segments, 1)) * 100 for c in counts]

    def _push_perception(self, active):
        n = len(active)
        for i, v in enumerate(active):
            if i + 1 < n:
                lead_pos, lead_speed = active[i + 1].kinematics.pos_m, active[i + 1].kinematics.speed_mps
            elif self.use_periodic and n > 1:
                lead_pos = active[0].kinematics.pos_m + self.road.segment_length_m
                lead_speed = active[0].kinematics.speed_mps
            else:
                lead_pos, lead_speed = NO_LEADER_DISTANCE, 0.0
            v.perception.push(v.kinematics.pos_m, v.kinematics.speed_mps, lead_pos, lead_speed)

    def _detect_crossings(self):
        zone_end, road_len = self.obs_zone.end_m, self.road.segment_length_m
        for v in self.vehicles:
            if v.has_exited: continue
            traj = self.history["vehicles"][v.vehicle_id]["pos"]
            if len(traj) < 2: continue
            p_prev, p_now = traj[-2], traj[-1]
            if self.use_periodic:
                p_prev_n, p_now_n = p_prev % road_len, p_now % road_len
                if (p_now_n < p_prev_n and (p_prev_n < zone_end or zone_end <= p_now_n)) or \
                   (p_prev_n < zone_end <= p_now_n):
                    self.zone_crossing_times.append(self.time)
            elif p_prev < zone_end <= p_now:
                self.zone_crossing_times.append(self.time)

    def _get_exit_flow(self):
        self.zone_crossing_times = [t for t in self.zone_crossing_times if t > self.time - self.flow_window_s]
        time_since_warmup = self.time - self.warmup_time_actual
        
        effective_window = min(self.flow_window_s, max(5.0, time_since_warmup))
        
        if effective_window < 1.0:
            return 0.0
        
        window_fraction = effective_window / self.flow_window_s
        raw_flow = (len(self.zone_crossing_times) / effective_window) * 3600.0
        
        if window_fraction < 0.5:
            raw_flow *= (window_fraction * 2) 
        
        return raw_flow

    def _calc_gaps(self, active):
        in_zone = sorted(self.obs_zone.get_vehicles(active, self.road.segment_length_m, self.use_periodic),
                        key=lambda v: v.kinematics.pos_m)
        if len(in_zone) < 2: return 0.0, 0.0, float('inf')
        gaps, headways, min_gap = [], [], float('inf')
        for i in range(len(in_zone) - 1):
            gap = in_zone[i + 1].kinematics.pos_m - in_zone[i].kinematics.pos_m - in_zone[i].vehicle_length_m
            if gap > 0:
                gaps.append(gap)
                min_gap = min(min_gap, gap)
                if in_zone[i].kinematics.speed_mps > 0.1:
                    headways.append(gap / in_zone[i].kinematics.speed_mps)
        return np.mean(gaps) if gaps else 0.0, np.mean(headways) if headways else 0.0, min_gap if min_gap < float('inf') else 0.0

    def _check_safety(self, active):
        for i in range(len(active) - 1):
            gap = active[i + 1].kinematics.pos_m - active[i].kinematics.pos_m - active[i].vehicle_length_m
            if self.use_periodic and gap < 0: gap += self.road.segment_length_m
            if gap < 0.5: self.collision_events.append((self.time, active[i].vehicle_id, gap))
            elif active[i].kinematics.speed_mps > 0.1 and gap / active[i].kinematics.speed_mps < 0.5:
                self.near_miss_events.append((self.time, active[i].vehicle_id, gap))

    def step(self):
        active = sorted([v for v in self.vehicles if not v.has_exited], key=lambda v: v.kinematics.pos_m)
        if not active: self.time += self.dt; return
        
        self._push_perception(active)
        
        for v in active:
            v.kinematics.accel_mps2 = float(np.clip(
                v.driver.get_desired_acceleration(v, self.road, self.dt),
                -v.max_decel_mps2, v.max_accel_mps2))
        
        self._apply_perturbation()
        
        for v in active:
            v.kinematics.speed_mps = float(np.clip(
                v.kinematics.speed_mps + v.kinematics.accel_mps2 * self.dt,
                0.0, v.get_max_speed(self.road.speedlimit_mps)))
            v.kinematics.pos_m += v.kinematics.speed_mps * self.dt
        
        if self.use_periodic:
            for v in active: v.kinematics.pos_m %= self.road.segment_length_m
        else:
            for v in self.vehicles:
                if not v.has_exited and v.kinematics.pos_m > self.road.segment_length_m:
                    v.has_exited = True; self.exited_vehicles.append(v)
            if self.headway_target_s and self.time >= self.next_inflow_time:
                if all(vv.has_exited or vv.kinematics.pos_m > 30.0 for vv in self.vehicles):
                    nv = self._create_vehicle(0.0, self.config["p_autonomous"])
                    nv.perception = PerceptionBuffer(max(2, int(1.5 / self.dt) + 5))
                    nv.perception.push(nv.kinematics.pos_m, nv.kinematics.speed_mps, NO_LEADER_DISTANCE, 0.0)
                self.next_inflow_time += self.headway_target_s
        
        self._check_safety(active)
        self.stability_buffer.append(np.mean([v.kinematics.speed_mps for v in active]))
        
        if not self.warmup_complete and self.time >= self.min_warmup_time and self._check_stability():
            self.warmup_complete = True
            self.warmup_time_actual = self.time
            for vid in self.history["vehicles"]:
                self.history["vehicles"][vid] = {"pos": [], "speed": [], "accel": [],
                                                 "autonomous": self.history["vehicles"][vid]["autonomous"]}
            self.history["time"] = []
            for k in self.history["aggregates"]: self.history["aggregates"][k] = []
            self.zone_crossing_times = []
            self.collision_events = []
            self.near_miss_events = []
        
        if self.warmup_complete: self._record_state()
        self.time += self.dt

    def _record_state(self):
        self.history["time"].append(self.time - self.warmup_time_actual)
        for v in self.vehicles:
            self.history["vehicles"][v.vehicle_id]["pos"].append(v.kinematics.pos_m)
            self.history["vehicles"][v.vehicle_id]["speed"].append(v.kinematics.speed_mps)
            self.history["vehicles"][v.vehicle_id]["accel"].append(v.kinematics.accel_mps2)
        
        zone_vehs = self.obs_zone.get_vehicles(self.vehicles, self.road.segment_length_m, self.use_periodic)
        active = [v for v in self.vehicles if not v.has_exited]
        zone_density = self.obs_zone.get_density(self.vehicles, self.road.segment_length_m, self.use_periodic)
        zone_speed = np.mean([v.kinematics.speed_mps for v in zone_vehs]) if zone_vehs else 0.0
        
        self._detect_crossings()
        avg_gap, avg_hw, min_gap = self._calc_gaps(active)
        local_d, local_pct = self._get_local_densities()
        total_d = len(active) / (self.road.segment_length_m / 1000.0)
        
        humans = [v for v in zone_vehs if not v.autonomous]
        avs = [v for v in zone_vehs if v.autonomous]
        all_speeds = [v.kinematics.speed_mps for v in active]
        
        agg = self.history["aggregates"]
        agg["zone_density_vpkm"].append(zone_density)
        agg["zone_density_pct"].append(zone_density / self.max_safe_density * 100 if self.max_safe_density > 0 else 0)
        agg["zone_speed_mps"].append(zone_speed)
        agg["zone_flow_vph"].append(zone_density * zone_speed * 3.6)
        agg["zone_exit_flow_vph"].append(self._get_exit_flow())
        agg["local_densities"].append(local_d)
        agg["local_density_pcts"].append(local_pct)
        agg["total_density_vpkm"].append(total_d)
        agg["total_density_pct"].append(total_d / self.max_safe_density * 100 if self.max_safe_density > 0 else 0)
        agg["avg_gap_m"].append(avg_gap)
        agg["avg_time_headway_s"].append(avg_hw)
        agg["min_gap_m"].append(min_gap)
        agg["speed_variance"].append(np.var([v.kinematics.speed_mps for v in zone_vehs]) if len(zone_vehs) > 1 else 0)
        agg["human_avg_speed"].append(np.mean([v.kinematics.speed_mps for v in humans]) if humans else 0)
        agg["av_avg_speed"].append(np.mean([v.kinematics.speed_mps for v in avs]) if avs else 0)
        agg["human_count"].append(len(humans))
        agg["av_count"].append(len(avs))
        agg["all_min_speed_mps"].append(min(all_speeds) if all_speeds else 0)
        agg["all_max_speed_mps"].append(max(all_speeds) if all_speeds else 0)
        agg["all_mean_speed_mps"].append(np.mean(all_speeds) if all_speeds else 0)
        
        sample_speed, sample_accel = 0.0, 0.0
        for v in active:
            if v.vehicle_id == self.sample_car_id:
                sample_speed, sample_accel = v.kinematics.speed_mps, v.kinematics.accel_mps2
                break
        agg["sample_car_speed_mps"].append(sample_speed)
        agg["sample_car_accel_mps2"].append(sample_accel)

    def run(self, duration_s, progress_cb=None):
        max_steps = int((self.min_warmup_time + duration_s) * 2 / self.dt)
        start = time_module.time()
        steps_after = 0
        target = int(duration_s / self.dt)
        
        for i in range(max_steps):
            self.step()
            if self.warmup_complete:
                steps_after += 1
                if steps_after >= target: break
            if progress_cb and i % 50 == 0:
                progress_cb(min(steps_after / target if self.warmup_complete else 0.3, 1.0))
        
        if progress_cb: progress_cb(1.0)
        return time_module.time() - start

    def get_metrics(self):
        if not self.history["time"]: return {"error": "No data"}
        agg = self.history["aggregates"]
        speeds = np.array(agg["zone_speed_mps"])
        avg_speed = np.mean(speeds) if len(speeds) else 0.0
        tti = (self.road.segment_length_m / avg_speed) / (self.road.segment_length_m / self.road.speedlimit_mps) if avg_speed > 0 else 1.0
        
        return {
            "zone_avg_density_vpkm": float(np.mean(agg["zone_density_vpkm"])) if agg["zone_density_vpkm"] else 0,
            "zone_avg_density_pct": float(np.mean(agg["zone_density_pct"])) if agg["zone_density_pct"] else 0,
            "zone_avg_speed_kmh": float(avg_speed * 3.6),
            "zone_min_speed_kmh": float(np.min(speeds) * 3.6) if len(speeds) else 0,
            "zone_max_speed_kmh": float(np.max(speeds) * 3.6) if len(speeds) else 0,
            "zone_avg_exit_flow_vph": float(np.mean(agg["zone_exit_flow_vph"])) if agg["zone_exit_flow_vph"] else 0,
            "avg_gap_m": float(np.mean(agg["avg_gap_m"])) if agg["avg_gap_m"] else 0,
            "min_gap_m": float(np.min(agg["min_gap_m"])) if agg["min_gap_m"] else 0,
            "avg_time_headway_s": float(np.mean(agg["avg_time_headway_s"])) if agg["avg_time_headway_s"] else 0,
            "avg_speed_variance": float(np.mean(agg["speed_variance"])) if agg["speed_variance"] else 0,
            "collision_events": len(self.collision_events),
            "near_miss_events": len(self.near_miss_events),
            "travel_time_index": float(tti),
            "warmup_time_detected": self.warmup_time_actual,
            "road_max_capacity": self.max_capacity,
            "road_max_safe_density_vpkm": self.max_safe_density,
            "human_avg_speed_kmh": float(np.mean([s for s in agg["human_avg_speed"] if s > 0]) * 3.6) if any(s > 0 for s in agg["human_avg_speed"]) else 0,
            "av_avg_speed_kmh": float(np.mean([s for s in agg["av_avg_speed"] if s > 0]) * 3.6) if any(s > 0 for s in agg["av_avg_speed"]) else 0,
            "total_density_vpkm": float(np.mean(agg["total_density_vpkm"])) if agg["total_density_vpkm"] else 0,
            "active_vehicles": len([v for v in self.vehicles if not v.has_exited]),
            "wave_speed_kmh": float(self._estimate_wave_speed() * 3.6),
            "sim_duration_s": float(self.history["time"][-1]) if self.history["time"] else 0,
            "free_flow_speed_kmh": float(self.road.speedlimit_mps * 3.6),
        }

    def _estimate_wave_speed(self):
        ld, times = self.history["aggregates"]["local_densities"], self.history["time"]
        if len(ld) < 50: return 0.0
        try:
            dm = np.array(ld)
            thresh = np.percentile(dm, 80)
            wp = [(times[i], np.argmax(dm[i]) * self.segment_len) for i in range(0, len(times), 10) 
                  if np.max(dm[i]) > thresh]
            if len(wp) < 3: return 0.0
            t, x = np.array([p[0] for p in wp]), np.array([p[1] for p in wp])
            xu = np.zeros_like(x); xu[0] = x[0]
            for i in range(1, len(x)):
                d = x[i] - x[i - 1]
                if d > self.road.segment_length_m / 2: d -= self.road.segment_length_m
                elif d < -self.road.segment_length_m / 2: d += self.road.segment_length_m
                xu[i] = xu[i - 1] + d
            return stats.linregress(t, xu)[0] if len(t) > 2 else 0.0
        except: return 0.0


def run_density_sweep(base_config, vehicle_counts, duration_s, progress_cb=None):
    results = []
    for i, n in enumerate(vehicle_counts):
        cfg = {**base_config, "num_vehicles": n, "seed": base_config.get("seed", 42) + i}
        np.random.seed(cfg["seed"])
        sim = TrafficSimulation(cfg)
        sim.run(duration_s)
        agg = sim.history["aggregates"]
        if agg["zone_density_vpkm"]:
            step = max(1, len(agg["zone_density_vpkm"]) // 50)
            for j in range(0, len(agg["zone_density_vpkm"]), step):
                results.append({"num_vehicles": n, "density_vpkm": agg["zone_density_vpkm"][j],
                               "density_pct": agg["zone_density_pct"][j], "speed_kmh": agg["zone_speed_mps"][j] * 3.6,
                               "flow_vph": agg["zone_flow_vph"][j], "exit_flow_vph": agg["zone_exit_flow_vph"][j]})
        if progress_cb: progress_cb((i + 1) / len(vehicle_counts))
    return pd.DataFrame(results)


def rolling_smooth(arr, window):
    return pd.Series(arr).rolling(window, min_periods=1, center=True).mean().values if window > 1 and len(arr) >= window else arr


def plot_time_space(sim, ax=None, unwrap=True):
    if ax is None: fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    times = np.array(sim.history["time"])
    if len(times) == 0: ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes); return ax
    road_len = sim.road.segment_length_m
    for vid, data in sim.history["vehicles"].items():
        if not data["pos"]: continue
        pos = np.array(data["pos"]); t = times[:len(pos)]
        if sim.use_periodic and unwrap:
            pu = np.zeros_like(pos); pu[0] = pos[0] % road_len
            for j in range(1, len(pos)):
                d = pos[j] - pos[j - 1]
                if d < -road_len / 2: d += road_len
                elif d > road_len / 2: d -= road_len
                pu[j] = pu[j - 1] + d
            pos = pu
        elif sim.use_periodic: pos = pos % road_len
        ax.plot(t, pos, color="#2196F3" if data["autonomous"] else "#F44336", alpha=0.6 if data["autonomous"] else 0.4, linewidth=0.6)
    ax.axhline(y=sim.obs_zone.start_m, color='green', linestyle='--', linewidth=2, alpha=0.8)
    ax.axhline(y=sim.obs_zone.end_m, color='green', linestyle='--', linewidth=2, alpha=0.8)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Position (m)"); ax.set_title("Time-Space Diagram"); ax.grid(True, alpha=0.3)
    ax.legend(handles=[Patch(facecolor='#2196F3', alpha=0.6, label='AV'), Patch(facecolor='#F44336', alpha=0.4, label='Human')], loc='upper right', fontsize=9)
    return ax


def plot_fundamental_diagram(sim=None, sweep_df=None, ax=None, smooth=1):
    if ax is None: fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    if sweep_df is not None and len(sweep_df) > 0:
        ax.scatter(sweep_df["density_vpkm"], sweep_df["flow_vph"], alpha=0.3, s=12, c='blue', edgecolors='none', label='Data')
        max_d = sweep_df["density_vpkm"].max(); v_f = 100
    elif sim is not None:
        d = rolling_smooth(np.array(sim.history["aggregates"]["zone_density_vpkm"]), smooth)
        f = rolling_smooth(np.array(sim.history["aggregates"]["zone_flow_vph"]), smooth)
        if len(d) == 0: ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes); return ax
        m = (d > 0) & (f >= 0) & ~np.isnan(d) & ~np.isnan(f)
        ax.scatter(d[m], f[m], alpha=0.4, s=15, c='blue', edgecolors='none', label='Data')
        max_d = np.max(d[m]) if np.any(m) else 50; v_f = sim.road.speedlimit_mps * 3.6
    else: ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes); return ax
    k = np.linspace(0.1, min(max_d * 1.5, 140), 100)
    ax.plot(k, k * v_f * (1 - k / 150), 'r--', alpha=0.5, linewidth=1.5, label='Greenshields')
    ax.set_xlabel("Density (veh/km)"); ax.set_ylabel("Flow (veh/h)"); ax.set_title("Fundamental Diagram"); ax.grid(True, alpha=0.3); ax.legend(fontsize=9); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    return ax


def plot_speed_density(sim=None, sweep_df=None, ax=None, smooth=1):
    if ax is None: fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    if sweep_df is not None and len(sweep_df) > 0:
        ax.scatter(sweep_df["density_vpkm"], sweep_df["speed_kmh"], alpha=0.3, s=12, c='green', edgecolors='none', label='Data')
        max_d = sweep_df["density_vpkm"].max(); v_f = 100
    elif sim is not None:
        d = rolling_smooth(np.array(sim.history["aggregates"]["zone_density_vpkm"]), smooth)
        s = rolling_smooth(np.array(sim.history["aggregates"]["zone_speed_mps"]) * 3.6, smooth)
        if len(d) == 0: ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes); return ax
        m = (d > 0) & ~np.isnan(s) & ~np.isnan(d)
        ax.scatter(d[m], s[m], alpha=0.4, s=15, c='green', edgecolors='none', label='Data')
        max_d = np.max(d[m]) if np.any(m) else 50; v_f = sim.road.speedlimit_mps * 3.6
    else: ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes); return ax
    k = np.linspace(0.1, min(max_d * 1.5, 150), 100)
    ax.plot(k, v_f * (1 - k / 150), 'r--', alpha=0.5, linewidth=1.5, label='Greenshields')
    ax.set_xlabel("Density (veh/km)"); ax.set_ylabel("Speed (km/h)"); ax.set_title("Speed-Density"); ax.grid(True, alpha=0.3); ax.legend(fontsize=9); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    return ax


def plot_metrics_time(sim, ax=None, smooth=1):
    if ax is None: fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    times = np.array(sim.history["time"])
    if len(times) == 0: ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes); return ax
    speeds = rolling_smooth(np.array(sim.history["aggregates"]["zone_speed_mps"]) * 3.6, smooth)
    densities = rolling_smooth(np.array(sim.history["aggregates"]["zone_density_vpkm"]), smooth)
    flows = rolling_smooth(np.array(sim.history["aggregates"]["zone_exit_flow_vph"]), smooth)
    ax1, ax2, ax3 = ax, ax.twinx(), ax.twinx(); ax3.spines["right"].set_position(("outward", 55))
    l1, = ax1.plot(times, speeds, 'b-', alpha=0.8, linewidth=1.5)
    l2, = ax2.plot(times, densities, 'r-', alpha=0.8, linewidth=1.5)
    l3, = ax3.plot(times, flows, 'g-', alpha=0.8, linewidth=1.5)
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Speed (km/h)", color='b'); ax2.set_ylabel("Density (veh/km)", color='r'); ax3.set_ylabel("Exit Flow (veh/h)", color='g')
    for a, c in [(ax1, 'b'), (ax2, 'r'), (ax3, 'g')]: a.tick_params(axis='y', labelcolor=c)
    ax1.set_title("Traffic Metrics Over Time"); ax1.grid(True, alpha=0.3); ax1.legend([l1, l2, l3], ['Speed', 'Density', 'Flow'], loc='upper left', fontsize=9)
    return ax1


def plot_density_capacity(sim, ax=None, smooth=1):
    if ax is None: fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    times = np.array(sim.history["time"])
    if len(times) == 0: ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes); return ax
    zd = rolling_smooth(np.array(sim.history["aggregates"]["zone_density_pct"]), smooth)
    td = rolling_smooth(np.array(sim.history["aggregates"]["total_density_pct"]), smooth)
    ax.plot(times, zd, 'b-', label='Obs Zone', alpha=0.8, linewidth=1.5)
    ax.plot(times, td, 'r--', label='Total Road', alpha=0.6, linewidth=1.5)
    ax.axhline(y=100, color='red', linestyle='-', alpha=0.5, linewidth=2, label='100% Capacity')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Density (% capacity)"); ax.set_title(f"Road Density as % of Capacity (T={BASE_IDM_PARAMS['T']:.1f}s)"); ax.grid(True, alpha=0.3); ax.legend(fontsize=9); ax.set_ylim(bottom=0)
    return ax


def plot_density_distribution(sim, ax=None, time_idx=-1):
    if ax is None: fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    ld = sim.history["aggregates"]["local_densities"]
    if not ld: ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes); return ax
    idx = time_idx if time_idx >= 0 else len(ld) - 1
    d = np.array(ld[idx])
    centers = np.array([(i + 0.5) * sim.segment_len for i in range(len(d))])
    bars = ax.bar(centers, d, width=sim.segment_len * 0.8, alpha=0.7, edgecolor='darkblue', linewidth=0.5)
    for bar, dens in zip(bars, d):
        bar.set_facecolor('#F44336' if dens > sim.max_safe_density * 0.9 else '#FF9800' if dens > sim.max_safe_density * 0.6 else '#4CAF50')
    ax.axvspan(sim.obs_zone.start_m, sim.obs_zone.end_m, alpha=0.2, color='cyan', label='Obs Zone')
    ax.axhline(y=sim.max_safe_density, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'Capacity ({sim.max_safe_density:.0f})')
    ax.set_xlabel("Position (m)"); ax.set_ylabel("Density (veh/km)"); ax.set_title(f"Density Distribution at t={sim.history['time'][idx]:.1f}s"); ax.grid(True, alpha=0.3, axis='y'); ax.legend(fontsize=9); ax.set_xlim(0, sim.road.segment_length_m); ax.set_ylim(bottom=0)
    return ax


def plot_vehicle_comparison(sim, ax=None, smooth=1):
    if ax is None: fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    times = np.array(sim.history["time"])
    if len(times) == 0: ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes); return ax
    hs = rolling_smooth(np.array(sim.history["aggregates"]["human_avg_speed"]) * 3.6, smooth)
    avs = rolling_smooth(np.array(sim.history["aggregates"]["av_avg_speed"]) * 3.6, smooth)
    hs[hs == 0] = np.nan; avs[avs == 0] = np.nan
    ax.plot(times, hs, 'r-', label='Human', alpha=0.8, linewidth=1.5)
    ax.plot(times, avs, 'b-', label='AV', alpha=0.8, linewidth=1.5)
    ax.axhline(y=sim.road.speedlimit_mps * 3.6, color='gray', linestyle='--', alpha=0.5, label='Limit')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Speed (km/h)"); ax.set_title("Human vs AV Speeds"); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    return ax


def plot_gap_headway(sim, ax=None, smooth=1):
    if ax is None: fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    times = np.array(sim.history["time"])
    if len(times) == 0: ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes); return ax
    gaps = rolling_smooth(np.array(sim.history["aggregates"]["avg_gap_m"]), smooth)
    hw = rolling_smooth(np.array(sim.history["aggregates"]["avg_time_headway_s"]), smooth)
    ax1, ax2 = ax, ax.twinx()
    l1, = ax1.plot(times, gaps, 'b-', alpha=0.8, linewidth=1.5)
    l2, = ax2.plot(times, hw, color='orange', alpha=0.8, linewidth=1.5)
    ax2.axhline(y=BASE_IDM_PARAMS["T"], color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Gap (m)", color='b'); ax2.set_ylabel("Headway (s)", color='orange')
    ax1.tick_params(axis='y', labelcolor='b'); ax2.tick_params(axis='y', labelcolor='orange')
    ax1.set_title(f"Following Distance (T={BASE_IDM_PARAMS['T']:.1f}s baseline)"); ax1.grid(True, alpha=0.3); ax1.legend([l1, l2], ['Gap', 'Headway'], fontsize=9)
    return ax1


def plot_density_heatmap(sim, ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    times, ld = np.array(sim.history["time"]), sim.history["aggregates"]["local_densities"]
    if not ld or len(times) == 0: ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes); return ax
    dm = np.array(ld).T
    cmap = LinearSegmentedColormap.from_list('traffic', ['#2E7D32', '#FFEB3B', '#FF9800', '#F44336', '#B71C1C'])
    im = ax.imshow(dm, aspect='auto', origin='lower', extent=[times[0], times[-1], 0, sim.road.segment_length_m], cmap=cmap, interpolation='bilinear')
    ax.axhline(y=sim.obs_zone.start_m, color='cyan', linestyle='--', linewidth=2)
    ax.axhline(y=sim.obs_zone.end_m, color='cyan', linestyle='--', linewidth=2)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Position (m)"); ax.set_title("Density Heatmap (diagonal = backward waves)")
    plt.colorbar(im, ax=ax, label="Density (veh/km)")
    return ax


def plot_stability(sim, ax=None, smooth=1):
    if ax is None: fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    times = np.array(sim.history["time"])
    if len(times) == 0: ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes); return ax
    sv = rolling_smooth(np.array(sim.history["aggregates"]["speed_variance"]), smooth)
    ax.fill_between(times, 0, sv, alpha=0.4, color='purple'); ax.plot(times, sv, 'purple', alpha=0.8, linewidth=1.5)
    ax.axhline(y=5.0, color='red', linestyle='--', alpha=0.5, label='Instability')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Speed Variance (m^2/s^2)"); ax.set_title("Flow Stability"); ax.grid(True, alpha=0.3); ax.legend(fontsize=9); ax.set_ylim(bottom=0)
    return ax


def plot_netlogo_correctness(sim, ax=None, normalize=True):
    if ax is None: fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    times = np.array(sim.history["time"])
    if len(times) == 0: ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes); return ax
    agg = sim.history["aggregates"]
    sample, min_s, max_s, mean_s = np.array(agg["sample_car_speed_mps"]), np.array(agg["all_min_speed_mps"]), np.array(agg["all_max_speed_mps"]), np.array(agg["all_mean_speed_mps"])
    if normalize and sim.road.speedlimit_mps > 0:
        sample, min_s, max_s, mean_s = sample / sim.road.speedlimit_mps, min_s / sim.road.speedlimit_mps, max_s / sim.road.speedlimit_mps, mean_s / sim.road.speedlimit_mps
        ylabel = "Speed (normalized)"
    else:
        sample, min_s, max_s, mean_s = sample * 3.6, min_s * 3.6, max_s * 3.6, mean_s * 3.6
        ylabel = "Speed (km/h)"
    ax.plot(times, sample, 'r-', linewidth=2, label='Sample Car', alpha=0.9)
    ax.plot(times, min_s, 'b-', linewidth=1.5, label='Min', alpha=0.7)
    ax.plot(times, max_s, 'g-', linewidth=1.5, label='Max', alpha=0.7)
    ax.plot(times, mean_s, 'k-', linewidth=1.5, label='Mean', alpha=0.8)
    ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel); ax.set_title("NetLogo Correctness Comparison"); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    if normalize: ax.set_ylim(0, 1.1)
    return ax


def plot_sample_car_acceleration(sim, ax=None, smooth=1):
    if ax is None: fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    times = np.array(sim.history["time"])
    if len(times) == 0: ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes); return ax
    accel = rolling_smooth(np.array(sim.history["aggregates"]["sample_car_accel_mps2"]), smooth)
    sample_type = "Unknown"
    for v in sim.vehicles:
        if v.vehicle_id == sim.sample_car_id: sample_type = "AV" if v.autonomous else "Human"; break
    ax.fill_between(times, 0, accel, where=(accel >= 0), color='green', alpha=0.3)
    ax.fill_between(times, 0, accel, where=(accel < 0), color='red', alpha=0.3)
    ax.plot(times, accel, 'k-', linewidth=1, alpha=0.8)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Acceleration (m/s^2)"); ax.set_title(f"Sample Car Acceleration ({sample_type})"); ax.grid(True, alpha=0.3)
    if len(accel) > 1:
        ax.text(0.02, 0.98, f"Std: {np.std(accel):.2f}\nJerk: {np.sqrt(np.mean(np.diff(accel)**2))/sim.dt:.2f}", transform=ax.transAxes, fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    return ax


def plot_safety_metrics(sim, ax=None, smooth=1):
    if ax is None: fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    times = np.array(sim.history["time"])
    if len(times) == 0: ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes); return ax
    mg = rolling_smooth(np.array(sim.history["aggregates"]["min_gap_m"]), smooth)
    ax.plot(times, mg, 'b-', alpha=0.8, linewidth=1.5, label='Min Gap')
    ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Safe')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Critical')
    if sim.collision_events:
        ct = [e[0] - sim.warmup_time_actual for e in sim.collision_events]
        cg = [e[2] for e in sim.collision_events]
        ax.scatter(ct, cg, color='red', s=50, marker='x', label=f'Collisions ({len(sim.collision_events)})', zorder=5)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Min Gap (m)"); ax.set_title("Safety Metrics"); ax.grid(True, alpha=0.3); ax.legend(fontsize=9); ax.set_ylim(bottom=0)
    return ax


def build_dataframes(sim):
    agg = pd.DataFrame({k: sim.history["aggregates"][k] for k in ["zone_density_vpkm", "zone_density_pct", "zone_speed_mps", "zone_flow_vph", "zone_exit_flow_vph", "avg_gap_m", "min_gap_m", "avg_time_headway_s", "speed_variance"]})
    agg.insert(0, "time_s", sim.history["time"])
    rows = []
    for vid, rec in sim.history["vehicles"].items():
        for i in range(len(rec["pos"])):
            rows.append({"time_s": sim.history["time"][i] if i < len(sim.history["time"]) else np.nan, "vehicle_id": vid, "autonomous": rec["autonomous"], "pos_m": rec["pos"][i], "speed_mps": rec["speed"][i], "accel_mps2": rec["accel"][i]})
    return pd.DataFrame(rows), agg


def main():
    st.title("Traffic Jam - IKT467 Final Project")
    st.markdown("**Mixed Autonomous and Human-Driven Traffic Flow Analysis**")

    with st.sidebar:
        st.header("Configuration")
        mode = st.radio("Mode", ["Single Run", "Density Sweep"])
        scenario = st.selectbox("Scenario", list(SCENARIO_CONFIGS.keys()))
        scenario_cfg = SCENARIO_CONFIGS[scenario]
        boundary = st.radio("Boundary", ["periodic", "open"])
        
        st.subheader("Road")
        road_len = st.number_input("Length (m)", 1000, 10000, 5000, 500)
        speed_lim = st.slider("Speed Limit (km/h)", 50, 130, 100, 10)
        
        st.subheader("Vehicles")
        default_n = scenario_cfg.get("num_vehicles", 50)
        num_veh = st.slider("Count", 20, 200, default_n, 10) if mode == "Single Run" else default_n
        p_av = st.slider("AV Ratio", 0.0, 1.0, 0.5, 0.05)
        
        st.subheader("Drivers")
        human_rt = st.slider("Human Reaction (s)", 0.5, 2.5, 1.2, 0.1)
        human_agg = st.selectbox("Human Aggression", ["calm", "normal", "aggressive"])
        av_rt = st.slider("AV Reaction (s)", 0.1, 1.0, 0.4, 0.05)
        av_delay = st.slider("AV Comm Delay (s)", 0.01, 0.2, 0.06, 0.01)
        
        st.subheader("Engine")
        dt = st.slider("Time Step (s)", 0.05, 0.5, 0.1, 0.05)
        flow_win = st.slider("Flow Window (s)", 5, 60, 20, 5)
        seed = st.number_input("Seed", 0, 1000000, 42)
        smooth_s = st.slider("Plot Smoothing (s)", 0, 30, 5)
        demand = st.slider("Demand (veh/h)", 200, 2400, 1200, 100) if boundary == "open" else 0
        
        st.subheader("Duration")
        duration = st.slider("Duration (s)", 60, 600, 300, 30)
        run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    if run_btn:
        config = {"scenario": scenario, "boundary_type": boundary, "road_length_m": int(road_len), "speed_limit_mps": float(speed_lim) / 3.6, "num_vehicles": int(num_veh), "p_autonomous": float(p_av), "human_reaction_time_s": float(human_rt), "human_aggression": human_agg, "av_reaction_time_s": float(av_rt), "av_comm_delay_s": float(av_delay), "dt": float(dt), "flow_window_s": float(flow_win), "seed": int(seed), "demand_vph": int(demand)}
        if "sample_car_type" in scenario_cfg: config["sample_car_type"] = scenario_cfg["sample_car_type"]
        if "p_autonomous" in scenario_cfg: config["p_autonomous"] = scenario_cfg["p_autonomous"]; st.info(f"Scenario sets AV ratio to {scenario_cfg['p_autonomous']*100:.0f}% for wave analysis")
        if "sample_car_type" in scenario_cfg: st.info(f"Tracking {scenario_cfg['sample_car_type'].upper()} sample car")
        
        np.random.seed(config["seed"])
        for w in validate_config(config): st.warning(w)
        
        sweep_df = None
        if mode == "Density Sweep":
            st.info("Running density sweep..."); prog = st.progress(0)
            sweep_df = run_density_sweep(config, list(range(20, 160, 15)), duration // 2, lambda p: prog.progress(p))
            np.random.seed(config["seed"]); config["num_vehicles"] = 50
        
        with st.spinner(f"Running ({duration}s)..."):
            prog_main = st.progress(0); sim = TrafficSimulation(config); wall = sim.run(duration, lambda p: prog_main.progress(p))
        
        st.success(f"Done in {wall:.1f}s ({sim.time/wall:.0f}x realtime)"); st.info(f"Warmup: {sim.warmup_time_actual:.1f}s")
        
        st.header("KPIs"); metrics = sim.get_metrics()
        if "error" in metrics: st.error(metrics["error"]); return
        
        st.caption(f"Obs Zone: {sim.obs_zone.start_m:.0f}-{sim.obs_zone.end_m:.0f}m | Capacity: {metrics['road_max_capacity']} veh")
        c1, c2, c3, c4 = st.columns(4); c1.metric("Density", f"{metrics['zone_avg_density_vpkm']:.1f} veh/km"); c2.metric("Density %", f"{metrics['zone_avg_density_pct']:.0f}%"); c3.metric("Speed", f"{metrics['zone_avg_speed_kmh']:.1f} km/h"); c4.metric("TTI", f"{metrics['travel_time_index']:.2f}")
        c5, c6, c7, c8 = st.columns(4); c5.metric("Gap", f"{metrics['avg_gap_m']:.1f}m"); c6.metric("Headway", f"{metrics['avg_time_headway_s']:.2f}s"); c7.metric("Flow", f"{metrics['zone_avg_exit_flow_vph']:.0f} veh/h"); c8.metric("Wave", f"{metrics['wave_speed_kmh']:.1f} km/h")
        c9, c10, c11, c12 = st.columns(4); c9.metric("Collisions", metrics['collision_events']); c10.metric("Near Miss", metrics['near_miss_events']); c11.metric("Min Gap", f"{metrics['min_gap_m']:.1f}m"); c12.metric("Warmup", f"{metrics['warmup_time_detected']:.1f}s")
        
        st.header("Plots"); smooth_steps = max(1, int(smooth_s / dt))
        tabs = st.tabs(["Time-Space", "Fundamental", "Metrics", "Density", "Vehicles", "Heatmap", "Safety", "Correctness"])
        
        with tabs[0]:
            c1, c2 = st.columns(2)
            with c1: fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT)); plot_time_space(sim, ax, True); st.pyplot(fig); plt.close()
            with c2: fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT)); plot_time_space(sim, ax, False); st.pyplot(fig); plt.close()
        
        with tabs[1]:
            c1, c2 = st.columns(2)
            with c1: fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT)); plot_fundamental_diagram(sim, sweep_df, ax, smooth_steps); st.pyplot(fig); plt.close()
            with c2: fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT)); plot_speed_density(sim, sweep_df, ax, smooth_steps); st.pyplot(fig); plt.close()
        
        with tabs[2]:
            fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT)); plot_metrics_time(sim, ax, smooth_steps); st.pyplot(fig); plt.close()
            fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT)); plot_gap_headway(sim, ax, smooth_steps); st.pyplot(fig); plt.close()
        
        with tabs[3]:
            fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT)); plot_density_capacity(sim, ax, smooth_steps); st.pyplot(fig); plt.close()
            max_idx = len(sim.history["time"]) - 1
            if max_idx > 0: time_idx = st.slider("Time", 0, max_idx, max_idx)
            else: time_idx = 0
            fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT)); plot_density_distribution(sim, ax, time_idx); st.pyplot(fig); plt.close()
        
        with tabs[4]:
            fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT)); plot_vehicle_comparison(sim, ax, smooth_steps); st.pyplot(fig); plt.close()
            st.markdown(f"Human: **{metrics['human_avg_speed_kmh']:.1f}** km/h | AV: **{metrics['av_avg_speed_kmh']:.1f}** km/h")
        
        with tabs[5]:
            c1, c2 = st.columns(2)
            with c1: fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT)); plot_density_heatmap(sim, ax); st.pyplot(fig); plt.close()
            with c2: fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT)); plot_stability(sim, ax, smooth_steps); st.pyplot(fig); plt.close()
        
        with tabs[6]:
            fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT)); plot_safety_metrics(sim, ax, smooth_steps); st.pyplot(fig); plt.close()
            st.markdown(f"**Parameters:** Human T={BASE_IDM_PARAMS['T']*HUMAN_AGGRESSION_MODIFIERS[human_agg]['T_factor']:.1f}s | AV T={BASE_IDM_PARAMS['T']*AV_BEHAVIOR_PARAMS['T_factor']:.1f}s")
        
        with tabs[7]:
            sample_type = "Unknown"
            for v in sim.vehicles:
                if v.vehicle_id == sim.sample_car_id: sample_type = "AV" if v.autonomous else "Human"; break
            st.info(f"Sample: {sample_type} (ID: {sim.sample_car_id})")
            c1, c2 = st.columns(2)
            with c1: fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT)); plot_netlogo_correctness(sim, ax, True); st.pyplot(fig); plt.close()
            with c2: fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT)); plot_netlogo_correctness(sim, ax, False); st.pyplot(fig); plt.close()
            fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT * 0.7)); plot_sample_car_acceleration(sim, ax, smooth_steps); st.pyplot(fig); plt.close()
        
        with st.expander("All Metrics"):
            c1, c2 = st.columns(2)
            for i, (k, v) in enumerate(metrics.items()): (c1 if i < len(metrics) // 2 else c2).write(f"**{k}**: {v:.4f}" if isinstance(v, float) else f"**{k}**: {v}")
        
        st.header("Export"); traj_df, agg_df = build_dataframes(sim)
        c1, c2, c3 = st.columns(3)
        c1.download_button("Config (JSON)", json.dumps(config, indent=2), f"config_{datetime.now():%Y%m%d_%H%M%S}.json", "application/json")
        c2.download_button("Aggregates (CSV)", agg_df.to_csv(index=False), f"agg_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
        c3.download_button("Trajectories (CSV)", traj_df.to_csv(index=False), f"traj_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
        if sweep_df is not None: st.download_button("Sweep (CSV)", sweep_df.to_csv(index=False), f"sweep_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")

    else:
        st.info("Select pre-configured scenarios, or configure custom parameters and press 'Run Simulation'")
        st.markdown("""
        ### About
        
        This simulation uses the Intelligent Driver Model (IDM) for car-following behavior.

        
        **Scenarios:** \n
            1. free_flow (stable)
            2. phantom_jam (emergent jam)
            3. driver_validation (human vs autonomous)
            4. bottleneck (speed reduction zone)
            5. correctness_* scenarios (NetLogo validation)
        """)


if __name__ == "__main__":
    main()

# =============================================================================
# USAGE
# =============================================================================
#
# To run this simulation (CLI):
#   streamlit run IKT467_traffic_jam_model.py
#
# =============================================================================
