"""
simulate_stadium.py — v7 FINAL
================================
Same v7 architecture as airport and mall:
  switch_agent_journey + per-agent minute counters
  agents_in_polygon() called ONLY at snapshot (every 60s)
  occupancy_ratio-based labels (size-independent)

STADIUM PATTERNS:
  Pre-match (3h before): gates open, spectators arrive in waves
  Match (90 min + 15 min HT): seated, minimal corridor movement
  Half-time (15 min): concourse / food / toilets surge
  Full-time: mass exit from all zones simultaneously
  Post-match (1h): trickle clearance

ZONES (10): main_gate_north, main_gate_south, concourse_lower,
            concourse_upper, seating_north_stand, seating_south_stand,
            seating_east_stand, seating_west_stand, food_beverage_kiosks,
            pitch_perimeter

SCENARIOS (6) — matching image cards exactly:
  CAT A (Normal / weekday):
    stadium_prematch_steady   — Pre-match steady entry
    stadium_midmatch_seated   — Mid-match seated state
    stadium_halftime_rush     — Half-time concourse rush
    stadium_one_gate_bottleneck — One gate entry bottleneck
    stadium_fulltime_exit     — Full-time exit surge
    stadium_winning_goal_surge — Winning goal crowd surge

  CAT B (Holiday / festival):
    stadium_festival_preshow   — Festival concert pre-show spread
    stadium_oversold_entry_jam — Over-sold concert entry jam
    stadium_mosh_pit_compression — Mosh pit compression surge
    stadium_emergency_evacuation — Emergency evacuation during show

LABEL LOGIC:
  0 = Normal    : occupancy_ratio < 0.40
  1 = Congested : 0.40 <= occupancy_ratio < 0.75
  2 = Surge     : occupancy_ratio >= 0.75

FEATURE COLUMNS: identical to airport/mall — fully compatible for concat.
"""

import jupedsim as jps
import numpy as np
import pandas as pd
import shapely.geometry as sg
from pathlib import Path
import math, random, warnings, os, sys

warnings.filterwarnings("ignore")
random.seed(42); np.random.seed(42)

OUT_DIR = Path("stadium_output")
OUT_DIR.mkdir(exist_ok=True)

class _Q:
    def __enter__(self): self._o = sys.stdout; sys.stdout = open(os.devnull, 'w')
    def __exit__(self, *_): sys.stdout.close(); sys.stdout = self._o

# ══════════════════════════════════════════════════════════════════════════════
# 1. ZONES
# ══════════════════════════════════════════════════════════════════════════════
# Layout (220 x 120 m walkable area):
#  North gate: top-centre entrance
#  South gate: bottom-centre entrance
#  Concourse lower runs around inner ring
#  Concourse upper runs around outer ring
#  Four stands fill the corners / sides
#  Food kiosks embedded in concourse lower
#  Pitch perimeter is a thin strip around the pitch (not walkable in sim — agents
#    who breach it are captured there — used for pitch-rush events)

ZONES = {
    "main_gate_north":     {"polygon": sg.Polygon([(80, 0),(140, 0),(140,14),(80,14)]),
                            "capacity": 800,  "num_exits": 8},
    "main_gate_south":     {"polygon": sg.Polygon([(80,106),(140,106),(140,120),(80,120)]),
                            "capacity": 800,  "num_exits": 8},
    "concourse_lower":     {"polygon": sg.Polygon([(30,14),(190,14),(190,50),(30,50)]),
                            "capacity": 4000, "num_exits": 12},
    "concourse_upper":     {"polygon": sg.Polygon([(30,70),(190,70),(190,106),(30,106)]),
                            "capacity": 4000, "num_exits": 12},
    "seating_north_stand": {"polygon": sg.Polygon([(0, 0),(80, 0),(80,60),(0,60)]),
                            "capacity": 8000, "num_exits": 10},
    "seating_south_stand": {"polygon": sg.Polygon([(140, 60),(220,60),(220,120),(140,120)]),
                            "capacity": 8000, "num_exits": 10},
    "seating_east_stand":  {"polygon": sg.Polygon([(190, 0),(220, 0),(220,60),(190,60)]),
                            "capacity": 6000, "num_exits": 8},
    "seating_west_stand":  {"polygon": sg.Polygon([(0, 60),(30,60),(30,120),(0,120)]),
                            "capacity": 6000, "num_exits": 8},
    "food_beverage_kiosks":{"polygon": sg.Polygon([(30,50),(190,50),(190,70),(30,70)]),
                            "capacity": 2000, "num_exits": 16},
    "pitch_perimeter":     {"polygon": sg.Polygon([(80,14),(140,14),(140,106),(80,106)]),
                            "capacity": 500,  "num_exits": 4},
}

NEIGHBORS = {
    "main_gate_north":     ["concourse_lower", "seating_north_stand", "seating_east_stand"],
    "main_gate_south":     ["concourse_upper", "seating_south_stand", "seating_west_stand"],
    "concourse_lower":     ["main_gate_north","food_beverage_kiosks","seating_north_stand",
                            "seating_east_stand"],
    "concourse_upper":     ["main_gate_south","food_beverage_kiosks","seating_south_stand",
                            "seating_west_stand"],
    "seating_north_stand": ["main_gate_north","concourse_lower","food_beverage_kiosks"],
    "seating_south_stand": ["main_gate_south","concourse_upper","food_beverage_kiosks"],
    "seating_east_stand":  ["main_gate_north","concourse_lower","food_beverage_kiosks"],
    "seating_west_stand":  ["main_gate_south","concourse_upper","food_beverage_kiosks"],
    "food_beverage_kiosks":["concourse_lower","concourse_upper","seating_north_stand",
                            "seating_south_stand","seating_east_stand","seating_west_stand"],
    "pitch_perimeter":     ["concourse_lower","concourse_upper","food_beverage_kiosks"],
}

WALKABLE = sg.Polygon([(1,1),(219,1),(219,119),(1,119)])

# ══════════════════════════════════════════════════════════════════════════════
# 2. BASE DWELL (minutes)
# ══════════════════════════════════════════════════════════════════════════════

BASE_DWELL_MIN = {
    "main_gate_north":      (0.5,  1.5),
    "main_gate_south":      (0.5,  1.5),
    "concourse_lower":      (1.0,  3.0),
    "concourse_upper":      (1.0,  3.0),
    "seating_north_stand":  (40.0,55.0),   # seated for match duration
    "seating_south_stand":  (40.0,55.0),
    "seating_east_stand":   (40.0,55.0),
    "seating_west_stand":   (40.0,55.0),
    "food_beverage_kiosks": (3.0,  8.0),
    "pitch_perimeter":      (0.5,  2.0),
}

DWELL_MULT = {
    # ── CAT A — Normal / weekday ──────────────────────────────────────────────

    # "gate_entry_rate low-medium, density spreading evenly"
    "stadium_prematch_steady": {**{z: 1.0 for z in ZONES},
        "main_gate_north":     1.5,
        "main_gate_south":     1.5,
        "concourse_lower":     1.2,
        "concourse_upper":     1.2,
        "food_beverage_kiosks":1.3},

    # "almost all spectators seated, minimal movement, only food/toilet zones active"
    # density very low in corridors, count stable, flow near zero
    "stadium_midmatch_seated": {**{z: 0.05 for z in ZONES},
        "seating_north_stand": 1.0,
        "seating_south_stand": 1.0,
        "seating_east_stand":  1.0,
        "seating_west_stand":  1.0,
        "food_beverage_kiosks":0.4,
        "concourse_lower":     0.1,
        "concourse_upper":     0.1},

    # "all spectators exit seats simultaneously — concourse and toilets overwhelmed for 15 min"
    # sudden flow_out spike all zones, 15-min window
    "stadium_halftime_rush": {**{z: 1.0 for z in ZONES},
        "seating_north_stand": 0.15,   # exits seats fast
        "seating_south_stand": 0.15,
        "seating_east_stand":  0.15,
        "seating_west_stand":  0.15,
        "concourse_lower":     3.5,
        "concourse_upper":     3.5,
        "food_beverage_kiosks":4.5},

    # "late arrivals cluster at single entry gate — scanning slow, queue building up into street"
    # gate_entry_rate high at one gate (north), other gates low
    "stadium_one_gate_bottleneck": {**{z: 1.0 for z in ZONES},
        "main_gate_north":     8.0,
        "main_gate_south":     0.3,
        "concourse_lower":     2.0,
        "seating_north_stand": 1.2,
        "seating_east_stand":  1.2},

    # "match ends: 30,000 spectators move to exits simultaneously"
    # time_to_event_end=0, all zones count high, flow_out max
    "stadium_fulltime_exit": {**{z: 0.08 for z in ZONES},
        "main_gate_north":     5.5,
        "main_gate_south":     5.5,
        "concourse_lower":     1.0,
        "concourse_upper":     1.0},

    # "emotional goal — spectators surge toward pitch barrier, standing area density spike"
    # sudden density spike in pitch_perimeter, speed spike, noise_max
    "stadium_winning_goal_surge": {**{z: 1.0 for z in ZONES},
        "pitch_perimeter":     12.0,
        "concourse_lower":     2.5,
        "concourse_upper":     2.5,
        "food_beverage_kiosks":1.5,
        "seating_north_stand": 0.4,
        "seating_south_stand": 0.4},

    # ── CAT B — Holiday / festival ────────────────────────────────────────────

    # "early arrivals exploring venue, merchandise and food zones busy, seating zones filling gradually"
    # is_festival=1, time_since_event_start low, density spreading
    "stadium_festival_preshow": {**{z: 1.2 for z in ZONES},
        "food_beverage_kiosks":2.2,
        "concourse_lower":     1.8,
        "concourse_upper":     1.8,
        "main_gate_north":     1.5,
        "main_gate_south":     1.5},

    # "more tickets sold than gate capacity can process — entry queues massive, all critical"
    # tickets_sold > zone_capacity, gate_entry_rate max
    "stadium_oversold_entry_jam": {**{z: 1.0 for z in ZONES},
        "main_gate_north":     12.0,
        "main_gate_south":     12.0,
        "concourse_lower":     3.5,
        "concourse_upper":     3.5,
        "food_beverage_kiosks":2.0},

    # "front standing zone compresses as crowd surges toward stage at peak song"
    # front zone density x6, speed near zero, noise_max
    "stadium_mosh_pit_compression": {**{z: 1.0 for z in ZONES},
        "pitch_perimeter":     18.0,
        "concourse_lower":     3.0,
        "seating_north_stand": 2.0,
        "seating_south_stand": 2.0,
        "food_beverage_kiosks":1.5},

    # "medical/security emergency mid-show — 50,000 move simultaneously, panic"
    # exit_blocked varies, all flow_out spike, panic noise
    "stadium_emergency_evacuation": {**{z: 0.05 for z in ZONES},
        "main_gate_north":     6.0,
        "main_gate_south":     6.0,
        "concourse_lower":     0.8,
        "concourse_upper":     0.8,
        "pitch_perimeter":     0.1},
}

# ══════════════════════════════════════════════════════════════════════════════
# 3. SPAWN RATE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def G(h, p, w, ht):
    return max(0.0, ht * math.exp(-0.5 * ((h - p) / max(w, 0.01)) ** 2))

# Pre-match steady entry (kick-off at h15):
# gradual spread from h13, peak h14-14:30
# "gate_entry_rate low-medium, density spreading evenly"
def spawn_prematch_steady(h):
    if h < 12 or h >= 16: return 0.0
    return max(0.3, G(h,13,0.8,14) + G(h,14,0.6,18) + 2.0)

# Mid-match seated (h15-h17):
# almost no movement — agents are in seats, minimal new arrivals
def spawn_midmatch_seated(h):
    if h < 14 or h >= 18: return 0.0
    if 15 <= h <= 17: return 0.4   # tiny trickle of late arrivals / re-entrants
    return max(0.0, G(h,14,0.5,8))

# Half-time rush (h16 burst):
# existing agents rush to concourse — no new spawn needed
# we keep spawn low; dwell changes drive the effect
def spawn_halftime_rush(h):
    if h < 13 or h >= 20: return 0.0
    base = G(h,14,0.8,16) + 2.0
    ht_burst = G(h,16,0.25,30) if 15 <= h <= 17 else 0.0
    return max(0.0, base + ht_burst)

# One gate bottleneck (pre-match):
# big wave at h14 funneled through only north gate
def spawn_one_gate_bottleneck(h):
    if h < 12 or h >= 17: return 0.0
    return max(0.5, G(h,14,0.4,26) + G(h,13,0.5,14) + 3.0)

# Full-time exit (h17):
# all agents already present — spawn drops to 0 at match end
def spawn_fulltime_exit(h):
    if h >= 17: return 0.0
    if h < 12:  return 0.0
    return max(0.3, G(h,14,0.6,16) + 2.0)

# Winning goal surge (h16):
# normal match attendance — goal announced at h16
def spawn_winning_goal_surge(h):
    if h < 12 or h >= 20: return 0.0
    return max(0.3, G(h,13,0.8,14) + G(h,14,0.6,18) + 2.0)

# Festival concert pre-show (is_festival=1):
# early arrivals from h16, venue opens h17, show h19
# "density spreading gradually, merch/food zones busy"
def spawn_festival_preshow(h):
    if h < 15 or h >= 23: return 0.0
    return max(1.0, G(h,17,1.2,18) + G(h,18,1.0,16) + G(h,19,0.8,14) + 4.0)

# Over-sold concert entry jam (is_festival=1):
# massive gate pressure — tickets_sold > capacity
# spike at h18-h19 as crowd masses at gates
def spawn_oversold_entry_jam(h):
    if h < 15 or h >= 23: return 0.0
    return max(2.0, G(h,18,0.5,34) + G(h,19,0.5,30) + G(h,17,0.8,18) + 6.0)

# Mosh pit compression (festival, h20 peak song):
# venue full — spawn drops, but existing crowd compresses forward
def spawn_mosh_pit_compression(h):
    if h < 15 or h >= 24: return 0.0
    normal = G(h,17,1.2,18) + G(h,18,1.0,16) + 4.0
    return max(0.5, normal)

# Emergency evacuation (mid-show h20-h21):
# show was running — all agents already inside, spawn stops at emergency
def spawn_emergency_evacuation(h):
    if h >= 20: return 0.0   # emergency announced h20 — no more entry
    if h < 15:  return 0.0
    return max(1.0, G(h,17,1.2,16) + G(h,18,1.0,14) + 4.0)

# ══════════════════════════════════════════════════════════════════════════════
# 4. SCENARIOS — all 10 matching image cards exactly
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    # ── CAT A — Normal day / weekday / regular operations ────────────────────

    # "gate_entry_rate low-medium, density spreading evenly across seating zones,
    #  food/beverage areas moderately busy"
    "stadium_prematch_steady": {
        "name": "Pre-Match Steady Entry", "cat": "A", "model": "CFSM",
        "event_type": "none", "event_scale": 0,
        "is_holiday": 0, "is_festival": 0,
        "weather": "clear", "temperature_c": 25,
        "spawn_fn": spawn_prematch_steady,
        "ev_start": 15, "ev_end": 17,
        "exit_blocked_h": -1, "exit_blocked_z": [],
    },

    # "almost all spectators seated, minimal movement, only food/toilet zones active,
    #  density very low in corridors, count stable, flow near zero"
    "stadium_midmatch_seated": {
        "name": "Mid-Match Seated State", "cat": "A", "model": "CFSM",
        "event_type": "none", "event_scale": 0,
        "is_holiday": 0, "is_festival": 0,
        "weather": "clear", "temperature_c": 25,
        "spawn_fn": spawn_midmatch_seated,
        "ev_start": 15, "ev_end": 17,
        "exit_blocked_h": -1, "exit_blocked_z": [],
    },

    # "all spectators exit seats simultaneously at half-time — concourse and
    #  toilets overwhelmed for 15 min. sudden flow_out spike all zones"
    "stadium_halftime_rush": {
        "name": "Half-Time Concourse Rush", "cat": "A", "model": "CFSMv2",
        "event_type": "halftime", "event_scale": 2,
        "is_holiday": 0, "is_festival": 0,
        "weather": "clear", "temperature_c": 25,
        "spawn_fn": spawn_halftime_rush,
        "ev_start": 16, "ev_end": 16,
        "exit_blocked_h": -1, "exit_blocked_z": [],
    },

    # "late arrivals cluster at single entry gate, scanning slow,
    #  queue building up into street-facing zone.
    #  gate_entry_rate high at one gate, other gates low"
    "stadium_one_gate_bottleneck": {
        "name": "One Gate Entry Bottleneck", "cat": "A", "model": "CFSMv2",
        "event_type": "none", "event_scale": 1,
        "is_holiday": 0, "is_festival": 0,
        "weather": "clear", "temperature_c": 24,
        "spawn_fn": spawn_one_gate_bottleneck,
        "ev_start": 14, "ev_end": 15,
        "exit_blocked_h": -1, "exit_blocked_z": [],
    },

    # "match ends — all 30,000+ spectators move to exits simultaneously.
    #  time_to_event_end=0, all zones count high, flow_out max"
    "stadium_fulltime_exit": {
        "name": "Full-Time Exit Surge", "cat": "A", "model": "CFSMv2",
        "event_type": "none", "event_scale": 2,
        "is_holiday": 0, "is_festival": 0,
        "weather": "clear", "temperature_c": 25,
        "spawn_fn": spawn_fulltime_exit,
        "ev_start": 15, "ev_end": 17,
        "exit_blocked_h": -1, "exit_blocked_z": [],
    },

    # "emotional goal causes spectators to surge toward pitch barrier.
    #  standing area density spike, speed spike, noise high"
    "stadium_winning_goal_surge": {
        "name": "Winning Goal Crowd Surge", "cat": "A", "model": "CFSMv2",
        "event_type": "goal_surge", "event_scale": 2,
        "is_holiday": 0, "is_festival": 0,
        "weather": "clear", "temperature_c": 25,
        "spawn_fn": spawn_winning_goal_surge,
        "ev_start": 16, "ev_end": 17,
        "exit_blocked_h": -1, "exit_blocked_z": [],
    },

    # ── CAT B — Holiday / festival day / event-driven ─────────────────────────

    # "early arrivals exploring venue, merchandise and food zones busy,
    #  seating zones filling gradually.
    #  is_festival=1, time_since_event_start low, density spreading"
    "stadium_festival_preshow": {
        "name": "Festival Concert Pre-Show Spread", "cat": "B", "model": "CFSMv2",
        "event_type": "concert", "event_scale": 2,
        "is_holiday": 1, "is_festival": 1,
        "weather": "clear", "temperature_c": 24,
        "spawn_fn": spawn_festival_preshow,
        "ev_start": 19, "ev_end": 23,
        "exit_blocked_h": -1, "exit_blocked_z": [],
    },

    # "more tickets sold than gate capacity can process — entry queues massive,
    #  entry zones at critical density.
    #  tickets_sold > zone_capacity, gate_entry_rate max"
    "stadium_oversold_entry_jam": {
        "name": "Over-Sold Concert Entry Jam", "cat": "B", "model": "CFSMv2",
        "event_type": "concert", "event_scale": 3,
        "is_holiday": 1, "is_festival": 1,
        "weather": "clear", "temperature_c": 24,
        "spawn_fn": spawn_oversold_entry_jam,
        "ev_start": 19, "ev_end": 23,
        "exit_blocked_h": -1, "exit_blocked_z": [],
    },

    # "front standing zone compresses as crowd surges toward stage at peak song.
    #  barriers at pressure limit.
    #  is_festival=1, front_zone density x6, speed near zero, noise_max"
    "stadium_mosh_pit_compression": {
        "name": "Mosh Pit Compression Surge", "cat": "B", "model": "CFSMv2",
        "event_type": "concert", "event_scale": 3,
        "is_holiday": 1, "is_festival": 1,
        "weather": "clear", "temperature_c": 23,
        "spawn_fn": spawn_mosh_pit_compression,
        "ev_start": 19, "ev_end": 23,
        "exit_blocked_h": -1, "exit_blocked_z": [],
    },

    # "medical/security emergency mid-show triggers announcement — 50,000 crowd
    #  moves to exits simultaneously. exit_blocked varies, all flow_out spike, panic noise"
    "stadium_emergency_evacuation": {
        "name": "Emergency Evacuation During Show", "cat": "B", "model": "CFSMv2",
        "event_type": "emergency", "event_scale": 3,
        "is_holiday": 1, "is_festival": 1,
        "weather": "clear", "temperature_c": 23,
        "spawn_fn": spawn_emergency_evacuation,
        "ev_start": 19, "ev_end": 20,
        "exit_blocked_h": 20,
        "exit_blocked_z": ["main_gate_north"],   # one gate blocked → redirected pressure
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# 5. SIMULATION FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def make_simulation(model_key):
    m = {
        "CFSM": jps.CollisionFreeSpeedModel(
            strength_neighbor_repulsion=8.0, range_neighbor_repulsion=0.1,
            strength_geometry_repulsion=5.0, range_geometry_repulsion=0.02),
        "CFSMv2": jps.CollisionFreeSpeedModelV2(),
    }
    return jps.Simulation(model=m[model_key], geometry=WALKABLE, dt=0.1)

def make_agent_params(mk, pos, jid, sid, spd, rad):
    kw = dict(position=pos, journey_id=jid, stage_id=sid, desired_speed=spd)
    if mk == "CFSM":
        return jps.CollisionFreeSpeedModelAgentParameters(**kw, radius=rad, time_gap=1.0)
    return jps.CollisionFreeSpeedModelV2AgentParameters(**kw, radius=rad, time_gap=1.0)

# ══════════════════════════════════════════════════════════════════════════════
# 6. JOURNEY BUILDER
# ══════════════════════════════════════════════════════════════════════════════

WP_POS = {
    "gate_n":    (110,   7),
    "gate_s":    (110, 113),
    "conc_lo":   (110,  32),
    "conc_up":   (110,  88),
    "seat_n":    ( 40,  30),
    "seat_s":    (180,  90),
    "seat_e":    (205,  30),
    "seat_w":    ( 15,  90),
    "food":      (110,  60),
    "pitch":     (110,  60),
}

# Waiting-set queue positions per zone
QUEUE_POS = {
    "main_gate_north":     [(84+i*4,  7)  for i in range(15)],
    "main_gate_south":     [(84+i*4,113)  for i in range(15)],
    "concourse_lower":     [(34+i*13,32)  for i in range(12)],
    "concourse_upper":     [(34+i*13,88)  for i in range(12)],
    "seating_north_stand": [(5+i*7,  30)  for i in range(11)],
    "seating_south_stand": [(145+i*6,90)  for i in range(11)],
    "seating_east_stand":  [(193+i*4,30)  for i in range(7)],
    "seating_west_stand":  [(3+i*4,  90)  for i in range(7)],
    "food_beverage_kiosks":[(34+i*13,60)  for i in range(12)],
    "pitch_perimeter":     [(84+i*8, 16)  for i in range(8)],
}

# Separate concourse waiting stages per journey type
# (same fix pattern as mall corridor)
QUEUE_COR_N_SEAT  = [(34+i*13, 22) for i in range(12)]   # via north to seating
QUEUE_COR_S_SEAT  = [(34+i*13, 98) for i in range(12)]   # via south to seating
QUEUE_COR_FOOD    = [(34+i*13, 38) for i in range(12)]   # to food kiosks
QUEUE_COR_PITCH   = [(84+i*8,  44) for i in range(8)]    # to pitch / standing zone

EXIT_POLY_N = [(82,1),(138,1),(138,13),(82,13)]
EXIT_POLY_S = [(82,107),(138,107),(138,119),(82,119)]

def build_journeys(sim, scenario_id):
    T    = jps.Transition.create_fixed_transition
    mult = DWELL_MULT[scenario_id]

    wp = {k: sim.add_waypoint_stage(v, 1.5) for k, v in WP_POS.items()}

    ws = {z: sim.add_waiting_set_stage(QUEUE_POS[z]) for z in QUEUE_POS}

    # Per-type concourse waiting stages (avoid dead-zone routing bug)
    ws_cor_n_seat  = sim.add_waiting_set_stage(QUEUE_COR_N_SEAT)
    ws_cor_s_seat  = sim.add_waiting_set_stage(QUEUE_COR_S_SEAT)
    ws_cor_food    = sim.add_waiting_set_stage(QUEUE_COR_FOOD)
    ws_cor_pitch   = sim.add_waiting_set_stage(QUEUE_COR_PITCH)

    for sid in ws.values():
        sim.get_stage(sid).state = jps.WaitingSetState.ACTIVE
    for sid in [ws_cor_n_seat, ws_cor_s_seat, ws_cor_food, ws_cor_pitch]:
        sim.get_stage(sid).state = jps.WaitingSetState.ACTIVE

    exit_n = sim.add_exit_stage(EXIT_POLY_N)
    exit_s = sim.add_exit_stage(EXIT_POLY_S)

    dwell_min = {}
    for zone, (mn, mx) in BASE_DWELL_MIN.items():
        dwell_min[zone] = random.uniform(mn, mx) * mult.get(zone, 1.0)
    dwell_min["concourse_lower"] = random.uniform(1.0, 3.0) * mult.get("concourse_lower", 1.0)
    dwell_min["concourse_upper"] = random.uniform(1.0, 3.0) * mult.get("concourse_upper", 1.0)

    def mj(*stages):
        jd = jps.JourneyDescription(list(stages))
        for i in range(len(stages) - 1):
            jd.set_transition_for_stage(stages[i], T(stages[i+1]))
        return sim.add_journey(jd), stages[0]

    release_info = {}
    # After gate: go to concourse then seat (north)
    release_info[ws["main_gate_north"]] = mj(
        wp["conc_lo"], ws_cor_n_seat, wp["seat_n"], ws["seating_north_stand"], exit_n)
    release_info[ws["main_gate_south"]] = mj(
        wp["conc_up"], ws_cor_s_seat, wp["seat_s"], ws["seating_south_stand"], exit_s)

    # Concourse releases
    release_info[ws_cor_n_seat] = mj(wp["seat_n"], ws["seating_north_stand"], exit_n)
    release_info[ws_cor_s_seat] = mj(wp["seat_s"], ws["seating_south_stand"], exit_s)
    release_info[ws_cor_food]   = mj(wp["food"],   ws["food_beverage_kiosks"], exit_n)
    release_info[ws_cor_pitch]  = mj(wp["pitch"],  ws["pitch_perimeter"],      exit_n)

    # Zone releases
    release_info[ws["concourse_lower"]]     = mj(exit_n)
    release_info[ws["concourse_upper"]]     = mj(exit_s)
    release_info[ws["seating_north_stand"]] = mj(wp["conc_lo"], exit_n)
    release_info[ws["seating_south_stand"]] = mj(wp["conc_up"], exit_s)
    release_info[ws["seating_east_stand"]]  = mj(wp["conc_lo"], exit_n)
    release_info[ws["seating_west_stand"]]  = mj(wp["conc_up"], exit_s)
    release_info[ws["food_beverage_kiosks"]]= mj(exit_n)
    release_info[ws["pitch_perimeter"]]     = mj(exit_n)

    ws_to_zone_extra = {
        ws_cor_n_seat: "concourse_lower",
        ws_cor_s_seat: "concourse_upper",
        ws_cor_food:   "food_beverage_kiosks",
        ws_cor_pitch:  "pitch_perimeter",
    }

    # ── JOURNEY A: North spectator — gate_n -> concourse_lo -> seating_n -> exit
    jd_A = jps.JourneyDescription([wp["gate_n"], ws["main_gate_north"],
                                    wp["conc_lo"], ws_cor_n_seat,
                                    wp["seat_n"],  ws["seating_north_stand"], exit_n])
    jd_A.set_transition_for_stage(wp["gate_n"],                T(ws["main_gate_north"]))
    jd_A.set_transition_for_stage(ws["main_gate_north"],       T(wp["conc_lo"]))
    jd_A.set_transition_for_stage(wp["conc_lo"],               T(ws_cor_n_seat))
    jd_A.set_transition_for_stage(ws_cor_n_seat,               T(wp["seat_n"]))
    jd_A.set_transition_for_stage(wp["seat_n"],                T(ws["seating_north_stand"]))
    jd_A.set_transition_for_stage(ws["seating_north_stand"],   T(exit_n))
    jid_A = sim.add_journey(jd_A)

    # ── JOURNEY B: South spectator — gate_s -> concourse_up -> seating_s -> exit
    jd_B = jps.JourneyDescription([wp["gate_s"], ws["main_gate_south"],
                                    wp["conc_up"], ws_cor_s_seat,
                                    wp["seat_s"],  ws["seating_south_stand"], exit_s])
    jd_B.set_transition_for_stage(wp["gate_s"],                T(ws["main_gate_south"]))
    jd_B.set_transition_for_stage(ws["main_gate_south"],       T(wp["conc_up"]))
    jd_B.set_transition_for_stage(wp["conc_up"],               T(ws_cor_s_seat))
    jd_B.set_transition_for_stage(ws_cor_s_seat,               T(wp["seat_s"]))
    jd_B.set_transition_for_stage(wp["seat_s"],                T(ws["seating_south_stand"]))
    jd_B.set_transition_for_stage(ws["seating_south_stand"],   T(exit_s))
    jid_B = sim.add_journey(jd_B)

    # ── JOURNEY C: Food visitor — gate_n -> concourse_lo -> food_kiosks -> exit
    jd_C = jps.JourneyDescription([wp["gate_n"], ws["main_gate_north"],
                                    wp["conc_lo"], ws_cor_food,
                                    wp["food"],    ws["food_beverage_kiosks"], exit_n])
    jd_C.set_transition_for_stage(wp["gate_n"],                T(ws["main_gate_north"]))
    jd_C.set_transition_for_stage(ws["main_gate_north"],       T(wp["conc_lo"]))
    jd_C.set_transition_for_stage(wp["conc_lo"],               T(ws_cor_food))
    jd_C.set_transition_for_stage(ws_cor_food,                 T(wp["food"]))
    jd_C.set_transition_for_stage(wp["food"],                  T(ws["food_beverage_kiosks"]))
    jd_C.set_transition_for_stage(ws["food_beverage_kiosks"],  T(exit_n))
    jid_C = sim.add_journey(jd_C)

    # ── JOURNEY D: Standing / pitch zone — gate_n -> concourse_lo -> pitch_perimeter -> exit
    jd_D = jps.JourneyDescription([wp["gate_n"], ws["main_gate_north"],
                                    wp["conc_lo"], ws_cor_pitch,
                                    wp["pitch"],   ws["pitch_perimeter"], exit_n])
    jd_D.set_transition_for_stage(wp["gate_n"],                T(ws["main_gate_north"]))
    jd_D.set_transition_for_stage(ws["main_gate_north"],       T(wp["conc_lo"]))
    jd_D.set_transition_for_stage(wp["conc_lo"],               T(ws_cor_pitch))
    jd_D.set_transition_for_stage(ws_cor_pitch,                T(wp["pitch"]))
    jd_D.set_transition_for_stage(wp["pitch"],                 T(ws["pitch_perimeter"]))
    jd_D.set_transition_for_stage(ws["pitch_perimeter"],       T(exit_n))
    jid_D = sim.add_journey(jd_D)

    journeys = {
        "north_spectator": (jid_A, wp["gate_n"]),
        "south_spectator": (jid_B, wp["gate_s"]),
        "food_visitor":    (jid_C, wp["gate_n"]),
        "standing":        (jid_D, wp["gate_n"]),
    }
    return journeys, ws, ws_to_zone_extra, release_info, dwell_min

# ══════════════════════════════════════════════════════════════════════════════
# 7. SPAWN
# ══════════════════════════════════════════════════════════════════════════════

J_NAMES   = ["north_spectator","south_spectator","food_visitor","standing"]
J_WEIGHTS = [0.40, 0.40, 0.12, 0.08]
ENT_POLY  = sg.Polygon([(84,1),(136,1),(136,13),(84,13)])
MAX_AGENTS = 2000

def try_spawn(sim, mk, n, journeys):
    if n <= 0 or sim.agent_count() >= MAX_AGENTS: return 0
    gn = len(list(sim.agents_in_polygon(ZONES["main_gate_north"]["polygon"])))
    gs = len(list(sim.agents_in_polygon(ZONES["main_gate_south"]["polygon"])))
    if (gn + gs) >= (800 + 800) * 0.85: return 0
    n = min(n, MAX_AGENTS - sim.agent_count(), 25)
    try:
        positions = jps.distribute_by_number(polygon=ENT_POLY, number_of_agents=n,
            distance_to_agents=0.55, distance_to_polygon=0.50,
            seed=random.randint(0, 99999), max_iterations=20000)
    except Exception: return 0
    spawned = 0
    for pos in positions:
        jt = random.choices(J_NAMES, weights=J_WEIGHTS, k=1)[0]
        jid, fs = journeys[jt]
        with _Q():
            try:
                p = make_agent_params(mk, pos, jid, fs,
                    random.uniform(0.9, 1.6), random.uniform(0.21, 0.27))
                sim.add_agent(p); spawned += 1
            except Exception: pass
    return spawned

# ══════════════════════════════════════════════════════════════════════════════
# 8. SNAPSHOT + DWELL MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def snapshot_and_manage(sim, minute, sc, prev_ids, ws, ws_to_zone_extra,
                         release_info, dwell_min, agent_min_ctr):
    hour = minute // 60
    snaps = []
    curr_ids = {}

    ws_to_zone = {v: k for k, v in ws.items()}
    ws_to_zone.update(ws_to_zone_extra)

    for zname, zinfo in ZONES.items():
        curr_ids[zname] = set(sim.agents_in_polygon(zinfo["polygon"]))

    active_ids = set()
    to_release = []
    for ag in sim.agents():
        aid = ag.id; sid = ag.stage_id; active_ids.add(aid)
        if sid in ws_to_zone:
            agent_min_ctr[aid] = agent_min_ctr.get(aid, 0) + 1
            zone_name  = ws_to_zone[sid]
            threshold  = dwell_min.get(zone_name, 2.0) * random.uniform(0.90, 1.10)
            if agent_min_ctr[aid] >= threshold:
                if sid in release_info:
                    to_release.append((aid, release_info[sid]))
                agent_min_ctr.pop(aid, None)
        else:
            agent_min_ctr.pop(aid, None)
    for k in [k for k in agent_min_ctr if k not in active_ids]:
        del agent_min_ctr[k]
    for aid, (rel_jid, rel_stage) in to_release:
        try: sim.switch_agent_journey(aid, rel_jid, rel_stage)
        except Exception: pass

    ev_s = sc["ev_start"]; ev_e = sc["ev_end"]

    for zname, zinfo in ZONES.items():
        agent_set = curr_ids[zname]
        count = len(agent_set)
        area  = zinfo["polygon"].area

        speeds = []
        for aid in list(agent_set)[:40]:
            try:
                m = sim.agent(aid).model
                if hasattr(m, "velocity"):
                    vx, vy = m.velocity; speeds.append(math.sqrt(vx**2 + vy**2))
                elif hasattr(m, "speed"):         speeds.append(float(m.speed))
                elif hasattr(m, "desired_speed"): speeds.append(float(m.desired_speed) * 0.65)
            except: pass
        avg_spd = round(float(np.mean(speeds)) if speeds else 0.0, 4)

        density = count / area
        occ_r   = count / zinfo["capacity"]
        is_blocked = (1 if sc["exit_blocked_h"] != -1
                      and hour >= sc["exit_blocked_h"]
                      and zname in sc["exit_blocked_z"] else 0)
        exits_open = max(1, zinfo["num_exits"] - is_blocked)
        t_s = (ev_s - hour) if ev_s != -1 else -1
        t_e = (ev_e - hour) if ev_e != -1 else -1
        prev = prev_ids.get(zname, set())
        fi   = len(agent_set - prev)
        fo   = len(prev - agent_set)
        imb  = fi - fo
        ratio = fi / (fo + 1e-6)
        pres  = density * max(abs(imb), 0.1)

        snaps.append({
            "zone_name": zname, "minute": minute, "hour": hour,
            "count": count, "density": round(density, 5),
            "occupancy_ratio": round(min(occ_r, 2.0), 4),
            "avg_speed": avg_spd,
            "flow_in": fi, "flow_out": fo,
            "flow_imbalance": imb,
            "flow_ratio": round(ratio, 4),
            "avg_dwelltime": dwell_min.get(zname, 1.0),
            "pressure_score": round(pres, 5),
            "exit_blocked_flag": is_blocked,
            "exits_open_count": exits_open,
            "time_to_event_start": t_s,
            "time_to_event_end": t_e,
        })
    return snaps, curr_ids

# ══════════════════════════════════════════════════════════════════════════════
# 9. RUN ONE SCENARIO
# ══════════════════════════════════════════════════════════════════════════════

def run_scenario(scenario_id):
    sc = SCENARIOS[scenario_id]; mk = sc["model"]; sfn = sc["spawn_fn"]
    print(f"\n{'='*65}")
    print(f"  {scenario_id}")
    print(f"  {sc['name']}  |  Model:{mk}  Cat:{sc['cat']}")
    print(f"{'='*65}")

    sim = make_simulation(mk)
    journeys, ws, ws_to_zone_extra, rel_info, dwell_m = build_journeys(sim, scenario_id)

    spawn_acc     = 0.0
    prev_ids      = {z: set() for z in ZONES}
    agent_min_ctr = {}
    all_snaps     = []
    minute_num    = 0

    for step in range(864000):  # 24h * 3600s / 0.1s_dt
        hour_now = int(step * 0.1 / 3600)

        if step % 600 == 0:   # every 60 sim-seconds = 1 snapshot-minute
            rate = sfn(hour_now); spawn_acc += rate
            n = int(spawn_acc);  spawn_acc -= n
            if n > 0: try_spawn(sim, mk, n, journeys)

            if step > 0:
                snaps, prev_ids = snapshot_and_manage(
                    sim, minute_num, sc, prev_ids, ws, ws_to_zone_extra,
                    rel_info, dwell_m, agent_min_ctr)
                all_snaps.extend(snaps)
                if minute_num % 60 == 0:
                    h = minute_num // 60
                    counts = {s["zone_name"]: s["count"] for s in snaps}
                    top = sorted(counts.items(), key=lambda x: -x[1])[:3]
                    top_s = ", ".join(f"{z}={c}" for z, c in top)
                    print(f"    Hour {h:02d}/24  agents={sim.agent_count():4d}  top:{top_s}")
            minute_num += 1

        try: sim.iterate()
        except RuntimeError: pass

    print(f"  Done — {len(all_snaps)} snapshots")
    return all_snaps

# ══════════════════════════════════════════════════════════════════════════════
# 10. AGGREGATE MINUTE → HOURLY  (occupancy_ratio labels + clipped dcr)
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_hourly(minute_snaps, scenario_id):
    sc   = SCENARIOS[scenario_id]
    df   = pd.DataFrame(minute_snaps)
    rows = []

    for zname in ZONES:
        zdf   = df[df["zone_name"] == zname].copy()
        zinfo = ZONES[zname]
        area  = zinfo["polygon"].area

        for hour in range(24):
            hdf = zdf[zdf["hour"] == hour]

            if hdf.empty:
                count_m = dens_max = spd_m = occ_m = 0.0
                fi = fo = imb = 0
                ratio = dwell = pres = 0.0
                blocked = 0; exits = zinfo["num_exits"]
            else:
                count_m  = hdf["count"].mean()
                dens_max = hdf["density"].max()
                spd_m    = hdf["avg_speed"].mean()
                occ_m    = hdf["occupancy_ratio"].max()
                fi       = int(hdf["flow_in"].sum())
                fo       = int(hdf["flow_out"].sum())
                imb      = fi - fo      # sum first, then subtract
                ratio    = fi / (fo + 1e-6)
                dwell    = hdf["avg_dwelltime"].mean()
                pres     = float(hdf["pressure_score"].max())
                blocked  = int(hdf["exit_blocked_flag"].max())
                exits    = int(hdf["exits_open_count"].min())

            if hour > 0:
                ph       = zdf[zdf["hour"] == hour - 1]
                count_t1 = ph["count"].mean()  if not ph.empty else 0.0
                dens_t1  = ph["density"].max() if not ph.empty else 0.0
                fi_t1    = int(ph["flow_in"].sum()) if not ph.empty else 0
            else:
                count_t1 = dens_t1 = fi_t1 = 0.0

            count_trend  = count_m - count_t1
            raw_dens_chg = dens_max - dens_t1
            dens_chg     = float(np.clip(raw_dens_chg, -50.0, 50.0))

            rdf = zdf[zdf["hour"].between(max(0, hour-3), max(0, hour-1))]
            r3h = rdf["count"].mean() if not rdf.empty else count_m

            nd, nf = [], []
            for nb in NEIGHBORS.get(zname, []):
                nbdf = df[(df["zone_name"] == nb) & (df["hour"] == hour)]
                if not nbdf.empty:
                    nd.append(nbdf["density"].mean())
                    nf.append(nbdf["flow_in"].sum())
            nb_dens = float(np.mean(nd)) if nd else 0.0
            nb_fi   = float(np.sum(nf))  if nf else 0.0

            ev_s = sc["ev_start"]; ev_e = sc["ev_end"]
            t_s  = (ev_s - hour) if ev_s != -1 else -1
            t_e  = (ev_e - hour) if ev_e != -1 else -1

            # Occupancy-ratio-based labels (size-independent)
            if   occ_m < 0.40:  label = 0   # Normal    : < 40% capacity
            elif occ_m < 0.75:  label = 1   # Congested : 40–75%
            else:               label = 2   # Surge     : > 75%

            rows.append({
                "zone_name": zname, "venue_type": "stadium",
                "count": round(count_m, 2), "density": round(dens_max, 5),
                "occupancy_ratio": round(occ_m, 4), "avg_speed": round(spd_m, 4),
                "flow_in": fi, "avg_dwelltime": round(dwell, 2),
                "flow_out": fo, "flow_imbalance": imb,
                "flow_ratio": round(ratio, 4),
                "count_t-1": round(count_t1, 2), "density_t-1": round(dens_t1, 5),
                "flow_in_t-1": int(fi_t1), "count_trend": round(count_trend, 2),
                "density_change_rate": round(dens_chg, 5),
                "rolling_count_3h": round(r3h, 2),
                "neighbor_density_avg": round(nb_dens, 5),
                "neighbor_flow_in": round(nb_fi, 2),
                "event_type": sc["event_type"], "event_scale": sc["event_scale"],
                "time_to_event_start": t_s, "time_to_event_end": t_e,
                "hour_of_day": hour, "day_of_week": 1,
                "is_weekend": sc.get("is_holiday", 0),
                "pressure_score": round(pres, 5),
                "exit_blocked_flag": blocked, "exits_open_count": exits,
                "zone_capacity": zinfo["capacity"],
                "zone_area_sqm": round(area, 2), "num_exits": zinfo["num_exits"],
                "weather": sc["weather"], "temperature_c": sc["temperature_c"],
                "scenario_id": scenario_id, "scenario_name": sc["name"],
                "model_used": sc["model"], "category": sc["cat"], "label": label,
            })
    return pd.DataFrame(rows)

FEATURE_COLS = [
    "zone_name","venue_type","count","density","occupancy_ratio","avg_speed",
    "flow_in","avg_dwelltime","flow_out","flow_imbalance","flow_ratio",
    "count_t-1","density_t-1","flow_in_t-1","count_trend","density_change_rate",
    "rolling_count_3h","neighbor_density_avg","neighbor_flow_in",
    "event_type","event_scale","time_to_event_start","time_to_event_end",
    "hour_of_day","day_of_week","is_weekend","pressure_score",
    "exit_blocked_flag","exits_open_count","zone_capacity","zone_area_sqm",
    "num_exits","weather","temperature_c",
    "scenario_id","scenario_name","model_used","category","label",
]

def main():
    all_hourly = []
    for sid in SCENARIOS:
        snaps  = run_scenario(sid)
        hourly = aggregate_hourly(snaps, sid).reindex(columns=FEATURE_COLS)
        path   = OUT_DIR / f"{sid}_hourly.csv"
        hourly.to_csv(path, index=False)
        print(f"  Saved {path}  [{len(hourly)} rows]")
        all_hourly.append(hourly)

    master = pd.concat(all_hourly, ignore_index=True).reindex(columns=FEATURE_COLS)
    master.to_csv(OUT_DIR / "master_stadium.csv", index=False)

    expected = len(SCENARIOS) * len(ZONES) * 24
    print(f"\n{'='*65}")
    print(f"  master_stadium.csv: {len(master)}/{expected} rows")
    print(f"\n  Label distribution:")
    for lbl, cnt in master["label"].value_counts().sort_index().items():
        name = {0:"Normal",1:"Congested",2:"Surge"}[lbl]
        print(f"    [{lbl}] {name:<12}: {cnt:5d}  ({cnt/len(master)*100:.1f}%)")
    print(f"\n  Scenario breakdown:")
    for sid in SCENARIOS:
        sub = master[master["scenario_id"] == sid]
        print(f"    {sid:<40} 0={len(sub[sub.label==0]):3d} "
              f"1={len(sub[sub.label==1]):3d} 2={len(sub[sub.label==2]):3d}")
    print(f"{'='*65}")

if __name__ == "__main__":
    main()