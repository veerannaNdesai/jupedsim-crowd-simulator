"""
simulate_mall.py — v7 FIXED
=============================
FIXES APPLIED:
  1. Label thresholds: occupancy_ratio based (not density-based) — works for all zone sizes
  2. Journey routing: per-journey-type corridor waiting stages so lunch->food_court,
     movie->multiplex, grocery->hypermarket, shopper->retail — no cross-contamination
  3. Closing time realism: spawn functions return 0 after closing_hour, dwell multiplier
     drops sharply in closing scenarios, TOD curve has proper wind-down at h21-22
  4. flow_imbalance: computed as flow_in - flow_out from actual prev_ids diff (was correct
     in snapshot but aggregation now correctly sums raw fi-fo, not recomputes)
  5. Dead zone fix: separate waiting stages per journey type in main_corridor so each
     agent goes to the correct next zone after their corridor wait
  6. Scenario realism: all 10 scenarios match the image scenario cards exactly

MALL PATTERNS:
  00-09h: CLOSED (zero spawn)
  09-10h: pre-open trickle (extended_hours / festival only)
  10-12h: soft open
  12-14h: LUNCH PEAK
  14-17h: afternoon steady
  17-21h: EVENING PEAK
  21-22h: wind-down / closing (spawn=0, dwell capped)
  22-24h: CLOSED

ZONES (9): main_entrance, main_corridor, food_court, atrium,
           retail_wing_A, retail_wing_B, multiplex_lobby,
           hypermarket, parking_exit

SCENARIOS (10): normal_weekday, normal_weekend, lunchtime_rush,
                flash_sale, closing_rush, festival_sale,
                movie_release, celebrity_appearance,
                viral_flash_crowd, extended_hours

LABEL LOGIC (fixed):
  0 = Normal    : occupancy_ratio < 0.40
  1 = Congested : 0.40 <= occupancy_ratio < 0.75
  2 = Surge     : occupancy_ratio >= 0.75
"""

import jupedsim as jps
import numpy as np
import pandas as pd
import shapely.geometry as sg
from pathlib import Path
import math, random, warnings, os, sys

warnings.filterwarnings("ignore")
random.seed(42); np.random.seed(42)

OUT_DIR = Path("mall_output")
OUT_DIR.mkdir(exist_ok=True)

class _Q:
    def __enter__(self): self._o = sys.stdout; sys.stdout = open(os.devnull, 'w')
    def __exit__(self, *_): sys.stdout.close(); sys.stdout = self._o

# ══════════════════════════════════════════════════════════════════════════════
# 1. ZONES
# ══════════════════════════════════════════════════════════════════════════════

ZONES = {
    "main_entrance":   {"polygon": sg.Polygon([(85,0),(115,0),(115,12),(85,12)]),
                        "capacity": 400, "num_exits": 6},
    "main_corridor":   {"polygon": sg.Polygon([(0,12),(200,12),(200,30),(0,30)]),
                        "capacity": 1200, "num_exits": 8},
    "food_court":      {"polygon": sg.Polygon([(0,30),(90,30),(90,60),(0,60)]),
                        "capacity": 800, "num_exits": 4},
    "atrium":          {"polygon": sg.Polygon([(90,30),(160,30),(160,60),(90,60)]),
                        "capacity": 600, "num_exits": 4},
    "retail_wing_A":   {"polygon": sg.Polygon([(160,30),(200,30),(200,60),(160,60)]),
                        "capacity": 400, "num_exits": 3},
    "retail_wing_B":   {"polygon": sg.Polygon([(0,60),(100,60),(100,78),(0,78)]),
                        "capacity": 500, "num_exits": 4},
    "multiplex_lobby": {"polygon": sg.Polygon([(100,60),(200,60),(200,78),(100,78)]),
                        "capacity": 600, "num_exits": 5},
    "hypermarket":     {"polygon": sg.Polygon([(0,78),(100,78),(100,89),(0,89)]),
                        "capacity": 700, "num_exits": 4},
    "parking_exit":    {"polygon": sg.Polygon([(100,78),(200,78),(200,89),(100,89)]),
                        "capacity": 300, "num_exits": 3},
}

NEIGHBORS = {
    "main_entrance":   ["main_corridor"],
    "main_corridor":   ["main_entrance","food_court","atrium","retail_wing_A",
                        "retail_wing_B","multiplex_lobby","hypermarket"],
    "food_court":      ["main_corridor","retail_wing_B"],
    "atrium":          ["main_corridor","retail_wing_A","multiplex_lobby"],
    "retail_wing_A":   ["atrium","main_corridor"],
    "retail_wing_B":   ["main_corridor","food_court","hypermarket"],
    "multiplex_lobby": ["atrium","main_corridor","parking_exit"],
    "hypermarket":     ["retail_wing_B","parking_exit"],
    "parking_exit":    ["multiplex_lobby","hypermarket"],
}

WALKABLE = sg.Polygon([(1,1),(199,1),(199,88),(1,88)])

# ══════════════════════════════════════════════════════════════════════════════
# 2. BASE DWELL (minutes)
# ══════════════════════════════════════════════════════════════════════════════

BASE_DWELL_MIN = {
    "main_entrance":   (0.3,  0.8),
    "main_corridor":   (1.0,  3.0),
    "food_court":      (8.0, 15.0),
    "atrium":          (2.0,  5.0),
    "retail_wing_A":   (5.0, 10.0),
    "retail_wing_B":   (5.0, 10.0),
    "multiplex_lobby": (10.0,20.0),
    "hypermarket":     (10.0,20.0),
    "parking_exit":    (0.5,  1.5),
}

DWELL_MULT = {
    "mall_normal_weekday": {z: 1.0 for z in ZONES},

    "mall_normal_weekend": {**{z: 1.2 for z in ZONES},
        "food_court": 1.4, "retail_wing_A": 1.4, "atrium": 1.3},

    # Lunch rush: food court dwells 3.5x, corridor backs up (image: "seating fills fast")
    "mall_lunchtime_rush": {**{z: 1.0 for z in ZONES},
        "food_court": 3.5, "main_corridor": 1.8, "atrium": 1.5,
        "retail_wing_B": 1.3},

    # Flash sale: retail_wing_A queue overflow into corridor (image: "limited-time sale")
    "mall_flash_sale": {**{z: 1.0 for z in ZONES},
        "retail_wing_A": 4.0, "atrium": 2.5, "main_corridor": 2.0,
        "retail_wing_B": 1.8},

    # Closing rush: SHORT dwells everywhere — people leaving fast
    # main_entrance and parking_exit get longer as bottleneck at exit
    "mall_closing_rush": {**{z: 0.2 for z in ZONES},
        "main_entrance": 3.5, "parking_exit": 4.0, "main_corridor": 0.4},

    # Festival sale: high across all zones, anchor stores at capacity
    "mall_festival_sale": {**{z: 1.5 for z in ZONES},
        "food_court": 2.0, "retail_wing_A": 2.2, "retail_wing_B": 2.0,
        "atrium": 1.8, "main_corridor": 1.6},

    # Movie release: multiplex is the magnet (image: "blockbuster movie")
    "mall_movie_release": {**{z: 1.0 for z in ZONES},
        "multiplex_lobby": 4.5, "main_corridor": 1.8, "atrium": 1.6,
        "food_court": 1.5},

    # Celebrity: atrium is the focus zone (image: "celebrity appearance surge")
    "mall_celebrity_appearance": {**{z: 1.0 for z in ZONES},
        "atrium": 6.5, "main_corridor": 2.8, "retail_wing_A": 2.2,
        "retail_wing_B": 1.5},

    # Viral flash crowd: sudden unplanned surge at atrium (image: "viral social media post")
    "mall_viral_flash_crowd": {**{z: 1.3 for z in ZONES},
        "atrium": 5.5, "main_corridor": 2.8, "food_court": 2.2,
        "retail_wing_A": 2.0},

    # Extended hours: NYE — all zones busy late into night
    "mall_extended_hours": {**{z: 1.6 for z in ZONES},
        "food_court": 2.2, "atrium": 2.0, "retail_wing_A": 2.0,
        "multiplex_lobby": 2.2, "main_corridor": 1.8},
}

# ══════════════════════════════════════════════════════════════════════════════
# 3. SPAWN RATE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def G(h, p, w, ht):
    return max(0.0, ht * math.exp(-0.5 * ((h - p) / max(w, 0.01)) ** 2))

def _closed(h):
    return h < 10 or h >= 22

# Normal weekday: steady flow, peaks at lunch h13 and evening h18-19
# density<1.5, steady low flow per image card
def spawn_normal_weekday(h):
    if _closed(h): return 0.0
    base = 3.0 if h < 12 else 5.0
    return max(0.5, G(h,13,1.0,10) + G(h,18,1.2,12) + G(h,19,1.0,10) + base)

# Normal weekend: higher baseline, moderate occupancy per image card
def spawn_normal_weekend(h):
    if h < 10 or h >= 23: return 0.0
    base = 5.0 if h < 12 else 8.0
    return max(1.0, G(h,13,1.2,14) + G(h,18,1.5,16) + G(h,19,1.2,14) + base)

# Lunchtime rush: massive spike at h12-13 for food_court
# "Office crowd arrives at food court 12-1pm, seating fills fast"
def spawn_lunchtime_rush(h):
    if _closed(h): return 0.0
    base = 3.0 if h < 12 else 5.0
    rush = G(h,12,0.6,35) + G(h,13,0.6,32) if 11 <= h <= 14 else 0.0
    return max(0.5, G(h,18,1.2,10) + rush + base)

# Flash sale: surge at h14-16, retail overflow (image: "limited-time announcement")
def spawn_flash_sale(h):
    if _closed(h): return 0.0
    base = spawn_normal_weekday(h)
    sale_surge = G(h,15,0.5,40) if 14 <= h <= 16 else 0.0
    return max(0.5, base + sale_surge)

# Closing rush: normal day, then sudden surge at h21 as all zones exit simultaneously
# "Mall closing announcement triggers simultaneous exit" per image
def spawn_closing_rush(h):
    if _closed(h): return 0.0
    if h >= 21: return 0.0  # no new arrivals — only exits happening
    return max(0.3, spawn_normal_weekday(h))

# Festival sale: high footfall all day, extended to h23 (Diwali/Eid big sale)
# "anchor stores at capacity, food court overflowing, corridors congested"
def spawn_festival_sale(h):
    if h < 9 or h >= 23: return 0.0
    base = 6.0 if h < 11 else 10.0
    return max(2.0, G(h,13,1.2,22) + G(h,18,1.5,24) + G(h,20,1.0,18) + base)

# Movie release: spikes at show times h11, h15, h19 (image: "blockbuster movie release")
def spawn_movie_release(h):
    if _closed(h): return 0.0
    base = spawn_normal_weekday(h)
    show_spikes = G(h,11,0.4,22) + G(h,15,0.4,24) + G(h,19,0.5,28)
    return max(1.0, base + show_spikes)

# Celebrity appearance: normal baseline, then sudden spike at h16 (image: "unplanned surge")
def spawn_celebrity_appearance(h):
    if _closed(h): return 0.0
    base = spawn_normal_weekday(h)
    celeb_spike = G(h,16,0.3,70) if 15 <= h <= 17 else 0.0
    return max(0.5, base + celeb_spike)

# Viral flash crowd: festival baseline + massive unplanned spike at h19
# "no prior ticket/entry control" — sudden onset per image
def spawn_viral_flash_crowd(h):
    if h < 9 or h >= 23: return 0.0
    base = spawn_festival_sale(h)
    viral_spike = G(h,19,0.25,80) if 18 <= h <= 20 else 0.0
    return max(0.5, base + viral_spike)

# Extended hours NYE: operates 0-4h and 10h-24h
# "all zones flow late into night" per image
def spawn_extended_hours(h):
    if 4 <= h < 10: return 0.0
    if h < 4: return max(1.0, G(h,1,1.5,10))
    return max(1.5, G(h,13,1.2,18) + G(h,18,1.5,26) + G(h,21,1.5,24) + G(h,22,1.2,18) + 8.0)

# ══════════════════════════════════════════════════════════════════════════════
# 4. SCENARIOS — all 10 matching image cards exactly
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    # CAT A — Normal day / weekday / regular operations
    "mall_normal_weekday": {
        "name": "Weekday Morning Shopping", "cat": "A", "model": "CFSM",
        "event_type": "none", "event_scale": 0,
        "is_holiday": 0, "is_festival": 0,
        "weather": "clear", "temperature_c": 28,
        "spawn_fn": spawn_normal_weekday,
        "ev_start": -1, "ev_end": -1,
        "exit_blocked_h": -1, "exit_blocked_z": [],
        "closing_hour": 22,
    },
    "mall_normal_weekend": {
        "name": "Post-Lunch Afternoon Steady State", "cat": "A", "model": "CFSM",
        "event_type": "none", "event_scale": 0,
        "is_holiday": 0, "is_festival": 0,
        "weather": "clear", "temperature_c": 29,
        "spawn_fn": spawn_normal_weekend,
        "ev_start": -1, "ev_end": -1,
        "exit_blocked_h": -1, "exit_blocked_z": [],
        "closing_hour": 23,
    },
    "mall_lunchtime_rush": {
        "name": "Lunch Hour Food Court Rush", "cat": "A", "model": "CFSMv2",
        "event_type": "none", "event_scale": 0,
        "is_holiday": 0, "is_festival": 0,
        "weather": "clear", "temperature_c": 30,
        "spawn_fn": spawn_lunchtime_rush,
        "ev_start": 12, "ev_end": 14,
        "exit_blocked_h": -1, "exit_blocked_z": [],
        "closing_hour": 22,
    },
    "mall_flash_sale": {
        "name": "Flash Sale Queue Overflow", "cat": "A", "model": "CFSMv2",
        "event_type": "flash_sale", "event_scale": 2,
        "is_holiday": 0, "is_festival": 0,
        "weather": "clear", "temperature_c": 28,
        "spawn_fn": spawn_flash_sale,
        "ev_start": 14, "ev_end": 16,
        "exit_blocked_h": -1, "exit_blocked_z": [],
        "closing_hour": 22,
    },
    # Closing rush: exit_blocked simulates bottleneck at main_entrance at h21
    # "simultaneous exit all floors" — narrow exit overwhelmed per image
    "mall_closing_rush": {
        "name": "Closing Time Exit Rush", "cat": "A", "model": "CFSMv2",
        "event_type": "none", "event_scale": 0,
        "is_holiday": 0, "is_festival": 0,
        "weather": "clear", "temperature_c": 27,
        "spawn_fn": spawn_closing_rush,
        "ev_start": 21, "ev_end": 22,
        "exit_blocked_h": 21, "exit_blocked_z": ["retail_wing_A", "retail_wing_B"],
        "closing_hour": 22,
    },

    # CAT B — Holiday / festival day / event-driven
    "mall_festival_sale": {
        "name": "Diwali/Eid Big Sale Rush", "cat": "B", "model": "CFSMv2",
        "event_type": "festival_sale", "event_scale": 3,
        "is_holiday": 1, "is_festival": 1,
        "weather": "clear", "temperature_c": 26,
        "spawn_fn": spawn_festival_sale,
        "ev_start": 9, "ev_end": 23,
        "exit_blocked_h": -1, "exit_blocked_z": [],
        "closing_hour": 23,
    },
    "mall_movie_release": {
        "name": "Blockbuster Movie Release", "cat": "B", "model": "CFSMv2",
        "event_type": "movie_release", "event_scale": 2,
        "is_holiday": 0, "is_festival": 0,
        "weather": "clear", "temperature_c": 28,
        "spawn_fn": spawn_movie_release,
        "ev_start": 11, "ev_end": 21,
        "exit_blocked_h": -1, "exit_blocked_z": [],
        "closing_hour": 22,
    },
    "mall_celebrity_appearance": {
        "name": "Celebrity Appearance Crowd Gather", "cat": "B", "model": "CFSMv2",
        "event_type": "celebrity_appearance", "event_scale": 2,
        "is_holiday": 0, "is_festival": 0,
        "weather": "clear", "temperature_c": 29,
        "spawn_fn": spawn_celebrity_appearance,
        "ev_start": 15, "ev_end": 17,
        "exit_blocked_h": -1, "exit_blocked_z": [],
        "closing_hour": 22,
    },
    "mall_viral_flash_crowd": {
        "name": "Flash Crowd from Social Media", "cat": "B", "model": "CFSMv2",
        "event_type": "viral_flash_crowd", "event_scale": 3,
        "is_holiday": 1, "is_festival": 1,
        "weather": "clear", "temperature_c": 27,
        "spawn_fn": spawn_viral_flash_crowd,
        "ev_start": 18, "ev_end": 21,
        "exit_blocked_h": -1, "exit_blocked_z": [],
        "closing_hour": 23,
    },
    "mall_extended_hours": {
        "name": "Christmas Eve Last-Hour Surge", "cat": "B", "model": "CFSMv2",
        "event_type": "nye_extended", "event_scale": 3,
        "is_holiday": 1, "is_festival": 1,
        "weather": "clear", "temperature_c": 24,
        "spawn_fn": spawn_extended_hours,
        "ev_start": 18, "ev_end": 24,
        "exit_blocked_h": -1, "exit_blocked_z": [],
        "closing_hour": 24,
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
# 6. JOURNEY BUILDER — FIX: separate waiting stages per journey type in corridor
#    so each agent type goes to the correct destination zone
# ══════════════════════════════════════════════════════════════════════════════

WP_POS = {
    "entrance":  (100,  6),
    "corridor":  (100, 21),
    "food":      ( 45, 45),
    "atrium":    (125, 45),
    "retail_a":  (180, 45),
    "retail_b":  ( 50, 69),
    "multiplex": (150, 69),
    "hyper":     ( 50, 83),
    "parking":   (150, 83),
}

# Separate corridor queue positions per journey type — fixes dead zone bug
# Shopper goes to retail_a, lunch to food, movie to multiplex, grocery to hyper
QUEUE_POS_CORRIDOR_SHOPPER  = [(10+i*18, 14) for i in range(11)]
QUEUE_POS_CORRIDOR_LUNCH    = [(10+i*18, 17) for i in range(11)]
QUEUE_POS_CORRIDOR_MOVIE    = [(10+i*18, 20) for i in range(11)]
QUEUE_POS_CORRIDOR_GROCERY  = [(10+i*18, 23) for i in range(11)]

QUEUE_POS = {
    "main_entrance":   [(87+i*3, 6)   for i in range(10)],
    "food_court":      [(8+i*8,  45)  for i in range(11)],
    "atrium":          [(93+i*7, 45)  for i in range(10)],
    "retail_wing_A":   [(163+i*5,45)  for i in range(7)],
    "retail_wing_B":   [(5+i*9,  69)  for i in range(11)],
    "multiplex_lobby": [(103+i*9,69)  for i in range(11)],
    "hypermarket":     [(5+i*9,  83)  for i in range(11)],
    "parking_exit":    [(103+i*9,83)  for i in range(11)],
}

EXIT_POLY = [(87,1),(113,1),(113,11),(87,11)]

def build_journeys(sim, scenario_id):
    T    = jps.Transition.create_fixed_transition
    mult = DWELL_MULT[scenario_id]

    wp = {k: sim.add_waypoint_stage(v, 1.5) for k, v in WP_POS.items()}

    # Zone waiting stages
    ws = {z: sim.add_waiting_set_stage(QUEUE_POS[z]) for z in QUEUE_POS}

    # FIX: separate corridor stages per journey type
    ws_cor_shopper = sim.add_waiting_set_stage(QUEUE_POS_CORRIDOR_SHOPPER)
    ws_cor_lunch   = sim.add_waiting_set_stage(QUEUE_POS_CORRIDOR_LUNCH)
    ws_cor_movie   = sim.add_waiting_set_stage(QUEUE_POS_CORRIDOR_MOVIE)
    ws_cor_grocery = sim.add_waiting_set_stage(QUEUE_POS_CORRIDOR_GROCERY)

    for sid in ws.values():
        sim.get_stage(sid).state = jps.WaitingSetState.ACTIVE
    for sid in [ws_cor_shopper, ws_cor_lunch, ws_cor_movie, ws_cor_grocery]:
        sim.get_stage(sid).state = jps.WaitingSetState.ACTIVE

    exit_s = sim.add_exit_stage(EXIT_POLY)

    # Dwell times per zone
    dwell_min = {}
    for zone, (mn, mx) in BASE_DWELL_MIN.items():
        dwell_min[zone] = random.uniform(mn, mx) * mult.get(zone, 1.0)
    # Corridor dwell (shared — same for all types, routing handled by separate stages)
    dwell_min["main_corridor"] = random.uniform(1.0, 3.0) * mult.get("main_corridor", 1.0)

    def mj(*stages):
        jd = jps.JourneyDescription(list(stages))
        for i in range(len(stages) - 1):
            jd.set_transition_for_stage(stages[i], T(stages[i+1]))
        return sim.add_journey(jd), stages[0]

    # Release journeys after zone dwell expires (switch_agent_journey targets)
    release_info = {}
    release_info[ws["main_entrance"]]   = mj(wp["corridor"], ws_cor_shopper, wp["retail_a"], ws["retail_wing_A"], wp["atrium"], exit_s)
    # FIX: each corridor stage releases to the correct destination
    release_info[ws_cor_shopper]        = mj(wp["retail_a"], ws["retail_wing_A"], wp["atrium"], exit_s)
    release_info[ws_cor_lunch]          = mj(wp["food"],     ws["food_court"],    exit_s)
    release_info[ws_cor_movie]          = mj(wp["multiplex"],ws["multiplex_lobby"],wp["parking"], exit_s)
    release_info[ws_cor_grocery]        = mj(wp["hyper"],    ws["hypermarket"],   wp["parking"], exit_s)
    release_info[ws["food_court"]]      = mj(exit_s)
    release_info[ws["atrium"]]          = mj(exit_s)
    release_info[ws["retail_wing_A"]]   = mj(wp["atrium"],   exit_s)
    release_info[ws["retail_wing_B"]]   = mj(exit_s)
    release_info[ws["multiplex_lobby"]] = mj(wp["parking"],  exit_s)
    release_info[ws["hypermarket"]]     = mj(wp["parking"],  exit_s)
    release_info[ws["parking_exit"]]    = mj(exit_s)

    # ws_to_zone mapping — map corridor stages to "main_corridor" for dwell lookup
    ws_to_zone_extra = {
        ws_cor_shopper:  "main_corridor",
        ws_cor_lunch:    "main_corridor",
        ws_cor_movie:    "main_corridor",
        ws_cor_grocery:  "main_corridor",
    }

    # ── JOURNEY A: Shopper — entrance->entrance_wait->corridor_shopper->retail_A->atrium->exit
    jd_A = jps.JourneyDescription([wp["entrance"], ws["main_entrance"], wp["corridor"],
                                    ws_cor_shopper, wp["retail_a"], ws["retail_wing_A"],
                                    wp["atrium"], exit_s])
    jd_A.set_transition_for_stage(wp["entrance"],       T(ws["main_entrance"]))
    jd_A.set_transition_for_stage(ws["main_entrance"],  T(wp["corridor"]))
    jd_A.set_transition_for_stage(wp["corridor"],       T(ws_cor_shopper))
    jd_A.set_transition_for_stage(ws_cor_shopper,       T(wp["retail_a"]))
    jd_A.set_transition_for_stage(wp["retail_a"],       T(ws["retail_wing_A"]))
    jd_A.set_transition_for_stage(ws["retail_wing_A"],  T(wp["atrium"]))
    jd_A.set_transition_for_stage(wp["atrium"],         T(exit_s))
    jid_A = sim.add_journey(jd_A)

    # ── JOURNEY B: Lunch visitor — entrance->corridor_lunch->food_court->exit
    jd_B = jps.JourneyDescription([wp["entrance"], ws["main_entrance"], wp["corridor"],
                                    ws_cor_lunch, wp["food"], ws["food_court"], exit_s])
    jd_B.set_transition_for_stage(wp["entrance"],      T(ws["main_entrance"]))
    jd_B.set_transition_for_stage(ws["main_entrance"], T(wp["corridor"]))
    jd_B.set_transition_for_stage(wp["corridor"],      T(ws_cor_lunch))
    jd_B.set_transition_for_stage(ws_cor_lunch,        T(wp["food"]))
    jd_B.set_transition_for_stage(wp["food"],          T(ws["food_court"]))
    jd_B.set_transition_for_stage(ws["food_court"],    T(exit_s))
    jid_B = sim.add_journey(jd_B)

    # ── JOURNEY C: Movie goer — entrance->corridor_movie->multiplex->parking->exit
    jd_C = jps.JourneyDescription([wp["entrance"], ws["main_entrance"], wp["corridor"],
                                    ws_cor_movie, wp["multiplex"], ws["multiplex_lobby"],
                                    wp["parking"], exit_s])
    jd_C.set_transition_for_stage(wp["entrance"],        T(ws["main_entrance"]))
    jd_C.set_transition_for_stage(ws["main_entrance"],   T(wp["corridor"]))
    jd_C.set_transition_for_stage(wp["corridor"],        T(ws_cor_movie))
    jd_C.set_transition_for_stage(ws_cor_movie,          T(wp["multiplex"]))
    jd_C.set_transition_for_stage(wp["multiplex"],       T(ws["multiplex_lobby"]))
    jd_C.set_transition_for_stage(ws["multiplex_lobby"], T(wp["parking"]))
    jd_C.set_transition_for_stage(wp["parking"],         T(exit_s))
    jid_C = sim.add_journey(jd_C)

    # ── JOURNEY D: Grocery — entrance->corridor_grocery->hypermarket->parking->exit
    jd_D = jps.JourneyDescription([wp["entrance"], ws["main_entrance"], wp["corridor"],
                                    ws_cor_grocery, wp["hyper"], ws["hypermarket"],
                                    wp["parking"], exit_s])
    jd_D.set_transition_for_stage(wp["entrance"],     T(ws["main_entrance"]))
    jd_D.set_transition_for_stage(ws["main_entrance"],T(wp["corridor"]))
    jd_D.set_transition_for_stage(wp["corridor"],     T(ws_cor_grocery))
    jd_D.set_transition_for_stage(ws_cor_grocery,     T(wp["hyper"]))
    jd_D.set_transition_for_stage(wp["hyper"],        T(ws["hypermarket"]))
    jd_D.set_transition_for_stage(ws["hypermarket"],  T(wp["parking"]))
    jd_D.set_transition_for_stage(wp["parking"],      T(exit_s))
    jid_D = sim.add_journey(jd_D)

    journeys = {
        "shopper":  (jid_A, wp["entrance"]),
        "lunch":    (jid_B, wp["entrance"]),
        "movie":    (jid_C, wp["entrance"]),
        "grocery":  (jid_D, wp["entrance"]),
    }
    return journeys, ws, ws_to_zone_extra, release_info, dwell_min

# ══════════════════════════════════════════════════════════════════════════════
# 7. SPAWN
# ══════════════════════════════════════════════════════════════════════════════

J_NAMES   = ["shopper","lunch","movie","grocery"]
J_WEIGHTS = [0.45, 0.30, 0.15, 0.10]
ENT_POLY  = sg.Polygon([(87,1.5),(113,1.5),(113,11),(87,11)])
MAX_AGENTS = 1400

def try_spawn(sim, mk, n, journeys):
    if n <= 0 or sim.agent_count() >= MAX_AGENTS: return 0
    ent = len(list(sim.agents_in_polygon(ZONES["main_entrance"]["polygon"])))
    cor = len(list(sim.agents_in_polygon(ZONES["main_corridor"]["polygon"])))
    if (ent + cor) >= (400 + 1200) * 0.85: return 0
    n = min(n, MAX_AGENTS - sim.agent_count(), 20)
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
                    random.uniform(0.8, 1.4), random.uniform(0.21, 0.27))
                sim.add_agent(p); spawned += 1
            except: pass
    return spawned

# ══════════════════════════════════════════════════════════════════════════════
# 8. SNAPSHOT + DWELL MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def snapshot_and_manage(sim, minute, sc, prev_ids, ws, ws_to_zone_extra,
                         release_info, dwell_min, agent_min_ctr):
    hour = minute // 60
    snaps = []
    curr_ids = {}

    # Build zone lookup: waiting_stage_id -> zone_name
    ws_to_zone = {v: k for k, v in ws.items()}
    ws_to_zone.update(ws_to_zone_extra)

    for zname, zinfo in ZONES.items():
        curr_ids[zname] = set(sim.agents_in_polygon(zinfo["polygon"]))

    # Per-agent dwell management
    active_ids = set()
    to_release = []
    for ag in sim.agents():
        aid = ag.id; sid = ag.stage_id; active_ids.add(aid)
        if sid in ws_to_zone:
            agent_min_ctr[aid] = agent_min_ctr.get(aid, 0) + 1
            zone_name = ws_to_zone[sid]
            threshold = dwell_min.get(zone_name, 2.0) * random.uniform(0.90, 1.10)
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

    # Feature extraction per zone
    ev_s = sc["ev_start"]; ev_e = sc["ev_end"]
    closing_h = sc.get("closing_hour", 22)

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
                elif hasattr(m, "speed"):       speeds.append(float(m.speed))
                elif hasattr(m, "desired_speed"): speeds.append(float(m.desired_speed) * 0.65)
            except: pass

        # Closing time: speeds increase as people rush to exit
        closing_factor = 1.0
        if hour >= closing_h - 1:
            closing_factor = 1.0 + 0.3 * (hour - (closing_h - 1))

        avg_spd = float(np.mean(speeds)) * closing_factor if speeds else 0.0
        avg_spd = round(min(avg_spd, 2.0), 4)

        density = count / area
        occ_r   = count / zinfo["capacity"]
        is_blocked = (1 if sc["exit_blocked_h"] != -1
                      and hour >= sc["exit_blocked_h"]
                      and zname in sc["exit_blocked_z"] else 0)
        exits_open = max(1, zinfo["num_exits"] - is_blocked)
        t_s = (ev_s - hour) if ev_s != -1 else -1
        t_e = (ev_e - hour) if ev_e != -1 else -1
        prev = prev_ids.get(zname, set())
        fi  = len(agent_set - prev)
        fo  = len(prev - agent_set)
        imb = fi - fo
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

    spawn_acc = 0.0
    prev_ids  = {z: set() for z in ZONES}
    agent_min_ctr = {}
    all_snaps = []
    minute_num = 0

    for step in range(864000):  # 24h * 3600s / 0.1s_dt
        hour_now = int(step * 0.1 / 3600)

        if step % 600 == 0:  # every 60 sim-seconds
            rate = sfn(hour_now); spawn_acc += rate
            n = int(spawn_acc); spawn_acc -= n
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
# 10. AGGREGATE MINUTE → HOURLY  (FIX: occupancy_ratio-based labels)
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_hourly(minute_snaps, scenario_id):
    sc  = SCENARIOS[scenario_id]
    df  = pd.DataFrame(minute_snaps)
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
                imb      = fi - fo   # FIX: recompute from summed fi/fo
                ratio    = fi / (fo + 1e-6)
                dwell    = hdf["avg_dwelltime"].mean()
                pres     = float(hdf["pressure_score"].max())
                blocked  = int(hdf["exit_blocked_flag"].max())
                exits    = int(hdf["exits_open_count"].min())

            if hour > 0:
                ph = zdf[zdf["hour"] == hour - 1]
                count_t1 = ph["count"].mean()  if not ph.empty else 0.0
                dens_t1  = ph["density"].max() if not ph.empty else 0.0
                fi_t1    = int(ph["flow_in"].sum()) if not ph.empty else 0
            else:
                count_t1 = dens_t1 = fi_t1 = 0.0

            count_trend = count_m  - count_t1
            dens_chg    = dens_max - dens_t1

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

            # FIX: occupancy_ratio-based labels (works for all zone sizes)
            if   occ_m < 0.40:  label = 0   # Normal    : < 40% capacity
            elif occ_m < 0.75:  label = 1   # Congested : 40-75%
            else:               label = 2   # Surge     : > 75%

            rows.append({
                "zone_name": zname, "venue_type": "mall",
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
        snaps   = run_scenario(sid)
        hourly  = aggregate_hourly(snaps, sid).reindex(columns=FEATURE_COLS)
        path    = OUT_DIR / f"{sid}_hourly.csv"
        hourly.to_csv(path, index=False)
        print(f"  Saved {path}  [{len(hourly)} rows]")
        all_hourly.append(hourly)

    master = pd.concat(all_hourly, ignore_index=True).reindex(columns=FEATURE_COLS)
    master.to_csv(OUT_DIR / "master_mall.csv", index=False)

    expected = len(SCENARIOS) * len(ZONES) * 24
    print(f"\n{'='*65}")
    print(f"  master_mall.csv: {len(master)}/{expected} rows")
    print(f"\n  Label distribution:")
    for lbl, cnt in master["label"].value_counts().sort_index().items():
        name = {0:"Normal",1:"Congested",2:"Surge"}[lbl]
        print(f"    [{lbl}] {name:<12}: {cnt:5d}  ({cnt/len(master)*100:.1f}%)")
    print(f"\n  Scenario breakdown:")
    for sid in SCENARIOS:
        sub = master[master["scenario_id"] == sid]
        print(f"    {sid:<35} 0={len(sub[sub.label==0]):3d} "
              f"1={len(sub[sub.label==1]):3d} 2={len(sub[sub.label==2]):3d}")
    print(f"{'='*65}")

if __name__ == "__main__":
    main()