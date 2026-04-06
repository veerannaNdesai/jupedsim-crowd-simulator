"""
mall_geometry.py
----------------
Defines the complete Mall geometry:
  - Walkable area (union of all zones)
  - Per-zone polygons mapped to the center-circle schema
  - Entry/Exit points
  - Zone metadata (area, capacity, num_exits, neighbors)

All units are in metres. The mall footprint is ~120m x 80m.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union


# ---------------------------------------------------------------------------
# Zone metadata
# ---------------------------------------------------------------------------

@dataclass
class ZoneMeta:
    zone_name: str
    zone_type: str          # maps to center-circle schema
    polygon: Polygon
    capacity: int
    num_exits: int
    neighbors: List[str] = field(default_factory=list)

    @property
    def area_sqm(self) -> float:
        return round(self.polygon.area, 2)

    def contains_point(self, x: float, y: float) -> bool:
        return self.polygon.contains(Point(x, y))


# ---------------------------------------------------------------------------
# Exit point definition
# ---------------------------------------------------------------------------

@dataclass
class ExitPoint:
    name: str
    polygon: Polygon   # JuPedSim exit = polygon region


# ---------------------------------------------------------------------------
# Build mall geometry
# ---------------------------------------------------------------------------

def build_mall_geometry() -> Tuple[Polygon, Dict[str, ZoneMeta], List[ExitPoint]]:
    """
    Returns:
        walkable_area  : full walkable Polygon for JuPedSim
        zones          : dict[zone_name -> ZoneMeta]
        exit_points    : list of ExitPoint
    """

    # ------------------------------------------------------------------
    # Zone polygons  (x_min, y_min, x_max, y_max)
    # Mall layout (top-down view):
    #
    #   [Entry/Exit North]
    #   [         Lobby          ]
    #   [Retail L][Pathway][Retail R]
    #   [Queue ][ServicePt][WaitArea]
    #   [FoodCourt     ][EventArea ]
    #   [Entry/Exit South]
    # ------------------------------------------------------------------

    def rect(x0, y0, x1, y1) -> Polygon:
        return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    # --- Core mall zones ---
    lobby               = rect(10, 60, 110, 80)          # 100×20 = 2000m²
    retail_left         = rect(10, 30, 45, 60)           # 35×30 = 1050m²
    pedestrian_pathway  = rect(45, 30, 75, 60)           # 30×30 = 900m²
    retail_right        = rect(75, 30, 110, 60)          # 35×30 = 1050m²
    queue_area          = rect(10, 10, 40, 30)           # 30×20 = 600m²
    service_point       = rect(40, 10, 75, 30)           # 35×20 = 700m²
    waiting_area        = rect(75, 10, 110, 30)          # 35×20 = 700m²
    food_court          = rect(10, -20, 60, 10)          # 50×30 = 1500m²
    event_area          = rect(60, -20, 110, 10)         # 50×30 = 1500m²

    # --- Entry/Exit corridors (walkable, not zones) ---
    entry_exit_north    = rect(45, 80, 75, 90)           # North entrance
    entry_exit_south    = rect(45, -30, 75, -20)         # South entrance
    entry_exit_east     = rect(110, 20, 120, 50)         # East entrance
    entry_exit_west     = rect(0, 20, 10, 50)            # West entrance

    # --- Walkable union ---
    all_polygons = [
        lobby, retail_left, pedestrian_pathway, retail_right,
        queue_area, service_point, waiting_area,
        food_court, event_area,
        entry_exit_north, entry_exit_south,
        entry_exit_east, entry_exit_west,
    ]
    walkable_area = unary_union(all_polygons)
    # Ensure single polygon (convex hull fallback)
    if isinstance(walkable_area, MultiPolygon):
        walkable_area = walkable_area.convex_hull

    # --- Zone registry ---
    zones: Dict[str, ZoneMeta] = {
        "Lobby": ZoneMeta(
            zone_name="Lobby",
            zone_type="Lobby",
            polygon=lobby,
            capacity=600,
            num_exits=2,
            neighbors=["RetailArea_L", "PedestrianPathway", "RetailArea_R", "EntryExitPoint_N"],
        ),
        "RetailArea_L": ZoneMeta(
            zone_name="RetailArea_L",
            zone_type="RetailArea",
            polygon=retail_left,
            capacity=300,
            num_exits=1,
            neighbors=["Lobby", "PedestrianPathway", "QueueArea"],
        ),
        "PedestrianPathway": ZoneMeta(
            zone_name="PedestrianPathway",
            zone_type="PedestrianPathway",
            polygon=pedestrian_pathway,
            capacity=250,
            num_exits=0,
            neighbors=["Lobby", "RetailArea_L", "RetailArea_R", "ServicePoint", "QueueArea", "WaitingArea"],
        ),
        "RetailArea_R": ZoneMeta(
            zone_name="RetailArea_R",
            zone_type="RetailArea",
            polygon=retail_right,
            capacity=300,
            num_exits=1,
            neighbors=["Lobby", "PedestrianPathway", "WaitingArea"],
        ),
        "QueueArea": ZoneMeta(
            zone_name="QueueArea",
            zone_type="QueueArea",
            polygon=queue_area,
            capacity=150,
            num_exits=1,
            neighbors=["RetailArea_L", "PedestrianPathway", "ServicePoint", "FoodCourt"],
        ),
        "ServicePoint": ZoneMeta(
            zone_name="ServicePoint",
            zone_type="ServicePoint",
            polygon=service_point,
            capacity=180,
            num_exits=1,
            neighbors=["QueueArea", "PedestrianPathway", "WaitingArea", "FoodCourt"],
        ),
        "WaitingArea": ZoneMeta(
            zone_name="WaitingArea",
            zone_type="WaitingArea",
            polygon=waiting_area,
            capacity=200,
            num_exits=1,
            neighbors=["RetailArea_R", "PedestrianPathway", "ServicePoint", "EventArea"],
        ),
        "FoodCourt": ZoneMeta(
            zone_name="FoodCourt",
            zone_type="RetailArea",
            polygon=food_court,
            capacity=400,
            num_exits=1,
            neighbors=["QueueArea", "ServicePoint", "EventArea", "EntryExitPoint_S"],
        ),
        "EventArea": ZoneMeta(
            zone_name="EventArea",
            zone_type="EventArea",
            polygon=event_area,
            capacity=500,
            num_exits=1,
            neighbors=["WaitingArea", "ServicePoint", "FoodCourt", "EntryExitPoint_S"],
        ),
        "EntryExitPoint_N": ZoneMeta(
            zone_name="EntryExitPoint_N",
            zone_type="EntryExitPoint",
            polygon=entry_exit_north,
            capacity=100,
            num_exits=1,
            neighbors=["Lobby"],
        ),
        "EntryExitPoint_S": ZoneMeta(
            zone_name="EntryExitPoint_S",
            zone_type="EntryExitPoint",
            polygon=entry_exit_south,
            capacity=100,
            num_exits=1,
            neighbors=["FoodCourt", "EventArea"],
        ),
        "EntryExitPoint_E": ZoneMeta(
            zone_name="EntryExitPoint_E",
            zone_type="EntryExitPoint",
            polygon=entry_exit_east,
            capacity=80,
            num_exits=1,
            neighbors=["WaitingArea", "RetailArea_R"],
        ),
        "EntryExitPoint_W": ZoneMeta(
            zone_name="EntryExitPoint_W",
            zone_type="EntryExitPoint",
            polygon=entry_exit_west,
            capacity=80,
            num_exits=1,
            neighbors=["RetailArea_L", "QueueArea"],
        ),
    }

    # --- Exit points (JuPedSim exit polygons) ---
    exit_points: List[ExitPoint] = [
        ExitPoint(name="Exit_North", polygon=rect(48, 88, 72, 92)),
        ExitPoint(name="Exit_South", polygon=rect(48, -32, 72, -28)),
        ExitPoint(name="Exit_East",  polygon=rect(118, 28, 122, 42)),
        ExitPoint(name="Exit_West",  polygon=rect(-2, 28, 2, 42)),
    ]

    return walkable_area, zones, exit_points


def get_zone_for_position(
    x: float, y: float, zones: Dict[str, ZoneMeta]
) -> Optional[str]:
    """Return zone_name for coordinates (x,y), or None if outside all zones."""
    pt = Point(x, y)
    for name, zmeta in zones.items():
        if zmeta.polygon.contains(pt):
            return name
    return None
