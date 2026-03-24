"""Static airport configuration for Metro Hub International (MHI).

Inspired by JFK: 4 runways in 2 pairs of parallels, 60 gates across 4 terminals.
All parameters sourced from FAA Airport Capacity Profiles and Order 7110.65.
"""

from __future__ import annotations

from dataclasses import dataclass
from models import ADG, WakeCategory, WeatherCondition


# ---------------------------------------------------------------------------
# Runway definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Runway:
    id: str
    heading: int        # magnetic heading (degrees)
    length_ft: int
    ils_equipped: bool
    parallel_group: str  # e.g. "13" groups 13L and 13R


# Physical runways — inspired by JFK International Airport.
# Source: FAA Airport/Facility Directory, AirNav KJFK.
# JFK actual: 13L/31R (10,000 ft), 13R/31L (14,511 ft),
#             04L/22R (12,079 ft physical pavement; 04L TORA is 11,351 ft
#                      due to displaced threshold — FAA AIP, AOPA chart supplement),
#             04R/22L (8,400 ft).
RUNWAYS: dict[str, Runway] = {
    "13L": Runway("13L", 130, 10000, True, "13"),
    "13R": Runway("13R", 130, 14511, True, "13"),
    "04L": Runway("04L", 40,  12079, True, "04"),
    "04R": Runway("04R", 40,  8400,  True, "04"),
    # Reciprocals share physical pavement
    "31R": Runway("31R", 310, 10000, True, "31"),
    "31L": Runway("31L", 310, 14511, True, "31"),
    "22R": Runway("22R", 220, 12079, True, "22"),
    "22L": Runway("22L", 220, 8400,  True, "22"),
}


# ---------------------------------------------------------------------------
# Runway configurations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunwayConfig:
    """An operational runway configuration."""
    name: str
    arrival_runways: tuple[str, ...]
    departure_runways: tuple[str, ...]
    wind_range: tuple[int, int]  # (min_heading, max_heading) for suitability
    vfr_arr_per_hour: int
    vfr_dep_per_hour: int
    description: str


# Runway configurations and VFR capacity estimates.
# Source: FAA JFK Airport Capacity Profile (2014).
# Note: FAA slot control limits JFK to 81 scheduled ops/hr (14 CFR 93 Subpart K,
# Federal Register 89 FR 41486, May 13, 2024). This simulation uses physical capacity.
RUNWAY_CONFIGS: dict[str, RunwayConfig] = {
    "south_flow": RunwayConfig(
        name="south_flow",
        arrival_runways=("13L", "13R"),
        departure_runways=("31L",),
        wind_range=(90, 200),
        vfr_arr_per_hour=60,
        vfr_dep_per_hour=36,
        description="Primary south flow: arrivals 13L/13R, departures 31L",
    ),
    "north_flow": RunwayConfig(
        name="north_flow",
        arrival_runways=("31R", "31L"),
        departure_runways=("22R",),
        wind_range=(270, 360),
        vfr_arr_per_hour=60,
        vfr_dep_per_hour=36,
        description="North flow: arrivals 31R/31L, departures 22R",
    ),
    "west_flow": RunwayConfig(
        name="west_flow",
        arrival_runways=("22L", "31L"),
        departure_runways=("22R", "13R"),
        wind_range=(200, 270),
        vfr_arr_per_hour=50,
        vfr_dep_per_hour=30,
        description="Crosswind west flow: split operations",
    ),
    "ifr_south": RunwayConfig(
        name="ifr_south",
        arrival_runways=("13R",),
        departure_runways=("31L",),
        wind_range=(90, 200),
        vfr_arr_per_hour=36,
        vfr_dep_per_hour=24,
        description="IFR reduced: single arrival runway 13R, departures 31L",
    ),
    "emergency": RunwayConfig(
        name="emergency",
        arrival_runways=("13R",),
        departure_runways=("13R",),
        wind_range=(0, 360),
        vfr_arr_per_hour=20,
        vfr_dep_per_hour=12,
        description="Emergency single-runway ops: 13R only",
    ),
}

DEFAULT_CONFIG = "south_flow"


# ---------------------------------------------------------------------------
# Gate definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Gate:
    id: str
    terminal: str
    max_adg: ADG
    position: int  # 0-based index within terminal (for taxi time calc)


def _make_gates() -> dict[str, Gate]:
    gates: dict[str, Gate] = {}
    # Terminal A: 15 gates, regional jets (ADG III — serves CRJ-900 and E175)
    for i in range(1, 16):
        gid = f"A{i}"
        gates[gid] = Gate(gid, "A", ADG.III, i - 1)
    # Terminal B: 20 gates, narrow-body (ADG III)
    for i in range(1, 21):
        gid = f"B{i}"
        gates[gid] = Gate(gid, "B", ADG.III, i - 1)
    # Terminal C: 15 gates, wide-body (ADG V)
    for i in range(1, 16):
        gid = f"C{i}"
        gates[gid] = Gate(gid, "C", ADG.V, i - 1)
    # Terminal D: 10 gates, all aircraft (ADG VI)
    for i in range(1, 11):
        gid = f"D{i}"
        gates[gid] = Gate(gid, "D", ADG.VI, i - 1)
    return gates


GATES: dict[str, Gate] = _make_gates()

TERMINALS = {
    "A": {"count": 15, "max_adg": ADG.III, "description": "Regional (E175, CRJ-900)"},
    "B": {"count": 20, "max_adg": ADG.III, "description": "Narrow-body & regional (B737, A320, E175)"},
    "C": {"count": 15, "max_adg": ADG.V, "description": "Wide-body (B757, B767, B777, B787, A330)"},
    "D": {"count": 10, "max_adg": ADG.VI, "description": "All aircraft incl. A380"},
}


# ---------------------------------------------------------------------------
# Wake turbulence separation (FAA Order 7110.65)
# ---------------------------------------------------------------------------

# Minimum separation in minutes for same-runway successive operations.
# Source: FAA Order JO 7110.65 Section 5-5-4 (distance-based NM values).
# Converted to time-based using ~140 kt approach speed:
#   SUPER->HEAVY: 6 NM ~ 2.6 min (rounded to 3 min)
#   SUPER->LARGE: 7 NM ~ 3.0 min (rounded to 4 min)
#   HEAVY->HEAVY: 4 NM ~ 1.7 min (rounded to 2 min)
#   HEAVY->LARGE/SMALL: 5 NM ~ 2.1 min (rounded to 3 min)
#   B757->SMALL: 4 NM ~ 1.7 min (rounded to 3 min, conservative)
# NOTE: Real ATC uses distance-based radar separation, not time-based.
# Time conversion is a simplification for this discrete-time simulation.
WAKE_SEPARATION_MINUTES: dict[tuple[WakeCategory, WakeCategory], int] = {
    # Leader SUPER
    (WakeCategory.SUPER, WakeCategory.SUPER): 2,
    (WakeCategory.SUPER, WakeCategory.HEAVY): 3,
    (WakeCategory.SUPER, WakeCategory.B757):  3,
    (WakeCategory.SUPER, WakeCategory.LARGE): 4,
    (WakeCategory.SUPER, WakeCategory.SMALL): 4,
    # Leader HEAVY
    (WakeCategory.HEAVY, WakeCategory.SUPER): 2,
    (WakeCategory.HEAVY, WakeCategory.HEAVY): 2,
    (WakeCategory.HEAVY, WakeCategory.B757):  2,
    (WakeCategory.HEAVY, WakeCategory.LARGE): 3,
    (WakeCategory.HEAVY, WakeCategory.SMALL): 3,
    # Leader B757
    (WakeCategory.B757, WakeCategory.SUPER): 2,
    (WakeCategory.B757, WakeCategory.HEAVY): 2,
    (WakeCategory.B757, WakeCategory.B757):  2,
    (WakeCategory.B757, WakeCategory.LARGE): 2,
    (WakeCategory.B757, WakeCategory.SMALL): 3,
    # Leader LARGE
    (WakeCategory.LARGE, WakeCategory.SUPER): 2,
    (WakeCategory.LARGE, WakeCategory.HEAVY): 2,
    (WakeCategory.LARGE, WakeCategory.B757):  2,
    (WakeCategory.LARGE, WakeCategory.LARGE): 2,
    (WakeCategory.LARGE, WakeCategory.SMALL): 2,
    # Leader SMALL
    (WakeCategory.SMALL, WakeCategory.SUPER): 2,
    (WakeCategory.SMALL, WakeCategory.HEAVY): 2,
    (WakeCategory.SMALL, WakeCategory.B757):  2,
    (WakeCategory.SMALL, WakeCategory.LARGE): 2,
    (WakeCategory.SMALL, WakeCategory.SMALL): 2,
}


def get_wake_separation(leader: WakeCategory, follower: WakeCategory) -> int:
    """Return minimum separation in minutes between leader and follower."""
    return WAKE_SEPARATION_MINUTES.get((leader, follower), 2)


# ---------------------------------------------------------------------------
# Weather capacity multipliers
# ---------------------------------------------------------------------------

# Weather capacity multipliers.
# Source: FAA Airport Capacity Profiles methodology (AC 150/5060-5).
# CLEAR=100%, MVFR=85%, IFR=67%, LOW_IFR=50%, THUNDERSTORM=25%.
WEATHER_CAPACITY_MULTIPLIER: dict[WeatherCondition, float] = {
    WeatherCondition.CLEAR: 1.0,
    WeatherCondition.MVFR: 0.85,
    WeatherCondition.IFR: 0.67,
    WeatherCondition.LOW_IFR: 0.50,
    WeatherCondition.THUNDERSTORM: 0.25,
}


def capacity_for_weather(
    config: RunwayConfig, weather: WeatherCondition
) -> tuple[int, int]:
    """Return (arrivals_per_hour, departures_per_hour) adjusted for weather."""
    mult = WEATHER_CAPACITY_MULTIPLIER[weather]
    arr = max(1, int(config.vfr_arr_per_hour * mult))
    dep = max(1, int(config.vfr_dep_per_hour * mult))
    return arr, dep


def capacity_per_step(
    config: RunwayConfig, weather: WeatherCondition, step_duration: int
) -> tuple[int, int]:
    """Return (max_arrivals, max_departures) allowed per time step."""
    arr_hr, dep_hr = capacity_for_weather(config, weather)
    # Convert from per-hour to per-step
    arr = max(1, int(arr_hr * step_duration / 60))
    dep = max(1, int(dep_hr * step_duration / 60))
    return arr, dep


# ---------------------------------------------------------------------------
# Taxi time estimation
# ---------------------------------------------------------------------------

# Base taxi times in minutes by terminal (approximate, based on JFK data)
_TAXI_TIME_BASE: dict[str, dict[str, int]] = {
    # Terminal -> {runway_group -> base_minutes}
    "A": {"13": 12, "31": 18, "04": 15, "22": 20},
    "B": {"13": 10, "31": 15, "04": 12, "22": 17},
    "C": {"13": 15, "31": 12, "04": 18, "22": 14},
    "D": {"13": 18, "31": 10, "04": 20, "22": 12},
}


def get_taxi_time(runway_id: str, gate_id: str) -> int:
    """Return taxi time in minutes between a runway and a gate.

    Uses terminal-to-runway-group base times with a small gate-position offset.
    """
    terminal = gate_id[0]  # "A", "B", "C", or "D"
    # Runway group: strip L/R suffix to get group (e.g. "13L" -> "13")
    rwy_group = runway_id.rstrip("LR")

    base = _TAXI_TIME_BASE.get(terminal, {}).get(rwy_group, 15)

    # Small offset based on gate position within terminal (0-2 min)
    gate_num = int(gate_id[1:]) if gate_id[1:].isdigit() else 0
    offset = gate_num % 3  # 0, 1, or 2 extra minutes

    return base + offset


# ---------------------------------------------------------------------------
# Go-around probability
# ---------------------------------------------------------------------------

# Go-around probability.
# Source: NASA TM-20240008006 "Statistical Analysis of Recent Go Around Flight Data".
# U.S. FY2023 rate: ~3.9 per 1,000 arrivals (0.39%). Code uses 0.5% — slightly
# conservative but within realistic range for a busy hub.
# Thunderstorm total (3.5%) is elevated for simulation impact; empirical data
# shows thunderstorms cause ~3-5x increase in go-around rate.
GO_AROUND_BASE_PROB = 0.005  # 0.5% base rate

GO_AROUND_WEATHER_BONUS: dict[WeatherCondition, float] = {
    WeatherCondition.CLEAR: 0.0,
    WeatherCondition.MVFR: 0.005,
    WeatherCondition.IFR: 0.01,
    WeatherCondition.LOW_IFR: 0.02,
    WeatherCondition.THUNDERSTORM: 0.03,
}


def go_around_probability(weather: WeatherCondition) -> float:
    """Return probability of a go-around given current weather."""
    return GO_AROUND_BASE_PROB + GO_AROUND_WEATHER_BONUS.get(weather, 0.0)
