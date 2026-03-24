"""Data models for the Air Traffic Control environment.

Enums, dataclasses, aircraft performance data, and Pydantic task spec.
All parameters sourced from FAA/ICAO operational data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class WakeCategory(str, Enum):
    """FAA wake turbulence categories.

    Source: FAA Order JO 7110.65 Section 5-5-4.
    SUPER: A380 only (FAA AC 90-23G).
    HEAVY: >= 300,000 lbs MTOW.
    B757: Special category due to unique vortex characteristics.
    LARGE: 41,000 - 300,000 lbs MTOW.
    SMALL: <= 41,000 lbs MTOW.
    """
    SUPER = "SUPER"    # A380
    HEAVY = "HEAVY"    # >= 300,000 lbs MTOW (B747, B777, B787, A330, B767)
    B757 = "B757"      # Special category for B757
    LARGE = "LARGE"    # 41,000 – 300,000 lbs (B737, A320, E175, CRJ)
    SMALL = "SMALL"    # <= 41,000 lbs


class ADG(int, Enum):
    """FAA Airplane Design Group (based on wingspan / tail height).

    Source: FAA AC 150/5300-13B, Table 1-1.
    Group determined by the more restrictive of wingspan or tail height.
    """
    I = 1    # < 49 ft wingspan
    II = 2   # 49 – < 79 ft
    III = 3  # 79 – < 118 ft (B737, A320, E175, CRJ-900)
    IV = 4   # 118 – < 171 ft (B757, B767)
    V = 5    # 171 – < 214 ft (B777, B787, A330)
    VI = 6   # 214 – < 262 ft (A380)


class FlightPhase(str, Enum):
    """Flight state machine phases."""
    # Arrival phases
    APPROACHING = "APPROACHING"  # In airspace, needs runway assignment
    HOLDING = "HOLDING"          # In holding pattern (fuel burning)
    ON_FINAL = "ON_FINAL"        # Cleared for approach, landing imminent
    LANDED = "LANDED"            # Touched down, vacating runway
    TAXIING_IN = "TAXIING_IN"    # Taxiing to gate
    AT_GATE = "AT_GATE"          # At gate (turnaround in progress)
    # Departure phases
    READY = "READY"              # Turnaround complete, awaiting departure clearance
    PUSHBACK = "PUSHBACK"        # Pushing back from gate
    TAXIING_OUT = "TAXIING_OUT"  # Taxiing to runway
    DEPARTED = "DEPARTED"        # Airborne
    # Special
    DIVERTED = "DIVERTED"        # Diverted to alternate airport


class WeatherCondition(str, Enum):
    """Weather conditions affecting airport capacity."""
    CLEAR = "CLEAR"              # VFR, full capacity
    MVFR = "MVFR"                # Marginal VFR, slightly reduced
    IFR = "IFR"                  # Instrument conditions, ~33% reduction
    LOW_IFR = "LOW_IFR"          # Low IFR, ~50% reduction
    THUNDERSTORM = "THUNDERSTORM"  # Severe, ~75% reduction


class Priority(str, Enum):
    """Flight priority levels."""
    NORMAL = "NORMAL"
    MEDICAL = "MEDICAL"       # Priority handling, small bonus
    EMERGENCY = "EMERGENCY"   # Must be handled immediately


# ---------------------------------------------------------------------------
# Aircraft performance data
# ---------------------------------------------------------------------------
# Sources:
#   Fuel burn rates: ICAO Engine Emissions Databank, Boeing/Airbus ops data.
#     B737-800 hold ~2,400 kg/hr confirmed by operational planning figures.
#     A380 hold 100 kg/min is a conservative simplification; real cruise is
#     ~10,000-12,000 kg/hr, but holding at lower altitude/speed is ~6,000 kg/hr.
#   Turnaround times: IATA Ground Handling Manual (narrow-body 35-50 min,
#     wide-body 75-150 min).
#   Passenger counts: Typical 2-class configurations per manufacturer data.
#   Wingspans: Boeing/Airbus type certificate data sheets.
#   Wake categories: FAA Order JO 7110.65 Section 5-5-4.
#   ADG groups: FAA AC 150/5300-13B Table 1-1.
# ---------------------------------------------------------------------------

AIRCRAFT_DATA: dict[str, dict] = {
    "B737-800": {
        "wake": WakeCategory.LARGE,
        "adg": ADG.III,
        "fuel_rate_hold": 40,    # kg/min in holding pattern
        "fuel_rate_taxi": 9,     # kg/min on ground (both engines)
        "turnaround": 45,        # minutes at gate
        "pax": 160,
        "wingspan_ft": 112,      # 112 ft 7 in without winglets (Boeing type certificate)
    },
    "A320": {
        "wake": WakeCategory.LARGE,
        "adg": ADG.III,
        "fuel_rate_hold": 37,
        "fuel_rate_taxi": 10,
        "turnaround": 45,
        "pax": 150,
        "wingspan_ft": 112,
    },
    "B757-200": {
        "wake": WakeCategory.B757,
        "adg": ADG.IV,           # 124 ft wingspan > 118 ft ADG III ceiling (FAA AC 150/5300-13)
        "fuel_rate_hold": 44,
        "fuel_rate_taxi": 11,
        "turnaround": 50,
        "pax": 200,
        "wingspan_ft": 124,
    },
    "B767-300": {
        "wake": WakeCategory.HEAVY,
        "adg": ADG.IV,
        "fuel_rate_hold": 55,
        "fuel_rate_taxi": 14,
        "turnaround": 75,
        "pax": 269,
        "wingspan_ft": 156,
    },
    "B777-300ER": {
        "wake": WakeCategory.HEAVY,
        "adg": ADG.V,
        "fuel_rate_hold": 90,
        "fuel_rate_taxi": 18,
        "turnaround": 120,
        "pax": 396,              # 2-class configuration (777-300ER)
        "wingspan_ft": 213,      # 212 ft 7 in (64.8 m), -300ER extended wing
    },
    "B787-9": {
        "wake": WakeCategory.HEAVY,
        "adg": ADG.V,            # 197 ft wingspan > 171 ft ADG IV ceiling (FAA AC 150/5300-13)
        "fuel_rate_hold": 60,
        "fuel_rate_taxi": 13,
        "turnaround": 90,
        "pax": 296,
        "wingspan_ft": 197,
    },
    "A330-300": {
        "wake": WakeCategory.HEAVY,
        "adg": ADG.V,            # 198 ft wingspan > 171 ft ADG IV ceiling (FAA AC 150/5300-13)
        "fuel_rate_hold": 65,
        "fuel_rate_taxi": 15,
        "turnaround": 90,
        "pax": 300,
        "wingspan_ft": 198,
    },
    "A380": {
        "wake": WakeCategory.SUPER,
        "adg": ADG.VI,
        "fuel_rate_hold": 100,
        "fuel_rate_taxi": 20,
        "turnaround": 150,
        "pax": 525,
        "wingspan_ft": 262,
    },
    "E175": {
        "wake": WakeCategory.LARGE,
        "adg": ADG.III,           # 85 ft wingspan > 79 ft ADG II ceiling (FAA AC 150/5300-13)
        "fuel_rate_hold": 20,
        "fuel_rate_taxi": 5,
        "turnaround": 35,
        "pax": 76,
        "wingspan_ft": 85,
    },
    "CRJ-900": {
        "wake": WakeCategory.LARGE,
        "adg": ADG.III,          # 81 ft wingspan > 79 ft ADG II ceiling (FAA AC 150/5300-13)
        "fuel_rate_hold": 18,
        "fuel_rate_taxi": 5,
        "turnaround": 30,
        "pax": 76,
        "wingspan_ft": 81,       # 81 ft 7 in (24.9 m) per Bombardier specs
    },
}

# Aircraft type weights for schedule generation
AIRCRAFT_WEIGHTS: dict[str, float] = {
    "B737-800": 0.25,
    "A320": 0.25,
    "B757-200": 0.05,
    "B767-300": 0.08,
    "B777-300ER": 0.05,
    "B787-9": 0.07,
    "A330-300": 0.05,
    "A380": 0.02,
    "E175": 0.10,
    "CRJ-900": 0.08,
}

# Airline callsign prefixes
AIRLINE_PREFIXES = [
    "AAL", "UAL", "DAL", "SWA", "JBU", "ASA", "NKS", "FFT",
    "BAW", "DLH", "AFR", "ANA", "JAL", "KAL", "SIA", "QFA",
    "SKW", "RPA", "ENY", "ASH",
]

# Minimum connection time in minutes.
# Source: IATA SSIM Chapter 8 / Resolution 765; JFK domestic MCT is 45-60 min.
MCT_DOMESTIC = 60
MCT_INTERNATIONAL = 120  # JFK international MCT is 120-135 min


# ---------------------------------------------------------------------------
# Flight dataclass
# ---------------------------------------------------------------------------

@dataclass
class Flight:
    """Represents a single flight in the simulation."""
    id: str
    callsign: str
    aircraft_type: str          # key into AIRCRAFT_DATA
    phase: FlightPhase
    is_arrival: bool
    scheduled_time: int         # minutes from sim start
    phase_timer: int            # minutes remaining in current phase
    fuel_remaining: float       # kg
    delay_minutes: int          # accumulated delay
    priority: Priority
    connecting_flights: list[str] = field(default_factory=list)
    assigned_runway: Optional[str] = None
    assigned_gate: Optional[str] = None
    hold_start_time: Optional[int] = None  # when hold began
    activated: bool = False     # whether flight has entered the sim

    @property
    def data(self) -> dict:
        return AIRCRAFT_DATA[self.aircraft_type]

    @property
    def wake(self) -> WakeCategory:
        return self.data["wake"]

    @property
    def adg(self) -> ADG:
        return self.data["adg"]

    @property
    def fuel_rate(self) -> float:
        """Current fuel burn rate in kg/min based on phase."""
        if self.phase == FlightPhase.HOLDING:
            return self.data["fuel_rate_hold"]
        elif self.phase in (FlightPhase.TAXIING_IN, FlightPhase.TAXIING_OUT,
                            FlightPhase.PUSHBACK):
            return self.data["fuel_rate_taxi"]
        elif self.phase == FlightPhase.ON_FINAL:
            return self.data["fuel_rate_hold"] * 0.7  # approach burn
        elif self.phase == FlightPhase.APPROACHING:
            return self.data["fuel_rate_hold"] * 0.8
        return 0.0  # at gate, departed, etc.

    @property
    def fuel_minutes_remaining(self) -> float:
        """Minutes of flight time remaining at current burn rate."""
        rate = self.fuel_rate
        if rate <= 0:
            return float("inf")
        return self.fuel_remaining / rate


# ---------------------------------------------------------------------------
# Metrics tracking
# ---------------------------------------------------------------------------

@dataclass
class StepDelta:
    """Metrics change for a single time step."""
    flights_completed: int = 0
    delay_added: float = 0
    hold_fuel_burned: float = 0
    connections_missed: int = 0
    safety_violations: int = 0
    emergencies_handled: int = 0
    diversions: int = 0
    go_arounds: int = 0


@dataclass
class Metrics:
    """Cumulative simulation metrics for reward computation."""
    flights_completed: int = 0
    total_delay: float = 0
    missed_connections: int = 0
    total_connections: int = 0
    excess_fuel: float = 0          # fuel burned in holds (kg)
    max_possible_excess: float = 1  # set during init
    safety_violations: int = 0
    diversions: int = 0
    go_arounds: int = 0
    emergencies_handled: int = 0
    ground_stop_minutes: int = 0

    # Snapshot for step_delta computation
    _prev: Optional[dict] = field(default=None, repr=False)

    def snapshot(self) -> dict:
        return {
            "flights_completed": self.flights_completed,
            "total_delay": self.total_delay,
            "missed_connections": self.missed_connections,
            "excess_fuel": self.excess_fuel,
            "safety_violations": self.safety_violations,
            "emergencies_handled": self.emergencies_handled,
            "diversions": self.diversions,
            "go_arounds": self.go_arounds,
        }

    def begin_step(self) -> None:
        self._prev = self.snapshot()

    def step_delta(self) -> StepDelta:
        if self._prev is None:
            return StepDelta()
        now = self.snapshot()
        return StepDelta(
            flights_completed=now["flights_completed"] - self._prev["flights_completed"],
            delay_added=now["total_delay"] - self._prev["total_delay"],
            hold_fuel_burned=now["excess_fuel"] - self._prev["excess_fuel"],
            connections_missed=now["missed_connections"] - self._prev["missed_connections"],
            safety_violations=now["safety_violations"] - self._prev["safety_violations"],
            emergencies_handled=now["emergencies_handled"] - self._prev["emergencies_handled"],
            diversions=now["diversions"] - self._prev["diversions"],
            go_arounds=now["go_arounds"] - self._prev["go_arounds"],
        )


# ---------------------------------------------------------------------------
# Task spec (Pydantic)
# ---------------------------------------------------------------------------

class TaskSpec(BaseModel):
    """Task specification for a single ATC scenario."""
    id: str
    seed: int
    scenario_type: str
    num_steps: int = 48           # 48 * 5 min = 4 hours
    step_duration: int = 5        # minutes per step
    weather_timeline: list[str]   # WeatherCondition values per step
    wind_timeline: list[int]      # wind direction (degrees) per step
    flight_count: int = 200       # approximate number of flights
