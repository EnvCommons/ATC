"""Core Air Traffic Control simulation engine.

Pure Python, zero OpenReward dependencies. Fully testable independently.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from models import (
    AIRCRAFT_DATA,
    AIRCRAFT_WEIGHTS,
    AIRLINE_PREFIXES,
    MCT_DOMESTIC,
    ADG,
    Flight,
    FlightPhase,
    Metrics,
    Priority,
    TaskSpec,
    WakeCategory,
    WeatherCondition,
)
from airport import (
    DEFAULT_CONFIG,
    GATES,
    RUNWAY_CONFIGS,
    RUNWAYS,
    capacity_per_step,
    get_taxi_time,
    get_wake_separation,
    go_around_probability,
)


# Critical fuel threshold: auto-divert if fuel < this many minutes.
# Note: FAA 14 CFR 91.167 requires 45 min reserve for IFR flight.
# 15 min is a simulation mechanic representing emergency diversion trigger.
FUEL_CRITICAL_MINUTES = 15.0

# Phase durations in minutes
APPROACH_TO_FINAL_DURATION = 5   # time on final approach
LANDING_ROLLOUT_DURATION = 2     # runway occupancy after touchdown
PUSHBACK_DURATION = 5            # pushback from gate


class ATCSimulation:
    """Discrete-event ATC simulation engine.

    Manages flight schedules, runway operations, gate assignments,
    weather effects, and safety constraint enforcement.
    """

    def __init__(self, task_spec: TaskSpec) -> None:
        self.seed = task_spec.seed
        self.rng = np.random.RandomState(task_spec.seed)
        self.step_count = 0
        self.max_steps = task_spec.num_steps
        self.step_duration = task_spec.step_duration
        self.clock = 0  # minutes from simulation start
        self.scenario_type = task_spec.scenario_type

        # Pre-generated timelines
        self.weather_timeline = [
            WeatherCondition(w) for w in task_spec.weather_timeline
        ]
        self.wind_timeline = task_spec.wind_timeline

        # Current state
        self.current_weather = self.weather_timeline[0]
        self.current_wind = self.wind_timeline[0]

        # Flight management
        self.flights: dict[str, Flight] = {}
        self.all_flight_ids: list[str] = []  # ordered by scheduled_time

        # Runway state
        self.active_config_name = DEFAULT_CONFIG
        self.active_config = RUNWAY_CONFIGS[DEFAULT_CONFIG]
        # Track last operation on each runway: (time, wake_category)
        self.runway_last_op: dict[str, tuple[int, WakeCategory]] = {}
        # Track operations this step per runway
        self.step_runway_ops: dict[str, int] = {}

        # Gate occupancy: gate_id -> (flight_id, available_at_time)
        self.gate_occupancy: dict[str, Optional[tuple[str, int]]] = {
            gid: None for gid in GATES
        }

        # Ground stop
        self.ground_stop_until = 0

        # Metrics
        self.metrics = Metrics()

        # Connection tracking: set of (arrival_id, departure_id) pairs
        self.connections: set[tuple[str, str]] = set()
        self.missed_connection_pairs: set[tuple[str, str]] = set()

        # Events log for observation
        self.step_events: list[str] = []

        # Generate flights
        self._generate_flight_schedule(task_spec.flight_count)

        # Set initial config based on wind
        self._auto_select_config()

    # ------------------------------------------------------------------
    # Flight schedule generation
    # ------------------------------------------------------------------

    def _generate_flight_schedule(self, target_count: int) -> None:
        """Generate a realistic flight schedule from the seed."""
        aircraft_types = list(AIRCRAFT_WEIGHTS.keys())
        weights = np.array([AIRCRAFT_WEIGHTS[t] for t in aircraft_types])
        weights = weights / weights.sum()

        total_time = self.max_steps * self.step_duration  # total sim minutes
        num_arrivals = target_count // 2
        num_departures = target_count - num_arrivals

        flight_counter = 0

        # Generate arrivals
        for _ in range(num_arrivals):
            flight_counter += 1
            fid = f"F{flight_counter:04d}"
            ac_type = self.rng.choice(aircraft_types, p=weights)
            ac_data = AIRCRAFT_DATA[ac_type]

            # Scheduled time: spread across simulation window
            sched = int(self.rng.uniform(0, total_time))

            # Fuel: 45-120 minutes of holding fuel
            fuel_minutes = self.rng.uniform(45, 120)
            fuel_kg = fuel_minutes * ac_data["fuel_rate_hold"]

            # Priority
            priority = Priority.NORMAL
            if self.rng.random() < 0.005:  # 0.5% emergency
                priority = Priority.EMERGENCY
            elif self.rng.random() < 0.02:  # 2% medical
                priority = Priority.MEDICAL

            # Callsign
            prefix = self.rng.choice(AIRLINE_PREFIXES)
            num = self.rng.randint(100, 9999)
            callsign = f"{prefix}{num}"

            flight = Flight(
                id=fid,
                callsign=callsign,
                aircraft_type=ac_type,
                phase=FlightPhase.APPROACHING,
                is_arrival=True,
                scheduled_time=sched,
                phase_timer=10,  # 10 min before auto-hold
                fuel_remaining=fuel_kg,
                delay_minutes=0,
                priority=priority,
            )
            self.flights[fid] = flight

        # Generate departures: these start at gates, ready after turnaround
        for _ in range(num_departures):
            flight_counter += 1
            fid = f"F{flight_counter:04d}"
            ac_type = self.rng.choice(aircraft_types, p=weights)
            ac_data = AIRCRAFT_DATA[ac_type]

            # Scheduled departure time
            sched = int(self.rng.uniform(0, total_time))

            # Departures don't burn holding fuel, set generous amount
            fuel_kg = 5000.0

            priority = Priority.NORMAL
            if self.rng.random() < 0.005:
                priority = Priority.EMERGENCY
            elif self.rng.random() < 0.02:
                priority = Priority.MEDICAL

            prefix = self.rng.choice(AIRLINE_PREFIXES)
            num = self.rng.randint(100, 9999)
            callsign = f"{prefix}{num}"

            flight = Flight(
                id=fid,
                callsign=callsign,
                aircraft_type=ac_type,
                phase=FlightPhase.READY,
                is_arrival=False,
                scheduled_time=sched,
                phase_timer=0,
                fuel_remaining=fuel_kg,
                delay_minutes=0,
                priority=priority,
            )
            self.flights[fid] = flight

        # Sort flight IDs by scheduled time
        self.all_flight_ids = sorted(
            self.flights.keys(),
            key=lambda fid: self.flights[fid].scheduled_time,
        )

        # Generate connections (~20% of arrivals connect to a departure)
        arrival_ids = [
            fid for fid in self.all_flight_ids if self.flights[fid].is_arrival
        ]
        departure_ids = [
            fid for fid in self.all_flight_ids if not self.flights[fid].is_arrival
        ]

        num_connections = int(len(arrival_ids) * 0.2)
        if departure_ids and num_connections > 0:
            conn_arrivals = self.rng.choice(
                arrival_ids,
                size=min(num_connections, len(arrival_ids)),
                replace=False,
            )
            conn_departures = self.rng.choice(
                departure_ids,
                size=min(num_connections, len(departure_ids)),
                replace=False,
            )
            for arr_id, dep_id in zip(conn_arrivals, conn_departures):
                self.flights[arr_id].connecting_flights.append(dep_id)
                self.flights[dep_id].connecting_flights.append(arr_id)
                self.connections.add((arr_id, dep_id))

        self.metrics.total_connections = len(self.connections)

        # Estimate max possible excess fuel (for normalization)
        total_hold_fuel = sum(
            f.data["fuel_rate_hold"] * 60  # 60 min max hold each
            for f in self.flights.values()
            if f.is_arrival
        )
        self.metrics.max_possible_excess = max(total_hold_fuel, 1.0)

    # ------------------------------------------------------------------
    # Config selection
    # ------------------------------------------------------------------

    def _auto_select_config(self) -> None:
        """Select best runway config based on current wind direction."""
        wind = self.current_wind
        best_config = DEFAULT_CONFIG

        for name, cfg in RUNWAY_CONFIGS.items():
            if name == "emergency":
                continue
            lo, hi = cfg.wind_range
            if lo <= hi:
                if lo <= wind <= hi:
                    best_config = name
                    break
            else:  # wraps around 360
                if wind >= lo or wind <= hi:
                    best_config = name
                    break

        self.active_config_name = best_config
        self.active_config = RUNWAY_CONFIGS[best_config]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Reset simulation and return initial observation."""
        self.step_count = 0
        self.clock = 0
        self.current_weather = self.weather_timeline[0]
        self.current_wind = self.wind_timeline[0]
        self._auto_select_config()
        self.step_events = []
        return self.get_observation()

    # ------------------------------------------------------------------
    # Agent actions
    # ------------------------------------------------------------------

    def sequence_flight(self, flight_id: str, runway_id: str) -> dict:
        """Assign a flight to a runway for arrival or departure."""
        if flight_id not in self.flights:
            return {"success": False, "error": f"Unknown flight: {flight_id}"}

        flight = self.flights[flight_id]

        # Validate runway exists
        if runway_id not in RUNWAYS:
            return {"success": False, "error": f"Unknown runway: {runway_id}"}

        # Check flight is in correct phase
        if flight.is_arrival:
            if flight.phase not in (FlightPhase.APPROACHING, FlightPhase.HOLDING):
                return {
                    "success": False,
                    "error": f"Flight {flight_id} is in phase {flight.phase.value}, "
                             f"cannot sequence for arrival",
                }
            # Check runway is in active arrival runways
            if runway_id not in self.active_config.arrival_runways:
                return {
                    "success": False,
                    "error": f"Runway {runway_id} is not an active arrival runway. "
                             f"Active: {', '.join(self.active_config.arrival_runways)}",
                }
        else:
            if flight.phase != FlightPhase.READY:
                return {
                    "success": False,
                    "error": f"Flight {flight_id} is in phase {flight.phase.value}, "
                             f"cannot sequence for departure",
                }
            # Check ground stop
            if self.ground_stop_until > self.clock:
                return {
                    "success": False,
                    "error": f"Ground stop in effect until T+{self.ground_stop_until}. "
                             f"Cannot clear departures.",
                }
            if runway_id not in self.active_config.departure_runways:
                return {
                    "success": False,
                    "error": f"Runway {runway_id} is not an active departure runway. "
                             f"Active: {', '.join(self.active_config.departure_runways)}",
                }

        # Check wake separation
        sep_result = self._check_wake_separation(flight, runway_id)
        if sep_result is not None:
            # Record violation but still allow (with penalty)
            self.metrics.safety_violations += 1
            self.step_events.append(
                f"SAFETY VIOLATION: Wake separation breach on {runway_id} "
                f"({flight.callsign} {flight.wake.value} too close to previous)"
            )

        # Assign runway
        flight.assigned_runway = runway_id

        if flight.is_arrival:
            flight.phase = FlightPhase.ON_FINAL
            flight.phase_timer = APPROACH_TO_FINAL_DURATION
            if flight.hold_start_time is not None:
                flight.hold_start_time = None
        else:
            flight.phase = FlightPhase.PUSHBACK
            flight.phase_timer = PUSHBACK_DURATION

        # Record operation on runway
        self.runway_last_op[runway_id] = (self.clock, flight.wake)
        self.step_runway_ops[runway_id] = (
            self.step_runway_ops.get(runway_id, 0) + 1
        )

        action = "arrival" if flight.is_arrival else "departure"
        return {
            "success": True,
            "message": f"Flight {flight.callsign} ({flight.aircraft_type}) "
                       f"sequenced for {action} on runway {runway_id}",
            "flight_id": flight_id,
            "runway": runway_id,
        }

    def assign_gate(self, flight_id: str, gate_id: str) -> dict:
        """Assign a gate to a flight."""
        if flight_id not in self.flights:
            return {"success": False, "error": f"Unknown flight: {flight_id}"}

        flight = self.flights[flight_id]

        if gate_id not in GATES:
            return {"success": False, "error": f"Unknown gate: {gate_id}"}

        gate = GATES[gate_id]

        # Check flight phase
        if flight.phase not in (
            FlightPhase.APPROACHING, FlightPhase.HOLDING,
            FlightPhase.ON_FINAL, FlightPhase.LANDED, FlightPhase.TAXIING_IN,
        ):
            return {
                "success": False,
                "error": f"Flight {flight_id} in phase {flight.phase.value}, "
                         f"cannot assign gate",
            }

        # Check ADG compatibility
        if flight.adg.value > gate.max_adg.value:
            return {
                "success": False,
                "error": f"Aircraft {flight.aircraft_type} (ADG {flight.adg.name}) "
                         f"too large for gate {gate_id} (max ADG {gate.max_adg.name})",
            }

        # Check gate availability
        occ = self.gate_occupancy[gate_id]
        if occ is not None:
            occ_fid, avail_at = occ
            if avail_at > self.clock:
                return {
                    "success": False,
                    "error": f"Gate {gate_id} occupied by {occ_fid} until T+{avail_at}",
                }

        flight.assigned_gate = gate_id
        return {
            "success": True,
            "message": f"Gate {gate_id} assigned to {flight.callsign} "
                       f"({flight.aircraft_type})",
            "flight_id": flight_id,
            "gate": gate_id,
        }

    def hold_flight(self, flight_id: str) -> dict:
        """Put a flight in holding pattern."""
        if flight_id not in self.flights:
            return {"success": False, "error": f"Unknown flight: {flight_id}"}

        flight = self.flights[flight_id]

        if flight.phase == FlightPhase.HOLDING:
            return {
                "success": True,
                "message": f"{flight.callsign} already in holding pattern",
            }

        if flight.phase != FlightPhase.APPROACHING:
            return {
                "success": False,
                "error": f"Flight {flight_id} in phase {flight.phase.value}, "
                         f"can only hold APPROACHING flights",
            }

        flight.phase = FlightPhase.HOLDING
        flight.hold_start_time = self.clock
        flight.assigned_runway = None
        return {
            "success": True,
            "message": f"{flight.callsign} entering holding pattern. "
                       f"Fuel remaining: {flight.fuel_minutes_remaining:.0f} min",
        }

    def divert_flight(self, flight_id: str) -> dict:
        """Divert a flight to alternate airport."""
        if flight_id not in self.flights:
            return {"success": False, "error": f"Unknown flight: {flight_id}"}

        flight = self.flights[flight_id]

        if flight.phase not in (FlightPhase.APPROACHING, FlightPhase.HOLDING):
            return {
                "success": False,
                "error": f"Flight {flight_id} in phase {flight.phase.value}, "
                         f"can only divert APPROACHING or HOLDING flights",
            }

        flight.phase = FlightPhase.DIVERTED
        flight.assigned_runway = None
        self.metrics.diversions += 1

        # Check if this breaks any connections
        for arr_id, dep_id in list(self.connections):
            if arr_id == flight_id and (arr_id, dep_id) not in self.missed_connection_pairs:
                self.missed_connection_pairs.add((arr_id, dep_id))
                self.metrics.missed_connections += 1
                self.step_events.append(
                    f"MISSED CONNECTION: Diversion of {flight.callsign} "
                    f"causes missed connection to {self.flights[dep_id].callsign}"
                )

        return {
            "success": True,
            "message": f"{flight.callsign} diverted to alternate airport",
        }

    def set_runway_config(self, config_name: str) -> dict:
        """Change active runway configuration."""
        if config_name not in RUNWAY_CONFIGS:
            available = ", ".join(RUNWAY_CONFIGS.keys())
            return {
                "success": False,
                "error": f"Unknown config: {config_name}. Available: {available}",
            }

        old_config = self.active_config_name
        self.active_config_name = config_name
        self.active_config = RUNWAY_CONFIGS[config_name]

        # Clear runway last-op tracking (new config, new runways)
        self.runway_last_op.clear()

        self.step_events.append(
            f"Config changed: {old_config} -> {config_name}"
        )

        return {
            "success": True,
            "message": f"Runway configuration changed to {config_name}: "
                       f"{self.active_config.description}",
            "old_config": old_config,
            "new_config": config_name,
        }

    def issue_ground_stop(self, duration_minutes: int) -> dict:
        """Issue a ground stop for the specified duration."""
        duration_minutes = max(5, min(60, duration_minutes))
        self.ground_stop_until = self.clock + duration_minutes
        self.metrics.ground_stop_minutes += duration_minutes

        self.step_events.append(
            f"GROUND STOP issued for {duration_minutes} min (until T+{self.ground_stop_until})"
        )

        return {
            "success": True,
            "message": f"Ground stop issued for {duration_minutes} minutes. "
                       f"No departures until T+{self.ground_stop_until}.",
        }

    # ------------------------------------------------------------------
    # Time advancement
    # ------------------------------------------------------------------

    def advance(self) -> dict:
        """Advance simulation by one time step.

        Returns observation dict.
        """
        self.metrics.begin_step()
        self.step_events = []
        self.step_runway_ops = {}

        # Update weather and wind
        if self.step_count < len(self.weather_timeline):
            new_weather = self.weather_timeline[self.step_count]
            if new_weather != self.current_weather:
                self.step_events.append(
                    f"Weather changed: {self.current_weather.value} -> {new_weather.value}"
                )
            self.current_weather = new_weather

        if self.step_count < len(self.wind_timeline):
            new_wind = self.wind_timeline[self.step_count]
            if abs(new_wind - self.current_wind) > 30:
                self.step_events.append(
                    f"Wind shift: {self.current_wind}° -> {new_wind}°"
                )
            self.current_wind = new_wind

        # Activate new flights
        self._activate_flights()

        # Process all flight phase transitions
        self._process_flights()

        # Check capacity
        self._check_capacity()

        # Check connections
        self._check_connections()

        # Advance clock
        self.clock += self.step_duration
        self.step_count += 1

        return self.get_observation()

    def _activate_flights(self) -> None:
        """Activate flights whose scheduled time has arrived."""
        for fid in self.all_flight_ids:
            flight = self.flights[fid]
            if flight.activated:
                continue
            if flight.scheduled_time <= self.clock:
                flight.activated = True
                if flight.is_arrival:
                    self.step_events.append(
                        f"New arrival: {flight.callsign} ({flight.aircraft_type}) "
                        f"{'[EMERGENCY]' if flight.priority == Priority.EMERGENCY else ''}"
                    )
                else:
                    # Departure: needs gate assignment to become visible
                    # Auto-assign to first compatible available gate
                    gate_assigned = self._auto_assign_departure_gate(flight)
                    if gate_assigned:
                        self.step_events.append(
                            f"Departure ready: {flight.callsign} ({flight.aircraft_type}) "
                            f"at gate {flight.assigned_gate}"
                        )

    def _auto_assign_departure_gate(self, flight: Flight) -> bool:
        """Auto-assign a departure to an available compatible gate."""
        for gid, occ in self.gate_occupancy.items():
            gate = GATES[gid]
            if flight.adg.value <= gate.max_adg.value:
                if occ is None or occ[1] <= self.clock:
                    flight.assigned_gate = gid
                    # Departure occupies gate until it pushes back
                    self.gate_occupancy[gid] = (flight.id, self.clock + 9999)
                    return True
        return False

    def _process_flights(self) -> None:
        """Process all active flights through their state machine."""
        for fid in list(self.flights.keys()):
            flight = self.flights[fid]
            if not flight.activated:
                continue
            if flight.phase in (FlightPhase.DEPARTED, FlightPhase.DIVERTED):
                continue

            self._process_single_flight(flight)

    def _process_single_flight(self, flight: Flight) -> None:
        """Process a single flight's phase transition."""
        dt = self.step_duration

        if flight.phase == FlightPhase.APPROACHING:
            # Burn fuel
            flight.fuel_remaining -= flight.fuel_rate * dt
            flight.phase_timer -= dt

            # Auto-hold if not sequenced within window
            if flight.phase_timer <= 0 and flight.assigned_runway is None:
                flight.phase = FlightPhase.HOLDING
                flight.hold_start_time = self.clock
                self.step_events.append(
                    f"{flight.callsign} auto-entered holding (no runway assigned)"
                )

        elif flight.phase == FlightPhase.HOLDING:
            # Burn fuel at holding rate
            fuel_burned = flight.data["fuel_rate_hold"] * dt
            flight.fuel_remaining -= fuel_burned
            self.metrics.excess_fuel += fuel_burned
            flight.delay_minutes += dt
            self.metrics.total_delay += dt

            # Check fuel critical
            if flight.fuel_minutes_remaining < FUEL_CRITICAL_MINUTES:
                flight.phase = FlightPhase.DIVERTED
                self.metrics.diversions += 1
                self.step_events.append(
                    f"FUEL CRITICAL: {flight.callsign} auto-diverted "
                    f"(fuel: {flight.fuel_minutes_remaining:.0f} min)"
                )
                # Check connections
                for arr_id, dep_id in list(self.connections):
                    if arr_id == flight.id and (arr_id, dep_id) not in self.missed_connection_pairs:
                        self.missed_connection_pairs.add((arr_id, dep_id))
                        self.metrics.missed_connections += 1

        elif flight.phase == FlightPhase.ON_FINAL:
            flight.fuel_remaining -= flight.fuel_rate * dt
            flight.phase_timer -= dt

            if flight.phase_timer <= 0:
                # Go-around check
                ga_prob = go_around_probability(self.current_weather)
                if self.rng.random() < ga_prob:
                    go_around_runway = flight.assigned_runway  # save before clearing
                    flight.phase = FlightPhase.APPROACHING
                    flight.phase_timer = 10
                    flight.assigned_runway = None
                    flight.delay_minutes += 5
                    self.metrics.total_delay += 5
                    self.metrics.go_arounds += 1
                    self.step_events.append(
                        f"GO-AROUND: {flight.callsign} on {go_around_runway}"
                    )
                else:
                    # Landed successfully
                    flight.phase = FlightPhase.LANDED
                    flight.phase_timer = LANDING_ROLLOUT_DURATION
                    self.step_events.append(
                        f"LANDED: {flight.callsign} on runway "
                        f"{flight.assigned_runway}"
                    )

        elif flight.phase == FlightPhase.LANDED:
            flight.phase_timer -= dt
            if flight.phase_timer <= 0:
                if flight.assigned_gate is not None:
                    taxi_time = get_taxi_time(
                        flight.assigned_runway or "13R",
                        flight.assigned_gate,
                    )
                    flight.phase = FlightPhase.TAXIING_IN
                    flight.phase_timer = taxi_time
                    fuel_burned = flight.data["fuel_rate_taxi"] * taxi_time
                    flight.fuel_remaining -= fuel_burned
                else:
                    # No gate assigned, wait
                    flight.delay_minutes += dt
                    self.metrics.total_delay += dt

        elif flight.phase == FlightPhase.TAXIING_IN:
            flight.phase_timer -= dt
            if flight.phase_timer <= 0:
                flight.phase = FlightPhase.AT_GATE
                turnaround = flight.data["turnaround"]
                flight.phase_timer = turnaround

                # Occupy gate
                if flight.assigned_gate:
                    avail_at = self.clock + turnaround + 5  # +5 for buffer
                    self.gate_occupancy[flight.assigned_gate] = (
                        flight.id, avail_at
                    )

                self.metrics.flights_completed += 1
                self.step_events.append(
                    f"AT GATE: {flight.callsign} arrived at {flight.assigned_gate}"
                )

                # Handle emergency
                if flight.priority == Priority.EMERGENCY:
                    self.metrics.emergencies_handled += 1

        elif flight.phase == FlightPhase.AT_GATE:
            flight.phase_timer -= dt
            # Turnaround in progress (for arrivals becoming future departures)
            # We don't model the turnaround->departure chain in this sim

        elif flight.phase == FlightPhase.READY:
            if not flight.activated:
                return

            # Waiting for departure clearance
            if flight.assigned_runway is not None:
                # Already sequenced but not yet pushing back
                # This shouldn't normally happen (sequence sets phase to PUSHBACK)
                pass
            else:
                # Accumulate delay if past scheduled time
                if self.clock > flight.scheduled_time:
                    excess = min(dt, self.clock - flight.scheduled_time)
                    if flight.delay_minutes == 0 and excess > 0:
                        flight.delay_minutes += excess
                        self.metrics.total_delay += excess
                    elif flight.delay_minutes > 0:
                        flight.delay_minutes += dt
                        self.metrics.total_delay += dt

        elif flight.phase == FlightPhase.PUSHBACK:
            flight.phase_timer -= dt
            if flight.phase_timer <= 0:
                # Free the gate
                if flight.assigned_gate and flight.assigned_gate in self.gate_occupancy:
                    self.gate_occupancy[flight.assigned_gate] = None

                taxi_time = get_taxi_time(
                    flight.assigned_runway or "13R",
                    flight.assigned_gate or "B1",
                )
                flight.phase = FlightPhase.TAXIING_OUT
                flight.phase_timer = taxi_time

        elif flight.phase == FlightPhase.TAXIING_OUT:
            flight.phase_timer -= dt
            if flight.phase_timer <= 0:
                flight.phase = FlightPhase.DEPARTED
                self.metrics.flights_completed += 1
                self.step_events.append(
                    f"DEPARTED: {flight.callsign} from runway "
                    f"{flight.assigned_runway}"
                )
                if flight.priority == Priority.EMERGENCY:
                    self.metrics.emergencies_handled += 1

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------

    def _check_wake_separation(
        self, flight: Flight, runway_id: str
    ) -> Optional[str]:
        """Check wake separation for a flight on a runway.

        Returns error string if violation, None if OK.
        """
        last = self.runway_last_op.get(runway_id)
        if last is None:
            return None

        last_time, last_wake = last
        required_sep = get_wake_separation(last_wake, flight.wake)
        actual_sep = self.clock - last_time

        if actual_sep < required_sep:
            return (
                f"Wake separation violation on {runway_id}: "
                f"{actual_sep}min < {required_sep}min required "
                f"({last_wake.value} -> {flight.wake.value})"
            )
        return None

    def _check_capacity(self) -> None:
        """Check if runway operations this step exceed capacity."""
        max_arr, max_dep = capacity_per_step(
            self.active_config, self.current_weather, self.step_duration
        )

        arr_ops = 0
        dep_ops = 0
        for rwy_id, ops in self.step_runway_ops.items():
            if rwy_id in self.active_config.arrival_runways:
                arr_ops += ops
            if rwy_id in self.active_config.departure_runways:
                dep_ops += ops

        if arr_ops > max_arr:
            self.step_events.append(
                f"CAPACITY WARNING: {arr_ops} arrivals exceeds max {max_arr} "
                f"for current weather"
            )

        if dep_ops > max_dep:
            self.step_events.append(
                f"CAPACITY WARNING: {dep_ops} departures exceeds max {max_dep} "
                f"for current weather"
            )

    def _check_connections(self) -> None:
        """Check for missed passenger connections."""
        for arr_id, dep_id in list(self.connections):
            if (arr_id, dep_id) in self.missed_connection_pairs:
                continue

            arr = self.flights[arr_id]
            dep = self.flights[dep_id]

            # Only check if arrival is significantly delayed
            if arr.phase in (FlightPhase.DEPARTED, FlightPhase.DIVERTED):
                continue
            if dep.phase == FlightPhase.DEPARTED:
                continue

            # If departure has already left and arrival hasn't arrived
            if dep.phase == FlightPhase.DEPARTED and arr.phase not in (
                FlightPhase.AT_GATE, FlightPhase.DEPARTED
            ):
                self.missed_connection_pairs.add((arr_id, dep_id))
                self.metrics.missed_connections += 1
                self.step_events.append(
                    f"MISSED CONNECTION: {arr.callsign} -> {dep.callsign}"
                )
                continue

            # If arrival delay makes MCT impossible
            if arr.delay_minutes > 0 and dep.scheduled_time > 0:
                arr_est_gate = (
                    arr.scheduled_time + arr.delay_minutes + 20
                )  # +20 for taxi+deplane
                dep_time = dep.scheduled_time
                buffer = dep_time - arr_est_gate
                if buffer < 0 and dep.phase == FlightPhase.READY:
                    # Connection is broken
                    self.missed_connection_pairs.add((arr_id, dep_id))
                    self.metrics.missed_connections += 1
                    self.step_events.append(
                        f"MISSED CONNECTION: {arr.callsign} delay "
                        f"({arr.delay_minutes}min) breaks connection "
                        f"to {dep.callsign}"
                    )

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def compute_step_reward(self) -> float:
        """Compute reward for the current step based on metrics delta."""
        delta = self.metrics.step_delta()

        throughput = delta.flights_completed * 0.05
        delay_pen = delta.delay_added * -0.002
        fuel_pen = delta.hold_fuel_burned * -0.0001
        conn_pen = delta.connections_missed * -0.1
        safety_pen = delta.safety_violations * -1.0
        emerg_bonus = delta.emergencies_handled * 0.5

        return throughput + delay_pen + fuel_pen + conn_pen + safety_pen + emerg_bonus

    def compute_final_reward(self) -> float:
        """Compute normalized final episode reward in [0, 1]."""
        m = self.metrics
        total_flights = sum(
            1 for f in self.flights.values() if f.activated
        )
        if total_flights == 0:
            return 0.5  # no flights = neutral

        # Throughput: fraction of activated flights completed
        throughput_score = m.flights_completed / max(total_flights, 1)

        # Delay: 1 - (avg_delay / 120), clamped
        if m.flights_completed > 0:
            avg_delay = m.total_delay / m.flights_completed
        else:
            avg_delay = 120
        delay_score = max(0.0, 1.0 - avg_delay / 120.0)

        # Connections: fraction preserved
        if m.total_connections > 0:
            conn_score = max(0.0, 1.0 - m.missed_connections / m.total_connections)
        else:
            conn_score = 1.0

        # Fuel efficiency
        fuel_score = max(
            0.0, 1.0 - m.excess_fuel / max(m.max_possible_excess, 1.0)
        )

        # Safety
        if m.safety_violations == 0:
            safety_score = 1.0
        else:
            safety_score = max(0.0, 1.0 - m.safety_violations * 0.2)

        # Weighted combination
        raw = (
            throughput_score * 0.30
            + delay_score * 0.25
            + conn_score * 0.20
            + fuel_score * 0.15
            + safety_score * 0.10
        )

        # Safety override
        if m.safety_violations > 0:
            raw *= 0.5

        return round(max(0.0, min(1.0, raw)), 4)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_observation(self) -> dict:
        """Build full airport state observation."""
        # Categorize flights by phase
        approaching = []
        holding = []
        on_final = []
        at_gate = []
        ready = []
        taxiing = []

        for fid, f in self.flights.items():
            if not f.activated:
                continue
            if f.phase == FlightPhase.APPROACHING:
                approaching.append(f)
            elif f.phase == FlightPhase.HOLDING:
                holding.append(f)
            elif f.phase == FlightPhase.ON_FINAL:
                on_final.append(f)
            elif f.phase in (FlightPhase.AT_GATE,):
                at_gate.append(f)
            elif f.phase == FlightPhase.READY:
                ready.append(f)
            elif f.phase in (FlightPhase.TAXIING_IN, FlightPhase.TAXIING_OUT,
                             FlightPhase.PUSHBACK, FlightPhase.LANDED):
                taxiing.append(f)

        # Available gates
        available_gates = []
        for gid, occ in self.gate_occupancy.items():
            if occ is None or occ[1] <= self.clock:
                available_gates.append(gid)

        # Capacity
        arr_cap, dep_cap = capacity_per_step(
            self.active_config, self.current_weather, self.step_duration
        )

        # Connections at risk
        connections_at_risk = []
        for arr_id, dep_id in self.connections:
            if (arr_id, dep_id) in self.missed_connection_pairs:
                continue
            arr = self.flights[arr_id]
            dep = self.flights[dep_id]
            if arr.phase in (FlightPhase.AT_GATE, FlightPhase.DEPARTED):
                continue
            if dep.phase == FlightPhase.DEPARTED:
                continue
            if arr.delay_minutes > 0 or arr.phase in (
                FlightPhase.HOLDING, FlightPhase.APPROACHING
            ):
                connections_at_risk.append((arr, dep))

        return {
            "step": self.step_count,
            "max_steps": self.max_steps,
            "clock": self.clock,
            "weather": self.current_weather.value,
            "wind": self.current_wind,
            "config": self.active_config_name,
            "config_desc": self.active_config.description,
            "arr_runways": list(self.active_config.arrival_runways),
            "dep_runways": list(self.active_config.departure_runways),
            "arr_capacity_per_step": arr_cap,
            "dep_capacity_per_step": dep_cap,
            "ground_stop": self.ground_stop_until > self.clock,
            "ground_stop_until": self.ground_stop_until,
            "approaching": [self._flight_dict(f) for f in approaching],
            "holding": [self._flight_dict(f) for f in holding],
            "on_final": [self._flight_dict(f) for f in on_final],
            "at_gate": [self._flight_dict(f) for f in at_gate],
            "ready": [self._flight_dict(f) for f in ready],
            "taxiing": [self._flight_dict(f) for f in taxiing],
            "available_gates": sorted(available_gates),
            "connections_at_risk": [
                {
                    "arrival": arr.callsign,
                    "arrival_id": arr.id,
                    "departure": dep.callsign,
                    "departure_id": dep.id,
                    "arrival_delay": arr.delay_minutes,
                }
                for arr, dep in connections_at_risk
            ],
            "events": list(self.step_events),
            "metrics": {
                "flights_completed": self.metrics.flights_completed,
                "total_delay": self.metrics.total_delay,
                "missed_connections": self.metrics.missed_connections,
                "total_connections": self.metrics.total_connections,
                "safety_violations": self.metrics.safety_violations,
                "diversions": self.metrics.diversions,
                "go_arounds": self.metrics.go_arounds,
                "ground_stop_minutes": self.metrics.ground_stop_minutes,
            },
        }

    def _flight_dict(self, f: Flight) -> dict:
        """Convert a Flight to a dict for observation."""
        return {
            "id": f.id,
            "callsign": f.callsign,
            "aircraft_type": f.aircraft_type,
            "wake": f.wake.value,
            "adg": f.adg.name,
            "phase": f.phase.value,
            "is_arrival": f.is_arrival,
            "scheduled_time": f.scheduled_time,
            "delay_minutes": f.delay_minutes,
            "fuel_minutes": round(f.fuel_minutes_remaining, 1),
            "priority": f.priority.value,
            "assigned_runway": f.assigned_runway,
            "assigned_gate": f.assigned_gate,
            "connecting_flights": f.connecting_flights,
            "phase_timer": f.phase_timer,
        }
