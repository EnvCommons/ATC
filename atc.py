"""Air Traffic Control environment for OpenReward.

An agent manages arrivals, departures, gate assignments, holds, diversions,
and runway configurations during weather disruptions at a realistic hub airport.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from openreward.environments import Environment, JSONObject, ToolOutput, tool, TextBlock

from models import TaskSpec, WeatherCondition
from simulation import ATCSimulation
from scenarios import ALL_TASKS
from airport import RUNWAY_CONFIGS, GATES, TERMINALS


# ---------------------------------------------------------------------------
# Tool parameter models
# ---------------------------------------------------------------------------

class SequenceFlightParams(BaseModel, extra="forbid"):
    flight_id: str = Field(description="ID of the flight to sequence (e.g., 'F0001')")
    runway_id: str = Field(description="Runway to assign (e.g., '13L', '31R', '04R')")


class AssignGateParams(BaseModel, extra="forbid"):
    flight_id: str = Field(description="ID of the arriving flight (e.g., 'F0001')")
    gate_id: str = Field(description="Gate to assign (e.g., 'B12', 'D3')")


class HoldFlightParams(BaseModel, extra="forbid"):
    flight_id: str = Field(description="ID of the flight to put in holding pattern")


class DivertFlightParams(BaseModel, extra="forbid"):
    flight_id: str = Field(description="ID of the flight to divert to alternate airport")


class SetRunwayConfigParams(BaseModel, extra="forbid"):
    config_name: str = Field(
        description="Configuration name: south_flow, north_flow, west_flow, ifr_south, or emergency"
    )


class GroundStopParams(BaseModel, extra="forbid"):
    duration_minutes: int = Field(
        description="Duration of ground stop in minutes (5-60)", ge=5, le=60
    )


class EmptyParams(BaseModel, extra="forbid"):
    pass


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class ATCEnvironment(Environment):
    """Air Traffic Control simulation environment.

    Multi-turn environment where the agent manages airport operations
    across 48 five-minute time steps (4 hours).
    """

    def __init__(
        self, task_spec: JSONObject, secrets: dict[str, str] = {}
    ) -> None:
        super().__init__(task_spec)
        self.config = TaskSpec.model_validate(task_spec)
        self.sim: ATCSimulation | None = None
        self.step_rewards: list[float] = []
        self.episode_done = False

    async def setup(self) -> None:
        """Initialize simulation from task spec."""
        self.sim = ATCSimulation(self.config)
        self.sim.reset()
        self.step_rewards = []
        self.episode_done = False

    async def teardown(self) -> None:
        self.sim = None

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    async def get_prompt(self) -> List[TextBlock]:
        """Return detailed prompt with airport layout and instructions."""
        assert self.sim is not None

        initial_obs = self._format_observation(self.sim.get_observation())

        # Weather outlook
        weather_counts: dict[str, int] = {}
        for w in self.config.weather_timeline:
            weather_counts[w] = weather_counts.get(w, 0) + 1
        wx_outlook = ", ".join(
            f"{k}: {v} steps" for k, v in sorted(weather_counts.items())
        )

        prompt = f"""You are an air traffic controller at Metro Hub International (MHI).

SCENARIO: {self.config.scenario_type.replace('_', ' ').title()} (Seed {self.config.seed})
DURATION: {self.config.num_steps} time steps of {self.config.step_duration} minutes each (4 hours total)
FLIGHTS: ~{self.config.flight_count} flights (arrivals + departures)
WEATHER OUTLOOK: {wx_outlook}

== AIRPORT LAYOUT ==

RUNWAYS (4 total, 2 pairs of parallels):
  13L/31R: 10,000 ft, ILS equipped
  13R/31L: 14,511 ft, ILS equipped
  04L/22R: 11,351 ft, ILS equipped
  04R/22L: 8,400 ft, ILS equipped

RUNWAY CONFIGURATIONS:
  south_flow:  Arrivals 13L, 13R | Departures 31L     | Wind 090-200°
  north_flow:  Arrivals 31R, 31L | Departures 22R      | Wind 270-360°
  west_flow:   Arrivals 22L, 31L | Departures 22R, 13R | Wind 200-270°
  ifr_south:   Arrivals 13R      | Departures 31L      | IFR reduced ops
  emergency:   All ops on 13R    |                      | Single runway

TERMINALS & GATES (60 total):
  Terminal A: 15 gates (A1-A15)  | Max ADG III | Regional jets (CRJ-900, E175)
  Terminal B: 20 gates (B1-B20)  | Max ADG III | Narrow-body & regional (B737, A320, E175, CRJ-900)
  Terminal C: 15 gates (C1-C15)  | Max ADG V   | Wide-body (B757, B767, B777, B787, A330)
  Terminal D: 10 gates (D1-D10)  | Max ADG VI  | All aircraft incl. A380

== YOUR OBJECTIVES ==

You must balance five objectives (weights for final score):
  1. THROUGHPUT (30%): Maximize flights processed (landed + departed)
  2. DELAY REDUCTION (25%): Minimize total delay minutes
  3. PASSENGER CONNECTIONS (20%): Prevent missed connections (MCT = 60 min)
  4. FUEL EFFICIENCY (15%): Minimize excess fuel burn (especially holds)
  5. SAFETY (10% + 0.5x multiplier): Zero wake separation violations

== TOOLS AVAILABLE ==

1. view_status()         - View full airport status (weather, flights, gates, connections)
2. sequence_flight(flight_id, runway_id) - Assign runway to arrival or departure
3. assign_gate(flight_id, gate_id)       - Assign gate to arriving flight
4. hold_flight(flight_id)               - Put flight in holding pattern
5. divert_flight(flight_id)             - Divert flight to alternate airport
6. set_runway_config(config_name)       - Change runway configuration
7. advance_time()                       - Advance simulation 5 minutes (returns step reward)
8. end_shift()                          - End simulation early (returns final reward)

== KEY RULES ==

WAKE TURBULENCE SEPARATION (same runway, consecutive ops):
  SUPER -> HEAVY/B757: 3 min  |  SUPER -> LARGE: 4 min
  HEAVY -> LARGE/SMALL: 3 min |  HEAVY -> HEAVY: 2 min
  Same category or smaller behind larger: 2 min minimum
  VIOLATION = safety penalty (final reward halved if any violations)

GATE COMPATIBILITY (Aircraft Design Group):
  ADG III gates (Terminal A): CRJ-900, E175 (regional)
  ADG III gates (Terminal B): B737, A320, E175, CRJ-900
  ADG V gates (Terminal C): + B757, B767, B777, B787, A330
  ADG VI gates (Terminal D): All aircraft including A380

FLIGHT PHASES:
  Arrivals:   APPROACHING -> ON_FINAL (5 min) -> LANDED -> TAXIING_IN -> AT_GATE
  Departures: READY -> PUSHBACK (5 min) -> TAXIING_OUT -> DEPARTED
  Holding:    Burns fuel at high rate; auto-diverts if fuel < 15 min

CONNECTIONS: If an arrival's delay makes MCT (60 min) impossible for
  connecting passengers, the connection is lost (-0.1 reward per miss).

WEATHER CAPACITY IMPACT:
  CLEAR: 100% | MVFR: 85% | IFR: 67% | LOW_IFR: 50% | THUNDERSTORM: 25%

== STRATEGY HINTS ==

- Prioritize EMERGENCY flights immediately (large bonus for quick handling)
- Match runway config to wind direction for maximum capacity
- Pre-assign gates to arrivals early to avoid taxi delays
- In bad weather, consider holding some flights and using ground stops
- Watch fuel levels on holding flights - divert before critical
- Manage connections: sequence connecting arrivals before their departures leave
- Use advance_time() to step the simulation forward after making decisions

== CURRENT STATUS ==

{initial_obs}

Begin managing the airport. Review the status, then make decisions and advance time."""

        return [TextBlock(text=prompt)]

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @tool
    async def view_status(self, params: EmptyParams) -> ToolOutput:
        """View current airport status including weather, flights, gates, and connections."""
        assert self.sim is not None
        obs = self.sim.get_observation()
        text = self._format_observation(obs)
        return ToolOutput(
            blocks=[TextBlock(text=text)],
            metadata=obs,
            reward=0.0,
            finished=False,
        )

    @tool
    async def sequence_flight(self, params: SequenceFlightParams) -> ToolOutput:
        """Sequence a flight for arrival or departure on a specified runway."""
        assert self.sim is not None
        if self.episode_done:
            return self._done_response()

        result = self.sim.sequence_flight(params.flight_id, params.runway_id)
        text = result.get("message") or result.get("error", "Unknown error")
        if not result["success"]:
            text = f"Error: {text}"

        return ToolOutput(
            blocks=[TextBlock(text=text)],
            metadata=result,
            reward=0.0,
            finished=False,
        )

    @tool
    async def assign_gate(self, params: AssignGateParams) -> ToolOutput:
        """Assign a gate to an arriving flight."""
        assert self.sim is not None
        if self.episode_done:
            return self._done_response()

        result = self.sim.assign_gate(params.flight_id, params.gate_id)
        text = result.get("message") or result.get("error", "Unknown error")
        if not result["success"]:
            text = f"Error: {text}"

        return ToolOutput(
            blocks=[TextBlock(text=text)],
            metadata=result,
            reward=0.0,
            finished=False,
        )

    @tool
    async def hold_flight(self, params: HoldFlightParams) -> ToolOutput:
        """Put a flight in a holding pattern."""
        assert self.sim is not None
        if self.episode_done:
            return self._done_response()

        result = self.sim.hold_flight(params.flight_id)
        text = result.get("message") or result.get("error", "Unknown error")
        if not result["success"]:
            text = f"Error: {text}"

        return ToolOutput(
            blocks=[TextBlock(text=text)],
            metadata=result,
            reward=0.0,
            finished=False,
        )

    @tool
    async def divert_flight(self, params: DivertFlightParams) -> ToolOutput:
        """Divert a flight to an alternate airport."""
        assert self.sim is not None
        if self.episode_done:
            return self._done_response()

        result = self.sim.divert_flight(params.flight_id)
        text = result.get("message") or result.get("error", "Unknown error")
        if not result["success"]:
            text = f"Error: {text}"

        return ToolOutput(
            blocks=[TextBlock(text=text)],
            metadata=result,
            reward=0.0,
            finished=False,
        )

    @tool
    async def set_runway_config(self, params: SetRunwayConfigParams) -> ToolOutput:
        """Change the active runway configuration."""
        assert self.sim is not None
        if self.episode_done:
            return self._done_response()

        result = self.sim.set_runway_config(params.config_name)
        text = result.get("message") or result.get("error", "Unknown error")
        if not result["success"]:
            text = f"Error: {text}"

        return ToolOutput(
            blocks=[TextBlock(text=text)],
            metadata=result,
            reward=0.0,
            finished=False,
        )

    @tool
    async def issue_ground_stop(self, params: GroundStopParams) -> ToolOutput:
        """Issue a ground stop, preventing departures for the specified duration."""
        assert self.sim is not None
        if self.episode_done:
            return self._done_response()

        result = self.sim.issue_ground_stop(params.duration_minutes)
        text = result.get("message") or result.get("error", "Unknown error")

        return ToolOutput(
            blocks=[TextBlock(text=text)],
            metadata=result,
            reward=0.0,
            finished=False,
        )

    @tool
    async def advance_time(self, params: EmptyParams) -> ToolOutput:
        """Advance the simulation by one time step (5 minutes).

        Processes all flight phase transitions, weather changes, and
        returns the step reward.
        """
        assert self.sim is not None
        if self.episode_done:
            return self._done_response()

        obs = self.sim.advance()
        step_reward = self.sim.compute_step_reward()
        self.step_rewards.append(step_reward)

        cumulative = sum(self.step_rewards)

        if self.sim.step_count >= self.sim.max_steps:
            self.episode_done = True
            final_reward = self.sim.compute_final_reward()
            text = self._format_observation(obs)
            text += f"\n\n=== SHIFT COMPLETE ===\n"
            text += f"Final Reward: {final_reward:.4f}\n"
            text += self._format_final_summary()

            return ToolOutput(
                blocks=[TextBlock(text=text)],
                metadata={**obs, "final_reward": final_reward},
                reward=final_reward,
                finished=True,
            )

        text = self._format_observation(obs)
        text += f"\nStep Reward: {step_reward:+.4f} | Cumulative: {cumulative:+.4f}"

        return ToolOutput(
            blocks=[TextBlock(text=text)],
            metadata={**obs, "step_reward": step_reward, "cumulative_reward": cumulative},
            reward=step_reward,
            finished=False,
        )

    @tool
    async def end_shift(self, params: EmptyParams) -> ToolOutput:
        """End the simulation early and receive final reward."""
        assert self.sim is not None
        if self.episode_done:
            return self._done_response()

        self.episode_done = True
        final_reward = self.sim.compute_final_reward()
        text = f"Shift ended early at step {self.sim.step_count}/{self.sim.max_steps}.\n"
        text += f"Final Reward: {final_reward:.4f}\n"
        text += self._format_final_summary()

        return ToolOutput(
            blocks=[TextBlock(text=text)],
            metadata={"final_reward": final_reward, "steps_completed": self.sim.step_count},
            reward=final_reward,
            finished=True,
        )

    # ------------------------------------------------------------------
    # Task listing
    # ------------------------------------------------------------------

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split not in ALL_TASKS:
            raise ValueError(f"Unknown split: {split}. Available: train, test")
        return ALL_TASKS[split]

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _done_response(self) -> ToolOutput:
        return ToolOutput(
            blocks=[TextBlock(text="Shift has already ended. No more actions allowed.")],
            metadata={"error": "episode_finished"},
            reward=0.0,
            finished=True,
        )

    def _format_observation(self, obs: dict) -> str:
        """Format observation dict as human-readable text."""
        lines = []

        # Header
        clock_h = obs["clock"] // 60
        clock_m = obs["clock"] % 60
        time_str = f"{12 + clock_h:02d}:{clock_m:02d}"
        lines.append(
            f"=== MHI AIRPORT STATUS [Step {obs['step']}/{obs['max_steps']}, "
            f"Time: {time_str}] ==="
        )

        # Weather
        lines.append(
            f"WEATHER: {obs['weather']} | Wind: {obs['wind']}°"
        )

        # Config
        lines.append(
            f"CONFIG: {obs['config']} ({obs['config_desc']})"
        )
        lines.append(
            f"  Arrivals: {', '.join(obs['arr_runways'])} | "
            f"Departures: {', '.join(obs['dep_runways'])} | "
            f"Capacity: {obs['arr_capacity_per_step']} arr/{obs['dep_capacity_per_step']} dep per step"
        )

        # Ground stop
        if obs["ground_stop"]:
            lines.append(f"GROUND STOP: Active until T+{obs['ground_stop_until']}")

        # Events
        if obs["events"]:
            lines.append("")
            lines.append("EVENTS:")
            for event in obs["events"]:
                lines.append(f"  >> {event}")

        # Approaching flights
        if obs["approaching"]:
            lines.append(f"\nAPPROACHING ({len(obs['approaching'])} flights):")
            for f in obs["approaching"]:
                pri = "[!EMERGENCY] " if f["priority"] == "EMERGENCY" else (
                    "[MEDICAL] " if f["priority"] == "MEDICAL" else ""
                )
                delay = f"[+{f['delay_minutes']}min]" if f["delay_minutes"] > 0 else "[ON TIME]"
                wake = f" {f['wake']}" if f["wake"] in ("SUPER", "HEAVY") else ""
                lines.append(
                    f"  {pri}{f['id']} {f['callsign']:8s} {f['aircraft_type']:10s} "
                    f"Fuel:{f['fuel_minutes']:.0f}min  {delay}{wake}"
                )

        # Holding flights
        if obs["holding"]:
            lines.append(f"\nHOLDING ({len(obs['holding'])} flights):")
            for f in obs["holding"]:
                pri = "[!EMERGENCY] " if f["priority"] == "EMERGENCY" else ""
                wake = f" {f['wake']}" if f["wake"] in ("SUPER", "HEAVY") else ""
                lines.append(
                    f"  {pri}{f['id']} {f['callsign']:8s} {f['aircraft_type']:10s} "
                    f"Fuel:{f['fuel_minutes']:.0f}min  [+{f['delay_minutes']}min delay]{wake}"
                )

        # On final
        if obs["on_final"]:
            lines.append(f"\nON FINAL ({len(obs['on_final'])} flights):")
            for f in obs["on_final"]:
                lines.append(
                    f"  {f['id']} {f['callsign']:8s} {f['aircraft_type']:10s} "
                    f"Runway: {f['assigned_runway']}  Landing in ~{f['phase_timer']}min"
                )

        # Taxiing
        if obs["taxiing"]:
            lines.append(f"\nTAXIING ({len(obs['taxiing'])} flights):")
            for f in obs["taxiing"]:
                direction = "IN" if f["phase"] in ("TAXIING_IN", "LANDED") else "OUT"
                lines.append(
                    f"  {f['id']} {f['callsign']:8s} {f['aircraft_type']:10s} "
                    f"Taxi {direction}  Gate: {f['assigned_gate'] or '--'}  "
                    f"~{f['phase_timer']}min"
                )

        # At gate
        if obs["at_gate"]:
            lines.append(f"\nAT GATE ({len(obs['at_gate'])} flights):")
            for f in obs["at_gate"][:10]:  # limit display
                lines.append(
                    f"  {f['id']} {f['callsign']:8s} {f['aircraft_type']:10s} "
                    f"Gate: {f['assigned_gate']}  Turnaround: {f['phase_timer']}min left"
                )
            if len(obs["at_gate"]) > 10:
                lines.append(f"  ... and {len(obs['at_gate']) - 10} more")

        # Ready for departure
        if obs["ready"]:
            lines.append(f"\nREADY FOR DEPARTURE ({len(obs['ready'])} flights):")
            for f in obs["ready"]:
                pri = "[!EMERGENCY] " if f["priority"] == "EMERGENCY" else ""
                rwy = f["assigned_runway"] or "--"
                delay = f"[+{f['delay_minutes']}min]" if f["delay_minutes"] > 0 else "[ON TIME]"
                wake = f" {f['wake']}" if f["wake"] in ("SUPER", "HEAVY") else ""
                lines.append(
                    f"  {pri}{f['id']} {f['callsign']:8s} {f['aircraft_type']:10s} "
                    f"Gate: {f['assigned_gate'] or '--'}  Runway: {rwy}  {delay}{wake}"
                )

        # Available gates (summarized by terminal)
        if obs["available_gates"]:
            by_terminal: dict[str, list[str]] = {}
            for g in obs["available_gates"]:
                t = g[0]
                by_terminal.setdefault(t, []).append(g)
            gate_str = " | ".join(
                f"{t}: {','.join(gs)}" for t, gs in sorted(by_terminal.items())
            )
            lines.append(f"\nAVAILABLE GATES: {gate_str}")

        # Connections at risk
        if obs["connections_at_risk"]:
            lines.append(f"\nCONNECTIONS AT RISK ({len(obs['connections_at_risk'])}):")
            for c in obs["connections_at_risk"][:5]:
                lines.append(
                    f"  {c['arrival']} ({c['arrival_id']}) -> "
                    f"{c['departure']} ({c['departure_id']}) "
                    f"[arrival +{c['arrival_delay']}min delay]"
                )

        # Metrics summary
        m = obs["metrics"]
        lines.append(
            f"\nMETRICS: Completed:{m['flights_completed']} | "
            f"Delay:{m['total_delay']:.0f}min | "
            f"Missed:{m['missed_connections']}/{m['total_connections']} conn | "
            f"Violations:{m['safety_violations']} | "
            f"Diversions:{m['diversions']} | "
            f"Go-arounds:{m['go_arounds']}"
        )

        return "\n".join(lines)

    def _format_final_summary(self) -> str:
        """Format final performance summary."""
        assert self.sim is not None
        m = self.sim.metrics
        total = sum(1 for f in self.sim.flights.values() if f.activated)

        lines = [
            "\n=== PERFORMANCE SUMMARY ===",
            f"Flights Completed: {m.flights_completed}/{total}",
            f"Total Delay: {m.total_delay:.0f} minutes",
            f"Average Delay: {m.total_delay / max(m.flights_completed, 1):.1f} min/flight",
            f"Missed Connections: {m.missed_connections}/{m.total_connections}",
            f"Excess Fuel Burned: {m.excess_fuel:.0f} kg",
            f"Safety Violations: {m.safety_violations}",
            f"Diversions: {m.diversions}",
            f"Go-Arounds: {m.go_arounds}",
            f"Ground Stop Minutes: {m.ground_stop_minutes}",
            f"Emergencies Handled: {m.emergencies_handled}",
        ]
        return "\n".join(lines)
