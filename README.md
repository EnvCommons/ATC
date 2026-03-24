# ATC

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/ATC)

## Description

An air traffic control simulation where an agent manages arrivals, departures, gate assignments, holding patterns, diversions, and runway configurations during weather disruptions at a realistic hub airport (Metro Hub International, inspired by JFK). The environment features a huge combinatorial action space, hard safety constraints, stochastic weather and go-arounds, and multi-objective rewards spanning throughput, delay reduction, fuel efficiency, passenger connections, and safety margins.

Note: this is a synthetic environment that is majority AI-generated; please test thoroughly before using in an RL pipeline.

## Capabilities

- Combinatorial scheduling of 150-300 flights across multiple runways and 60 gates
- Safety constraint satisfaction (FAA wake turbulence separation, gate ADG compatibility)
- Weather adaptation (5 weather conditions affecting capacity from 100% to 25%)
- Multi-objective optimization balancing throughput, delays, fuel, connections, and safety
- Dynamic runway configuration management based on wind conditions
- Ground stop and holding pattern management during severe weather

## Compute Requirements

No additional compute requirements.

## License

MIT

## Tasks

30 total tasks across 10 scenario types:

| Scenario | Description | Train | Test |
|----------|------------|-------|------|
| clear_day | Normal traffic, good weather | 2 | 1 |
| morning_fog | IFR conditions clearing to VFR | 2 | 1 |
| thunderstorm | Building thunderstorm with capacity reduction | 2 | 1 |
| snow_event | Snow with LOW_IFR and runway clearing | 2 | 1 |
| wind_shift | Wind direction change requiring config switch | 2 | 1 |
| peak_traffic | Holiday-level traffic volume (250-300 flights) | 2 | 1 |
| compound_wx | Multiple weather types in rapid succession | 2 | 1 |
| cascading_delay | Normal weather but systemic early delays | 2 | 1 |
| runway_closure | Reduced capacity from planned maintenance | 2 | 1 |
| crosswind | Strong crosswinds limiting runway options | 2 | 1 |

Each task runs for 48 time steps of 5 minutes each (4 hours of simulated operations).

## Reward Structure

Dense per-step rewards on each `advance_time()` call, plus a final normalized episode reward in [0, 1].

**Final reward** is a weighted combination of five objectives:

| Component | Weight | Metric |
|-----------|--------|--------|
| Throughput | 30% | Fraction of flights completed (landed + departed) |
| Delay Reduction | 25% | 1 - (avg_delay / 120 min), clamped to [0,1] |
| Passenger Connections | 20% | Fraction of connections preserved |
| Fuel Efficiency | 15% | 1 - (excess_fuel / max_excess), clamped |
| Safety | 10% | 1.0 if zero violations, degrades by 0.2 per violation |

**Safety override**: Any wake separation violation applies a 0.5x multiplier to the entire final reward.

No LLM grader is used; all rewards are computed algorithmically from simulation state.

## Data

No external data files. Flight schedules and weather timelines are generated procedurally from deterministic seeds at module load time. Aircraft performance parameters (fuel burn rates, turnaround times, wake categories) are sourced from FAA/ICAO operational data and embedded in the code.

## Tools

8 tools available to the agent:

| Tool | Description |
|------|-------------|
| `view_status()` | View full airport state: weather, flights by phase, gates, connections |
| `sequence_flight(flight_id, runway_id)` | Assign a runway to an arriving or departing flight |
| `assign_gate(flight_id, gate_id)` | Assign a gate to an arriving flight |
| `hold_flight(flight_id)` | Put a flight in a holding pattern |
| `divert_flight(flight_id)` | Divert a flight to an alternate airport |
| `set_runway_config(config_name)` | Change active runway configuration |
| `issue_ground_stop(duration_minutes)` | Issue a ground stop preventing departures |
| `advance_time()` | Advance simulation by 5 minutes (returns step reward) |
| `end_shift()` | End simulation early (returns final reward) |

## Time Horizon

Multi-turn, open-ended within a 48-step window. The agent makes multiple tool calls per step (view status, sequence flights, assign gates) before advancing time. A typical episode involves 48 `advance_time()` calls plus action calls per step.

## Environment Difficulty

[To be determined after baseline evaluation]

## Other Environment Requirements

- OpenAI API key required for `test_agent.py` (the environment itself requires no external API keys)

## Safety

This is a simulated environment with no real-world impact.

## Parameter Sources

All aviation parameters are sourced from FAA, ICAO, IATA, and manufacturer operational data. Key citations:

| Parameter | Value | Source |
|-----------|-------|--------|
| ADG classifications | Groups I–VI by wingspan | FAA AC 150/5300-13B, Table 1-1 |
| Wake turbulence categories | SUPER/HEAVY/B757/LARGE/SMALL | FAA Order JO 7110.65, Section 5-5-4 |
| Wake separation (time-based) | 2–4 min | FAA JO 7110.65 NM values converted at ~140 kt approach speed |
| Runway lengths | 8,400–14,511 ft | FAA Airport/Facility Directory (AirNav KJFK); 04L/22R physical length is 12,079 ft |
| VFR arrival capacity | ~60/hr | FAA JFK Airport Capacity Profile (2014) |
| Weather capacity multipliers | 25%–100% | FAA AC 150/5060-5, Airport Capacity Profiles methodology |
| Go-around base rate | 0.5% | NASA TM-20240008006 (U.S. FY2023 actual: 0.39%) |
| B737-800 holding fuel burn | 40 kg/min (~2,400 kg/hr) | ICAO Engine Emissions Databank; Boeing operational data |
| Turnaround times | 30–150 min | IATA Ground Handling Manual (narrow-body 35–50, wide-body 75–150 min) |
| Domestic MCT | 60 min | IATA SSIM Chapter 8 / Resolution 765; JFK domestic MCT 45–60 min |
| International MCT | 120 min | IATA SSIM Chapter 8; JFK international MCT 120–135 min |
| Passenger counts | 76–525 pax | Manufacturer 2-class configuration data |
| Fuel critical threshold | 15 min | Simulation mechanic (FAA 14 CFR 91.167 requires 45 min IFR reserve) |

**Simplifications**: Time-based wake separation (real ATC uses distance-based radar); discrete 5-minute steps; A380 holding fuel rate (100 kg/min) is conservative vs. cruise (~200 kg/min); JFK slot limit of 81 ops/hr (14 CFR 93 Subpart K) not enforced — simulation uses physical runway capacity.

## Citations

```bibtex
@dataset{GRATC,
  author = {General Reasoning Inc. Team},
  title = {ATC: Air Traffic Control Environment for OpenReward},
  year = {2025},
  publisher = {OpenReward},
}
```
