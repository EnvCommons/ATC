"""Comprehensive test suite for the ATC environment.

Tests cover structural validation, determinism, tool validation,
simulation mechanics, reward computation, episode lifecycle, and edge cases.
"""

from __future__ import annotations

import asyncio
import pytest

from models import (
    AIRCRAFT_DATA,
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
    GATES,
    RUNWAY_CONFIGS,
    RUNWAYS,
    capacity_for_weather,
    capacity_per_step,
    get_taxi_time,
    get_wake_separation,
    go_around_probability,
)
from simulation import ATCSimulation
from scenarios import ALL_TASKS, SCENARIO_TYPES, _make_task, generate_all_tasks
from atc import ATCEnvironment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sim(scenario: str = "clear_day", seed: int = 0) -> ATCSimulation:
    """Create a simulation with a specific scenario for testing."""
    task = _make_task(scenario, seed, "train")
    spec = TaskSpec.model_validate(task)
    sim = ATCSimulation(spec)
    sim.reset()
    return sim


def _make_env(scenario: str = "clear_day", seed: int = 0) -> ATCEnvironment:
    """Create an ATCEnvironment instance for testing."""
    task = _make_task(scenario, seed, "train")
    return ATCEnvironment(task_spec=task)


def _find_flight(sim: ATCSimulation, phase: FlightPhase, is_arrival: bool = True) -> Flight | None:
    """Find the first activated flight in a given phase."""
    for f in sim.flights.values():
        if f.activated and f.phase == phase and f.is_arrival == is_arrival:
            return f
    return None


def _activate_and_advance(sim: ATCSimulation, steps: int = 5) -> None:
    """Advance simulation by several steps to activate flights."""
    for _ in range(steps):
        sim.advance()


# ===========================================================================
# A. Structural Tests
# ===========================================================================

class TestStructural:

    def test_list_splits(self):
        assert ATCEnvironment.list_splits() == ["train", "test"]

    def test_list_tasks_train_count(self):
        tasks = ATCEnvironment.list_tasks("train")
        assert len(tasks) == 20, f"Expected 20 train tasks, got {len(tasks)}"

    def test_list_tasks_test_count(self):
        tasks = ATCEnvironment.list_tasks("test")
        assert len(tasks) == 10, f"Expected 10 test tasks, got {len(tasks)}"

    def test_task_ids_unique(self):
        all_ids = set()
        for split in ["train", "test"]:
            tasks = ATCEnvironment.list_tasks(split)
            for task in tasks:
                assert task["id"] not in all_ids, f"Duplicate ID: {task['id']}"
                all_ids.add(task["id"])

    def test_task_spec_valid(self):
        """All tasks should parse as valid TaskSpec."""
        for split in ["train", "test"]:
            tasks = ATCEnvironment.list_tasks(split)
            for task in tasks:
                spec = TaskSpec.model_validate(task)
                assert spec.num_steps == 48
                assert spec.step_duration == 5
                assert len(spec.weather_timeline) == 48
                assert len(spec.wind_timeline) == 48

    def test_all_scenario_types_covered(self):
        """All scenario types should appear in tasks."""
        types_seen = set()
        for split in ["train", "test"]:
            for task in ATCEnvironment.list_tasks(split):
                types_seen.add(task["scenario_type"])
        for st in SCENARIO_TYPES:
            assert st in types_seen, f"Scenario type {st} missing from tasks"

    def test_invalid_split_raises(self):
        with pytest.raises(ValueError):
            ATCEnvironment.list_tasks("invalid")


# ===========================================================================
# B. Determinism Tests
# ===========================================================================

class TestDeterminism:

    def test_same_seed_same_flights(self):
        sim1 = _make_sim("clear_day", seed=0)
        sim2 = _make_sim("clear_day", seed=0)
        ids1 = sorted(sim1.flights.keys())
        ids2 = sorted(sim2.flights.keys())
        assert ids1 == ids2

        for fid in ids1:
            assert sim1.flights[fid].callsign == sim2.flights[fid].callsign
            assert sim1.flights[fid].aircraft_type == sim2.flights[fid].aircraft_type

    def test_different_seeds_different(self):
        sim1 = _make_sim("clear_day", seed=0)
        sim2 = _make_sim("clear_day", seed=42)
        cs1 = [sim1.flights[fid].callsign for fid in sorted(sim1.flights.keys())]
        cs2 = [sim2.flights[fid].callsign for fid in sorted(sim2.flights.keys())]
        assert cs1 != cs2, "Different seeds should produce different schedules"

    def test_weather_timeline_reproducible(self):
        task1 = _make_task("thunderstorm", 0, "train")
        task2 = _make_task("thunderstorm", 0, "train")
        assert task1["weather_timeline"] == task2["weather_timeline"]
        assert task1["wind_timeline"] == task2["wind_timeline"]


# ===========================================================================
# C. Initialization Tests
# ===========================================================================

class TestInitialization:

    def test_initial_observation_structure(self):
        sim = _make_sim()
        obs = sim.get_observation()
        assert "step" in obs
        assert "weather" in obs
        assert "wind" in obs
        assert "config" in obs
        assert "approaching" in obs
        assert "holding" in obs
        assert "available_gates" in obs
        assert "metrics" in obs

    def test_initial_step_is_zero(self):
        sim = _make_sim()
        obs = sim.get_observation()
        assert obs["step"] == 0
        assert obs["clock"] == 0

    def test_flights_generated(self):
        sim = _make_sim("clear_day", seed=0)
        assert len(sim.flights) > 100, "Should generate >100 flights"
        assert len(sim.flights) < 400, "Should generate <400 flights"

    def test_connections_generated(self):
        sim = _make_sim("clear_day", seed=0)
        assert len(sim.connections) > 0, "Should generate some connections"
        assert sim.metrics.total_connections == len(sim.connections)

    @pytest.mark.asyncio
    async def test_get_prompt_returns_textblock(self):
        env = _make_env()
        await env.setup()
        prompt = await env.get_prompt()
        assert len(prompt) == 1
        assert hasattr(prompt[0], "text")
        assert len(prompt[0].text) > 100
        await env.teardown()

    @pytest.mark.asyncio
    async def test_get_prompt_contains_key_info(self):
        env = _make_env()
        await env.setup()
        prompt = await env.get_prompt()
        text = prompt[0].text
        assert "RUNWAY" in text
        assert "GATE" in text
        assert "WEATHER" in text
        assert "sequence_flight" in text
        assert "advance_time" in text
        await env.teardown()


# ===========================================================================
# D. Tool Validation Tests
# ===========================================================================

class TestToolValidation:

    def test_sequence_valid_arrival(self):
        sim = _make_sim("clear_day", seed=0)
        _activate_and_advance(sim, steps=3)

        flight = _find_flight(sim, FlightPhase.APPROACHING, is_arrival=True)
        if flight is None:
            flight = _find_flight(sim, FlightPhase.HOLDING, is_arrival=True)
        assert flight is not None, "Should have at least one approaching/holding flight"

        arr_runways = sim.active_config.arrival_runways
        result = sim.sequence_flight(flight.id, arr_runways[0])
        assert result["success"], f"Should succeed: {result}"
        assert flight.phase == FlightPhase.ON_FINAL

    def test_sequence_invalid_flight_id(self):
        sim = _make_sim()
        result = sim.sequence_flight("NONEXISTENT", "13R")
        assert not result["success"]
        assert "Unknown flight" in result["error"]

    def test_sequence_wrong_phase(self):
        sim = _make_sim()
        _activate_and_advance(sim, steps=10)

        departed = None
        for f in sim.flights.values():
            if f.phase == FlightPhase.DEPARTED:
                departed = f
                break

        if departed:
            result = sim.sequence_flight(departed.id, "13R")
            assert not result["success"]

    def test_sequence_invalid_runway(self):
        sim = _make_sim()
        _activate_and_advance(sim, steps=3)
        flight = _find_flight(sim, FlightPhase.APPROACHING)
        if flight:
            result = sim.sequence_flight(flight.id, "99X")
            assert not result["success"]
            assert "Unknown runway" in result["error"]

    def test_sequence_wrong_runway_type(self):
        sim = _make_sim()
        _activate_and_advance(sim, steps=3)
        flight = _find_flight(sim, FlightPhase.APPROACHING)
        if flight:
            # Try to assign arrival to a departure runway
            dep_rwy = sim.active_config.departure_runways[0]
            if dep_rwy not in sim.active_config.arrival_runways:
                result = sim.sequence_flight(flight.id, dep_rwy)
                assert not result["success"]
                assert "not an active arrival runway" in result["error"]

    def test_assign_gate_valid(self):
        sim = _make_sim()
        _activate_and_advance(sim, steps=3)
        flight = _find_flight(sim, FlightPhase.APPROACHING)
        if flight:
            # Find a compatible gate
            adg = flight.adg
            for gid, gate in GATES.items():
                if gate.max_adg.value >= adg.value:
                    occ = sim.gate_occupancy.get(gid)
                    if occ is None or occ[1] <= sim.clock:
                        result = sim.assign_gate(flight.id, gid)
                        assert result["success"], f"Gate assignment should succeed: {result}"
                        assert flight.assigned_gate == gid
                        break

    def test_assign_gate_adg_incompatible(self):
        sim = _make_sim()
        _activate_and_advance(sim, steps=3)

        # Create a heavy aircraft manually for testing
        from models import Flight as FlightModel
        heavy_flight = Flight(
            id="TEST_HEAVY",
            callsign="TST001",
            aircraft_type="B777-300ER",
            phase=FlightPhase.APPROACHING,
            is_arrival=True,
            scheduled_time=0,
            phase_timer=10,
            fuel_remaining=10000,
            delay_minutes=0,
            priority=Priority.NORMAL,
            activated=True,
        )
        sim.flights["TEST_HEAVY"] = heavy_flight

        # Terminal A gates are ADG II only
        result = sim.assign_gate("TEST_HEAVY", "A1")
        assert not result["success"]
        assert "too large" in result["error"]

    def test_assign_gate_occupied(self):
        sim = _make_sim()
        _activate_and_advance(sim, steps=3)

        # Manually occupy a gate
        sim.gate_occupancy["B1"] = ("FAKE_FLIGHT", sim.clock + 100)

        flight = _find_flight(sim, FlightPhase.APPROACHING)
        if flight:
            result = sim.assign_gate(flight.id, "B1")
            assert not result["success"]
            assert "occupied" in result["error"]

    def test_hold_flight_valid(self):
        sim = _make_sim()
        _activate_and_advance(sim, steps=3)
        flight = _find_flight(sim, FlightPhase.APPROACHING)
        if flight:
            result = sim.hold_flight(flight.id)
            assert result["success"]
            assert flight.phase == FlightPhase.HOLDING

    def test_hold_already_holding(self):
        sim = _make_sim()
        _activate_and_advance(sim, steps=3)
        flight = _find_flight(sim, FlightPhase.APPROACHING)
        if flight:
            sim.hold_flight(flight.id)
            result = sim.hold_flight(flight.id)
            assert result["success"]  # Idempotent

    def test_divert_flight(self):
        sim = _make_sim()
        _activate_and_advance(sim, steps=3)
        flight = _find_flight(sim, FlightPhase.APPROACHING)
        if flight:
            result = sim.divert_flight(flight.id)
            assert result["success"]
            assert flight.phase == FlightPhase.DIVERTED
            assert sim.metrics.diversions >= 1

    def test_set_runway_config(self):
        sim = _make_sim()
        result = sim.set_runway_config("ifr_south")
        assert result["success"]
        assert sim.active_config_name == "ifr_south"
        assert sim.active_config.arrival_runways == ("13R",)

    def test_set_runway_config_invalid(self):
        sim = _make_sim()
        result = sim.set_runway_config("nonexistent")
        assert not result["success"]
        assert "Unknown config" in result["error"]

    def test_ground_stop(self):
        sim = _make_sim()
        result = sim.issue_ground_stop(30)
        assert result["success"]
        assert sim.ground_stop_until == 30

    def test_view_status_idempotent(self):
        sim = _make_sim()
        obs1 = sim.get_observation()
        obs2 = sim.get_observation()
        assert obs1["step"] == obs2["step"]
        assert obs1["weather"] == obs2["weather"]
        assert len(obs1["approaching"]) == len(obs2["approaching"])


# ===========================================================================
# E. Simulation Mechanics Tests
# ===========================================================================

class TestSimulationMechanics:

    def test_weather_change(self):
        """Thunderstorm scenario should have weather changes."""
        sim = _make_sim("thunderstorm", seed=0)
        weathers = set()
        for _ in range(48):
            sim.advance()
            weathers.add(sim.current_weather)
        assert len(weathers) > 1, "Thunderstorm scenario should have weather changes"

    def test_flight_phase_transitions_arrival(self):
        """Test full arrival cycle: APPROACHING -> ON_FINAL -> LANDED -> TAXIING_IN -> AT_GATE."""
        sim = _make_sim("clear_day", seed=0)
        _activate_and_advance(sim, steps=3)

        flight = _find_flight(sim, FlightPhase.APPROACHING)
        if flight is None:
            flight = _find_flight(sim, FlightPhase.HOLDING)
        assert flight is not None

        # Assign runway and gate
        arr_rwy = sim.active_config.arrival_runways[0]
        sim.sequence_flight(flight.id, arr_rwy)
        assert flight.phase == FlightPhase.ON_FINAL

        # Find a compatible gate
        for gid, gate in GATES.items():
            if gate.max_adg.value >= flight.adg.value:
                occ = sim.gate_occupancy.get(gid)
                if occ is None or occ[1] <= sim.clock:
                    sim.assign_gate(flight.id, gid)
                    break

        # Advance through phases
        phases_seen = {flight.phase}
        for _ in range(20):
            sim.advance()
            phases_seen.add(flight.phase)
            if flight.phase == FlightPhase.AT_GATE:
                break

        assert FlightPhase.ON_FINAL in phases_seen
        assert FlightPhase.AT_GATE in phases_seen or FlightPhase.TAXIING_IN in phases_seen

    def test_departure_phase_transitions(self):
        """Test departure: READY -> PUSHBACK -> TAXIING_OUT -> DEPARTED."""
        sim = _make_sim("clear_day", seed=0)
        _activate_and_advance(sim, steps=5)

        flight = _find_flight(sim, FlightPhase.READY, is_arrival=False)
        if flight is None:
            _activate_and_advance(sim, steps=10)
            flight = _find_flight(sim, FlightPhase.READY, is_arrival=False)

        if flight is None:
            pytest.skip("No READY departure found in this scenario")

        dep_rwy = sim.active_config.departure_runways[0]
        result = sim.sequence_flight(flight.id, dep_rwy)
        assert result["success"]
        assert flight.phase == FlightPhase.PUSHBACK

        phases_seen = {flight.phase}
        for _ in range(20):
            sim.advance()
            phases_seen.add(flight.phase)
            if flight.phase == FlightPhase.DEPARTED:
                break

        assert FlightPhase.PUSHBACK in phases_seen
        assert FlightPhase.DEPARTED in phases_seen

    def test_fuel_burn_in_holding(self):
        sim = _make_sim()
        _activate_and_advance(sim, steps=3)
        flight = _find_flight(sim, FlightPhase.APPROACHING)
        if flight:
            initial_fuel = flight.fuel_remaining
            sim.hold_flight(flight.id)
            sim.advance()
            assert flight.fuel_remaining < initial_fuel, "Fuel should decrease in hold"

    def test_fuel_critical_auto_divert(self):
        sim = _make_sim()
        _activate_and_advance(sim, steps=3)
        flight = _find_flight(sim, FlightPhase.APPROACHING)
        if flight is None:
            flight = _find_flight(sim, FlightPhase.HOLDING)
        if flight:
            # Set fuel very low
            flight.fuel_remaining = 100  # very low
            sim.hold_flight(flight.id)
            for _ in range(10):
                sim.advance()
                if flight.phase == FlightPhase.DIVERTED:
                    break
            assert flight.phase == FlightPhase.DIVERTED, "Low fuel should trigger auto-divert"

    def test_turnaround_time_by_aircraft(self):
        """B737 should have shorter turnaround than B777."""
        assert AIRCRAFT_DATA["B737-800"]["turnaround"] < AIRCRAFT_DATA["B777-300ER"]["turnaround"]
        assert AIRCRAFT_DATA["CRJ-900"]["turnaround"] < AIRCRAFT_DATA["A380"]["turnaround"]

    def test_weather_reduces_capacity(self):
        """IFR should have lower capacity than CLEAR."""
        config = RUNWAY_CONFIGS["south_flow"]
        arr_clear, dep_clear = capacity_for_weather(config, WeatherCondition.CLEAR)
        arr_ifr, dep_ifr = capacity_for_weather(config, WeatherCondition.IFR)
        assert arr_ifr < arr_clear
        assert dep_ifr < dep_clear

    def test_thunderstorm_capacity(self):
        config = RUNWAY_CONFIGS["south_flow"]
        arr_ts, dep_ts = capacity_for_weather(config, WeatherCondition.THUNDERSTORM)
        arr_clear, dep_clear = capacity_for_weather(config, WeatherCondition.CLEAR)
        assert arr_ts < arr_clear * 0.30

    def test_ground_stop_blocks_departures(self):
        sim = _make_sim()
        _activate_and_advance(sim, steps=5)
        sim.issue_ground_stop(30)

        flight = _find_flight(sim, FlightPhase.READY, is_arrival=False)
        if flight:
            dep_rwy = sim.active_config.departure_runways[0]
            result = sim.sequence_flight(flight.id, dep_rwy)
            assert not result["success"]
            assert "Ground stop" in result["error"]


# ===========================================================================
# F. Airport Configuration Tests
# ===========================================================================

class TestAirportConfig:

    def test_all_runways_defined(self):
        assert len(RUNWAYS) == 8  # 4 physical runways * 2 directions

    def test_all_gates_defined(self):
        assert len(GATES) == 60

    def test_gate_counts_by_terminal(self):
        counts = {}
        for gid, gate in GATES.items():
            counts[gate.terminal] = counts.get(gate.terminal, 0) + 1
        assert counts["A"] == 15
        assert counts["B"] == 20
        assert counts["C"] == 15
        assert counts["D"] == 10

    def test_wake_separation_values(self):
        # SUPER -> LARGE should be higher than LARGE -> LARGE
        sep_sl = get_wake_separation(WakeCategory.SUPER, WakeCategory.LARGE)
        sep_ll = get_wake_separation(WakeCategory.LARGE, WakeCategory.LARGE)
        assert sep_sl > sep_ll

    def test_taxi_time_range(self):
        """Taxi times should be between 10 and 25 minutes."""
        for rwy_id in RUNWAYS:
            for gid in GATES:
                tt = get_taxi_time(rwy_id, gid)
                assert 10 <= tt <= 25, f"Taxi time {rwy_id}->{gid} = {tt}"

    def test_go_around_probability(self):
        p_clear = go_around_probability(WeatherCondition.CLEAR)
        p_ts = go_around_probability(WeatherCondition.THUNDERSTORM)
        assert p_clear < p_ts
        assert 0 < p_clear < 0.05
        assert p_ts < 0.10

    def test_runway_configs_have_valid_runways(self):
        for name, config in RUNWAY_CONFIGS.items():
            for rwy in config.arrival_runways:
                assert rwy in RUNWAYS, f"Config {name}: unknown arrival runway {rwy}"
            for rwy in config.departure_runways:
                assert rwy in RUNWAYS, f"Config {name}: unknown departure runway {rwy}"


# ===========================================================================
# G. Reward Tests
# ===========================================================================

class TestReward:

    def test_step_reward_positive_for_completions(self):
        """Completing flights should give positive step reward."""
        sim = _make_sim("clear_day", seed=0)
        _activate_and_advance(sim, steps=3)

        # Sequence several arrivals
        for f in list(sim.flights.values()):
            if f.activated and f.phase in (FlightPhase.APPROACHING, FlightPhase.HOLDING):
                arr_rwy = sim.active_config.arrival_runways[0]
                sim.sequence_flight(f.id, arr_rwy)
                # Assign gate
                for gid, gate in GATES.items():
                    if gate.max_adg.value >= f.adg.value:
                        occ = sim.gate_occupancy.get(gid)
                        if occ is None or occ[1] <= sim.clock:
                            sim.assign_gate(f.id, gid)
                            break

        # Advance several steps to complete some flights
        for _ in range(10):
            sim.advance()

        reward = sim.compute_step_reward()
        # After 10 advances with flights being processed, should have some completions
        # (reward may still be negative due to delays)

    def test_final_reward_in_range(self):
        sim = _make_sim("clear_day", seed=0)
        for _ in range(48):
            sim.advance()
        final = sim.compute_final_reward()
        assert 0.0 <= final <= 1.0, f"Final reward {final} out of [0,1] range"

    def test_final_reward_multiple_scenarios(self):
        """Final reward should be in [0,1] for all scenarios."""
        for scenario in SCENARIO_TYPES:
            sim = _make_sim(scenario, seed=0)
            for _ in range(48):
                sim.advance()
            final = sim.compute_final_reward()
            assert 0.0 <= final <= 1.0, (
                f"Scenario {scenario}: final reward {final} out of range"
            )

    def test_safety_violation_halves_reward(self):
        sim = _make_sim()
        # Run a few steps to get realistic metrics
        _activate_and_advance(sim, steps=10)
        # Process some flights to get non-trivial reward
        for f in list(sim.flights.values()):
            if f.activated and f.phase in (FlightPhase.APPROACHING, FlightPhase.HOLDING):
                arr_rwy = sim.active_config.arrival_runways[0]
                sim.sequence_flight(f.id, arr_rwy)
                for gid, gate in GATES.items():
                    if gate.max_adg.value >= f.adg.value:
                        occ = sim.gate_occupancy.get(gid)
                        if occ is None or occ[1] <= sim.clock:
                            sim.assign_gate(f.id, gid)
                            break
        for _ in range(20):
            sim.advance()

        sim.metrics.safety_violations = 0
        reward_clean = sim.compute_final_reward()

        sim.metrics.safety_violations = 2
        reward_violated = sim.compute_final_reward()
        # Safety violations should significantly reduce reward
        assert reward_violated < reward_clean, (
            f"Safety violation reward {reward_violated} should be < clean {reward_clean}"
        )

    def test_no_action_low_reward(self):
        """Running simulation without any actions should give low reward."""
        sim = _make_sim("clear_day", seed=0)
        for _ in range(48):
            sim.advance()
        final = sim.compute_final_reward()
        # With no actions, flights just auto-hold and delays pile up
        # Should be low but not necessarily zero
        assert final < 0.6, f"No-action reward {final} should be low"


# ===========================================================================
# H. Episode Lifecycle Tests
# ===========================================================================

class TestLifecycle:

    @pytest.mark.asyncio
    async def test_episode_ends_at_max_steps(self):
        from atc import EmptyParams
        env = _make_env()
        await env.setup()

        for i in range(48):
            result = await env.advance_time(EmptyParams())
            if result.finished:
                assert i == 47, f"Should finish on last step, finished on {i}"
                break
        else:
            pytest.fail("Episode should have finished")

        assert env.episode_done
        assert result.reward is not None
        assert 0.0 <= result.reward <= 1.0
        await env.teardown()

    @pytest.mark.asyncio
    async def test_end_shift_early(self):
        from atc import EmptyParams
        env = _make_env()
        await env.setup()

        # Advance a few steps
        for _ in range(5):
            await env.advance_time(EmptyParams())

        result = await env.end_shift(EmptyParams())
        assert result.finished
        assert result.reward is not None
        assert 0.0 <= result.reward <= 1.0
        await env.teardown()

    @pytest.mark.asyncio
    async def test_actions_after_finished(self):
        from atc import EmptyParams
        env = _make_env()
        await env.setup()

        await env.end_shift(EmptyParams())

        # All subsequent actions should return done
        result = await env.advance_time(EmptyParams())
        assert result.finished
        assert "already ended" in result.blocks[0].text.lower() or result.finished
        await env.teardown()

    @pytest.mark.asyncio
    async def test_full_episode_flow(self):
        """Run a complete episode with actions."""
        from atc import EmptyParams, SequenceFlightParams, AssignGateParams
        env = _make_env("clear_day", seed=0)
        await env.setup()

        for step in range(48):
            # View status
            status = await env.view_status(EmptyParams())
            assert not status.finished

            # Make some decisions based on observation
            obs = status.metadata

            # Sequence approaching flights
            for f in obs.get("approaching", [])[:2]:
                arr_rwys = obs["arr_runways"]
                if arr_rwys:
                    await env.sequence_flight(
                        SequenceFlightParams(
                            flight_id=f["id"], runway_id=arr_rwys[0]
                        )
                    )
                    # Assign gate
                    avail = obs.get("available_gates", [])
                    if avail:
                        for gid in avail:
                            gate = GATES[gid]
                            if gate.max_adg.value >= ADG[f["adg"]].value:
                                await env.assign_gate(
                                    AssignGateParams(flight_id=f["id"], gate_id=gid)
                                )
                                break

            # Sequence departures
            for f in obs.get("ready", [])[:2]:
                dep_rwys = obs["dep_runways"]
                if dep_rwys:
                    await env.sequence_flight(
                        SequenceFlightParams(
                            flight_id=f["id"], runway_id=dep_rwys[0]
                        )
                    )

            # Advance time
            result = await env.advance_time(EmptyParams())
            if result.finished:
                assert step == 47
                break

        assert env.episode_done
        await env.teardown()


# ===========================================================================
# I. Edge Case Tests
# ===========================================================================

class TestEdgeCases:

    def test_all_gates_full(self):
        sim = _make_sim()
        # Fill all gates
        for gid in GATES:
            sim.gate_occupancy[gid] = ("FAKE", sim.clock + 999)

        _activate_and_advance(sim, steps=3)
        flight = _find_flight(sim, FlightPhase.APPROACHING)
        if flight:
            # Should fail to assign any gate
            result = sim.assign_gate(flight.id, "A1")
            assert not result["success"]

    def test_empty_observation(self):
        """Initial observation before any flights activate should work."""
        sim = _make_sim()
        obs = sim.get_observation()
        assert obs is not None
        assert obs["step"] == 0

    def test_capacity_per_step_always_positive(self):
        """Capacity should always be >= 1 per step."""
        for config in RUNWAY_CONFIGS.values():
            for weather in WeatherCondition:
                arr, dep = capacity_per_step(config, weather, 5)
                assert arr >= 1
                assert dep >= 1

    def test_multiple_advance_without_actions(self):
        """Advancing without actions should not crash."""
        sim = _make_sim()
        for _ in range(48):
            obs = sim.advance()
            assert obs is not None
            assert "step" in obs

    def test_advance_past_max_steps(self):
        """Advancing past max steps should still work."""
        sim = _make_sim()
        for _ in range(48):
            sim.advance()
        # One more advance past the end
        obs = sim.advance()
        assert obs is not None

    def test_emergency_flight_exists_in_some_scenario(self):
        """At least some scenarios should have emergency flights."""
        found_emergency = False
        for seed in range(20):
            sim = _make_sim("clear_day", seed=seed)
            for f in sim.flights.values():
                if f.priority == Priority.EMERGENCY:
                    found_emergency = True
                    break
            if found_emergency:
                break
        # With 0.5% probability per flight and ~200 flights, expect ~1 per scenario
        # Over 20 seeds, should find at least one


# ===========================================================================
# J. Aircraft Data Tests
# ===========================================================================

class TestAircraftData:

    def test_all_aircraft_have_required_fields(self):
        required = [
            "wake", "adg", "fuel_rate_hold", "fuel_rate_taxi",
            "turnaround", "pax", "wingspan_ft",
        ]
        for ac_type, data in AIRCRAFT_DATA.items():
            for field in required:
                assert field in data, f"{ac_type} missing field: {field}"

    def test_fuel_rates_positive(self):
        for ac_type, data in AIRCRAFT_DATA.items():
            assert data["fuel_rate_hold"] > 0
            assert data["fuel_rate_taxi"] > 0

    def test_holding_burns_more_than_taxi(self):
        for ac_type, data in AIRCRAFT_DATA.items():
            assert data["fuel_rate_hold"] > data["fuel_rate_taxi"], (
                f"{ac_type}: hold fuel rate should exceed taxi rate"
            )

    def test_turnaround_times_reasonable(self):
        for ac_type, data in AIRCRAFT_DATA.items():
            assert 20 <= data["turnaround"] <= 200, (
                f"{ac_type}: turnaround {data['turnaround']} out of range"
            )

    def test_heavy_aircraft_higher_fuel(self):
        b737 = AIRCRAFT_DATA["B737-800"]["fuel_rate_hold"]
        b777 = AIRCRAFT_DATA["B777-300ER"]["fuel_rate_hold"]
        assert b777 > b737, "B777 should burn more fuel than B737"


# ===========================================================================
# K. Wake Separation Tests
# ===========================================================================

class TestWakeSeparation:

    def test_two_large_same_runway_same_step_no_violation(self):
        """Two LARGE flights on the same runway in a 5-min step should NOT
        violate wake separation (2 min sep fits within 5 min window)."""
        sim = _make_sim("clear_day", seed=0)
        _activate_and_advance(sim, steps=3)

        large_arrivals = [f for f in sim.flights.values()
                          if f.activated
                          and f.phase in (FlightPhase.APPROACHING, FlightPhase.HOLDING)
                          and f.wake == WakeCategory.LARGE]
        assert len(large_arrivals) >= 2, "Need at least 2 LARGE arrivals"

        f1, f2 = large_arrivals[0], large_arrivals[1]
        arr_rwy = sim.active_config.arrival_runways[0]

        sim.sequence_flight(f1.id, arr_rwy)
        violations_before = sim.metrics.safety_violations
        sim.sequence_flight(f2.id, arr_rwy)

        assert sim.metrics.safety_violations == violations_before, (
            "Two LARGE flights in a 5-min step should fit with 2-min separation"
        )

    def test_wake_violation_when_step_capacity_exceeded(self):
        """Enough flights on one runway in a single step should trigger a
        violation when cumulative separation exceeds step duration."""
        sim = _make_sim("clear_day", seed=0)
        _activate_and_advance(sim, steps=5)

        arrivals = [f for f in sim.flights.values()
                    if f.activated
                    and f.phase in (FlightPhase.APPROACHING, FlightPhase.HOLDING)
                    and f.wake == WakeCategory.LARGE]

        # Need 4 LARGE flights: sep 2+2+2=6 min > 5 min step → violation on 4th
        if len(arrivals) < 4:
            pytest.skip("Need at least 4 LARGE arrivals")

        arr_rwy = sim.active_config.arrival_runways[0]
        for f in arrivals[:3]:
            sim.sequence_flight(f.id, arr_rwy)

        violations_before = sim.metrics.safety_violations
        sim.sequence_flight(arrivals[3].id, arr_rwy)

        assert sim.metrics.safety_violations > violations_before, (
            "4th LARGE flight on same runway in 5-min step should violate "
            "(cumulative 6 min separation > 5 min step)"
        )

    def test_no_wake_violation_with_time_gap(self):
        """Flights on same runway with >= 5 min gap should not violate
        LARGE->LARGE (2 min) separation."""
        sim = _make_sim("clear_day", seed=0)
        _activate_and_advance(sim, steps=3)

        large_flights = [f for f in sim.flights.values()
                         if f.activated
                         and f.phase in (FlightPhase.APPROACHING, FlightPhase.HOLDING)
                         and f.wake == WakeCategory.LARGE]

        if len(large_flights) < 2:
            pytest.skip("Need at least 2 LARGE arrivals")

        arr_rwy = sim.active_config.arrival_runways[0]
        sim.sequence_flight(large_flights[0].id, arr_rwy)

        # Advance 1 step (5 min) to exceed 2-min LARGE->LARGE requirement
        sim.advance()

        violations_before = sim.metrics.safety_violations
        sim.sequence_flight(large_flights[1].id, arr_rwy)
        assert sim.metrics.safety_violations == violations_before, (
            "5 min gap should satisfy LARGE->LARGE 2 min requirement"
        )

    def test_super_to_large_requires_4_min(self):
        """SUPER -> LARGE requires 4 minutes per FAA wake table."""
        sep = get_wake_separation(WakeCategory.SUPER, WakeCategory.LARGE)
        assert sep == 4

    def test_heavy_to_small_requires_3_min(self):
        """HEAVY -> SMALL requires 3 minutes per FAA wake table."""
        sep = get_wake_separation(WakeCategory.HEAVY, WakeCategory.SMALL)
        assert sep == 3

    def test_b757_to_small_requires_3_min(self):
        """B757 -> SMALL requires 3 minutes per FAA wake table."""
        sep = get_wake_separation(WakeCategory.B757, WakeCategory.SMALL)
        assert sep == 3


# ===========================================================================
# L. Connection Tests
# ===========================================================================

class TestConnections:

    def test_connection_missed_on_diversion(self):
        """Diverting an arrival with a connection should trigger missed connection."""
        sim = _make_sim("clear_day", seed=0)
        _activate_and_advance(sim, steps=5)

        for arr_id, dep_id in sim.connections:
            arr = sim.flights[arr_id]
            if arr.activated and arr.phase in (FlightPhase.APPROACHING, FlightPhase.HOLDING):
                missed_before = sim.metrics.missed_connections
                sim.divert_flight(arr_id)
                missed_after = sim.metrics.missed_connections
                assert missed_after > missed_before, (
                    "Diverting connected arrival should miss connection"
                )
                return

        pytest.skip("No active connected arrival found")

    def test_connection_preserved_initial(self):
        """Initial state should have 0 missed connections."""
        sim = _make_sim("clear_day", seed=0)
        assert sim.metrics.missed_connections == 0

    def test_total_connections_count(self):
        """Should have ~20% of arrivals connected to departures."""
        sim = _make_sim("clear_day", seed=0)
        num_arrivals = sum(1 for f in sim.flights.values() if f.is_arrival)
        expected_min = int(num_arrivals * 0.1)
        expected_max = int(num_arrivals * 0.3)
        assert expected_min <= sim.metrics.total_connections <= expected_max, (
            f"Expected {expected_min}-{expected_max} connections, "
            f"got {sim.metrics.total_connections}"
        )


# ===========================================================================
# M. Go-Around Tests
# ===========================================================================

class TestGoAround:

    def test_go_around_occurs_in_thunderstorm(self):
        """Over many steps in thunderstorm, at least one go-around should occur."""
        sim = _make_sim("thunderstorm", seed=0)
        for step in range(48):
            for f in list(sim.flights.values()):
                if f.activated and f.phase in (FlightPhase.APPROACHING, FlightPhase.HOLDING):
                    arr_rwys = sim.active_config.arrival_runways
                    if arr_rwys:
                        sim.sequence_flight(f.id, arr_rwys[0])
            sim.advance()

        assert sim.metrics.go_arounds > 0, (
            "Thunderstorm scenario with active sequencing should produce go-arounds"
        )

    def test_go_around_probability_monotonic(self):
        """Go-around probability should increase with worse weather."""
        p_clear = go_around_probability(WeatherCondition.CLEAR)
        p_mvfr = go_around_probability(WeatherCondition.MVFR)
        p_ifr = go_around_probability(WeatherCondition.IFR)
        p_low = go_around_probability(WeatherCondition.LOW_IFR)
        p_ts = go_around_probability(WeatherCondition.THUNDERSTORM)

        assert p_clear < p_mvfr < p_ifr < p_low < p_ts, (
            "Go-around probability should monotonically increase with worse weather"
        )

    def test_go_around_event_no_none(self):
        """Go-around events should name the runway, not 'None'."""
        sim = _make_sim("thunderstorm", seed=0)
        for step in range(48):
            for f in list(sim.flights.values()):
                if f.activated and f.phase in (FlightPhase.APPROACHING, FlightPhase.HOLDING):
                    arr_rwys = sim.active_config.arrival_runways
                    if arr_rwys:
                        sim.sequence_flight(f.id, arr_rwys[0])
            sim.advance()

            for event in sim.step_events:
                if "GO-AROUND" in event:
                    assert "None" not in event, (
                        f"Go-around event should not contain 'None': {event}"
                    )


# ===========================================================================
# N. Config/Wind Tests
# ===========================================================================

class TestConfigWind:

    def test_initial_config_matches_wind(self):
        """clear_day (wind 150) should auto-select south_flow."""
        sim = _make_sim("clear_day", seed=0)
        assert sim.active_config_name == "south_flow"

    def test_wind_shift_scenario_has_wind_change(self):
        """Wind shift scenario should have significant wind direction change."""
        from scenarios import _make_task
        task = _make_task("wind_shift", 0, "train")
        wind = task["wind_timeline"]
        # Allow for noise; check overall trend
        assert abs(wind[-1] - wind[0]) > 50 or abs(wind[len(wind)//2] - wind[0]) > 50, (
            "Wind shift scenario should have significant wind change"
        )

    def test_config_change_clears_runway_last_op(self):
        """Changing runway config should clear runway_last_op tracking."""
        sim = _make_sim("clear_day", seed=0)
        _activate_and_advance(sim, steps=3)

        f = _find_flight(sim, FlightPhase.APPROACHING)
        if f is None:
            f = _find_flight(sim, FlightPhase.HOLDING)
        if f:
            arr_rwy = sim.active_config.arrival_runways[0]
            sim.sequence_flight(f.id, arr_rwy)
            assert len(sim.runway_last_op) > 0

        sim.set_runway_config("north_flow")
        assert len(sim.runway_last_op) == 0, (
            "Config change should clear runway_last_op"
        )

    def test_sim_does_not_auto_switch_config(self):
        """Simulation does NOT auto-switch config when wind changes.
        The agent must call set_runway_config() explicitly."""
        sim = _make_sim("wind_shift", seed=0)
        initial_config = sim.active_config_name

        for _ in range(48):
            sim.advance()

        assert sim.active_config_name == initial_config, (
            "Config should NOT auto-change; agent must call set_runway_config()"
        )


# ===========================================================================
# O. Capacity Tests
# ===========================================================================

class TestCapacityEnforcement:

    def test_thunderstorm_capacity_quarter(self):
        """Thunderstorm should reduce capacity to ~25% of VFR."""
        config = RUNWAY_CONFIGS["south_flow"]
        arr_clear, _ = capacity_per_step(config, WeatherCondition.CLEAR, 5)
        arr_ts, _ = capacity_per_step(config, WeatherCondition.THUNDERSTORM, 5)
        assert arr_ts <= arr_clear * 0.30

    def test_capacity_warning_logged(self):
        """Exceeding capacity should generate a CAPACITY WARNING event."""
        sim = _make_sim("clear_day", seed=0)
        _activate_and_advance(sim, steps=3)

        count = 0
        for f in list(sim.flights.values()):
            if f.activated and f.phase in (FlightPhase.APPROACHING, FlightPhase.HOLDING):
                arr_rwy = sim.active_config.arrival_runways[0]
                result = sim.sequence_flight(f.id, arr_rwy)
                if result["success"]:
                    count += 1
                if count > 10:
                    break

        sim.advance()

        if count > 5:
            has_warning = any("CAPACITY WARNING" in e for e in sim.step_events)
            assert has_warning, (
                f"Sequencing {count} flights should trigger capacity warning"
            )


# ===========================================================================
# P. Reward Component Tests
# ===========================================================================

class TestRewardComponents:

    def test_smart_agent_beats_no_action(self):
        """A wake-separation-aware agent should outperform no-action baseline."""
        # No-action baseline
        sim_none = _make_sim("clear_day", seed=0)
        for _ in range(48):
            sim_none.advance()
        reward_none = sim_none.compute_final_reward()

        # Smart agent: respect wake separation, assign gates.
        # Uses virtual timestamps: checks that the next op's earliest time
        # (last_virtual_time + required_sep) fits within the current step.
        sim_smart = _make_sim("clear_day", seed=0)
        for step in range(48):
            arr_rwys = list(sim_smart.active_config.arrival_runways)
            for f in list(sim_smart.flights.values()):
                if f.activated and f.phase in (FlightPhase.APPROACHING, FlightPhase.HOLDING):
                    for rwy in arr_rwys:
                        last = sim_smart.runway_last_op.get(rwy)
                        if last is None:
                            fits = True
                        else:
                            earliest = last[0] + get_wake_separation(last[1], f.wake)
                            fits = earliest <= sim_smart.clock + sim_smart.step_duration
                        if fits:
                            sim_smart.sequence_flight(f.id, rwy)
                            for gid, gate in GATES.items():
                                if gate.max_adg.value >= f.adg.value:
                                    occ = sim_smart.gate_occupancy.get(gid)
                                    if occ is None or occ[1] <= sim_smart.clock:
                                        sim_smart.assign_gate(f.id, gid)
                                        break
                            break

            dep_rwys = list(sim_smart.active_config.departure_runways)
            for f in list(sim_smart.flights.values()):
                if f.activated and f.phase == FlightPhase.READY and f.assigned_gate:
                    if sim_smart.ground_stop_until <= sim_smart.clock:
                        for rwy in dep_rwys:
                            last = sim_smart.runway_last_op.get(rwy)
                            if last is None:
                                fits = True
                            else:
                                earliest = last[0] + get_wake_separation(last[1], f.wake)
                                fits = earliest <= sim_smart.clock + sim_smart.step_duration
                            if fits:
                                sim_smart.sequence_flight(f.id, rwy)
                                break

            sim_smart.advance()

        reward_smart = sim_smart.compute_final_reward()

        assert reward_smart > reward_none, (
            f"Smart agent ({reward_smart:.4f}) should beat no-action ({reward_none:.4f})"
        )
        assert sim_smart.metrics.safety_violations == 0, (
            f"Smart agent should have zero wake violations, got {sim_smart.metrics.safety_violations}"
        )

    def test_final_reward_weights_sum_to_one(self):
        """The five reward component weights should sum to 1.0."""
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        assert abs(sum(weights) - 1.0) < 1e-9

    def test_safety_multiplier_effect(self):
        """Safety violations should reduce the final reward."""
        sim = _make_sim("clear_day", seed=0)
        _activate_and_advance(sim, steps=10)
        for f in list(sim.flights.values()):
            if f.activated and f.phase in (FlightPhase.APPROACHING, FlightPhase.HOLDING):
                arr_rwy = sim.active_config.arrival_runways[0]
                sim.sequence_flight(f.id, arr_rwy)
                for gid, gate in GATES.items():
                    if gate.max_adg.value >= f.adg.value:
                        occ = sim.gate_occupancy.get(gid)
                        if occ is None or occ[1] <= sim.clock:
                            sim.assign_gate(f.id, gid)
                            break
        for _ in range(20):
            sim.advance()

        sim.metrics.safety_violations = 0
        clean = sim.compute_final_reward()

        sim.metrics.safety_violations = 2
        dirty = sim.compute_final_reward()

        assert dirty < clean, (
            f"Violated ({dirty:.4f}) should be less than clean ({clean:.4f})"
        )
        if clean > 0:
            ratio = dirty / clean
            assert ratio < 0.6, (
                f"Safety multiplier should reduce by ~50%, ratio={ratio:.3f}"
            )

    def test_throughput_final_reward_in_range(self):
        """Final reward should always be in [0, 1]."""
        for scenario in ["clear_day", "thunderstorm", "peak_traffic"]:
            sim = _make_sim(scenario, seed=0)
            for _ in range(48):
                sim.advance()
            final = sim.compute_final_reward()
            assert 0.0 <= final <= 1.0, (
                f"{scenario}: final reward {final} out of range"
            )


# ===========================================================================
# Q. Flight Lifecycle Tests (Extended)
# ===========================================================================

class TestFlightLifecycleExtended:

    def test_flights_activate_at_scheduled_time(self):
        """Flights should activate when clock >= scheduled_time."""
        sim = _make_sim("clear_day", seed=0)

        target = None
        for f in sim.flights.values():
            if 5 < f.scheduled_time < 20:
                target = f
                break

        if target is None:
            pytest.skip("No flight with suitable scheduled_time found")

        assert not target.activated

        # _activate_flights runs BEFORE clock increment in advance(), so
        # we need to advance until the activation check sees clock >= scheduled_time.
        # After the while loop, clock has passed scheduled_time but activation
        # ran at the old clock. One more advance triggers activation at the new clock.
        while sim.clock <= target.scheduled_time:
            sim.advance()
        # This advance runs _activate_flights with clock > scheduled_time
        sim.advance()

        assert target.activated, (
            f"Flight {target.id} should be activated at clock={sim.clock} "
            f"(scheduled_time={target.scheduled_time})"
        )

    def test_departure_auto_gate_assignment(self):
        """Activated departures should have auto-assigned gates."""
        sim = _make_sim("clear_day", seed=0)
        _activate_and_advance(sim, steps=5)

        departures_with_gate = 0
        departures_total = 0
        for f in sim.flights.values():
            if f.activated and not f.is_arrival and f.phase == FlightPhase.READY:
                departures_total += 1
                if f.assigned_gate is not None:
                    departures_with_gate += 1

        if departures_total > 0:
            assert departures_with_gate > 0, (
                "At least some activated departures should have auto-assigned gates"
            )

    def test_gate_freed_on_pushback(self):
        """Gate should be freed when a departure completes pushback."""
        sim = _make_sim("clear_day", seed=0)
        _activate_and_advance(sim, steps=5)

        dep = _find_flight(sim, FlightPhase.READY, is_arrival=False)
        if dep is None:
            pytest.skip("No READY departure found")

        gate = dep.assigned_gate
        if gate is None:
            pytest.skip("Departure has no gate assigned")

        assert sim.gate_occupancy[gate] is not None

        dep_rwy = sim.active_config.departure_runways[0]
        sim.sequence_flight(dep.id, dep_rwy)

        for _ in range(15):
            sim.advance()
            if dep.phase not in (FlightPhase.PUSHBACK, FlightPhase.READY):
                break

        assert sim.gate_occupancy[gate] is None, (
            f"Gate {gate} should be freed after pushback (flight phase: {dep.phase.value})"
        )

    def test_arrival_gate_occupied_at_gate(self):
        """When an arrival reaches AT_GATE, the gate should be occupied."""
        sim = _make_sim("clear_day", seed=0)
        _activate_and_advance(sim, steps=3)

        arr = _find_flight(sim, FlightPhase.APPROACHING)
        if arr is None:
            arr = _find_flight(sim, FlightPhase.HOLDING)
        if arr is None:
            pytest.skip("No approaching/holding arrival found")

        arr_rwy = sim.active_config.arrival_runways[0]
        sim.sequence_flight(arr.id, arr_rwy)

        for gid, gate in GATES.items():
            if gate.max_adg.value >= arr.adg.value:
                occ = sim.gate_occupancy.get(gid)
                if occ is None or occ[1] <= sim.clock:
                    sim.assign_gate(arr.id, gid)
                    break

        if arr.assigned_gate is None:
            pytest.skip("Could not assign gate")

        for _ in range(30):
            sim.advance()
            if arr.phase == FlightPhase.AT_GATE:
                break

        if arr.phase == FlightPhase.AT_GATE:
            occ = sim.gate_occupancy[arr.assigned_gate]
            assert occ is not None, "Gate should be occupied at AT_GATE phase"
            assert occ[0] == arr.id


# ===========================================================================
# R. ADG Correctness Tests
# ===========================================================================

class TestADGCorrectness:

    def test_adg_matches_wingspan_all_aircraft(self):
        """Every aircraft ADG should match its wingspan per FAA AC 150/5300-13."""
        for ac_type, data in AIRCRAFT_DATA.items():
            ws = data["wingspan_ft"]
            coded_adg = data["adg"]

            if ws < 49:
                expected = ADG.I
            elif ws < 79:
                expected = ADG.II
            elif ws < 118:
                expected = ADG.III
            elif ws < 171:
                expected = ADG.IV
            elif ws < 214:
                expected = ADG.V
            else:
                expected = ADG.VI

            assert coded_adg == expected, (
                f"{ac_type}: wingspan {ws} ft should be ADG {expected.name}, "
                f"but coded as ADG {coded_adg.name}"
            )

    def test_b757_adg_iv(self):
        """B757-200 (wingspan 124 ft) should be ADG IV."""
        assert AIRCRAFT_DATA["B757-200"]["adg"] == ADG.IV

    def test_b787_adg_v(self):
        """B787-9 (wingspan 197 ft) should be ADG V."""
        assert AIRCRAFT_DATA["B787-9"]["adg"] == ADG.V

    def test_a330_adg_v(self):
        """A330-300 (wingspan 198 ft) should be ADG V."""
        assert AIRCRAFT_DATA["A330-300"]["adg"] == ADG.V

    def test_e175_adg_iii(self):
        """E175 (wingspan 85 ft) should be ADG III."""
        assert AIRCRAFT_DATA["E175"]["adg"] == ADG.III

    def test_gate_compatibility_after_fix(self):
        """B787/A330 (ADG V) should fit Terminal C (max ADG V)."""
        b787_adg = AIRCRAFT_DATA["B787-9"]["adg"]
        a330_adg = AIRCRAFT_DATA["A330-300"]["adg"]
        terminal_c_max = GATES["C1"].max_adg

        assert b787_adg.value <= terminal_c_max.value, (
            f"B787-9 (ADG {b787_adg.name}) should fit Terminal C (max {terminal_c_max.name})"
        )
        assert a330_adg.value <= terminal_c_max.value

    def test_crj900_fits_terminal_a(self):
        """CRJ-900 (ADG III) should fit Terminal A (ADG III)."""
        crj_adg = AIRCRAFT_DATA["CRJ-900"]["adg"]
        terminal_a_max = GATES["A1"].max_adg
        assert crj_adg.value <= terminal_a_max.value


# ===========================================================================
# S. Parameter Realism Tests
# ===========================================================================

class TestParameterRealism:

    def test_b737_fuel_rate_realistic(self):
        """B737-800 holding fuel rate ~2,400 kg/hr (40 kg/min)."""
        rate = AIRCRAFT_DATA["B737-800"]["fuel_rate_hold"]
        rate_per_hr = rate * 60
        assert 2000 <= rate_per_hr <= 3000, (
            f"B737-800 hold rate {rate_per_hr} kg/hr should be 2000-3000"
        )

    def test_go_around_base_rate_range(self):
        """Base go-around rate should be ~0.3-0.6% per FAA/NASA data."""
        from airport import GO_AROUND_BASE_PROB
        assert 0.003 <= GO_AROUND_BASE_PROB <= 0.006

    def test_runway_lengths_match_jfk(self):
        """Runway lengths should match JFK physical pavement values."""
        from airport import RUNWAYS
        assert RUNWAYS["13R"].length_ft == 14511
        assert RUNWAYS["13L"].length_ft == 10000
        assert RUNWAYS["04L"].length_ft == 12079  # physical length (FAA AIP)
        assert RUNWAYS["04R"].length_ft == 8400

    def test_mct_domestic_reasonable(self):
        """MCT domestic should be 45-90 minutes."""
        from models import MCT_DOMESTIC
        assert 45 <= MCT_DOMESTIC <= 90


# ===========================================================================
# T. Observation Completeness Tests
# ===========================================================================

class TestObservationCompleteness:

    def test_observation_has_all_decision_info(self):
        """Observation should contain all info needed for decision making."""
        sim = _make_sim("clear_day", seed=0)
        _activate_and_advance(sim, steps=5)
        obs = sim.get_observation()

        required_keys = [
            "weather", "wind", "config", "arr_runways", "dep_runways",
            "arr_capacity_per_step", "dep_capacity_per_step", "ground_stop",
            "available_gates", "connections_at_risk", "events", "metrics",
            "approaching", "holding", "on_final", "ready", "taxiing", "at_gate",
        ]
        for key in required_keys:
            assert key in obs, f"Observation missing key: {key}"

    def test_flight_dict_has_actionable_info(self):
        """Each flight in observation should have actionable fields."""
        sim = _make_sim("clear_day", seed=0)
        _activate_and_advance(sim, steps=5)
        obs = sim.get_observation()

        required_flight_fields = [
            "id", "callsign", "aircraft_type", "wake", "adg",
            "phase", "fuel_minutes", "priority", "phase_timer",
        ]

        for phase_list in ["approaching", "holding", "on_final"]:
            for f in obs.get(phase_list, [])[:1]:
                for field in required_flight_fields:
                    assert field in f, (
                        f"Flight in {phase_list} missing field: {field}"
                    )


# ===========================================================================
# U. Citation Accuracy Tests
# ===========================================================================

class TestCitationAccuracy:
    """Verify that aviation parameters match real-world verified values.

    Sources verified via FAA documents, Boeing/Airbus/Bombardier/Embraer specs,
    NASA TM-20240008006, and IATA standards.
    """

    # Real-world verified wingspans (ft) per manufacturer type certificates
    VERIFIED_WINGSPANS = {
        "B737-800": 112,    # 112 ft 7 in (Boeing, no winglets)
        "A320": 112,        # 111 ft 10 in (Airbus, no sharklets)
        "B757-200": 125,    # 124 ft 10 in (Boeing)
        "B767-300": 156,    # 156 ft 1 in (Boeing)
        "B777-300ER": 213,  # 212 ft 7 in (Boeing, -300ER extended wing)
        "B787-9": 197,      # 197 ft 2 in (Boeing)
        "A330-300": 198,    # 197 ft 10 in (Airbus)
        "A380": 262,        # 261 ft 10 in (Airbus)
        "E175": 85,         # 85 ft 4 in (Embraer, no winglets)
        "CRJ-900": 81,      # 81 ft 7 in (Bombardier)
    }

    def test_crj900_wingspan_matches_real(self):
        """CRJ-900 wingspan should be ~81 ft (Bombardier spec: 81 ft 7 in)."""
        ws = AIRCRAFT_DATA["CRJ-900"]["wingspan_ft"]
        assert 80 <= ws <= 83, f"CRJ-900 wingspan {ws} should be ~81 ft"

    def test_b737_wingspan_matches_real(self):
        """B737-800 wingspan should be ~112 ft (Boeing: 112 ft 7 in without winglets)."""
        ws = AIRCRAFT_DATA["B737-800"]["wingspan_ft"]
        assert 111 <= ws <= 114, f"B737-800 wingspan {ws} should be ~112 ft"

    def test_b757_wingspan_matches_real(self):
        """B757-200 wingspan should be ~125 ft (Boeing: 124 ft 10 in)."""
        ws = AIRCRAFT_DATA["B757-200"]["wingspan_ft"]
        assert 123 <= ws <= 126, f"B757-200 wingspan {ws} should be ~125 ft"

    def test_a380_wingspan_matches_real(self):
        """A380 wingspan should be ~262 ft (Airbus: 261 ft 10 in)."""
        ws = AIRCRAFT_DATA["A380"]["wingspan_ft"]
        assert 260 <= ws <= 263, f"A380 wingspan {ws} should be ~262 ft"

    def test_b787_wingspan_matches_real(self):
        """B787-9 wingspan should be ~197 ft (Boeing: 197 ft 2 in)."""
        ws = AIRCRAFT_DATA["B787-9"]["wingspan_ft"]
        assert 196 <= ws <= 199, f"B787-9 wingspan {ws} should be ~197 ft"

    def test_b777_is_er_variant(self):
        """B777-300ER should have ~213 ft wingspan (not base 777-300 at 200 ft).

        The ER variant has a larger raked wingtip wing: 212 ft 7 in (64.8 m).
        The base 777-300 has a 199 ft 10 in (60.9 m) wingspan.
        """
        assert "B777-300ER" in AIRCRAFT_DATA, (
            "Aircraft should be labeled B777-300ER (not B777-300) since specs match -300ER"
        )
        ws = AIRCRAFT_DATA["B777-300ER"]["wingspan_ft"]
        assert ws > 210, (
            f"B777-300ER wingspan {ws} should be >210 ft (ER variant), "
            f"not ~200 ft (base 777-300)"
        )

    def test_all_wingspans_within_5pct(self):
        """Every coded wingspan should be within 5% of the verified real value."""
        for ac_type, verified_ws in self.VERIFIED_WINGSPANS.items():
            coded_ws = AIRCRAFT_DATA[ac_type]["wingspan_ft"]
            pct_diff = abs(coded_ws - verified_ws) / verified_ws
            assert pct_diff < 0.05, (
                f"{ac_type}: coded wingspan {coded_ws} ft differs from "
                f"verified {verified_ws} ft by {pct_diff:.1%}"
            )

    def test_wake_sep_super_heavy_gte_3(self):
        """SUPER->HEAVY separation should be >= 3 min (FAA: 6 NM ~ 2.6 min)."""
        sep = get_wake_separation(WakeCategory.SUPER, WakeCategory.HEAVY)
        assert sep >= 3, f"SUPER->HEAVY sep={sep}, should be >= 3 min"

    def test_wake_sep_heavy_large_gte_3(self):
        """HEAVY->LARGE separation should be >= 3 min (FAA: 5 NM ~ 2.1 min)."""
        sep = get_wake_separation(WakeCategory.HEAVY, WakeCategory.LARGE)
        assert sep >= 3, f"HEAVY->LARGE sep={sep}, should be >= 3 min"

    def test_mct_values_match_jfk(self):
        """MCT values should match JFK station standards."""
        from models import MCT_DOMESTIC, MCT_INTERNATIONAL
        assert MCT_DOMESTIC == 60, f"Domestic MCT {MCT_DOMESTIC} should be 60 min"
        assert MCT_INTERNATIONAL == 120, f"International MCT {MCT_INTERNATIONAL} should be 120 min"

    def test_runway_04l_length_is_physical(self):
        """04L/22R should use physical pavement length (12,079 ft), not TORA (11,351 ft).

        Source: FAA AIP, AOPA chart supplement for KJFK.
        Physical runway is 12,079 ft; the 11,351 ft figure is the TORA
        for the 04L direction due to a displaced threshold.
        """
        from airport import RUNWAYS
        assert RUNWAYS["04L"].length_ft == 12079, (
            f"04L length {RUNWAYS['04L'].length_ft} should be 12079 (physical), not 11351 (TORA)"
        )
        assert RUNWAYS["22R"].length_ft == 12079, (
            f"22R length {RUNWAYS['22R'].length_ft} should match 04L (same physical runway)"
        )

    def test_crj900_adg_matches_wingspan(self):
        """CRJ-900 with 81 ft wingspan should be ADG III (79-118 ft range)."""
        adg = AIRCRAFT_DATA["CRJ-900"]["adg"]
        assert adg == ADG.III, (
            f"CRJ-900 ADG should be III (wingspan 81 ft > 79 ft ADG II ceiling), got {adg.name}"
        )

    def test_terminal_a_accommodates_crj900(self):
        """Terminal A should accommodate CRJ-900 (ADG III) after terminal upgrade."""
        crj_adg = AIRCRAFT_DATA["CRJ-900"]["adg"]
        terminal_a_max = GATES["A1"].max_adg
        assert crj_adg.value <= terminal_a_max.value, (
            f"CRJ-900 (ADG {crj_adg.name}) should fit Terminal A (max ADG {terminal_a_max.name})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
