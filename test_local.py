"""Local integration test: runs a complete episode with a greedy heuristic
agent and logs the trajectory to a .jsonl file for inspection.

This test does NOT require the OpenReward platform or OpenAI API.
"""

import asyncio
import json
import time

from atc import (
    ATCEnvironment,
    AssignGateParams,
    DivertFlightParams,
    EmptyParams,
    HoldFlightParams,
    SequenceFlightParams,
    SetRunwayConfigParams,
)
from airport import GATES, RUNWAY_CONFIGS
from models import ADG, FlightPhase, WeatherCondition
from scenarios import _make_task


async def run_episode(scenario: str, seed: int, trajectory_path: str):
    """Run a full episode with a greedy heuristic controller."""
    task = _make_task(scenario, seed, "train")
    env = ATCEnvironment(task_spec=task)
    await env.setup()

    trajectory = open(trajectory_path, "w")

    # Get prompt
    prompt = await env.get_prompt()
    trajectory.write(json.dumps({
        "type": "prompt",
        "scenario": scenario,
        "seed": seed,
        "content_length": len(prompt[0].text),
        "timestamp": time.time(),
    }) + "\n")

    turn = 0
    finished = False
    total_reward = 0.0

    while not finished and turn < 500:
        # View status
        status = await env.view_status(EmptyParams())
        obs = status.metadata

        actions_taken = []

        # --- HEURISTIC POLICY ---

        # 1. Check if config matches wind
        wind = obs.get("wind", 150)
        current_config = obs.get("config", "south_flow")
        best_config = _best_config_for_wind(wind, obs.get("weather", "CLEAR"))
        if best_config != current_config:
            result = await env.set_runway_config(
                SetRunwayConfigParams(config_name=best_config)
            )
            actions_taken.append({"tool": "set_runway_config", "args": {"config_name": best_config}})
            # Re-fetch status after config change
            status = await env.view_status(EmptyParams())
            obs = status.metadata

        # 2. Handle emergency flights first
        for f in obs.get("approaching", []) + obs.get("holding", []):
            if f["priority"] == "EMERGENCY":
                arr_rwys = obs["arr_runways"]
                if arr_rwys:
                    r = await env.sequence_flight(
                        SequenceFlightParams(flight_id=f["id"], runway_id=arr_rwys[0])
                    )
                    actions_taken.append({
                        "tool": "sequence_flight",
                        "args": {"flight_id": f["id"], "runway_id": arr_rwys[0]},
                        "reason": "emergency",
                    })
                    _assign_best_gate(env, f, obs, actions_taken)

        # 3. Divert flights with very low fuel
        for f in obs.get("holding", []):
            if f["fuel_minutes"] < 20:
                r = await env.divert_flight(
                    DivertFlightParams(flight_id=f["id"])
                )
                actions_taken.append({
                    "tool": "divert_flight",
                    "args": {"flight_id": f["id"]},
                    "reason": "low_fuel",
                })

        # 4. Sequence approaching flights (up to capacity)
        arr_cap = obs.get("arr_capacity_per_step", 5)
        arr_rwys = obs.get("arr_runways", [])
        approaching = obs.get("approaching", [])
        holding = obs.get("holding", [])

        # Prioritize holding flights (they're burning fuel)
        to_sequence_arr = holding + approaching
        sequenced_count = 0

        for f in to_sequence_arr:
            if sequenced_count >= arr_cap:
                break
            if f["phase"] not in ("APPROACHING", "HOLDING"):
                continue
            # Use round-robin across arrival runways
            rwy = arr_rwys[sequenced_count % len(arr_rwys)] if arr_rwys else None
            if rwy:
                r = await env.sequence_flight(
                    SequenceFlightParams(flight_id=f["id"], runway_id=rwy)
                )
                if r.metadata.get("success"):
                    sequenced_count += 1
                    actions_taken.append({
                        "tool": "sequence_flight",
                        "args": {"flight_id": f["id"], "runway_id": rwy},
                    })
                    await _assign_best_gate_async(env, f, obs, actions_taken)

        # 5. Sequence departures (up to capacity)
        dep_cap = obs.get("dep_capacity_per_step", 3)
        dep_rwys = obs.get("dep_runways", [])
        ready = obs.get("ready", [])
        dep_count = 0

        for f in ready:
            if dep_count >= dep_cap:
                break
            if f["phase"] != "READY":
                continue
            rwy = dep_rwys[dep_count % len(dep_rwys)] if dep_rwys else None
            if rwy:
                r = await env.sequence_flight(
                    SequenceFlightParams(flight_id=f["id"], runway_id=rwy)
                )
                if r.metadata.get("success"):
                    dep_count += 1
                    actions_taken.append({
                        "tool": "sequence_flight",
                        "args": {"flight_id": f["id"], "runway_id": rwy},
                        "type": "departure",
                    })

        # 6. Advance time
        result = await env.advance_time(EmptyParams())
        reward = result.reward if result.reward else 0
        total_reward += reward
        finished = result.finished

        # Log to trajectory
        trajectory.write(json.dumps({
            "type": "step",
            "turn": turn,
            "step": obs["step"],
            "clock": obs["clock"],
            "weather": obs["weather"],
            "wind": obs["wind"],
            "config": obs["config"],
            "num_approaching": len(obs.get("approaching", [])),
            "num_holding": len(obs.get("holding", [])),
            "num_on_final": len(obs.get("on_final", [])),
            "num_ready": len(obs.get("ready", [])),
            "num_at_gate": len(obs.get("at_gate", [])),
            "connections_at_risk": len(obs.get("connections_at_risk", [])),
            "actions": actions_taken,
            "step_reward": reward,
            "total_reward": total_reward,
            "finished": finished,
            "metrics": obs.get("metrics", {}),
            "events": obs.get("events", []),
            "timestamp": time.time(),
        }) + "\n")
        trajectory.flush()

        print(
            f"Step {obs['step']:3d} | "
            f"Wx: {obs['weather']:12s} | "
            f"Arr: {len(approaching):2d} Hold: {len(holding):2d} "
            f"Ready: {len(ready):2d} | "
            f"Actions: {len(actions_taken):2d} | "
            f"Reward: {reward:+.4f} | "
            f"Total: {total_reward:+.4f}"
        )

        turn += 1

    # Final summary
    if not env.episode_done:
        final = await env.end_shift(EmptyParams())
        total_reward = final.reward
        finished = True

    summary = {
        "type": "summary",
        "scenario": scenario,
        "seed": seed,
        "total_turns": turn,
        "total_reward": total_reward,
        "finished": finished,
        "timestamp": time.time(),
    }
    trajectory.write(json.dumps(summary) + "\n")
    trajectory.close()

    await env.teardown()
    return total_reward


def _best_config_for_wind(wind: int, weather: str) -> str:
    """Select runway config based on wind and weather."""
    # If thunderstorm, use IFR config
    if weather in ("THUNDERSTORM", "LOW_IFR"):
        return "ifr_south"
    if weather == "IFR":
        return "ifr_south"

    # Match wind to config
    for name, cfg in RUNWAY_CONFIGS.items():
        if name in ("emergency", "ifr_south"):
            continue
        lo, hi = cfg.wind_range
        if lo <= hi:
            if lo <= wind <= hi:
                return name
        else:
            if wind >= lo or wind <= hi:
                return name
    return "south_flow"


async def _assign_best_gate_async(env, f, obs, actions_taken):
    """Assign the best available gate to a flight."""
    avail = obs.get("available_gates", [])
    try:
        adg_val = ADG[f["adg"]].value
    except (KeyError, TypeError):
        return

    for gid in avail:
        gate = GATES.get(gid)
        if gate and gate.max_adg.value >= adg_val:
            result = await env.assign_gate(
                AssignGateParams(flight_id=f["id"], gate_id=gid)
            )
            if result.metadata.get("success"):
                actions_taken.append({
                    "tool": "assign_gate",
                    "args": {"flight_id": f["id"], "gate_id": gid},
                })
            break


def _assign_best_gate(env, f, obs, actions_taken):
    """Sync wrapper — used for initial emergency handling."""
    pass  # handled in async version above


async def main():
    print("=" * 70)
    print("ATC ENVIRONMENT LOCAL TEST")
    print("=" * 70)

    scenarios_to_test = [
        ("clear_day", 0),
        ("thunderstorm", 0),
        ("morning_fog", 0),
        ("peak_traffic", 0),
    ]

    results = {}
    for scenario, seed in scenarios_to_test:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario} (seed={seed})")
        print(f"{'='*70}")

        traj_path = f"trajectory_{scenario}_{seed}.jsonl"
        reward = await run_episode(scenario, seed, traj_path)
        results[scenario] = reward
        print(f"\n  Final reward: {reward:.4f}")
        print(f"  Trajectory: {traj_path}")

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    for scenario, reward in results.items():
        print(f"  {scenario:20s}: {reward:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
