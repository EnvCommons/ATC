"""Scenario and task generation for the ATC environment.

Weather timelines are generated via Markov chains and frozen at generation
time for deterministic reproducibility.
"""

from __future__ import annotations

import numpy as np

from models import WeatherCondition

# ---------------------------------------------------------------------------
# Scenario types and their characteristics
# ---------------------------------------------------------------------------

SCENARIO_TYPES = [
    "clear_day",
    "morning_fog",
    "thunderstorm",
    "snow_event",
    "wind_shift",
    "peak_traffic",
    "compound_wx",
    "cascading_delay",
    "runway_closure",
    "crosswind",
]

# Weather Markov chain transition matrices per scenario.
# Each maps current_state -> {next_state: probability}.
_WX = WeatherCondition

WEATHER_TRANSITIONS: dict[str, dict[WeatherCondition, dict[WeatherCondition, float]]] = {
    "clear_day": {
        _WX.CLEAR: {_WX.CLEAR: 0.95, _WX.MVFR: 0.05},
        _WX.MVFR: {_WX.CLEAR: 0.70, _WX.MVFR: 0.30},
        _WX.IFR: {_WX.MVFR: 0.60, _WX.IFR: 0.40},
        _WX.LOW_IFR: {_WX.IFR: 0.50, _WX.LOW_IFR: 0.50},
        _WX.THUNDERSTORM: {_WX.IFR: 0.50, _WX.THUNDERSTORM: 0.50},
    },
    "morning_fog": {
        _WX.IFR: {_WX.IFR: 0.60, _WX.MVFR: 0.30, _WX.CLEAR: 0.10},
        _WX.MVFR: {_WX.CLEAR: 0.50, _WX.MVFR: 0.40, _WX.IFR: 0.10},
        _WX.CLEAR: {_WX.CLEAR: 0.95, _WX.MVFR: 0.05},
        _WX.LOW_IFR: {_WX.IFR: 0.60, _WX.LOW_IFR: 0.40},
        _WX.THUNDERSTORM: {_WX.IFR: 0.50, _WX.THUNDERSTORM: 0.50},
    },
    "thunderstorm": {
        _WX.CLEAR: {_WX.CLEAR: 0.60, _WX.MVFR: 0.30, _WX.IFR: 0.10},
        _WX.MVFR: {_WX.MVFR: 0.30, _WX.IFR: 0.40, _WX.THUNDERSTORM: 0.30},
        _WX.IFR: {_WX.IFR: 0.30, _WX.THUNDERSTORM: 0.50, _WX.MVFR: 0.20},
        _WX.THUNDERSTORM: {_WX.THUNDERSTORM: 0.60, _WX.IFR: 0.25, _WX.MVFR: 0.15},
        _WX.LOW_IFR: {_WX.IFR: 0.40, _WX.THUNDERSTORM: 0.40, _WX.LOW_IFR: 0.20},
    },
    "snow_event": {
        _WX.CLEAR: {_WX.CLEAR: 0.50, _WX.MVFR: 0.30, _WX.IFR: 0.20},
        _WX.MVFR: {_WX.MVFR: 0.40, _WX.IFR: 0.40, _WX.LOW_IFR: 0.20},
        _WX.IFR: {_WX.IFR: 0.50, _WX.LOW_IFR: 0.30, _WX.MVFR: 0.20},
        _WX.LOW_IFR: {_WX.LOW_IFR: 0.60, _WX.IFR: 0.30, _WX.MVFR: 0.10},
        _WX.THUNDERSTORM: {_WX.LOW_IFR: 0.50, _WX.THUNDERSTORM: 0.50},
    },
    "wind_shift": {
        _WX.CLEAR: {_WX.CLEAR: 0.80, _WX.MVFR: 0.20},
        _WX.MVFR: {_WX.CLEAR: 0.50, _WX.MVFR: 0.50},
        _WX.IFR: {_WX.MVFR: 0.40, _WX.IFR: 0.60},
        _WX.LOW_IFR: {_WX.IFR: 0.50, _WX.LOW_IFR: 0.50},
        _WX.THUNDERSTORM: {_WX.IFR: 0.50, _WX.THUNDERSTORM: 0.50},
    },
    "peak_traffic": {
        _WX.CLEAR: {_WX.CLEAR: 0.98, _WX.MVFR: 0.02},
        _WX.MVFR: {_WX.CLEAR: 0.80, _WX.MVFR: 0.20},
        _WX.IFR: {_WX.MVFR: 0.50, _WX.IFR: 0.50},
        _WX.LOW_IFR: {_WX.IFR: 0.50, _WX.LOW_IFR: 0.50},
        _WX.THUNDERSTORM: {_WX.IFR: 0.50, _WX.THUNDERSTORM: 0.50},
    },
    "compound_wx": {
        _WX.CLEAR: {_WX.CLEAR: 0.40, _WX.MVFR: 0.30, _WX.IFR: 0.20, _WX.THUNDERSTORM: 0.10},
        _WX.MVFR: {_WX.MVFR: 0.30, _WX.IFR: 0.30, _WX.THUNDERSTORM: 0.20, _WX.CLEAR: 0.20},
        _WX.IFR: {_WX.IFR: 0.30, _WX.THUNDERSTORM: 0.30, _WX.LOW_IFR: 0.20, _WX.MVFR: 0.20},
        _WX.LOW_IFR: {_WX.LOW_IFR: 0.40, _WX.THUNDERSTORM: 0.30, _WX.IFR: 0.30},
        _WX.THUNDERSTORM: {_WX.THUNDERSTORM: 0.40, _WX.IFR: 0.30, _WX.MVFR: 0.20, _WX.CLEAR: 0.10},
    },
    "cascading_delay": {
        _WX.CLEAR: {_WX.CLEAR: 0.90, _WX.MVFR: 0.10},
        _WX.MVFR: {_WX.CLEAR: 0.60, _WX.MVFR: 0.40},
        _WX.IFR: {_WX.MVFR: 0.50, _WX.IFR: 0.50},
        _WX.LOW_IFR: {_WX.IFR: 0.50, _WX.LOW_IFR: 0.50},
        _WX.THUNDERSTORM: {_WX.IFR: 0.50, _WX.THUNDERSTORM: 0.50},
    },
    "runway_closure": {
        _WX.CLEAR: {_WX.CLEAR: 0.85, _WX.MVFR: 0.15},
        _WX.MVFR: {_WX.CLEAR: 0.50, _WX.MVFR: 0.40, _WX.IFR: 0.10},
        _WX.IFR: {_WX.MVFR: 0.40, _WX.IFR: 0.60},
        _WX.LOW_IFR: {_WX.IFR: 0.50, _WX.LOW_IFR: 0.50},
        _WX.THUNDERSTORM: {_WX.IFR: 0.50, _WX.THUNDERSTORM: 0.50},
    },
    "crosswind": {
        _WX.CLEAR: {_WX.CLEAR: 0.70, _WX.MVFR: 0.30},
        _WX.MVFR: {_WX.CLEAR: 0.30, _WX.MVFR: 0.50, _WX.IFR: 0.20},
        _WX.IFR: {_WX.MVFR: 0.30, _WX.IFR: 0.50, _WX.LOW_IFR: 0.20},
        _WX.LOW_IFR: {_WX.IFR: 0.40, _WX.LOW_IFR: 0.60},
        _WX.THUNDERSTORM: {_WX.IFR: 0.50, _WX.THUNDERSTORM: 0.50},
    },
}

# Initial weather per scenario type
INITIAL_WEATHER: dict[str, WeatherCondition] = {
    "clear_day": _WX.CLEAR,
    "morning_fog": _WX.IFR,
    "thunderstorm": _WX.CLEAR,       # builds over time
    "snow_event": _WX.MVFR,
    "wind_shift": _WX.CLEAR,
    "peak_traffic": _WX.CLEAR,
    "compound_wx": _WX.CLEAR,
    "cascading_delay": _WX.CLEAR,
    "runway_closure": _WX.CLEAR,
    "crosswind": _WX.MVFR,
}

# Initial wind direction per scenario type
INITIAL_WIND: dict[str, int] = {
    "clear_day": 150,       # south flow
    "morning_fog": 140,     # south flow
    "thunderstorm": 130,    # south flow
    "snow_event": 320,      # north flow
    "wind_shift": 150,      # starts south, shifts
    "peak_traffic": 160,    # south flow
    "compound_wx": 200,     # west flow
    "cascading_delay": 140, # south flow
    "runway_closure": 150,  # south flow
    "crosswind": 240,       # west flow
}

# Flight count ranges per scenario
FLIGHT_COUNTS: dict[str, tuple[int, int]] = {
    "clear_day": (180, 220),
    "morning_fog": (150, 190),
    "thunderstorm": (160, 200),
    "snow_event": (140, 180),
    "wind_shift": (170, 210),
    "peak_traffic": (250, 300),
    "compound_wx": (150, 190),
    "cascading_delay": (180, 220),
    "runway_closure": (160, 200),
    "crosswind": (160, 200),
}


# ---------------------------------------------------------------------------
# Weather timeline generation
# ---------------------------------------------------------------------------

def _generate_weather_timeline(
    rng: np.random.RandomState,
    scenario_type: str,
    num_steps: int,
) -> list[str]:
    """Generate a weather timeline using a Markov chain."""
    transitions = WEATHER_TRANSITIONS[scenario_type]
    current = INITIAL_WEATHER[scenario_type]
    timeline = []

    for _ in range(num_steps):
        timeline.append(current.value)

        # Get transition probabilities
        trans = transitions.get(current, {current: 1.0})
        states = list(trans.keys())
        probs = np.array([trans[s] for s in states])
        probs = probs / probs.sum()  # normalize

        idx = rng.choice(len(states), p=probs)
        current = states[idx]

    return timeline


def _generate_wind_timeline(
    rng: np.random.RandomState,
    scenario_type: str,
    num_steps: int,
) -> list[int]:
    """Generate a wind direction timeline."""
    initial_wind = INITIAL_WIND[scenario_type]
    timeline = []

    if scenario_type == "wind_shift":
        # Wind gradually shifts from south to north over the simulation
        for step in range(num_steps):
            frac = step / max(num_steps - 1, 1)
            # Shift from 150° to 320° with some noise
            base = 150 + frac * 170
            noise = rng.normal(0, 5)
            wind = int((base + noise) % 360)
            timeline.append(wind)
    elif scenario_type == "crosswind":
        # Strong crosswinds with variability
        for step in range(num_steps):
            noise = rng.normal(0, 15)
            wind = int((240 + noise) % 360)
            timeline.append(wind)
    else:
        # Relatively stable wind with small fluctuations
        for step in range(num_steps):
            noise = rng.normal(0, 8)
            wind = int((initial_wind + noise) % 360)
            timeline.append(wind)

    return timeline


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

def _make_task(scenario_type: str, seed: int, split: str) -> dict:
    """Create a single task spec dict."""
    rng = np.random.RandomState(seed + hash(scenario_type) % 10000)
    num_steps = 48
    step_duration = 5

    weather = _generate_weather_timeline(rng, scenario_type, num_steps)
    wind = _generate_wind_timeline(rng, scenario_type, num_steps)

    lo, hi = FLIGHT_COUNTS[scenario_type]
    flight_count = int(rng.uniform(lo, hi))

    task_id = f"atc_{scenario_type}_{seed:03d}"

    return {
        "id": task_id,
        "seed": seed,
        "scenario_type": scenario_type,
        "num_steps": num_steps,
        "step_duration": step_duration,
        "weather_timeline": weather,
        "wind_timeline": wind,
        "flight_count": flight_count,
    }


def generate_all_tasks() -> dict[str, list[dict]]:
    """Generate all tasks for train and test splits."""
    train = []
    test = []

    for scenario_type in SCENARIO_TYPES:
        # Train: 2 seeds per scenario type = 20 train tasks
        for seed in range(2):
            task = _make_task(scenario_type, seed, "train")
            train.append(task)

        # Test: 1 seed per scenario type (different seed) = 10 test tasks
        task = _make_task(scenario_type, seed=100, split="test")
        test.append(task)

    return {"train": train, "test": test}


# Pre-generate at module load time
ALL_TASKS = generate_all_tasks()
