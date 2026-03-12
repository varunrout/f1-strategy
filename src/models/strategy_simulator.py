"""
strategy_simulator.py
=====================
Monte-Carlo race-strategy simulator for Domain 3.

Given per-lap tyre-degradation rates and traffic costs, simulate the
expected total race time for one-stop, two-stop, and three-stop strategies
and rank them.

Usage (as a library)
--------------------
    from src.models.strategy_simulator import StrategySimulator

    sim = StrategySimulator(total_laps=57, base_lap_time_s=90.0)
    results = sim.run(n_simulations=1000)
    print(results.sort_values("mean_race_time_s"))

Usage (CLI)
-----------
    python -m src.models.strategy_simulator \
        --total-laps 57 \
        --base-lap-time 90.0 \
        --n-sim 5000
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tyre-compound parameters (sensible F1 defaults; override via constructor)
# ---------------------------------------------------------------------------

COMPOUND_PARAMS: dict[str, dict] = {
    "SOFT": {
        "deg_rate_s_per_lap": 0.12,   # lap-time loss per lap on this compound
        "deg_rate_std": 0.03,
        "max_stint_laps": 25,
        "initial_advantage_s": -0.8,  # faster than MEDIUM at lap 1
    },
    "MEDIUM": {
        "deg_rate_s_per_lap": 0.07,
        "deg_rate_std": 0.02,
        "max_stint_laps": 35,
        "initial_advantage_s": 0.0,   # reference
    },
    "HARD": {
        "deg_rate_s_per_lap": 0.04,
        "deg_rate_std": 0.01,
        "max_stint_laps": 50,
        "initial_advantage_s": 0.5,   # slower than MEDIUM at lap 1
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Stint:
    compound: str
    start_lap: int
    end_lap: int

    @property
    def length(self) -> int:
        return self.end_lap - self.start_lap + 1


@dataclass
class Strategy:
    name: str
    stints: list[Stint]
    n_stops: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_stops = len(self.stints) - 1


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class StrategySimulator:
    """
    Monte-Carlo race-strategy simulator.

    Parameters
    ----------
    total_laps:
        Number of race laps.
    base_lap_time_s:
        Ideal (minimum) lap time in seconds on the reference compound (MEDIUM).
    pit_lane_time_s:
        Fixed pit-lane time loss per stop (seconds).
    sc_probability:
        Probability of a safety-car period per lap.
    sc_pit_benefit_s:
        Time benefit of pitting under SC vs green (seconds).
    fuel_correction_s_per_lap:
        Lap-time improvement per lap due to fuel burn (lighter car).
    compound_params:
        Override the default COMPOUND_PARAMS dict.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        total_laps: int = 57,
        base_lap_time_s: float = 90.0,
        pit_lane_time_s: float = 22.0,
        sc_probability: float = 0.04,
        sc_pit_benefit_s: float = 12.0,
        fuel_correction_s_per_lap: float = 0.04,
        compound_params: Optional[dict] = None,
        seed: Optional[int] = 42,
    ) -> None:
        self.total_laps = total_laps
        self.base_lap_time_s = base_lap_time_s
        self.pit_lane_time_s = pit_lane_time_s
        self.sc_probability = sc_probability
        self.sc_pit_benefit_s = sc_pit_benefit_s
        self.fuel_correction_s_per_lap = fuel_correction_s_per_lap
        self.compound_params = compound_params or COMPOUND_PARAMS
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Strategy generation
    # ------------------------------------------------------------------

    def _default_strategies(self) -> list[Strategy]:
        """Generate a representative set of 1-, 2- and 3-stop strategies."""
        L = self.total_laps
        strategies = []

        # 1-stop strategies
        for split in range(max(10, L // 3), min(L - 10, 2 * L // 3), 5):
            for c1, c2 in [("SOFT", "MEDIUM"), ("MEDIUM", "HARD"), ("SOFT", "HARD")]:
                strategies.append(
                    Strategy(
                        name=f"1S_{c1}_{split}_{c2}",
                        stints=[
                            Stint(c1, 1, split),
                            Stint(c2, split + 1, L),
                        ],
                    )
                )

        # 2-stop strategies
        for s1 in range(L // 4, L // 3, 5):
            for s2 in range(s1 + L // 4, s1 + L // 2, 5):
                if s2 >= L - 5:
                    continue
                for c1, c2, c3 in [
                    ("SOFT", "MEDIUM", "SOFT"),
                    ("SOFT", "HARD", "SOFT"),
                    ("MEDIUM", "HARD", "MEDIUM"),
                ]:
                    strategies.append(
                        Strategy(
                            name=f"2S_{c1}_{s1}_{c2}_{s2}_{c3}",
                            stints=[
                                Stint(c1, 1, s1),
                                Stint(c2, s1 + 1, s2),
                                Stint(c3, s2 + 1, L),
                            ],
                        )
                    )

        # 3-stop strategy (sample)
        q = L // 4
        strategies.append(
            Strategy(
                name="3S_SOFT_MED_SOFT_MED",
                stints=[
                    Stint("SOFT", 1, q),
                    Stint("MEDIUM", q + 1, 2 * q),
                    Stint("SOFT", 2 * q + 1, 3 * q),
                    Stint("MEDIUM", 3 * q + 1, L),
                ],
            )
        )
        return strategies

    # ------------------------------------------------------------------
    # Single-simulation lap time model
    # ------------------------------------------------------------------

    def _simulate_strategy(
        self,
        strategy: Strategy,
        sc_laps: set[int],
    ) -> float:
        """
        Simulate total race time for one strategy in one MC run.

        Returns total race time in seconds.
        """
        total_time = 0.0
        pit_cost = self.pit_lane_time_s

        for stint in strategy.stints:
            params = self.compound_params.get(stint.compound, self.compound_params["MEDIUM"])
            deg_rate = max(
                0.0,
                self.rng.normal(
                    params["deg_rate_s_per_lap"],
                    params["deg_rate_std"],
                ),
            )
            init_adv = params["initial_advantage_s"]

            for lap_offset, lap_num in enumerate(range(stint.start_lap, stint.end_lap + 1)):
                fuel_saving = self.fuel_correction_s_per_lap * (lap_num - 1)
                tyre_deg = deg_rate * lap_offset
                lap_time = (
                    self.base_lap_time_s
                    + init_adv
                    + tyre_deg
                    - fuel_saving
                )
                # Add small Gaussian noise
                lap_time += self.rng.normal(0, 0.15)
                total_time += max(lap_time, self.base_lap_time_s * 0.90)

            # Pit stop cost at the end of all but the last stint
            if stint is not strategy.stints[-1]:
                if (stint.end_lap + 1) in sc_laps:
                    total_time += pit_cost - self.sc_pit_benefit_s
                else:
                    total_time += pit_cost

        return total_time

    # ------------------------------------------------------------------
    # Public run method
    # ------------------------------------------------------------------

    def run(
        self,
        strategies: Optional[list[Strategy]] = None,
        n_simulations: int = 1000,
    ) -> pd.DataFrame:
        """
        Run the Monte-Carlo simulation for all strategies.

        Parameters
        ----------
        strategies:
            List of Strategy objects to evaluate.  If None, the default
            set of 1-, 2- and 3-stop strategies is used.
        n_simulations:
            Number of MC iterations per strategy.

        Returns
        -------
        DataFrame sorted by mean_race_time_s with columns:
            strategy_name, n_stops, stints_desc,
            mean_race_time_s, std_race_time_s,
            p10_race_time_s, p90_race_time_s,
            best_case_s, worst_case_s
        """
        if strategies is None:
            strategies = self._default_strategies()

        logger.info(
            "Simulating %d strategies × %d MC runs (total_laps=%d) …",
            len(strategies),
            n_simulations,
            self.total_laps,
        )

        records = []
        for strat in strategies:
            times = []
            for _ in range(n_simulations):
                # Generate SC laps for this simulation
                sc_laps = {
                    lap
                    for lap in range(1, self.total_laps + 1)
                    if self.rng.random() < self.sc_probability
                }
                t = self._simulate_strategy(strat, sc_laps)
                times.append(t)

            arr = np.array(times)
            records.append(
                {
                    "strategy_name": strat.name,
                    "n_stops": strat.n_stops,
                    "stints_desc": " | ".join(
                        f"{s.compound}[{s.start_lap}-{s.end_lap}]"
                        for s in strat.stints
                    ),
                    "mean_race_time_s": float(np.mean(arr)),
                    "std_race_time_s": float(np.std(arr)),
                    "p10_race_time_s": float(np.percentile(arr, 10)),
                    "p90_race_time_s": float(np.percentile(arr, 90)),
                    "best_case_s": float(np.min(arr)),
                    "worst_case_s": float(np.max(arr)),
                }
            )

        df = pd.DataFrame(records).sort_values("mean_race_time_s").reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
        return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte-Carlo F1 race-strategy simulator."
    )
    parser.add_argument("--total-laps", type=int, default=57)
    parser.add_argument("--base-lap-time", type=float, default=90.0)
    parser.add_argument("--pit-lane-time", type=float, default=22.0)
    parser.add_argument("--sc-probability", type=float, default=0.04)
    parser.add_argument("--n-sim", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top", type=int, default=10, help="Show top N strategies")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    sim = StrategySimulator(
        total_laps=args.total_laps,
        base_lap_time_s=args.base_lap_time,
        pit_lane_time_s=args.pit_lane_time,
        sc_probability=args.sc_probability,
        seed=args.seed,
    )
    results = sim.run(n_simulations=args.n_sim)
    print(f"\nTop {args.top} strategies by expected race time:\n")
    print(
        results.head(args.top).to_string(
            columns=[
                "rank",
                "strategy_name",
                "n_stops",
                "mean_race_time_s",
                "std_race_time_s",
                "p10_race_time_s",
                "p90_race_time_s",
            ],
            index=False,
        )
    )


if __name__ == "__main__":
    main()
