from __future__ import annotations

from collections import defaultdict
import time

import numpy as np

from dp.environments.JacksCarRental import CarState, JacksCarRental
from dp.environments.factory import make_jacks_small_env


def evaluate_pair(
    env: JacksCarRental,
    s: CarState,
    a: int,
    num_samples: int,
) -> tuple[float, float, int]:
    """
    Compare exact p(s', r | s, a) against Monte Carlo frequencies.

    Returns:
    - total variation distance
    - max absolute error for outcomes with non-negligible exact mass
    - number of distinct exact outcomes
    """
    exact = env._dynamics_cache[(s, a)]

    counts: dict[tuple[CarState, int], int] = defaultdict(int)
    for _ in range(num_samples):
        s_prime, r = env.do_action(s, a)
        counts[(s_prime, int(r))] += 1

    empirical = {k: v / num_samples for k, v in counts.items()}
    keys = set(exact.keys()) | set(empirical.keys())

    l1 = sum(abs(empirical.get(k, 0.0) - exact.get(k, 0.0)) for k in keys)
    tv = 0.5 * l1

    mass_threshold = 1e-3
    major_keys = [k for k, p in exact.items() if p >= mass_threshold]
    max_abs_major = max(
        (abs(empirical.get(k, 0.0) - exact[k]) for k in major_keys),
        default=0.0,
    )

    return tv, max_abs_major, len(exact)


def run_verification(
    env: JacksCarRental | None = None,
    num_samples: int = 30000,
    seed: int = 0,
) -> dict[str, float | int | tuple[CarState, int] | None]:
    np.random.seed(seed)

    if env is None:
        env = make_jacks_small_env()

    # Time this
    start_time = time.time()

    env._precompute_dynamics()

    end_time = time.time()
    print(f"Precomputation took {end_time - start_time:.2f} seconds")

    pairs: list[tuple[CarState, int]] = []
    for s in env.states:
        for a in env.get_actions(s):
            pairs.append((s, a))

    tvs: list[float] = []
    major_maxes: list[float] = []

    worst_pair: tuple[CarState, int] | None = None
    worst_tv = -1.0

    for s, a in pairs:
        tv, max_abs_major, num_outcomes = evaluate_pair(env, s, a, num_samples)
        tvs.append(tv)
        major_maxes.append(max_abs_major)

        if tv > worst_tv:
            worst_tv = tv
            worst_pair = (s, a)

        print(
            f"(s={s}, a={a}): outcomes={num_outcomes:3d}, "
            f"TV={tv:.4f}, max_abs_major={max_abs_major:.4f}"
        )

    mean_tv = float(np.mean(tvs)) if tvs else 0.0
    p95_tv = float(np.percentile(tvs, 95)) if tvs else 0.0
    max_tv = float(np.max(tvs)) if tvs else 0.0
    max_major = float(np.max(major_maxes)) if major_maxes else 0.0

    print("\nSummary")
    print(f"pairs={len(pairs)}, samples_per_pair={num_samples}")
    print(f"mean_TV={mean_tv:.4f}, p95_TV={p95_tv:.4f}, max_TV={max_tv:.4f}")
    print(f"max_abs_major={max_major:.4f}")
    print(f"worst_pair={worst_pair}")

    # A lightweight acceptance criterion for stochastic verification.
    assert mean_tv < 0.05, f"Mean TV too high: {mean_tv:.4f}"
    assert p95_tv < 0.08, f"P95 TV too high: {p95_tv:.4f}"

    return {
        "pairs": len(pairs),
        "samples_per_pair": num_samples,
        "mean_tv": mean_tv,
        "p95_tv": p95_tv,
        "max_tv": max_tv,
        "max_abs_major": max_major,
        "worst_pair": worst_pair,
        "precompute_seconds": end_time - start_time,
    }


def main() -> None:
    run_verification()


if __name__ == "__main__":
    main()