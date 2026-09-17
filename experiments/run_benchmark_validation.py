"""Review finding #3: BENCHMARK_SEQUENCE was defined but never run. This
validates the deterministic GA operators (independent of the timing harness)
against the Unger & Moult (1993) n=20 HP benchmark with known optimal energy
E*=-9 (9 non-consecutive H-H contacts). Longer run than the timing sweep
(more generations, restarts) since the point here is convergence quality,
not timing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpga.config import BENCHMARK_OPTIMAL_ENERGY, BENCHMARK_SEQUENCE, HPGAConfig
from hpga.island import Island


def run_one(seed: int, n_generations: int) -> dict:
    cfg = HPGAConfig(
        sequence=BENCHMARK_SEQUENCE,
        pop_size=256,
        n_generations=n_generations,
        n_workers=8,
        seed=seed,
    )
    island = Island(cfg)
    recorder = island.run()
    return recorder.summary()


def main() -> None:
    target_contacts = -BENCHMARK_OPTIMAL_ENERGY
    print(f"sequence={BENCHMARK_SEQUENCE}  length={len(BENCHMARK_SEQUENCE)}")
    print(f"known optimal energy E*={BENCHMARK_OPTIMAL_ENERGY}  (target = {target_contacts} H-H contacts)\n")

    n_generations = 500
    n_seeds = 5
    best_overall = -1e9
    results = []
    for seed in range(n_seeds):
        s = run_one(seed, n_generations)
        final = s["best_fitness_final"]
        results.append((seed, final))
        best_overall = max(best_overall, final)
        gaps = s["best_fitness_by_gen"]
        milestones = [0, 25, 50, 100, 200, 300, 400, n_generations - 1]
        trace = "  ".join(f"g{m}={gaps[m]:.0f}" for m in milestones if m < len(gaps))
        print(f"seed={seed}  final={final:.0f}  ({trace})")

    print(f"\ntarget (E*={BENCHMARK_OPTIMAL_ENERGY}) = {target_contacts} contacts")
    print(f"best achieved across {n_seeds} seeds x {n_generations} generations = {best_overall:.0f} contacts")
    print(f"gap to optimum = {target_contacts - best_overall:.0f} contacts")
    print(f"reached/approached optimum: {'YES' if best_overall >= target_contacts else 'NO'}")


if __name__ == "__main__":
    main()
