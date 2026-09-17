"""Validates the migrated 3D hp_model.py (simple cubic lattice) the same way
Phase 1's 2D model was validated in run_benchmark_validation.py: run the
deterministic GA (HPGA_OPERATOR_MODE unset -> deterministic, the default;
no LLM operators here by design) against a benchmark sequence with a known
optimal energy, across several seeds, and report whether it reaches or
approaches that optimum.

Benchmark: Mann, Will & Backofen (2008), "CPSP-tools -- exact and complete
algorithms for high-throughput 3D lattice protein studies", BMC
Bioinformatics 9:230, https://doi.org/10.1186/1471-2105-9-230, Table 2,
sequence S1 (n=27, plain HP model, unrestricted simple cubic lattice).
E*=-22 is PROVEN optimal (branch-and-bound, degeneracy=1), not
heuristic-best-known. See hpga/config.py and the Phase 2 README for the
full citation and the HPSC-variant caveat (Xue et al.'s base paper uses the
HP side-chain model, not plain 3D HP -- this validates plain 3D HP only).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpga.config import BENCHMARK_OPTIMAL_ENERGY_3D, BENCHMARK_SEQUENCE_3D, HPGAConfig
from hpga.island import Island


def run_one(seed: int, n_generations: int) -> dict:
    cfg = HPGAConfig(
        sequence=BENCHMARK_SEQUENCE_3D,
        pop_size=256,
        n_generations=n_generations,
        n_workers=8,
        seed=seed,
    )
    island = Island(cfg)
    recorder = island.run()
    return recorder.summary()


def main() -> None:
    target_contacts = -BENCHMARK_OPTIMAL_ENERGY_3D
    print(f"sequence={BENCHMARK_SEQUENCE_3D}  length={len(BENCHMARK_SEQUENCE_3D)}  (3D simple cubic lattice)")
    print(f"known optimal energy E*={BENCHMARK_OPTIMAL_ENERGY_3D}  (target = {target_contacts} H-H contacts, PROVEN via CPSP-tools branch-and-bound, degeneracy=1)\n")

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

    print(f"\ntarget (E*={BENCHMARK_OPTIMAL_ENERGY_3D}) = {target_contacts} contacts")
    print(f"best achieved across {n_seeds} seeds x {n_generations} generations = {best_overall:.0f} contacts")
    print(f"gap to optimum = {target_contacts - best_overall:.0f} contacts")
    print(f"reached/approached optimum: {'YES' if best_overall >= target_contacts else 'NO'}")


if __name__ == "__main__":
    main()
