"""Self-contained correctness check for the 3D migration, run because the
intended check (score CPSP-tools' own proven-optimal structure for the S1
benchmark through our fitness function) could not be completed: CPSP-tools'
HPstruct job queue (http://cpsp.informatik.uni-freiburg.de/HPstruct/) never
advanced past "Submitted & Queued" across two separate submissions
(jobID=3058336, waited 15 minutes; jobID=3478281, waited ~80s as a spot
check) -- a stalled server-side job runner, not a per-job fluke. See the
Phase 2 README's "3D migration" section for the full account, including
what this script does and does not establish relative to that blocked
check.

This script needs no external dependency at all. On a small (n=8) HP
instance it:

  1. Runs a from-scratch brute-force solver over ALL self-avoiding walks
     using the raw 6 cubic-lattice unit directions directly (+-x, +-y,
     +-z) -- NOT hp_model.genome_to_coords, NOT hp_model.fitness_from_coords,
     not any code from hpga/ at all -- with its own from-scratch
     pairwise-distance contact counter. This is a second, independent
     implementation of "what is the true optimal HH-contact count for this
     sequence," the same role CPSP-tools' proven optimum was meant to play,
     just computed locally instead of fetched from a stalled server.
  2. Exhaustively enumerates hpga's ENTIRE reduced-move genome space for the
     same sequence (5^6 = 15625 genomes -- small enough to brute force) using
     the actual production hp_model.evaluate_fitness, and takes its best.
  3. Compares the two. If they match, hp_model.py's genome encoding is both
     expressive enough to reach the true optimal shape AND scores it
     correctly -- for this instance, the full genome->coords->fitness
     pipeline is proven correct against independently-computed ground
     truth, not merely internally self-consistent.

Caveat, stated plainly: this proves correctness on a small, exhaustively
verified instance (n=8), not on the 27-mer benchmark itself, and it is not
the external CPSP-tools ground-truth match originally asked for. A bug in
the per-step rotation/contact logic would be expected to show up here too
(it's the same code applied fewer times), so a match here is real evidence
against a geometry bug at any scale -- but it is corroborating evidence, not
a substitute for the blocked external check. If CPSP-tools' queue recovers,
run experiments/verify_3d_benchmark_structure.py against the completed job
for the decisive external check.
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpga.config import BENCHMARK_SEQUENCE_3D
from hpga.hp_model import MOVES, evaluate_fitness

TEST_SEQUENCE = BENCHMARK_SEQUENCE_3D[:8]  # n=8, genome_length=6, arbitrary prefix of the real benchmark

_DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def _independent_contacts(coords: list[tuple[int, int, int]], sequence: str) -> int:
    """From-scratch contact counter: pairwise Manhattan distance, no
    neighbor-offset table, no dict indexing -- deliberately not sharing
    approach with hp_model.fitness_from_coords."""
    n = len(coords)
    c = 0
    for i in range(n):
        if sequence[i] != "H":
            continue
        for j in range(i + 2, n):
            if sequence[j] != "H":
                continue
            dx = abs(coords[i][0] - coords[j][0])
            dy = abs(coords[i][1] - coords[j][1])
            dz = abs(coords[i][2] - coords[j][2])
            if dx + dy + dz == 1:
                c += 1
    return c


def brute_force_optimal(sequence: str) -> tuple[int, list[tuple[int, int, int]]]:
    """Exhaustive DFS over all self-avoiding walks using the raw 6 lattice
    directions (no reduced-move alphabet, no excluded-reversal assumption --
    the rawest possible formulation of the problem)."""
    n = len(sequence)
    best = -1
    best_coords: list[tuple[int, int, int]] | None = None

    def rec(coords: list[tuple[int, int, int]], occupied: set[tuple[int, int, int]]) -> None:
        nonlocal best, best_coords
        if len(coords) == n:
            c = _independent_contacts(coords, sequence)
            if c > best:
                best = c
                best_coords = list(coords)
            return
        x, y, z = coords[-1]
        for dx, dy, dz in _DIRS:
            p = (x + dx, y + dy, z + dz)
            if p in occupied:
                continue
            occupied.add(p)
            coords.append(p)
            rec(coords, occupied)
            coords.pop()
            occupied.remove(p)

    rec([(0, 0, 0)], {(0, 0, 0)})
    assert best_coords is not None
    return best, best_coords


def main() -> None:
    seq = TEST_SEQUENCE
    print(f"test sequence: {seq}  (n={len(seq)})")

    true_opt, true_opt_coords = brute_force_optimal(seq)
    print(f"independent raw-6-direction brute force: true optimum = {true_opt} contacts")
    print(f"  (structure: {true_opt_coords})")

    genome_length = len(seq) - 2
    best_ga_space = -1e9
    best_genome = None
    for genome in itertools.product(MOVES, repeat=genome_length):
        f = evaluate_fitness(list(genome), seq)
        if f > best_ga_space:
            best_ga_space = f
            best_genome = genome
    print(f"exhaustive search of hp_model's own genome space (5^{genome_length}={5 ** genome_length} genomes,"
          f" via production evaluate_fitness): best = {best_ga_space}")
    print(f"  (best genome: {best_genome})")

    match = best_ga_space == true_opt
    print()
    print(f"hp_model's genome/fitness pipeline reaches the independently-computed true optimum: {match}")
    if not match:
        print("MISMATCH -- investigate before trusting hp_model.py on any instance.")


if __name__ == "__main__":
    main()
