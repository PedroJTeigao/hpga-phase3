"""Independent correctness check for the 3D migration, requested after
run_benchmark_validation_3d.py fell 2 contacts short of the proven optimum
(20/22 across 5 seeds): does our contact-counting/geometry code agree with
CPSP-tools on CPSP-tools' OWN published optimal structure for the benchmark
sequence, independent of whether our GA's *search* can find it?

If this scores 22, evaluate_fitness (via hp_model.fitness_from_coords) is
proven correct against ground truth, and the run_benchmark_validation_3d.py
shortfall is purely a GA search-budget gap. If it does not score 22, there is
a bug in the geometry/contact code that the rotation-identity and
2D-plane-reduction checks (run during the migration, see the README) did not
catch.

Structure source: CPSP-tools HPstruct (Mann, Will & Backofen 2008, BMC
Bioinformatics 9:230), submitted live against
http://cpsp.informatik.uni-freiburg.de/HPstruct/ for
sequence=HHHHHPHHPHPHPHPHPHPHHHHHHPH, lattice=CUB, structure_model=BB,
number_of_structures=1, jobID=3058336. Result gave energy=-22 (matching the
paper's Table 2 for this sequence) and one optimal structure as an
"absolute move string" (CPSP's own coordinate-free representation: each
letter is a FIXED global lattice direction, not relative to a body frame,
unlike our genome encoding).

Letter -> displacement mapping taken verbatim from CPSP's own
absoluteMoveToPDB.js (computeMove()), not guessed or reverse-engineered from
examples:
    F: z -= 1      B: z += 1
    L: x += 1      R: x -= 1
    U: y -= 1      D: y += 1
The move string has length n-1 for an n-residue sequence (one step per
residue after the first, same convention as our genome_length = n-2 plus the
implicit second-residue placement -- CPSP's string additionally encodes the
step from residue 0 to residue 1 that our genome takes as fixed/implicit).

This script does NOT go through hp_model.genome_to_coords or our relative
STRAIGHT/LEFT/RIGHT/UP/DOWN frame at all -- it builds coordinates directly
from CPSP's absolute move string using CPSP's own letter convention, then
scores those coordinates with hp_model.fitness_from_coords, the exact same
contact/collision code evaluate_fitness calls internally (not a
reimplementation of it). This isolates the question "is our contact-counting
correct" from "does our relative-move encoding happen to reproduce this
exact walk" -- the contact count is invariant to which coordinate
convention/starting orientation produced the walk, since HP contacts only
depend on the walk's shape.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpga.config import BENCHMARK_OPTIMAL_ENERGY_3D, BENCHMARK_SEQUENCE_3D
from hpga.hp_model import fitness_from_coords

_LETTER_TO_STEP = {
    "F": (0, 0, -1),
    "B": (0, 0, 1),
    "L": (1, 0, 0),
    "R": (-1, 0, 0),
    "U": (0, -1, 0),
    "D": (0, 1, 0),
}

# Filled in from the completed CPSP-tools HPstruct job (jobID=3058336).
CPSP_ABSOLUTE_MOVE_STRING = "PASTE_RESULT_HERE"


def absolute_move_string_to_coords(move_string: str) -> list[tuple[int, int, int]]:
    coords = [(0, 0, 0)]
    for ch in move_string:
        dx, dy, dz = _LETTER_TO_STEP[ch]
        x, y, z = coords[-1]
        coords.append((x + dx, y + dy, z + dz))
    return coords


def main() -> None:
    sequence = BENCHMARK_SEQUENCE_3D
    target_contacts = -BENCHMARK_OPTIMAL_ENERGY_3D

    coords = absolute_move_string_to_coords(CPSP_ABSOLUTE_MOVE_STRING)
    print(f"sequence length = {len(sequence)}, structure has {len(coords)} residues")
    assert len(coords) == len(sequence), "structure/sequence length mismatch -- move string wrong length"

    n_unique = len(set(coords))
    print(f"self-avoiding: {n_unique == len(coords)} ({n_unique}/{len(coords)} unique lattice points)")

    fitness = fitness_from_coords(coords, sequence)
    print(f"fitness_from_coords(...) = {fitness}")
    print(f"target (CPSP proven optimum) = {target_contacts} contacts")
    print("MATCH" if fitness == target_contacts else "MISMATCH -- geometry/contact bug")


if __name__ == "__main__":
    main()
