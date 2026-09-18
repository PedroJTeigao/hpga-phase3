"""Sequence genome model: variable-length amino-acid strings over the 20
canonical letters, scored by folding with ESMFold and TM-aligning against a
fixed real-protein target (esmfold/fitness.py). Counterpart to hp_model.py's
fixed-length lattice genome. Not imported directly by island.py / agents.py /
circles.py -- reached only through hpga.genome_model.SequenceGenomeModel, so
this module is free to represent a genome as a plain `str` rather than
matching hp_model.py's list[int] convention.

Target and length-bound configuration are env-var-driven module constants
here (HPGA_SEQ_MIN_LENGTH / HPGA_SEQ_MAX_LENGTH), not HPGAConfig fields --
matching operators.py's own convention for keeping model-specific knobs
(HPGA_LLM_MODEL, HPGA_LLM_TEMPERATURE, ...) out of the shared dataclass. The
target itself (esmfold/pdb_cache/7UR7.pdb) is not made configurable here:
this phase targets exactly the one candidate selected and validated for
reachability (native-sequence self-fold TM-score 0.9137, see esmfold/
fitness.py's docstring and RESULTS.md) -- a generic multi-target knob would
be speculative generality this phase doesn't use.
"""

import os
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"  # 20 canonical amino acids, one-letter codes

# Bounds chosen for this phase's target (7UR7, 63-residue resolved core):
# wide enough that the native length sits comfortably inside, narrow enough
# to bound crossover drift and ESMFold's per-eval cost (RESULTS.md sec 3.1:
# ~7x wall-time from 50 to 250 residues at this project's standard GA
# config) rather than leaving either unbounded.
MIN_LENGTH = int(os.environ.get("HPGA_SEQ_MIN_LENGTH", "30"))
MAX_LENGTH = int(os.environ.get("HPGA_SEQ_MAX_LENGTH", "80"))


def random_sequence(rng: random.Random, length: int | None = None) -> str:
    """A fresh random amino-acid sequence. `length`, if given, must already
    be within [MIN_LENGTH, MAX_LENGTH] -- this function enforces the bound,
    it doesn't clamp into it, so a caller passing an out-of-range length
    finds out immediately rather than getting a silently different genome
    than it asked for. If omitted, a length is drawn uniformly from that
    range, so an initial population isn't all one length by default."""
    if length is None:
        length = rng.randint(MIN_LENGTH, MAX_LENGTH)
    elif not (MIN_LENGTH <= length <= MAX_LENGTH):
        raise ValueError(f"length {length} outside [{MIN_LENGTH}, {MAX_LENGTH}]")
    return "".join(rng.choice(ALPHABET) for _ in range(length))


_fitness = None  # lazy ESMFoldFitness singleton


def _get_fitness():
    """Loading ESMFold is a ~9s, ~13.7GB-of-GPU operation (RESULTS.md sec 3)
    -- deferred to first use, not import time or construction time, and
    cached process-wide rather than per SequenceGenomeModel instance so
    constructing more than one of those (e.g. a probe script building its
    own alongside Island's) doesn't try to load the model twice onto a GPU
    that only fits one copy."""
    global _fitness
    if _fitness is None:
        from esmfold.fitness import ESMFoldFitness

        _fitness = ESMFoldFitness()
    return _fitness


def evaluate_fitness(sequence: str) -> float:
    """TM-score of `sequence`'s ESMFold-predicted structure against the
    configured target -- higher is better, same orientation as
    hp_model.evaluate_fitness's contact count."""
    return _get_fitness().score(sequence)


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance: the variable-length generalisation of the
    lattice model's Hamming-based diversity metric. Reduces to an exact
    Hamming count when len(a) == len(b) (every aligned position is either a
    match or a one-for-one substitution, and there is nothing left to
    insert or delete) -- a principled default, not an arbitrary distance
    choice, and the reason lattice-mode diversity logs and sequence-mode
    diversity logs are kept under DIFFERENT keys rather than one shared
    "diversity" field: the two are the same statistic only in the
    equal-length case, and conflating them under one name is how a
    fixed-length comparison and a variable-length one get silently averaged
    together later."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,  # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + (ca != cb),  # match / substitution
            )
        prev = curr
    return prev[-1]


def crossover(parent1: str, parent2: str, rate: float, rng: random.Random) -> tuple[str, str]:
    """Aligned relative-position single-point crossover (design option A):
    one cut FRACTION is drawn once and applied to each parent's own length
    independently, rather than a single shared position index (which is
    undefined when len(parent1) != len(parent2)) -- child1 = parent1[:cut1]
    + parent2[cut2:], child2 is the complement.

    Chosen over independent unaligned cuts (option B) or length-gated
    pairing (option C) because a shared fraction keeps each child's length
    *close to* a convex combination of the two parents' lengths (f*len(p1)
    + (1-f)*len(p2)): if both parents are within [MIN_LENGTH, MAX_LENGTH],
    the combination is too, so in the common case a child already lands
    in-bounds with no fixup needed. "Close to" because cut1/cut2 are
    rounded independently, which can push a child's length up to +-1
    outside that combination -- rare, but real, and left unhandled it
    compounds: a child one residue over MAX_LENGTH becomes a parent next
    generation, whose own children can then land two over, and so on. The
    retry loop below (re-draw f, not fix up the string) is the guard
    against that drift -- bounded at a handful of attempts since a fresh f
    landing out of bounds twice running is already unlikely, falling back
    to a no-op (unchanged parents) rather than truncating/padding a string,
    which would silently reach into indel territory `mutate` below
    deliberately stays out of.

    Also the natural variable-length generalisation of what the LLM
    'segment' affordance already collapses to in practice (PHASE2_RESULTS.md
    Sec 4.4: 36/40 live segment-crossover calls produced a single cut at or
    within one position of the exact midpoint, despite the format allowing
    arbitrary multi-segment splits) -- so matching that with an explicit
    single-cut design here doesn't give up expressiveness the model was
    actually using anyway.
    """
    if rng.random() > rate or (len(parent1) < 2 and len(parent2) < 2):
        return parent1, parent2
    for _ in range(8):
        f = rng.random()
        cut1 = min(len(parent1), max(0, round(f * len(parent1))))
        cut2 = min(len(parent2), max(0, round(f * len(parent2))))
        child1 = parent1[:cut1] + parent2[cut2:]
        child2 = parent2[:cut2] + parent1[cut1:]
        if MIN_LENGTH <= len(child1) <= MAX_LENGTH and MIN_LENGTH <= len(child2) <= MAX_LENGTH:
            return child1, child2
    return parent1, parent2


def mutate(sequence: str, rate: float, rng: random.Random) -> str:
    """Length-preserving point substitution -- deliberately no insertion or
    deletion in this first pass. Crossover (above) is already one
    length-changing pressure that needs bounding; adding indel mutation on
    top would make any observed length drift impossible to attribute to
    either operator specifically. Scoped out, not overlooked."""
    return "".join(rng.choice(ALPHABET) if rng.random() < rate else c for c in sequence)
