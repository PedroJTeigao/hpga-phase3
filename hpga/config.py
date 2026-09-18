"""Run configuration for the HPGA baseline."""

from dataclasses import dataclass, field


@dataclass
class HPGAConfig:
    """Parameters for one island's GA + master/slave dispatch loop.

    `sequence` is fixed at construction; `genome_length = len(sequence) - 2`
    (two residues are pinned by construction to remove the four-fold rotational
    and mirror symmetry of the lattice, see hp_model.py). Both only apply to
    `genome_model == "lattice"` -- see `genome_model` below.
    """

    sequence: str = ""
    pop_size: int = 64
    n_generations: int = 30
    n_workers: int = 4
    mutation_rate: float = 0.05
    crossover_rate: float = 0.9
    tournament_k: int = 3
    elitism: int = 2
    collision_penalty: float = 50.0
    seed: int = 0

    # Phase 4 hooks (DIBM / TDIM analogues) — unused in Phase 1, kept here so
    # later phases don't need to touch this dataclass's call sites.
    dispatch_channels: int = 1
    island_id: int = 0

    # Which GenomeModel this run uses (see hpga/genome_model.py): "lattice"
    # (default, unchanged Phase 1-3 behaviour -- fixed-length list[int] over
    # the 5-symbol move alphabet, hp_model.py) or "sequence" (variable-length
    # amino-acid string, hpga/sequence_model.py, scored by ESMFold+TM-align
    # against a fixed real-protein target). Model-specific parameters that
    # only "sequence" needs (length bounds, target structure) deliberately
    # do NOT live here -- they're env-var-driven constants in
    # hpga/sequence_model.py, matching how operators.py already keeps
    # LLM-operator config (HPGA_LLM_MODEL etc.) out of this dataclass rather
    # than growing it with fields most runs don't use.
    genome_model: str = "lattice"

    def __post_init__(self) -> None:
        if self.genome_model not in ("lattice", "sequence"):
            raise ValueError(
                f"genome_model must be 'lattice' or 'sequence', got {self.genome_model!r}"
            )
        if self.genome_model == "lattice" and len(self.sequence) < 4:
            raise ValueError("sequence must have at least 4 residues")
        if self.n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        if self.pop_size < 2:
            raise ValueError("pop_size must be >= 2")

    @property
    def genome_length(self) -> int:
        return len(self.sequence) - 2


# --- Reference sequences -------------------------------------------------

# Unger & Moult (1993) 2D HP benchmark, n=20, known optimal energy E* = -9.
# Used to sanity-check that selection/crossover/mutation actually converge
# toward a known-good answer, independent of the timing harness.
#
# Kept for reference/citation only in this file (Phase 2) -- hpga/hp_model.py
# was migrated to the 3D simple cubic lattice (see BENCHMARK_SEQUENCE_3D
# below and the Phase 2 README), so this 2D sequence is no longer paired
# with a runnable 2D fitness function here. Phase 1's hp_model.py is
# untouched and still evaluates this correctly.
BENCHMARK_SEQUENCE = "HPHPPHHPHPPHPHHPPHPH"
BENCHMARK_OPTIMAL_ENERGY = -9

# Mann, Will & Backofen (2008), "CPSP-tools -- exact and complete algorithms
# for high-throughput 3D lattice protein studies", BMC Bioinformatics 9:230,
# https://doi.org/10.1186/1471-2105-9-230 -- Table 2, sequence S1. Plain
# (backbone-only) HP model, unrestricted simple cubic lattice, n=27.
# E*=-22 (22 non-consecutive H-H contacts) is PROVEN optimal, not
# best-known: CPSP-tools' branch-and-bound (HPstruct) exhaustively proves no
# structure of this sequence forms more than 22 HH-contacts, and reports
# degeneracy=1 (a unique optimal structure up to lattice symmetry). This is
# the plain 3D HP model, not the HP side-chain model (HPSC, Benitez & Lopes
# 2009) that Xue et al. (NoCS 2014, this project's base paper) actually use
# for their Unger273d/Dill benchmarks -- see the Phase 2 README's note on
# that gap before citing this as validating the HPSC variant.
BENCHMARK_SEQUENCE_3D = "HHHHHPHHPHPHPHPHPHPHHHHHHPH"
BENCHMARK_OPTIMAL_ENERGY_3D = -22


def make_timing_sequence(length: int, seed: int = 0) -> str:
    """A longer, randomly generated H/P sequence with no known optimum.

    Used purely to give T_calc (one fitness evaluation) a realistic,
    non-negligible wall-clock cost relative to IPC dispatch overhead — a
    20-mer evaluates in low single-digit microseconds, which would make the
    dispatch channel saturate at N=2-3 regardless of anything interesting.
    This is a real HP-energy computation on a longer real chain, not a
    synthetic delay; the length is a reported, documented knob.
    """
    import random

    rng = random.Random(seed)
    return "".join(rng.choice("HP") for _ in range(length))
