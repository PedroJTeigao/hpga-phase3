"""The genome-type abstraction boundary: everything that differs between the
lattice genome (fixed-length list[int], 5-symbol move alphabet, hp_model.py)
and the sequence genome (variable-length str, 20-symbol amino-acid alphabet,
hpga/sequence_model.py) is isolated behind the `GenomeModel` interface below,
so circles.py, blackboard.py, and agents.py's diversity logging can operate
on "a genome" without knowing which kind is live.

Deliberately minimal for this first pass: only the operations that
circles.py/agents.py's non-LLM machinery already needs (constructing a fresh
genome, measuring fitness/distance) are part of the interface here. LLM
prompt-style dispatch (operators.py's mutate/crossover, at "full" / "position"
/ "diff" / "segment") is NOT included yet -- operators.py's LLM machinery is
untouched in this pass (see hpga/config.py's genome_model field docstring),
and guessing the right shape for that hook before actually exercising it
against the sequence model (next step: running
experiments/probe_operator_compliance.py against genome_model="sequence")
risks designing the wrong boundary. That step will show concretely what
operators.py needs; this interface grows to fit that evidence rather than
anticipating it.

Selection: `genome_model.build_genome_model(config)` constructs the right
model from `HPGAConfig.genome_model`; `set_active`/`current` hold a
process-local active instance (not a per-call env-var read like
HPGA_OPERATOR_MODE) because a GenomeModel can carry real state -- the
sequence model's ESMFold weights live on the GPU for the process's lifetime,
not something a bare string switch could represent. `current()` raises
rather than defaulting to lattice if nothing has called `set_active` yet: a
silent fallback here would mean a sequence-mode run silently measuring the
lattice model instead, not a cosmetic gap.
"""

import random
from typing import Protocol

from hpga.config import HPGAConfig

Genome = list[int] | str


class GenomeModel(Protocol):
    name: str

    def random_genome(self, rng: random.Random, length: int | None = None) -> Genome: ...

    def to_str(self, genome: Genome) -> str: ...

    def length(self, genome: Genome) -> int: ...

    def evaluate_fitness(self, genome: Genome) -> float: ...

    def distance(self, a: Genome, b: Genome) -> float:
        """A diversity/dissimilarity metric between two genomes of this
        model's type. Not assumed comparable across models -- see
        hpga/sequence_model.py's edit_distance docstring for why lattice and
        sequence runs deliberately log this under different keys."""
        ...

    def deterministic_crossover(
        self, p1: Genome, p2: Genome, rate: float, rng: random.Random
    ) -> tuple[Genome, Genome]: ...

    def deterministic_mutate(self, genome: Genome, rate: float, rng: random.Random) -> Genome: ...


class LatticeGenomeModel:
    """Adapter over the existing, unmodified hp_model.py / operators.py
    lattice implementation -- delegates every method to the exact functions
    Phase 1-3 already use, so behaviour is byte-identical to before this
    module existed. Imports are local to each method (not at module level)
    so importing hpga.genome_model doesn't require hpga.operators to import
    cleanly first -- avoids a needless import-order coupling between two
    modules that don't otherwise need one."""

    name = "lattice"

    def __init__(self, config: HPGAConfig):
        self._sequence = config.sequence
        self._genome_length = config.genome_length
        self._collision_penalty = config.collision_penalty

    def random_genome(self, rng: random.Random, length: int | None = None) -> Genome:
        from hpga.operators import random_genome

        return random_genome(length if length is not None else self._genome_length, rng)

    def to_str(self, genome: Genome) -> str:
        from hpga.operators import _genome_to_str

        return _genome_to_str(genome)  # type: ignore[arg-type]

    def length(self, genome: Genome) -> int:
        return len(genome)

    def evaluate_fitness(self, genome: Genome) -> float:
        from hpga.hp_model import evaluate_fitness

        return evaluate_fitness(genome, self._sequence, self._collision_penalty)  # type: ignore[arg-type]

    def distance(self, a: Genome, b: Genome) -> float:
        # Exact Hamming count -- the same pairwise term agents.py's
        # mean_pairwise_hamming already computes inline via zip(). Genomes
        # are fixed-length here by construction, so zip() never silently
        # truncates the way it would for the sequence model.
        return sum(x != y for x, y in zip(a, b))

    def deterministic_crossover(
        self, p1: Genome, p2: Genome, rate: float, rng: random.Random
    ) -> tuple[Genome, Genome]:
        from hpga.operators import crossover

        return crossover(p1, p2, rate, rng)  # type: ignore[arg-type]

    def deterministic_mutate(self, genome: Genome, rate: float, rng: random.Random) -> Genome:
        from hpga.operators import mutate

        return mutate(genome, rate, rng)  # type: ignore[arg-type]


class SequenceGenomeModel:
    """Adapter over hpga/sequence_model.py. Holds no state of its own --
    length bounds and the ESMFold target are sequence_model.py's own
    env-var-driven module constants (see that module's docstring), not
    config passed in here, so this constructor intentionally ignores most of
    `config` (only genome_model routing reads HPGAConfig; nothing in this
    model needs `config.sequence` or `config.collision_penalty`, which are
    lattice-specific)."""

    name = "sequence"

    def __init__(self, config: HPGAConfig):
        pass

    def random_genome(self, rng: random.Random, length: int | None = None) -> Genome:
        from hpga import sequence_model

        return sequence_model.random_sequence(rng, length)

    def to_str(self, genome: Genome) -> str:
        return genome  # type: ignore[return-value]  # already a plain str

    def length(self, genome: Genome) -> int:
        return len(genome)

    def evaluate_fitness(self, genome: Genome) -> float:
        from hpga import sequence_model

        return sequence_model.evaluate_fitness(genome)  # type: ignore[arg-type]

    def distance(self, a: Genome, b: Genome) -> float:
        from hpga import sequence_model

        return sequence_model.edit_distance(a, b)  # type: ignore[arg-type]

    def deterministic_crossover(
        self, p1: Genome, p2: Genome, rate: float, rng: random.Random
    ) -> tuple[Genome, Genome]:
        from hpga import sequence_model

        return sequence_model.crossover(p1, p2, rate, rng)  # type: ignore[arg-type]

    def deterministic_mutate(self, genome: Genome, rate: float, rng: random.Random) -> Genome:
        from hpga import sequence_model

        return sequence_model.mutate(genome, rate, rng)  # type: ignore[arg-type]


def build_genome_model(config: HPGAConfig) -> GenomeModel:
    if config.genome_model == "lattice":
        return LatticeGenomeModel(config)
    if config.genome_model == "sequence":
        return SequenceGenomeModel(config)
    raise ValueError(f"unknown genome_model: {config.genome_model!r}")


_active: GenomeModel | None = None


def set_active(model: GenomeModel) -> None:
    global _active
    _active = model


def current() -> GenomeModel:
    if _active is None:
        raise RuntimeError(
            "no active GenomeModel -- call genome_model.set_active(build_genome_model(config)) "
            "before anything that needs one"
        )
    return _active
