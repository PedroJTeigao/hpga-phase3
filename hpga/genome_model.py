"""The genome-type abstraction boundary: everything that differs between the
lattice genome (fixed-length list[int], 5-symbol move alphabet, hp_model.py)
and the sequence genome (variable-length str, 20-symbol amino-acid alphabet,
hpga/sequence_model.py) is isolated behind the `GenomeModel` interface below,
so that code written against `GenomeModel` need not know which kind is live.
(That is the goal, not yet the state of the codebase -- see "Not yet
dispatched" below: as of this writing operators.py's LLM entry points are the
only callers of this module's model-selection machinery.)

What the interface covers: constructing a fresh genome, measuring
fitness/distance, the deterministic crossover/mutate operators, and -- for
operators.py's `_llm_crossover` / `_llm_mutate` -- LLM prompt-style dispatch.
The latter is `plan_llm_crossover` / `plan_llm_mutate`: given the requested
style (HPGA_LLM_PROMPT_STYLE) and the genome(s), a model returns an
`LLMOpPlan` (system prompt, prompt builder, response parser, num_predict,
retry hint) that operators._run_llm_op runs unchanged. The retry / fallback /
logging / stats machinery, the RNG draw order, and the rate gate stay in
operators.py and are model-independent. The lattice plans are the original
5-symbol style code, still living in operators.py
(`_lattice_crossover_plan` / `_lattice_mutate_plan`, moved out of the two
entry points verbatim); the sequence plans live in sequence_model.py.

Selection: `genome_model.build_genome_model(config)` constructs the right
model from `HPGAConfig.genome_model`; `set_active`/`current` hold a
process-local active instance (not a per-call env-var read like
HPGA_OPERATOR_MODE) because a GenomeModel can carry real state -- the
sequence model's ESMFold weights live on the GPU for the process's lifetime,
not something a bare string switch could represent.

How operators.py's LLM entry points resolve the model -- and the one place
this module's original "never default to lattice" rule no longer holds:

  `current()` still raises if nothing has called `set_active`. But
  operators._llm_crossover/_llm_mutate use `active_or_none()` instead, and
  when it returns None they fall back to a lattice model IF the genome is a
  list. The reason is compatibility, not preference: the experiment scripts
  call these entry points directly (or via next_generation), never call
  set_active, and their outputs back published results, so requiring an
  active model would have meant editing every one of them.

  What still fails loudly: a `str` genome with no active model, and an active
  model whose `genome_type` doesn't match the genome passed in.

  The gap that remains, and can't be detected from inside operators.py: a run
  configured with genome_model="sequence" that never calls set_active AND
  hands these functions list genomes gets lattice prompts silently, exactly
  the "sequence run silently measuring lattice" failure the old rule existed
  to prevent. Sequence genomes are str, so ordinary sequence-model code
  can't reach it -- it needs a lattice-only code path feeding a
  sequence-configured run, which is precisely what the "Not yet dispatched"
  list below is. Closing that list closes the gap; until then, any driver
  for genome_model="sequence" must call
  `set_active(build_genome_model(config))` itself.

Not yet dispatched through this module (found while wiring the LLM entry
points; known scope of the follow-up task, deliberately left alone here):

  - operators.tournament_select: `list(population[best])` turns a str genome
    into a list of single characters.
  - operators.next_generation's elitism copy
    (`[list(population[i]) for i in ranked[:elitism]]`): same problem.
  - operators.random_genome / random_population: always lattice (MOVES).
  - operators.next_generation's deterministic branch calls the lattice
    crossover()/mutate() directly rather than model.deterministic_*.
  - operators.next_generation's agents/circles hooks assume fixed length
    (`genome_length = len(population[0])`).
  - agents.py / circles.py: their LLM calls hardcode the 5-letter alphabet
    (ops._CHAR_TO_MOVE, ops._genome_to_str, ops._extract_labelled, and the
    S/L/R/U/D parsers in agents._extract_refine_position and
    circles._extract_fold_line).
  - agents.mean_pairwise_hamming / log_diversity: zip()-based Hamming, which
    silently truncates on unequal lengths; GenomeModel.distance exists for
    exactly this and isn't used there.
  - Nothing outside operators.py's LLM entry points calls set_active or
    current() yet, including island.py.
"""

import random
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from hpga.config import HPGAConfig

Genome = list[int] | str


@dataclass(frozen=True)
class LLMOpPlan:
    """Everything operators._run_llm_op needs for one LLM crossover/mutate
    call at one prompt style. `parse` returns a genome (mutate), a
    (child1, child2) tuple (crossover), or None for a malformed response --
    None is what triggers retry/fallback there."""

    system: str
    build_prompt: Callable[[str], str]  # retry_hint -> prompt
    parse: Callable[[str], Any]
    num_predict: int
    retry_hint_text: str


class GenomeModel(Protocol):
    name: str
    genome_type: type  # list / str -- lets callers reject a mismatched genome

    def copy(self, genome: Genome) -> Genome:
        """A fresh copy safe to hand back where the caller expects its own
        genome (list(g) for lattice; the str itself for sequence)."""
        ...

    def plan_llm_crossover(self, style: str, p1: Genome, p2: Genome) -> LLMOpPlan: ...

    def plan_llm_mutate(self, style: str, genome: Genome, k: int) -> LLMOpPlan:
        """`k` = number of positions the operator must change, computed by
        the caller from the mutation rate (max(1, round(rate * length)))."""
        ...

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
    genome_type = list

    def __init__(self, config: HPGAConfig | None = None):
        # `config=None` is the config-less instance operators.py's LLM entry
        # points fall back to when no model is active (see module docstring):
        # the operator/LLM-plan methods below need nothing from config, only
        # random_genome / evaluate_fitness do, and those raise clearly if it's
        # missing rather than guessing a sequence or length.
        self._sequence = config.sequence if config is not None else None
        self._genome_length = config.genome_length if config is not None else None
        self._collision_penalty = config.collision_penalty if config is not None else None

    def _require_config(self, what: str) -> None:
        if self._sequence is None:
            raise RuntimeError(f"LatticeGenomeModel.{what} needs a model built from an HPGAConfig")

    def random_genome(self, rng: random.Random, length: int | None = None) -> Genome:
        from hpga.operators import random_genome

        if length is None:
            self._require_config("random_genome (without an explicit length)")
        return random_genome(length if length is not None else self._genome_length, rng)

    def to_str(self, genome: Genome) -> str:
        from hpga.operators import _genome_to_str

        return _genome_to_str(genome)  # type: ignore[arg-type]

    def length(self, genome: Genome) -> int:
        return len(genome)

    def evaluate_fitness(self, genome: Genome) -> float:
        from hpga.hp_model import evaluate_fitness

        self._require_config("evaluate_fitness")
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

    def copy(self, genome: Genome) -> Genome:
        return list(genome)

    def plan_llm_crossover(self, style: str, p1: Genome, p2: Genome) -> LLMOpPlan:
        from hpga.operators import _lattice_crossover_plan

        return _lattice_crossover_plan(style, p1, p2, len(p1))  # type: ignore[arg-type]

    def plan_llm_mutate(self, style: str, genome: Genome, k: int) -> LLMOpPlan:
        from hpga.operators import _lattice_mutate_plan

        return _lattice_mutate_plan(style, genome, len(genome), k)  # type: ignore[arg-type]


class SequenceGenomeModel:
    """Adapter over hpga/sequence_model.py. Holds no state of its own --
    length bounds and the ESMFold target are sequence_model.py's own
    env-var-driven module constants (see that module's docstring), not
    config passed in here, so this constructor intentionally ignores most of
    `config` (only genome_model routing reads HPGAConfig; nothing in this
    model needs `config.sequence` or `config.collision_penalty`, which are
    lattice-specific)."""

    name = "sequence"
    genome_type = str

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

    def copy(self, genome: Genome) -> Genome:
        return genome  # str is immutable

    def plan_llm_crossover(self, style: str, p1: Genome, p2: Genome) -> LLMOpPlan:
        from hpga import sequence_model

        return sequence_model.plan_llm_crossover(style, p1, p2)  # type: ignore[arg-type]

    def plan_llm_mutate(self, style: str, genome: Genome, k: int) -> LLMOpPlan:
        from hpga import sequence_model

        return sequence_model.plan_llm_mutate(style, genome, k)  # type: ignore[arg-type]


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


def active_or_none() -> GenomeModel | None:
    """`current()` without the raise -- for callers that have a defined
    fallback when nothing is active (operators.py's LLM entry points; see the
    module docstring for why they're allowed one)."""
    return _active


def current() -> GenomeModel:
    if _active is None:
        raise RuntimeError(
            "no active GenomeModel -- call genome_model.set_active(build_genome_model(config)) "
            "before anything that needs one"
        )
    return _active
