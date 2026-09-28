"""Fitness-aware sequence operator prompts: arm C's two prompts with the
genome's fitness added as one labelled line per sequence, and NOTHING else
changed.

Why a separate module. hpga/sequence_model.py's prompt text is what five
published arms (A-E, results/SEQUENCE_GA_REPORT.md) were measured with. This
module never edits it -- it takes the plan sequence_model.py builds and
inserts one line into the prompt, so "byte-identical apart from the fitness
line" is true by construction rather than by a copy that can drift. The
insertion asserts the line it inserts after (see _insert_lines), so a future
change to the base prompt's shape raises here instead of silently producing a
prompt that differs in more than the fitness line. strip_fitness_lines()
inverts the insertion, which is what experiments/verify_fitness_prompts.py
uses to prove byte-identity on every case it generates.

Scope: exactly the two styles arm C runs -- mutate "position", crossover
"segment". Any other style raises NotImplementedError rather than falling
through to an unlabelled prompt, following this codebase's existing rule
(hpga/sequence_model.py's docstring): a fallthrough would run a different
prompt than the one named in HPGA_LLM_PROMPT_STYLE and in every log record's
prompt_style field, which is what makes a compliance result unattributable.
With fitness=None every plan function here returns sequence_model.py's plan
unchanged, so this model is exactly arm C until a fitness context is passed.

WHAT THE NUMBER IS, AND WHY THERE ARE TWO CASES. Crossover's two inputs are
population members: the GA has folded both, so each parent's fitness is exact
and sits next to the sequence it belongs to. Mutate's input is a
post-crossover child the GA has NOT folded -- its fitness is computed one
generation later, after mutation, so at mutate time there is no true "this
sequence's fitness" to show and folding one per call would change the fold
budget and break comparability with arm C. So mutate has two cases, with
different labels, and which one applied is recorded per call (see
MutateFitness.log_fields -- without that field the arm cannot be analysed,
because the two cases are different treatments):

  inherited  the child differs from both parents: the line reports the
             LINEAGE's scores ("Parent fitness scores: a and b"), never the
             child's own, because the child's own is unknown.
  own        the child is byte-identical to one of its parents, so its
             fitness IS that parent's, exactly (fitness is a deterministic
             function of the genome string). The line says so
             ("Fitness of this sequence: x").

The "own" case is a property of the genome, not of the code path: it is true
whenever child == parent, however that arose -- the crossover rate gate
declining to recombine (~10% of calls at rate 0.9), identical parents, a
deterministic-crossover fallback that copied, or a recombination that
happened to reproduce a parent. Elites never reach mutate at all
(next_generation copies them straight into the new population), so this
module never has to describe a carried-through elite.

Both cases always show a fitness line. A line that appeared only when the
number was exact would confound "fitness shown" with "genome is a copy of a
parent" -- the copies are exactly the genomes crossover did not change -- so
presence is held constant and the basis is put in the label and the log.
"""

from dataclasses import dataclass, replace

from hpga import sequence_model as sm
from hpga.genome_model import LLMOpPlan, SequenceGenomeModel

# 5 decimals: the precision results/raw/*.json actually carries (TM-align
# emits 5, e.g. 0.46325). The 3-decimal {f:.3f} used by hpga/circles_sequence.py's
# coordination prompts would collapse fitness gaps of <0.001, and tournament
# parents routinely differ by that much -- exactly where this line is meant to
# be informative.
_DP = 5


def _fmt(value: float) -> str:
    return f"{value:.{_DP}f}"


# The lines this module inserts. Kept as prefixes so strip_fitness_lines() and
# the parity verifier can recognise an inserted line without re-deriving its
# formatting.
_OWN_LABEL = "Fitness of this sequence"
_INHERITED_LABEL = "Parent fitness scores"
_PARENT_LABEL = "Parent {} fitness"
FITNESS_LABELS = (_OWN_LABEL, _INHERITED_LABEL, "Parent 1 fitness", "Parent 2 fitness")

# Anchors: the line each fitness line is inserted immediately after, asserted
# at insertion time. These are the opening lines of sequence_model.py's
# _mutate_position_prompt and _crossover_segment_prompt.
_MUTATE_ANCHOR = "Sequence (0-indexed positions 0-"
_CROSSOVER_ANCHOR_1 = "Parent 1 (length "
_CROSSOVER_ANCHOR_2 = "Parent 2 (length "


@dataclass(frozen=True)
class MutateFitness:
    """What is known about the fitness of the genome mutate is about to change.

    basis "own": `values` is one number, the genome's own exact fitness.
    basis "inherited": `values` is the two parents' fitnesses, in (parent1,
    parent2) order -- the child's own is unknown. Two values are shown even
    when the parents are identical, so the prompt's shape does not vary with
    the population's convergence."""

    values: tuple[float, ...]
    basis: str

    def __post_init__(self):
        if self.basis == "own" and len(self.values) != 1:
            raise ValueError(f"basis 'own' takes exactly 1 value, got {len(self.values)}")
        if self.basis == "inherited" and len(self.values) != 2:
            raise ValueError(f"basis 'inherited' takes exactly 2 values, got {len(self.values)}")
        if self.basis not in ("own", "inherited"):
            raise ValueError(f"basis must be 'own' or 'inherited', got {self.basis!r}")

    def prompt_line(self) -> str:
        if self.basis == "own":
            return f"{_OWN_LABEL}: {_fmt(self.values[0])}"
        return f"{_INHERITED_LABEL}: {_fmt(self.values[0])} and {_fmt(self.values[1])}"

    def log_fields(self) -> dict:
        """Merged into every log record of the call this context was built for.
        `n_parents_shown` is 0 for basis 'own' -- the number shown is the
        sequence's own, not a parent's -- and 2 for 'inherited'."""
        return {
            "fitness_shown": True,
            "fitness_basis": self.basis,
            "fitness_values_shown": [round(v, _DP) for v in self.values],
            "n_parents_shown": 0 if self.basis == "own" else 2,
            "fitness_decimals": _DP,
        }


@dataclass(frozen=True)
class CrossoverFitness:
    """The two parents' exact fitnesses, one per sequence shown. basis is
    always 'own': each number is the exact fitness of the sequence it is
    printed next to, because both parents are folded population members."""

    parent1: float
    parent2: float

    def prompt_lines(self) -> tuple[str, str]:
        return (
            f"{_PARENT_LABEL.format(1)}: {_fmt(self.parent1)}",
            f"{_PARENT_LABEL.format(2)}: {_fmt(self.parent2)}",
        )

    def log_fields(self) -> dict:
        return {
            "fitness_shown": True,
            "fitness_basis": "own",
            "fitness_values_shown": [round(self.parent1, _DP), round(self.parent2, _DP)],
            "n_parents_shown": 2,
            "fitness_decimals": _DP,
        }


def _insert_lines(text: str, insertions: list[tuple[int, str, str]]) -> str:
    """Insert each `line` immediately after line `index` of `text`, asserting
    that line starts with `expect_prefix` first. Applied from the highest
    index down so earlier insertions don't shift later anchors.

    The assertion is the point: it makes "the fitness line is the ONLY
    difference" a checked property. If sequence_model.py's prompt is reworded
    so the anchor moves, this raises rather than inserting the line somewhere
    else."""
    lines = text.split("\n")
    for index, _, expect_prefix in insertions:
        if index >= len(lines) or not lines[index].startswith(expect_prefix):
            found = lines[index] if index < len(lines) else "<past end of prompt>"
            raise RuntimeError(
                f"fitness line cannot be placed: expected line {index} of the base prompt to start "
                f"with {expect_prefix!r}, found {found!r}. hpga/sequence_model.py's prompt shape "
                f"changed; update hpga/sequence_model_fitness.py's anchors deliberately rather than "
                f"letting the line land in the wrong place."
            )
    for index, line, _ in sorted(insertions, key=lambda t: -t[0]):
        lines.insert(index + 1, line)
    return "\n".join(lines)


def strip_fitness_lines(text: str) -> str:
    """Inverse of the insertion: drop every line this module could have
    inserted. Used by experiments/verify_fitness_prompts.py to prove a
    fitness-aware prompt equals the base prompt once the fitness line is
    removed."""
    kept = [ln for ln in text.split("\n") if not any(ln.startswith(f"{lab}:") for lab in FITNESS_LABELS)]
    return "\n".join(kept)


def plan_llm_mutate(style: str, genome: str, k: int, fitness: MutateFitness | None = None) -> LLMOpPlan:
    """sequence_model.plan_llm_mutate's plan with one fitness line inserted.
    fitness=None returns that plan untouched (this model is then arm C)."""
    base = sm.plan_llm_mutate(style, genome, k)
    if fitness is None:
        return base
    resolved = sm._resolve_style("mutate", style)
    if resolved != "position":
        raise NotImplementedError(
            f"the fitness-aware mutate prompt is defined only for style 'position' (arm C's style); "
            f"HPGA_LLM_PROMPT_STYLE={style!r} resolves to {resolved!r}"
        )
    line = fitness.prompt_line()

    def build_prompt(retry_hint: str) -> str:
        return _insert_lines(base.build_prompt(retry_hint), [(0, line, _MUTATE_ANCHOR)])

    # replace() keeps system, parse, num_predict and retry_hint_text exactly as
    # sequence_model.py built them: build_prompt is the only field that differs.
    return replace(base, build_prompt=build_prompt)


def plan_llm_crossover(
    style: str, parent1: str, parent2: str, fitness: CrossoverFitness | None = None
) -> LLMOpPlan:
    """sequence_model.plan_llm_crossover's plan with one fitness line inserted
    after each parent. fitness=None returns that plan untouched."""
    base = sm.plan_llm_crossover(style, parent1, parent2)
    if fitness is None:
        return base
    resolved = sm._resolve_style("crossover", style)
    if resolved != "segment":
        raise NotImplementedError(
            f"the fitness-aware crossover prompt is defined only for style 'segment' (arm C's style); "
            f"HPGA_LLM_PROMPT_STYLE={style!r} resolves to {resolved!r}"
        )
    line1, line2 = fitness.prompt_lines()

    def build_prompt(retry_hint: str) -> str:
        return _insert_lines(
            base.build_prompt(retry_hint),
            [(0, line1, _CROSSOVER_ANCHOR_1), (1, line2, _CROSSOVER_ANCHOR_2)],
        )

    return replace(base, build_prompt=build_prompt)


class FitnessAwareSequenceGenomeModel(SequenceGenomeModel):
    """SequenceGenomeModel with the two fitness-aware prompts. Everything else
    -- random_genome, evaluate_fitness, distance, the deterministic operators,
    copy -- is inherited unchanged, so the only difference from arm C is the
    prompt text and the extra log fields.

    `wants_fitness_context` is what operators.next_generation tests (by
    getattr, defaulting False) to decide whether to build fitness contexts at
    all. No other model defines it, so no other model's path changes."""

    name = "sequence_fitness"
    genome_type = str
    wants_fitness_context = True

    def plan_llm_crossover(self, style, p1, p2, fitness: CrossoverFitness | None = None) -> LLMOpPlan:
        return plan_llm_crossover(style, p1, p2, fitness)

    def plan_llm_mutate(self, style, genome, k, fitness: MutateFitness | None = None) -> LLMOpPlan:
        return plan_llm_mutate(style, genome, k, fitness)

    # --- context construction (called by operators.next_generation) ---------
    #
    # next_generation holds `population` and `fitnesses`; these two turn that
    # into the number(s) each prompt shows. Kept on the model, not in
    # operators.py, so operators.py needs no import from this module and no
    # knowledge of what a fitness context is -- it passes an opaque object
    # through and merges whatever log_fields() returns.

    @staticmethod
    def _lookup(genome: str, fitness_by_genome: dict) -> float:
        if genome not in fitness_by_genome:
            raise RuntimeError(
                f"no fitness recorded for a selected parent ({genome[:20]}...): next_generation's "
                f"fitness map should cover every member of the population it was given"
            )
        return fitness_by_genome[genome]

    def crossover_fitness_context(self, p1: str, p2: str, fitness_by_genome: dict) -> CrossoverFitness:
        return CrossoverFitness(self._lookup(p1, fitness_by_genome), self._lookup(p2, fitness_by_genome))

    def mutate_fitness_context(
        self, child: str, p1: str, p2: str, fitness_by_genome: dict
    ) -> MutateFitness:
        """'own' when the child is byte-identical to a parent -- then its
        fitness is that parent's, exactly. Otherwise 'inherited'."""
        f1, f2 = self._lookup(p1, fitness_by_genome), self._lookup(p2, fitness_by_genome)
        if child == p1:
            return MutateFitness((f1,), "own")
        if child == p2:
            return MutateFitness((f2,), "own")
        return MutateFitness((f1, f2), "inherited")
