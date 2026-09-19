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

LLM prompt styles (bottom of this module) are the 20-symbol counterparts of
operators.py's 5-symbol ones, reached through
SequenceGenomeModel.plan_llm_crossover / plan_llm_mutate: crossover "full"
and "segment", mutate "full" and "position". "diff" and "best" are not
implemented yet and raise NotImplementedError rather than silently falling
through to "full" (the lattice code's behaviour for names a given operator
doesn't recognise) -- a fallthrough would run a different prompt than the
one named in HPGA_LLM_PROMPT_STYLE and in every log record's prompt_style
field, which is exactly the kind of mismatch that makes a compliance result
unattributable. Names that are just not-for-this-operator (crossover asked
for "position", mutate asked for "segment") still fall through to "full",
matching the lattice contract that lets one env var value drive both
operators. Genomes are shown to the model space-separated, like the lattice
ones, so alphabet size is not the only thing that changed -- but it is meant to
be nearly so, and the exceptions are listed here so a compliance gap between
the two models isn't misattributed to the alphabet:

  1. DELIBERATE: the "segment" crossover prompt contains a worked example
     ('e.g. 0-40:1, 40-100:2') that the lattice segment prompt does not. The
     convention differs (percentage boundaries that consecutive segments
     share, because the parents have different lengths, vs. the lattice's
     inclusive position ranges) and is unintuitive enough that leaving it
     unexplained would fail for reasons unrelated to the alphabet. A
     sequence-vs-lattice difference in segment compliance is therefore
     alphabet + boundary convention + example, not alphabet alone.
  2. Prompts state the allowed child length range (MIN_LENGTH-MAX_LENGTH),
     since children aren't fixed-length; the lattice prompts state one
     exact length.
  3. Crossover "full" is gated on edit distance, not positional Hamming.
  4. Unknown style names raise instead of falling through (see above).
"""

import os
import random
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hpga.genome_model import LLMOpPlan

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


_fitness = None  # lazy esmfold.tm_fitness.TMFitness singleton


def _get_fitness():
    """Loading ESMFold is a ~9s, ~13.7GB-of-GPU operation (RESULTS.md sec 3)
    -- deferred to first use, not import time or construction time, and
    cached process-wide rather than per SequenceGenomeModel instance so
    constructing more than one of those (e.g. a probe script building its
    own alongside Island's) doesn't try to load the model twice onto a GPU
    that only fits one copy. The object is esmfold.tm_fitness's process-wide
    default TMFitness (ESMFold once, kept warm; TM-score against 7UR7
    normalised by the reference length), so anything else in the process that
    calls esmfold.tm_fitness.tm_fitness() shares this one predictor. It is
    in-process only: the multiprocessing worker pool (worker.py) never sees
    it, so a GPU-resident predictor cannot be replicated per worker."""
    global _fitness
    if _fitness is None:
        from esmfold.tm_fitness import get_default

        _fitness = get_default()
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


# --- LLM prompt styles ------------------------------------------------------
#
# 20-symbol counterparts of operators.py's lattice styles. Same shape: each
# plan_* function returns an LLMOpPlan that operators._run_llm_op runs
# unchanged, so retry/fallback/logging/stats are shared with the lattice path.

_LETTER = f"[{ALPHABET}{ALPHABET.lower()}]"
_LETTER_RUN = f"[{ALPHABET}{ALPHABET.lower()} \\t,]"  # same line only -- see _extract_labelled
_ALPHA_SET = "{" + ",".join(ALPHABET) + "}"

_IMPLEMENTED = {"crossover": ("full", "segment"), "mutate": ("full", "position")}
_NOT_YET = ("diff", "best")
_KNOWN = ("", "full", "diff", "position", "segment", "best")


def _resolve_style(op: str, style: str) -> str:
    """Map a HPGA_LLM_PROMPT_STYLE value to the style this operator runs.
    Not-yet-implemented styles and unknown names raise (see module docstring
    for why not fall through); names that belong to the other operator fall
    through to 'full', as in the lattice code."""
    if style in _NOT_YET:
        raise NotImplementedError(
            f"HPGA_LLM_PROMPT_STYLE={style!r} is not implemented for the sequence model "
            f"(implemented: crossover {_IMPLEMENTED['crossover']}, mutate {_IMPLEMENTED['mutate']})"
        )
    if style not in _KNOWN:
        raise ValueError(f"unknown HPGA_LLM_PROMPT_STYLE={style!r} (known: {_KNOWN[1:]})")
    return style if style in _IMPLEMENTED[op] else "full"


def _spaced(sequence: str) -> str:
    return " ".join(sequence)


def _crossover_min_diff() -> int:
    from hpga import operators  # lazy: one source of truth for the knob, and patchable there

    return operators.CROSSOVER_MIN_DIFF


def _extract_labelled(text: str, label: str) -> str | None:
    """Parse '<label>: <letters>' into an upper-cased string, or None. The
    letter run is kept to ONE line ([ \\t,], not \\s): with a 20-letter
    alphabet, the next line's own label ('CHILD2') is itself made of valid
    letters, so a newline-tolerant run would swallow it. Otherwise as lenient
    as the lattice parser -- letters may be separated by spaces, commas or
    nothing, and anything after the run on that line is ignored. Length
    checking is the caller's job (unlike the lattice version, sequence
    lengths aren't fixed)."""
    m = re.search(rf"{label}\s*:\s*({_LETTER}{_LETTER_RUN}*)", text)
    if not m:
        return None
    return "".join(c.upper() for c in re.findall(_LETTER, m.group(1)))


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


_POSITION_PAIR = re.compile(rf"POSITION\s*:\s*(\d+)\s*,\s*NEW\s*:\s*({_LETTER})", re.IGNORECASE)


def _extract_position_mutation(text: str, base: str, k: int) -> str | None:
    """Parse exactly `k` 'POSITION: <n>, NEW: <letter>' lines and apply them
    to `base`. Same rejections as the lattice parser: wrong pair count, a
    position out of range, a repeated position, or a NEW letter equal to the
    one already there."""
    pairs = _POSITION_PAIR.findall(text)
    if len(pairs) != k:
        return None
    result = list(base)
    seen: set[int] = set()
    for pos_str, letter in pairs:
        pos = int(pos_str)
        if pos < 0 or pos >= len(base) or pos in seen:
            return None
        letter = letter.upper()
        if letter == base[pos]:
            return None
        seen.add(pos)
        result[pos] = letter
    return "".join(result)


_SEGMENT_TRIPLE = re.compile(r"(\d+)\s*-\s*(\d+)\s*:\s*([12])")


def _extract_segment_crossover(text: str, p1: str, p2: str) -> tuple[str, str] | None:
    """Parse 'SEGMENTS: <start>-<end>:<parent>, ...'. The parents have
    different lengths, so -- unlike the lattice version, where segments are
    inclusive position ranges -- boundaries are PERCENTAGES along a sequence
    (0 = start, 100 = end), mapped onto each parent's own length with the
    same rounding as crossover() above (option A, aligned relative-position).
    Consecutive segments SHARE a boundary value ('0-40:1, 40-100:2'), so a
    valid declaration is a chain of boundaries 0 = b0 < b1 < ... < bn = 100
    and every parent is partitioned exactly, with no rounding gaps.
    Structural rules otherwise as the lattice parser: >=2 segments, both
    parents used. CHILD1 takes each segment from the named parent, CHILD2
    from the other. A child outside [MIN_LENGTH, MAX_LENGTH] is invalid --
    same bound crossover() enforces, and the model isn't told the exact
    lengths its cut will produce, so this can legitimately fail."""
    m = re.search(r"SEGMENTS\s*:\s*(.*)", text)
    if not m:
        return None
    triples = _SEGMENT_TRIPLE.findall(m.group(1))
    if len(triples) < 2:
        return None
    segments = sorted((int(s), int(e), int(p)) for s, e, p in triples)
    if segments[0][0] != 0 or segments[-1][1] != 100:
        return None
    if any(s >= e for s, e, _ in segments):
        return None
    for i in range(1, len(segments)):
        if segments[i][0] != segments[i - 1][1]:
            return None  # gap or overlap
    if {p for _, _, p in segments} != {1, 2}:
        return None
    child1 = child2 = ""
    for start, end, parent in segments:
        src1, src2 = (p1, p2) if parent == 1 else (p2, p1)
        child1 += src1[round(start * len(src1) / 100) : round(end * len(src1) / 100)]
        child2 += src2[round(start * len(src2) / 100) : round(end * len(src2) / 100)]
    if not (MIN_LENGTH <= len(child1) <= MAX_LENGTH and MIN_LENGTH <= len(child2) <= MAX_LENGTH):
        return None
    return child1, child2


def _crossover_sufficiently_mixed(c1: str, c2: str, p1: str, p2: str) -> bool:
    """Counterpart of operators._crossover_sufficiently_mixed with edit
    distance in place of positional Hamming (children and parents can differ
    in length, and zip() would silently truncate)."""
    d = _crossover_min_diff()
    return all(edit_distance(c, p) >= d for c in (c1, c2) for p in (p1, p2))


_CROSSOVER_SYSTEM = (
    "You recombine two parent protein sequences for a genetic algorithm that "
    "designs a protein chain. Each residue is one letter from the 20 standard "
    "amino-acid codes (A C D E F G H I K L M N P Q R S T V W Y), and sequences "
    "are written as letters separated by single spaces. Follow the requested "
    "output format exactly and output nothing else."
)

_MUTATE_SYSTEM = (
    "You mutate a protein sequence for a genetic algorithm that designs a "
    "protein chain. Each residue is one letter from the 20 standard "
    "amino-acid codes (A C D E F G H I K L M N P Q R S T V W Y), and sequences "
    "are written as letters separated by single spaces. Follow the requested "
    "output format exactly and output nothing else."
)

_RETRY_HINT_SEGMENT = (
    "\nIMPORTANT: your previous response did not match the required "
    "format -- the segments must run from 0 to 100 in increasing order, each "
    "starting exactly where the previous one ended (e.g. 0-40:1, 40-100:2), "
    "must use at least 2 segments, must use both parents at least once, and "
    "must give two children within the allowed length range. Respond with "
    "ONLY the SEGMENTS line above."
)


def _crossover_full_prompt(p1_str: str, p2_str: str, retry_hint: str) -> str:
    return f"""Parent 1: {p1_str}
Parent 2: {p2_str}

Produce two children, each between {MIN_LENGTH} and {MAX_LENGTH} letters long, using only the letters {_ALPHA_SET},
by recombining segments from Parent 1 and Parent 2 (mix contiguous or
interleaved segments from both parents, similar to genetic crossover). The
two children may have different lengths.

Respond with EXACTLY two lines and nothing else:
CHILD1: <{MIN_LENGTH}-{MAX_LENGTH} letters from {_ALPHA_SET} separated by single spaces>
CHILD2: <{MIN_LENGTH}-{MAX_LENGTH} letters from {_ALPHA_SET} separated by single spaces>
{retry_hint}"""


def _crossover_segment_prompt(p1_str: str, p2_str: str, n1: int, n2: int, retry_hint: str) -> str:
    return f"""Parent 1 (length {n1}): {p1_str}
Parent 2 (length {n2}): {p2_str}

Produce CHILD1 by dividing the sequences into contiguous segments and
assigning each segment to a source parent (1 or 2) -- CHILD1 takes its
letters from that parent for that stretch. CHILD2 is the complement:
wherever CHILD1 used Parent 1, CHILD2 uses Parent 2, and vice versa.

The parents have different lengths, so segments are given as percentages of
the way along a sequence: 0 is the start and 100 is the end of each parent,
and a boundary at 40 falls 40% of the way along Parent 1 and 40% of the way
along Parent 2. Each segment is <start>-<end>, where <end> is where that
segment stops and the next one begins, so consecutive segments share a
boundary value (e.g. 0-40:1, 40-100:2). Segments must start at 0, end at
100, and be in increasing order with no gaps or overlaps. Use at least 2
segments, and use both parents at least once. Both children must come out
between {MIN_LENGTH} and {MAX_LENGTH} letters long.

Respond with EXACTLY one line and nothing else:
SEGMENTS: <start>-<end>:<parent>, <start>-<end>:<parent>, ...
{retry_hint}"""


def _mutate_full_prompt(genome_str: str, length: int, k: int, retry_hint: str) -> str:
    return f"""Sequence: {genome_str}

Change EXACTLY {k} of the {length} positions to a different letter from
{_ALPHA_SET}; leave the other {length - k} positions unchanged. Choose which
{k} positions to change yourself.

Respond with EXACTLY one line and nothing else:
MUTATED: <{length} letters from {_ALPHA_SET} separated by single spaces>
{retry_hint}"""


def _mutate_position_prompt(genome_str: str, length: int, k: int, retry_hint: str) -> str:
    return f"""Sequence (0-indexed positions 0-{length - 1}): {genome_str}

Change exactly {k} position(s) in this sequence. For each one, name the
position and the new letter it becomes -- the new letter MUST differ from
whatever letter is currently at that position (copying the current letter
back is not a change). Do not restate the sequence.

Respond with EXACTLY {k} line(s) and nothing else, one change per line:
POSITION: <0-{length - 1}>, NEW: <one letter from {_ALPHA_SET}>
{retry_hint}"""


def plan_llm_crossover(style: str, parent1: str, parent2: str) -> LLMOpPlan:
    style = _resolve_style("crossover", style)
    p1_str, p2_str = _spaced(parent1), _spaced(parent2)
    n1, n2 = len(parent1), len(parent2)

    if style == "segment":
        num_predict = min(2048, max(32, 6 * max(n1, n2)))

        def build_prompt(retry_hint: str) -> str:
            return _crossover_segment_prompt(p1_str, p2_str, n1, n2, retry_hint)

        def parse(text: str):
            # Not gated by _crossover_sufficiently_mixed, for the same reason
            # as the lattice segment style: the child is built FROM the
            # model's declared cut, so matching it is true by construction
            # and only the declaration's structure needs checking.
            return _extract_segment_crossover(text, parent1, parent2)

        retry_hint_text = _RETRY_HINT_SEGMENT
    else:
        num_predict = min(2048, max(64, 8 * MAX_LENGTH))

        def build_prompt(retry_hint: str) -> str:
            return _crossover_full_prompt(p1_str, p2_str, retry_hint)

        def parse(text: str):
            c1 = _extract_labelled(text, "CHILD1")
            c2 = _extract_labelled(text, "CHILD2")
            if c1 is None or c2 is None:
                return None
            if not (MIN_LENGTH <= len(c1) <= MAX_LENGTH and MIN_LENGTH <= len(c2) <= MAX_LENGTH):
                return None
            if not _crossover_sufficiently_mixed(c1, c2, parent1, parent2):
                return None
            return (c1, c2)

        retry_hint_text = (
            "\nIMPORTANT: your previous response did not match the required "
            "format. Respond with ONLY the labelled line(s) above, each "
            f"containing between {MIN_LENGTH} and {MAX_LENGTH} of the 20 "
            "amino-acid letters separated by single spaces."
            f" Each child must differ from BOTH Parent 1 and Parent 2 in at "
            f"least {_crossover_min_diff()} positions -- do not copy one parent "
            f"unchanged (or nearly unchanged), actually recombine material "
            f"from both."
        )

    return LLMOpPlan(
        system=_CROSSOVER_SYSTEM, build_prompt=build_prompt, parse=parse,
        num_predict=num_predict, retry_hint_text=retry_hint_text,
    )


def plan_llm_mutate(style: str, genome: str, k: int) -> LLMOpPlan:
    style = _resolve_style("mutate", style)
    length = len(genome)
    genome_str = _spaced(genome)

    if style == "position":
        num_predict = min(2048, max(24, 10 * k))

        def build_prompt(retry_hint: str) -> str:
            return _mutate_position_prompt(genome_str, length, k, retry_hint)

        def parse(text: str):
            return _extract_position_mutation(text, genome, k)

        retry_hint_text = (
            f"\nIMPORTANT: your previous response did not match the "
            f"required format, gave a number of lines other than {k}, "
            f"repeated a position, or gave a NEW letter equal to the "
            f"current letter at that position (that is not a change). "
            f"Respond with ONLY the {k} labelled line(s) above."
        )
    else:
        num_predict = min(2048, max(32, 6 * length))

        def build_prompt(retry_hint: str) -> str:
            return _mutate_full_prompt(genome_str, length, k, retry_hint)

        def parse(text: str):
            parsed = _extract_labelled(text, "MUTATED")
            if parsed is None or len(parsed) != length or _hamming(parsed, genome) != k:
                return None
            return parsed

        retry_hint_text = (
            "\nIMPORTANT: your previous response did not match the required "
            "format. Respond with ONLY the labelled line(s) above, each "
            f"containing exactly {length} of the 20 amino-acid letters "
            "separated by single spaces."
            f" Exactly {k} of the {length} positions must differ from the "
            f"input Sequence -- not approximately {k}, exactly {k}."
        )

    return LLMOpPlan(
        system=_MUTATE_SYSTEM, build_prompt=build_prompt, parse=parse,
        num_predict=num_predict, retry_hint_text=retry_hint_text,
    )
