"""Genetic operators: tournament selection and elitism are always
deterministic. Crossover and mutation have two implementations selectable at
call time:

  - deterministic: single-point crossover, point mutation (Phase 1 behaviour,
    unchanged, still the default).
  - llm: crossover and mutation are LLM calls against a local Ollama server
    (Phase 2). Selection controls which parents are combined; the LLM only
    replaces the "how do these genes recombine/mutate" step, called once per
    crossover() and once per mutate() invocation -- the same call cardinality
    as the deterministic path, so LLM latency is genuinely substituting for
    near-zero deterministic compute rather than also multiplying call count.

Mode is chosen per call via the HPGA_OPERATOR_MODE env var ("deterministic",
the default, or "llm"), read fresh on every next_generation() call rather
than cached at import time, so a single process can run paired
deterministic/LLM comparisons back to back without a restart. This keeps the
switch entirely inside this module -- island.py's call to next_generation()
is unchanged, and so is its signature.
"""

import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from hpga.hp_model import MOVES
from hpga import agents
from hpga import circles
from hpga import genome_model
from hpga.genome_model import LLMOpPlan

Genome = list[int]

# --- Deterministic operators (Phase 1, unchanged) -------------------------


def random_genome(length: int, rng: random.Random) -> Genome:
    return [rng.choice(MOVES) for _ in range(length)]


def random_population(pop_size: int, genome_length: int, rng: random.Random) -> list[Genome]:
    return [random_genome(genome_length, rng) for _ in range(pop_size)]


def tournament_select(population: Sequence[Genome], fitnesses: Sequence[float], k: int, rng: random.Random) -> Genome:
    idxs = rng.sample(range(len(population)), k)
    best = max(idxs, key=lambda i: fitnesses[i])
    return list(population[best])


def crossover(parent1: Genome, parent2: Genome, rate: float, rng: random.Random) -> tuple[Genome, Genome]:
    if rng.random() > rate or len(parent1) < 2:
        return list(parent1), list(parent2)
    point = rng.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(genome: Genome, rate: float, rng: random.Random) -> Genome:
    return [rng.choice(MOVES) if rng.random() < rate else gene for gene in genome]


# --- LLM operators (Phase 2) -----------------------------------------------
#
# Move-code <-> letter mapping matches hp_model.MOVES = (STRAIGHT, LEFT,
# RIGHT, UP, DOWN) = (0, 1, 2, 3, 4) (3D migration -- see README.md's "3D
# migration" section; U/D checked against every regex/label below for
# collisions before picking them, none found: no existing format uses U or D
# as a delimiter, flag, or label-initial character). Genomes are represented
# to the model as space-separated letters ("S L R U D S ...") -- an encoding
# chosen only for prompt/parse convenience, not a change to the genome
# representation itself, which stays list[int] throughout, identical to the
# deterministic path.

_MOVE_TO_CHAR = {0: "S", 1: "L", 2: "R", 3: "U", 4: "D"}
_CHAR_TO_MOVE = {"S": 0, "L": 1, "R": 2, "U": 3, "D": 4}
_VALID_CHARS = re.compile(r"[SLRUD]", re.IGNORECASE)

LLM_MODEL = os.environ.get("HPGA_LLM_MODEL", "gemma4:12b")
LLM_HOST = os.environ.get("HPGA_OLLAMA_HOST", "http://localhost:11434")
LLM_TEMPERATURE = float(os.environ.get("HPGA_LLM_TEMPERATURE", "0.7"))
LLM_NUM_CTX = int(os.environ.get("HPGA_LLM_NUM_CTX", "4096"))
LLM_KEEP_ALIVE = os.environ.get("HPGA_LLM_KEEP_ALIVE", "30m")
LLM_REQUEST_TIMEOUT_S = float(os.environ.get("HPGA_LLM_TIMEOUT_S", "120"))
LLM_MAX_RETRIES = int(os.environ.get("HPGA_LLM_MAX_RETRIES", "2"))  # attempts beyond the first

# Minimum number of positions each crossover child must differ from BOTH
# parents, enforced post-hoc with reject-and-retry. Diagnosed against
# search-quality logs: at rate=0.7, 39/42 crossover calls with genuinely
# differing parents returned a child that was one parent verbatim (a strict
# c1 != p1 check would still admit a near-echo that changes 1 position to
# satisfy the letter of "recombine"; the threshold closes that gap too).
#
# ONLY APPLIES TO "full" AND "diff" STYLES as of Phase 3. "segment" style
# dropped this check entirely (see _llm_crossover) -- not recalibrated, but
# found mis-designed: experiments/calibrate_crossover_min_diff.py measured
# 5000 genuine deterministic crossovers (random parent pairs, genome_length
# 18, 5-symbol alphabet) against this threshold and found the CURRENT
# default (2) rejects 17.6% of them outright, on nothing but random parent
# pairs -- the best case for how different two parents can be. Raising the
# threshold doesn't fix this: a legitimate edge-adjacent cut and a near-echo
# dodge produce the identical diff count (both differ from one parent by as
# little as 1 position), so no single value of this metric separates them.
# For "segment" style specifically, that ambiguity doesn't need resolving,
# because the model declares its own cut ('SEGMENTS: 0-8:1, 9-17:2') and the
# child is built FROM that declaration in code rather than independently
# produced -- matching the declaration is true by construction, and the
# declaration's structural validity (exact coverage, >=2 segments, both
# parents used) is what actually needs checking, not the resulting
# magnitude. "full"/"diff" still restate letters independently (nothing
# declared to validate a construction against), so this threshold remains
# their fallback gate, uncalibrated exactly as before -- results/raw/
# crossover_min_diff_calibration.json has the full percentile/pass-rate
# table if it needs revisiting for those styles specifically.
CROSSOVER_MIN_DIFF = int(os.environ.get("HPGA_CROSSOVER_MIN_DIFF", "2"))

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "raw"
_DEFAULT_RUN_ID = f"{int(time.time())}_{os.getpid()}"  # fallback if HPGA_RUN_ID is never set

_client = None  # lazy ollama.Client singleton


def _operator_mode() -> str:
    return os.environ.get("HPGA_OPERATOR_MODE", "deterministic")


def _prompt_style() -> str:
    """'full' (default): model emits the entire child/mutated genome, same as
    the original Phase 2 prompts. 'diff': model emits only the positions that
    change from a stated reference genome, applied on top of a copy of that
    reference. 'position' (mutate only): like 'diff' but with nothing to
    restate at all -- one 'POSITION: <n>, NEW: <letter>' line per change,
    NEW constrained to differ from the current letter, so copying the input
    back is not an expressible answer. 'segment' (crossover only): the model
    never writes a letter at all -- it partitions positions into contiguous
    segments and names a source parent per segment ('SEGMENTS:
    <start>-<end>:<parent>, ...'), applied in code; CHILD2 is the
    parent-complement of CHILD1. Same affordance idea as 'position', applied
    to crossover instead of mutation. 'best': 'position' for mutate and
    'segment' for crossover simultaneously -- the two are otherwise mutually
    exclusive under a single env var (mutate doesn't recognize 'segment',
    crossover doesn't recognize 'position', both silently fall through to
    'full'), which meant no single value of this env var could give a full
    GA run the affordance fix for both operators at once. Added when Phase 3
    needed exactly that combination for a real run, not a hypothetical.
    All share retry/fallback/logging via _run_llm_op; only build_prompt/parse
    differ. Read fresh per call, like HPGA_OPERATOR_MODE."""
    return os.environ.get("HPGA_LLM_PROMPT_STYLE", "full")


def _get_client():
    global _client
    if _client is None:
        try:
            from ollama import Client
        except ImportError as exc:
            raise RuntimeError(
                "HPGA_OPERATOR_MODE=llm requires the 'ollama' package "
                "(pip install ollama) and a running Ollama server."
            ) from exc
        _client = Client(host=LLM_HOST, timeout=LLM_REQUEST_TIMEOUT_S)
    return _client


@dataclass
class OperatorCallStats:
    n_llm_calls: int = 0          # top-level crossover()/mutate() calls made in llm mode
    n_llm_requests: int = 0       # actual HTTP requests, incl. retries
    n_retries: int = 0
    n_failures: int = 0           # exhausted retries, fell back to deterministic
    n_skipped_no_op: int = 0      # crossover rate-gate said "no crossover", no LLM call needed
    total_latency_s: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    latencies_s: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        d = asdict(self)
        lat = d.pop("latencies_s")
        d["mean_latency_s"] = (sum(lat) / len(lat)) if lat else None
        d["max_latency_s"] = max(lat) if lat else None
        d["min_latency_s"] = min(lat) if lat else None
        return d


_stats = OperatorCallStats()
_stats_lock = threading.Lock()  # guards _stats mutations under HPGA_GA_DISPATCH_P > 1


def reset_operator_stats() -> None:
    """Call before a run whose LLM-operator stats you want isolated from any
    prior run in the same process (the paired deterministic/llm comparison
    script does this between the two runs)."""
    global _stats
    _stats = OperatorCallStats()


def get_operator_stats() -> dict:
    return _stats.summary()


def _log_path() -> Path:
    override = os.environ.get("HPGA_LLM_LOG_PATH")
    if override:
        return Path(override)
    run_id = os.environ.get("HPGA_RUN_ID") or _DEFAULT_RUN_ID
    return _RESULTS_DIR / f"llm_operator_calls_{run_id}.jsonl"


def _log_call(record: dict) -> None:
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _stats_lock:  # serializes the write so concurrent dispatch (P>1) can't interleave lines
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


def _genome_to_str(genome: Genome) -> str:
    return " ".join(_MOVE_TO_CHAR[g] for g in genome)


def _extract_labelled(text: str, label: str, expected_len: int) -> Genome | None:
    m = re.search(rf"{label}\s*:\s*([SLRUDslrud][SLRUDslrud\s,]*)", text)
    if not m:
        return None
    letters = _VALID_CHARS.findall(m.group(1))
    if len(letters) != expected_len:
        return None
    return [_CHAR_TO_MOVE[c.upper()] for c in letters]


_DIFF_PAIR = re.compile(r"(\d+)\s*:\s*([SLRUDslrud])")


def _extract_diff(text: str, label: str, base: Genome, length: int) -> Genome | None:
    """Parse '<label> ...: none' or '<label> ...: <pos>:<letter>, <pos>:<letter>, ...'
    and apply it on top of a copy of `base`. Unlike _extract_labelled, the
    model does not restate the whole genome -- only the positions that
    change from `base`, which is the point of the diff-style prompt (see
    experiments/run_diff_style_probe.py)."""
    m = re.search(rf"{label}[^:\n]*:\s*(.*)", text)
    if not m:
        return None
    content = m.group(1).strip()
    if re.fullmatch(r"none\.?", content, re.IGNORECASE):
        return list(base)
    pairs = _DIFF_PAIR.findall(content)
    if not pairs:
        return None
    result = list(base)
    for pos_str, letter in pairs:
        pos = int(pos_str)
        if pos < 0 or pos >= length:
            return None
        result[pos] = _CHAR_TO_MOVE[letter.upper()]
    return result


_POSITION_PAIR = re.compile(r"POSITION\s*:\s*(\d+)\s*,\s*NEW\s*:\s*([SLRUDslrud])", re.IGNORECASE)


def _extract_position_mutation(text: str, base: Genome, length: int, k: int) -> Genome | None:
    """Parse exactly `k` 'POSITION: <n>, NEW: <letter>' lines and apply them
    to a copy of `base`. Rejects: wrong pair count, a position outside
    range, a repeated position, or a NEW letter equal to the letter already
    at that position (there is nothing to "restate" in this format, so a
    same-letter pair is the only way left to fake a no-op)."""
    pairs = _POSITION_PAIR.findall(text)
    if len(pairs) != k:
        return None
    result = list(base)
    seen: set[int] = set()
    for pos_str, letter in pairs:
        pos = int(pos_str)
        if pos < 0 or pos >= length or pos in seen:
            return None
        letter_code = _CHAR_TO_MOVE[letter.upper()]
        if letter_code == base[pos]:
            return None
        seen.add(pos)
        result[pos] = letter_code
    return result


def _mutate_position_prompt(genome_str: str, length: int, k: int, retry_hint: str) -> str:
    return f"""Genome (0-indexed positions 0-{length - 1}): {genome_str}

Change exactly {k} position(s) in this genome. For each one, name the
position and the new letter it becomes -- the new letter MUST differ from
whatever letter is currently at that position (copying the current letter
back is not a change). Do not restate the genome.

Respond with EXACTLY {k} line(s) and nothing else, one change per line:
POSITION: <0-{length - 1}>, NEW: <S, L, R, U, or D>
{retry_hint}"""


def _diff_count(a: Genome, b: Genome) -> int:
    return sum(x != y for x, y in zip(a, b))


def _crossover_sufficiently_mixed(c1: Genome, c2: Genome, p1: Genome, p2: Genome) -> bool:
    """Reject a no-op or near-no-op crossover: each child must actually carry
    material from both parents, not just copy one (or copy one but flip a
    couple of positions to dodge a strict != check)."""
    return (
        _diff_count(c1, p1) >= CROSSOVER_MIN_DIFF
        and _diff_count(c1, p2) >= CROSSOVER_MIN_DIFF
        and _diff_count(c2, p1) >= CROSSOVER_MIN_DIFF
        and _diff_count(c2, p2) >= CROSSOVER_MIN_DIFF
    )


def _call_ollama(prompt: str, system: str, num_predict: int, seed: int) -> tuple[str, float, int, int]:
    client = _get_client()
    t0 = time.perf_counter()
    response = client.generate(
        model=LLM_MODEL,
        prompt=prompt,
        system=system,
        stream=False,
        think=False,
        options={
            "num_ctx": LLM_NUM_CTX,
            "num_predict": num_predict,
            "temperature": LLM_TEMPERATURE,
            "seed": seed,
        },
        keep_alive=LLM_KEEP_ALIVE,
    )
    latency = time.perf_counter() - t0
    text = getattr(response, "response", "") or ""
    tokens_in = getattr(response, "prompt_eval_count", None) or 0
    tokens_out = getattr(response, "eval_count", None) or 0
    return text, latency, tokens_in, tokens_out


_CROSSOVER_SYSTEM = (
    "You recombine two parent move sequences for a genetic algorithm that "
    "folds a protein chain on a 3D lattice. Each move is one letter: S "
    "(straight), L (turn left), R (turn right), U (turn up), or D (turn "
    "down), relative to the current heading and orientation. Follow the "
    "requested output format exactly and output nothing else."
)

_MUTATE_SYSTEM = (
    "You mutate a move sequence for a genetic algorithm that folds a protein "
    "chain on a 3D lattice. Each move is one letter: S (straight), L (turn "
    "left), R (turn right), U (turn up), or D (turn down), relative to the "
    "current heading and orientation. Follow the requested output format "
    "exactly and output nothing else."
)


def _crossover_prompt(p1_str: str, p2_str: str, length: int, retry_hint: str) -> str:
    return f"""Parent 1: {p1_str}
Parent 2: {p2_str}

Produce two children of length {length}, using only the letters S, L, R, U, D,
by recombining segments from Parent 1 and Parent 2 (mix contiguous or
interleaved segments from both parents, similar to genetic crossover).

Respond with EXACTLY two lines and nothing else:
CHILD1: <{length} letters from {{S,L,R,U,D}} separated by single spaces>
CHILD2: <{length} letters from {{S,L,R,U,D}} separated by single spaces>
{retry_hint}"""


def _mutate_prompt(genome_str: str, length: int, k: int, retry_hint: str) -> str:
    return f"""Genome: {genome_str}

Change EXACTLY {k} of the {length} positions to a different letter from
{{S,L,R,U,D}}; leave the other {length - k} positions unchanged. Choose which
{k} positions to change yourself.

Respond with EXACTLY one line and nothing else:
MUTATED: <{length} letters from {{S,L,R,U,D}} separated by single spaces>
{retry_hint}"""


def _crossover_diff_prompt(p1_str: str, p2_str: str, length: int, retry_hint: str) -> str:
    return f"""Parent 1 (0-indexed positions 0-{length - 1}): {p1_str}
Parent 2 (0-indexed positions 0-{length - 1}): {p2_str}

Produce two children of length {length} by recombining segments from Parent 1
and Parent 2 (mix contiguous or interleaved segments from both parents,
similar to genetic crossover). Describe CHILD1 as the positions where it
differs from Parent 1, and CHILD2 as the positions where it differs from
Parent 2 -- do NOT restate the full sequence, only the changed positions.

Respond with EXACTLY two lines and nothing else:
CHILD1 diff from Parent1: <pos>:<letter>, <pos>:<letter>, ... (or 'none' if CHILD1 equals Parent1)
CHILD2 diff from Parent2: <pos>:<letter>, <pos>:<letter>, ... (or 'none' if CHILD2 equals Parent2)
{retry_hint}"""


def _mutate_diff_prompt(genome_str: str, length: int, k: int, retry_hint: str) -> str:
    return f"""Genome (0-indexed positions 0-{length - 1}): {genome_str}

Change EXACTLY {k} of the {length} positions to a different letter from
{{S,L,R,U,D}}; leave the rest unchanged. Choose which {k} positions to change
yourself. Do NOT restate the full sequence, only the positions that change.

Respond with EXACTLY one line and nothing else:
CHANGES: <pos>:<letter>, <pos>:<letter>, ... ({k} pairs required)
{retry_hint}"""


def _run_llm_op(
    *,
    op: str,
    style: str,
    system: str,
    build_prompt,
    parse,
    rng: random.Random,
    num_predict: int,
    fallback,
    retry_hint_text: str,
    extra_log_fields: dict | None = None,
) -> Genome | tuple[Genome, Genome]:
    """extra_log_fields, when given, is merged into every _log_call() record
    this invocation produces (island id / agent id / role / generation, for
    hpga.agents' agent-generation calls -- see that module). None for
    crossover/mutate, unchanged from before this parameter existed."""
    extra = extra_log_fields or {}
    with _stats_lock:
        _stats.n_llm_calls += 1
    retry_hint = ""
    last_response = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        prompt = build_prompt(retry_hint)
        seed = rng.getrandbits(31)
        with _stats_lock:
            _stats.n_llm_requests += 1
        try:
            text, latency, tin, tout = _call_ollama(prompt, system, num_predict, seed)
        except Exception as exc:  # infra failure (server down, timeout): not "malformed output"
            _log_call({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "op": op, "prompt_style": style, "attempt": attempt, "seed": seed,
                "temperature": LLM_TEMPERATURE,
                "model": LLM_MODEL, "prompt": prompt, "error": str(exc),
                **extra,
            })
            raise
        last_response = text
        parsed = parse(text)
        with _stats_lock:
            _stats.total_latency_s += latency
            _stats.latencies_s.append(latency)
            _stats.total_tokens_in += tin
            _stats.total_tokens_out += tout
        _log_call({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "op": op, "prompt_style": style, "attempt": attempt, "seed": seed,
            "temperature": LLM_TEMPERATURE,
            "model": LLM_MODEL, "prompt": prompt, "response": text,
            "latency_s": latency, "tokens_in": tin, "tokens_out": tout,
            "valid": parsed is not None,
            **extra,
        })
        if parsed is not None:
            return parsed
        with _stats_lock:
            _stats.n_retries += 1
        retry_hint = retry_hint_text

    with _stats_lock:
        _stats.n_failures += 1
    _log_call({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "op": op, "prompt_style": style, "event": "fallback_to_deterministic",
        "last_response": last_response,
        **extra,
    })
    return fallback()


_RETRY_HINT_FULL = (
    "\nIMPORTANT: your previous response did not match the required "
    "format. Respond with ONLY the labelled line(s) above, each "
    "containing exactly the requested number of S/L/R/U/D letters."
)

_RETRY_HINT_DIFF = (
    "\nIMPORTANT: your previous response did not match the required "
    "format. Respond with ONLY the labelled line(s) above, each either "
    "'none' or a comma-separated list of <position>:<letter> pairs "
    "(position an integer in range, letter one of S/L/R/U/D) -- do not "
    "restate the full sequence."
)

_RETRY_HINT_SEGMENT = (
    "\nIMPORTANT: your previous response did not match the required "
    "format -- the segments must cover positions 0 through length-1 "
    "exactly once each, in increasing order, with no gaps or overlaps, "
    "must use at least 2 segments, and must use both parents at least "
    "once. Respond with ONLY the SEGMENTS line above."
)

_SEGMENT_TRIPLE = re.compile(r"(\d+)\s*-\s*(\d+)\s*:\s*([12])")


def _extract_segment_crossover(text: str, p1: Genome, p2: Genome, length: int) -> tuple[Genome, Genome] | None:
    """Parse 'SEGMENTS: <start>-<end>:<parent>, ...' -- a partition of
    positions 0..length-1 into contiguous, non-overlapping segments, each
    naming which parent CHILD1 draws from for that range. CHILD2 is the
    complement (opposite parent per segment). The model never writes a
    single letter; it only chooses segment boundaries and parent source,
    same idea as the position-style mutation fix applied to crossover."""
    m = re.search(r"SEGMENTS\s*:\s*(.*)", text)
    if not m:
        return None
    triples = _SEGMENT_TRIPLE.findall(m.group(1))
    if len(triples) < 2:
        return None
    segments = sorted((int(s), int(e), int(p)) for s, e, p in triples)
    if segments[0][0] != 0 or segments[-1][1] != length - 1:
        return None
    for i in range(1, len(segments)):
        if segments[i][0] != segments[i - 1][1] + 1:
            return None  # gap or overlap
    if {p for _, _, p in segments} != {1, 2}:
        return None
    child1 = [0] * length
    child2 = [0] * length
    for start, end, parent in segments:
        src1, src2 = (p1, p2) if parent == 1 else (p2, p1)
        for i in range(start, end + 1):
            child1[i] = src1[i]
            child2[i] = src2[i]
    return child1, child2


def _crossover_segment_prompt(p1_str: str, p2_str: str, length: int, retry_hint: str) -> str:
    return f"""Parent 1 (0-indexed positions 0-{length - 1}): {p1_str}
Parent 2 (0-indexed positions 0-{length - 1}): {p2_str}

Produce CHILD1 by dividing positions 0-{length - 1} into contiguous segments
and assigning each segment to a source parent (1 or 2) -- CHILD1 takes its
letters from that parent for those positions. CHILD2 is the complement:
wherever CHILD1 used Parent 1, CHILD2 uses Parent 2, and vice versa.
Segments must cover every position exactly once, in increasing order,
starting at 0 and ending at {length - 1}, with no gaps or overlaps. Use at
least 2 segments, and use both parents at least once.

Respond with EXACTLY one line and nothing else:
SEGMENTS: <start>-<end>:<parent>, <start>-<end>:<parent>, ...
{retry_hint}"""


def _lattice_crossover_plan(style: str, parent1: Genome, parent2: Genome, length: int) -> LLMOpPlan:
    """The 5-symbol crossover prompt styles: the style dispatch that used to
    live inline in _llm_crossover, moved here unchanged so the lattice model
    (genome_model.LatticeGenomeModel.plan_llm_crossover) can hand it back as
    a plan. Unknown `style` values fall through to 'full', as before."""
    p1_str, p2_str = _genome_to_str(parent1), _genome_to_str(parent2)

    def mix_hint(base_hint: str) -> str:
        return (
            base_hint
            + f" Each child must differ from BOTH Parent 1 and Parent 2 in at "
            f"least {CROSSOVER_MIN_DIFF} positions -- do not copy one parent "
            f"unchanged (or nearly unchanged), actually recombine material "
            f"from both."
        )

    if style in ("segment", "best"):
        num_predict = min(2048, max(32, 6 * length))

        def build_prompt(retry_hint: str) -> str:
            return _crossover_segment_prompt(p1_str, p2_str, length, retry_hint)

        def parse(text: str):
            # Deliberately NOT gated by _crossover_sufficiently_mixed: the
            # calibration in experiments/calibrate_crossover_min_diff.py
            # found no single diff-count threshold can separate a legitimate
            # edge-adjacent cut (child differs from one parent by as little
            # as 1 position, by construction of a valid declared segment)
            # from a near-echo dodge -- they produce the identical number.
            # Segment style has something full/diff don't: the model
            # DECLARES its cut ('SEGMENTS: 0-8:1, 9-17:2') and the child is
            # built FROM that declaration in code (below), never
            # independently produced and cross-checked -- so the child
            # matching its own declared segments is true by construction,
            # not something to additionally verify. What's left to validate
            # is the declaration's own structural well-formedness (exact
            # coverage of 0..length-1, >=2 segments, both parents used),
            # which _extract_segment_crossover already enforces -- a
            # degenerate single-segment or single-parent declaration still
            # correctly fails here, on structure, not on magnitude.
            return _extract_segment_crossover(text, parent1, parent2, length)

        retry_hint_text = _RETRY_HINT_SEGMENT
    elif style == "diff":
        num_predict = min(2048, max(48, 3 * length))

        def build_prompt(retry_hint: str) -> str:
            return _crossover_diff_prompt(p1_str, p2_str, length, retry_hint)

        def parse(text: str):
            c1 = _extract_diff(text, "CHILD1", parent1, length)
            c2 = _extract_diff(text, "CHILD2", parent2, length)
            if c1 is None or c2 is None:
                return None
            if not _crossover_sufficiently_mixed(c1, c2, parent1, parent2):
                return None
            return (c1, c2)

        retry_hint_text = mix_hint(_RETRY_HINT_DIFF)
    else:
        num_predict = min(2048, max(64, 8 * length))

        def build_prompt(retry_hint: str) -> str:
            return _crossover_prompt(p1_str, p2_str, length, retry_hint)

        def parse(text: str):
            c1 = _extract_labelled(text, "CHILD1", length)
            c2 = _extract_labelled(text, "CHILD2", length)
            if c1 is None or c2 is None:
                return None
            if not _crossover_sufficiently_mixed(c1, c2, parent1, parent2):
                return None
            return (c1, c2)

        retry_hint_text = mix_hint(_RETRY_HINT_FULL)

    return LLMOpPlan(
        system=_CROSSOVER_SYSTEM, build_prompt=build_prompt, parse=parse,
        num_predict=num_predict, retry_hint_text=retry_hint_text,
    )


def _model_for(genome) -> "genome_model.GenomeModel":
    """The GenomeModel the LLM entry points dispatch through. The active
    model if one is set; otherwise a config-less lattice model, but only for
    a list genome -- see hpga/genome_model.py's module docstring for why this
    departs from current()'s raise-if-unset rule, and what gap it leaves."""
    model = genome_model.active_or_none()
    if model is None:
        if isinstance(genome, str):
            raise RuntimeError(
                "str genome passed to an LLM operator with no active GenomeModel -- call "
                "genome_model.set_active(build_genome_model(config)) first; refusing to "
                "fall back to the lattice model for a sequence genome"
            )
        return _LEGACY_LATTICE
    if not isinstance(genome, model.genome_type):
        raise RuntimeError(
            f"active GenomeModel is {model.name!r} (genome_type {model.genome_type.__name__}) "
            f"but the operator was handed a {type(genome).__name__} genome"
        )
    return model


_LEGACY_LATTICE = genome_model.LatticeGenomeModel()


def _llm_crossover(
    parent1: Genome, parent2: Genome, rate: float, rng: random.Random, generation: int | None = None,
) -> tuple[Genome, Genome]:
    """`generation`, when given, is logged (with `identical_parents`) on
    every attempt this call makes -- diagnostic for the population-
    convergence mechanism documented in PHASE2_RESULTS.md sec 4.1 and
    re-observed in Phase 3: once two selected parents are literally
    identical, _crossover_sufficiently_mixed can never pass regardless of
    prompt style, so a rising identical-parent rate over a run's
    generations is a distinct, measurable cause of crossover fallback from
    a format/compliance failure. Optional and defaults to None so existing
    direct callers (test_operator_failure_path.py, ad-hoc scripts) are
    unaffected.

    Prompt building and response parsing come from the active GenomeModel
    (see _model_for); everything else -- the rate gate and its RNG draw,
    retries, fallback, logging, stats -- is model-independent and unchanged."""
    model = _model_for(parent1)
    identical_parents = list(parent1) == list(parent2)
    if rng.random() > rate or len(parent1) < 2:
        with _stats_lock:
            _stats.n_skipped_no_op += 1
        return model.copy(parent1), model.copy(parent2)

    style = _prompt_style()
    plan = model.plan_llm_crossover(style, parent1, parent2)

    def fallback():
        return model.deterministic_crossover(parent1, parent2, rate, rng)

    return _run_llm_op(
        op="crossover", style=style, system=plan.system, build_prompt=plan.build_prompt,
        parse=plan.parse, rng=rng, num_predict=plan.num_predict, fallback=fallback,
        retry_hint_text=plan.retry_hint_text,
        extra_log_fields={"generation": generation, "identical_parents": identical_parents},
    )


def _lattice_mutate_plan(style: str, genome: Genome, length: int, k: int) -> LLMOpPlan:
    """The 5-symbol mutate prompt styles: the style dispatch that used to
    live inline in _llm_mutate, moved here unchanged so the lattice model
    (genome_model.LatticeGenomeModel.plan_llm_mutate) can hand it back as a
    plan. Unknown `style` values (and 'segment', which mutate doesn't
    recognize) fall through to 'full', as before."""
    genome_str = _genome_to_str(genome)

    def count_hint(base_hint: str) -> str:
        return (
            base_hint
            + f" Exactly {k} of the {length} positions must differ from the "
            f"input Genome -- not approximately {k}, exactly {k}."
        )

    if style in ("position", "best"):
        num_predict = min(2048, max(24, 10 * k))

        def build_prompt(retry_hint: str) -> str:
            return _mutate_position_prompt(genome_str, length, k, retry_hint)

        def parse(text: str):
            return _extract_position_mutation(text, genome, length, k)

        retry_hint_text = (
            f"\nIMPORTANT: your previous response did not match the "
            f"required format, gave a number of lines other than {k}, "
            f"repeated a position, or gave a NEW letter equal to the "
            f"current letter at that position (that is not a change). "
            f"Respond with ONLY the {k} labelled line(s) above."
        )
    elif style == "diff":
        num_predict = min(2048, max(32, 3 * length))

        def build_prompt(retry_hint: str) -> str:
            return _mutate_diff_prompt(genome_str, length, k, retry_hint)

        def parse(text: str):
            parsed = _extract_diff(text, "CHANGES", genome, length)
            if parsed is None or _diff_count(parsed, genome) != k:
                return None
            return parsed

        retry_hint_text = count_hint(_RETRY_HINT_DIFF)
    else:
        num_predict = min(2048, max(32, 6 * length))

        def build_prompt(retry_hint: str) -> str:
            return _mutate_prompt(genome_str, length, k, retry_hint)

        def parse(text: str):
            parsed = _extract_labelled(text, "MUTATED", length)
            if parsed is None or _diff_count(parsed, genome) != k:
                return None
            return parsed

        retry_hint_text = count_hint(_RETRY_HINT_FULL)

    return LLMOpPlan(
        system=_MUTATE_SYSTEM, build_prompt=build_prompt, parse=parse,
        num_predict=num_predict, retry_hint_text=retry_hint_text,
    )


def _llm_mutate(genome: Genome, rate: float, rng: random.Random) -> Genome:
    """Prompt building and response parsing come from the active GenomeModel
    (see _model_for); retries, fallback, logging and stats are unchanged."""
    model = _model_for(genome)
    length = len(genome)
    style = _prompt_style()
    k = max(1, round(rate * length))
    plan = model.plan_llm_mutate(style, genome, k)

    def fallback():
        return model.deterministic_mutate(genome, rate, rng)

    return _run_llm_op(
        op="mutate", style=style, system=plan.system, build_prompt=plan.build_prompt,
        parse=plan.parse, rng=rng, num_predict=plan.num_predict, fallback=fallback,
        retry_hint_text=plan.retry_hint_text,
    )


# --- GA phase entry point ---------------------------------------------------


def _dispatch_degree() -> int:
    """DIBM analogue: number of concurrent operator-call 'injection channels'
    the master issues through instead of serialising through one. Read fresh
    per next_generation() call, like HPGA_OPERATOR_MODE. P=1 (default) is the
    original sequential path, byte-for-byte -- P>1 only changes anything when
    HPGA_OPERATOR_MODE=llm, since the deterministic path is already
    microseconds and has nothing to parallelise against."""
    return int(os.environ.get("HPGA_GA_DISPATCH_P", "1"))


def next_generation(
    population: Sequence[Genome],
    fitnesses: Sequence[float],
    pop_size: int,
    tournament_k: int,
    crossover_rate: float,
    mutation_rate: float,
    elitism: int,
    rng: random.Random,
) -> list[Genome]:
    """One GA phase: selection + crossover + mutation, run entirely on the
    island master (sequential, no worker involved -- this is the phase that
    competes with DIS for wall-clock time in the DIS/GA ratio measurement).

    Crossover and mutation are dispatched to either the deterministic or the
    LLM implementation based on HPGA_OPERATOR_MODE, read fresh here so a
    single process can run both modes back to back. Selection and elitism
    are always deterministic.

    Under HPGA_OPERATOR_MODE=llm with HPGA_GA_DISPATCH_P > 1, operator calls
    for separate reproduction units (one crossover + two mutates) are issued
    concurrently across a thread pool of size P, instead of one at a time --
    this is the paper's DIBM: the master borrows P injection channels instead
    of serialising through one. Selection (which mutates the shared `rng`)
    still happens sequentially first, so parent choice and per-job seeds are
    identical regardless of P; only the LLM calls themselves run concurrently,
    each against its own independently-seeded random.Random (see
    PHASE2_RESULTS.md §7.3 for the Stage 2 write-up -- measured on the
    pre-migration 2D/3-symbol model, see that section's disclosure note).
    P>1 in deterministic mode falls through to the same sequential loop below
    -- there is nothing to parallelise against microsecond-scale compute.

    Agents (optional, off by default -- see hpga/agents.py for the full
    design): when HPGA_OPERATOR_MODE=llm and HPGA_AGENTS_ENABLED=1,
    HPGA_N_AGENTS agent-produced genomes are ADDED on top of the ordinary
    tournament-select/crossover/mutate fill below, not carved out of it --
    the fill loop always runs to exactly pop_size first, identically to
    the agents-off baseline, then hpga.agents.run_agents() appends its A
    genomes afterward, so a run with agents enabled returns pop_size + A
    individuals, not pop_size (previously agents displaced non-elite
    slots, which meant the agents-on arm made fewer top-level LLM calls
    than the baseline -- a call-budget confound on any fitness comparison;
    additive avoids that by construction, at the cost of a growing
    population from generation 1 onward. island.py's call site, worker.py,
    and instrumentation.py are unaffected either way -- _dispatch_and_collect
    sizes its dispatch off len(population), not a hardcoded pop_size, so
    the larger population from generation 1 onward is handled without any
    change there). A generation counter is ticked unconditionally on every
    call (agents.tick_generation()) since island.py's own gen_id is not
    passed through; population diversity (mean pairwise Hamming distance)
    is logged when HPGA_LOG_DIVERSITY=1, independent of agents/LLM mode,
    so every arm of an agents-off/agents-no-comm/agents-comm comparison
    gets a diversity trace.

    Circles (Phase 3, optional, off by default -- see hpga/circles.py and
    hpga/blackboard.py): when HPGA_OPERATOR_MODE=llm and
    HPGA_CIRCLES_ENABLED=1, HPGA_N_CIRCLES * HPGA_AGENTS_PER_CIRCLE genomes
    are ADDED on top of the ordinary fill below, same additive contract as
    the agents feature above and for the same reason (a call-budget
    confound otherwise). New architecture, not an extension of the agents
    feature -- mutually exclusive with it (HPGA_AGENTS_ENABLED=1 together
    with HPGA_CIRCLES_ENABLED=1 raises, rather than silently running both
    and confounding either one's measurement).
    """
    mode = _operator_mode()
    use_llm = mode == "llm"
    P = _dispatch_degree()

    if (
        use_llm
        and os.environ.get("HPGA_AGENTS_ENABLED", "0") == "1"
        and os.environ.get("HPGA_CIRCLES_ENABLED", "0") == "1"
    ):
        raise RuntimeError(
            "HPGA_CIRCLES_ENABLED=1 and HPGA_AGENTS_ENABLED=1 are mutually "
            "exclusive -- both are additive coordination features and "
            "running them together would confound either one's measurement."
        )

    gen = agents.tick_generation()

    if os.environ.get("HPGA_LOG_DIVERSITY", "0") == "1" and population:
        agents.log_diversity(population, gen)

    ranked = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
    new_pop: list[Genome] = [list(population[i]) for i in ranked[:elitism]]

    if use_llm and P > 1:
        n_units = -(-(pop_size - len(new_pop)) // 2)  # ceil division: pairs needed
        jobs = []
        for _ in range(n_units):
            p1 = tournament_select(population, fitnesses, tournament_k, rng)
            p2 = tournament_select(population, fitnesses, tournament_k, rng)
            jobs.append((p1, p2, rng.getrandbits(31)))

        def run_job(job):
            p1, p2, seed = job
            job_rng = random.Random(seed)
            c1, c2 = _llm_crossover(p1, p2, crossover_rate, job_rng, generation=gen)
            c1 = _llm_mutate(c1, mutation_rate, job_rng)
            c2 = _llm_mutate(c2, mutation_rate, job_rng)
            return c1, c2

        with ThreadPoolExecutor(max_workers=P) as pool:
            for c1, c2 in pool.map(run_job, jobs):
                new_pop.append(c1)
                if len(new_pop) < pop_size:
                    new_pop.append(c2)
    else:
        while len(new_pop) < pop_size:
            p1 = tournament_select(population, fitnesses, tournament_k, rng)
            p2 = tournament_select(population, fitnesses, tournament_k, rng)
            if use_llm:
                c1, c2 = _llm_crossover(p1, p2, crossover_rate, rng, generation=gen)
                c1 = _llm_mutate(c1, mutation_rate, rng)
                c2 = _llm_mutate(c2, mutation_rate, rng)
            else:
                c1, c2 = crossover(p1, p2, crossover_rate, rng)
                c1 = mutate(c1, mutation_rate, rng)
                c2 = mutate(c2, mutation_rate, rng)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

    if use_llm and os.environ.get("HPGA_AGENTS_ENABLED", "0") == "1":
        genome_length = len(population[0]) if population else 0
        agent_genomes = agents.run_agents(population, fitnesses, pop_size, genome_length, gen, rng)
        new_pop.extend(agent_genomes)

    if use_llm and os.environ.get("HPGA_CIRCLES_ENABLED", "0") == "1":
        genome_length = len(population[0]) if population else 0
        circle_genomes = circles.run_circles(population, fitnesses, pop_size, genome_length, gen, rng)
        new_pop.extend(circle_genomes)

    return new_pop
