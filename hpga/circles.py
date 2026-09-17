"""Phase 3: circle-based agents over a shared blackboard.

New architecture, not an extension of hpga/agents.py (Phase 2's role/
central-comm design) -- three results from that module constrain this one
(see PHASE2_RESULTS.md sec 8): communication cost 312 extra tokens and
changed nothing measurable; the diversity effect came from the roles, not
the channel; passing folds between agents carried no reasoning the
receiver couldn't derive by evaluating the fold itself. This module never
imports hpga.agents' roles/communication/generation-counter machinery for
that reason -- the two things reused from it (mean_pairwise_hamming via
operators.next_generation()'s existing unconditional diversity-logging
call, and _approx_tokens here and in hpga/blackboard.py) are trivial, non-
architectural utilities, reused by explicit agreement, not a dependency on
Phase 2's design.

Circles: groups of HPGA_AGENTS_PER_CIRCLE agents, HPGA_N_CIRCLES groups.
Agents within a circle can consult each other (_consult, one call per
circle) before proposing; agents in different circles cannot -- consult()
never touches the blackboard, which is what makes that isolation true by
construction rather than by convention (see blackboard.py's read_live()
docstring). The blackboard is the only cross-circle channel, and carries
observations (reasoning), never candidates (folds) -- see blackboard.py.

Same additive-population contract as agents.run_agents(): run_circles()
always returns exactly n_circles * agents_per_circle genomes, appended
after next_generation()'s ordinary fill-to-pop_size loop, never carved out
of it, for the same reason agents.py is additive -- carving out slots would
make a circles-on run issue fewer top-level LLM calls than the circles-off
baseline, confounding any fitness comparison by call budget alone. Circle-
agent i (0-indexed, i = circle_id * agents_per_circle + agent_id) owns
population index pop_size + i from generation 1 onward, exactly like
agents.py's tail slots -- this module's per-slot fitness/genome history
(_last_fitness_by_slot, _last_genome_by_slot) is what lets observe() ground
its reasoning in a real before/after, without a second population copy.

Global directive only: the central circle's directive is broadcast
identically to every circle (confirmed against per-circle targeting, which
would already be the lateral task routing this phase explicitly excludes).

Central circle mode (HPGA_CENTRAL_MODE=deterministic, the default, or
llm): the deterministic baseline can only act on the structured fitness-
delta trend, not on the blackboard's free-text content -- there's no non-
LLM way to "read" an observation. So llm-vs-deterministic here is really
"informed by reasoning" vs. "informed by fitness trend alone," not the
same information through two decision procedures; read it as bounding what
the free-text channel buys over a numeric signal, not as a clean A/B.

Known open risk, not fixed here: propose()'s FOLD/RATIONALE prompt doesn't
use position/segment-style affordances (PHASE2_RESULTS.md sec 4.3), so it
could in principle collapse the way mutate/full did. It's less directly
analogous (propose asks for a new fold from context, not "restate with one
change"), but the RATIONALE line has an equivalent risk: the model could
restate the fitness numbers back as its "observation"/"rationale" instead
of inferring anything from them, and nothing here mechanically distinguishes
that from real reasoning -- watch for this in the smoke-run logs rather
than assuming the prompt avoids it.

Imports hpga.operators and hpga.blackboard lazily inside functions to avoid
a circular import (operators.py imports this module at the top level),
matching hpga/agents.py's existing pattern.
"""

import os
import random
import re
from typing import Sequence

from hpga.agents import _approx_tokens

Genome = list[int]

_last_fitness_by_slot: dict[int, float] = {}
_last_genome_by_slot: dict[int, Genome] = {}


def reset_circle_state() -> None:
    """Call before a run whose per-slot fitness/genome history should be
    isolated from any prior run in the same process -- mirrors
    agents.reset_agent_state() / operators.reset_operator_stats() /
    blackboard.reset_blackboard()."""
    global _last_fitness_by_slot, _last_genome_by_slot
    _last_fitness_by_slot = {}
    _last_genome_by_slot = {}


def _n_circles() -> int:
    return int(os.environ.get("HPGA_N_CIRCLES", "2"))


def _agents_per_circle() -> int:
    return int(os.environ.get("HPGA_AGENTS_PER_CIRCLE", "2"))


def _central_mode() -> str:
    return os.environ.get("HPGA_CENTRAL_MODE", "deterministic")


def _central_interval() -> int:
    return int(os.environ.get("HPGA_CENTRAL_INTERVAL", "1"))


def _curation_interval() -> int:
    return int(os.environ.get("HPGA_CURATION_INTERVAL", "5"))


def _extract_note(text: str, label: str) -> str | None:
    m = re.search(rf"{label}\s*:\s*(.+)", text, re.IGNORECASE)
    if not m:
        return None
    note = m.group(1).strip()
    return note if note else None


def _extract_fold_line(text: str, length: int) -> Genome | None:
    """Deliberately NOT ops._extract_labelled: that function's capture
    group ([SLRUDslrud][SLRUDslrud\\s,]*) has no newline anchor, so on a
    two-line response it greedily consumes into the next line whenever
    that line's first character happens to be a move letter -- exactly
    what "RATIONALE" does (its leading R matches). Caught by the pre-run
    dry-run stub, not by inspection: every propose() call failed to parse
    silently before this fix. `(.+)` without re.DOTALL stops at the
    newline by construction, which is all this needs -- every other
    _extract_labelled call site in operators.py only ever has one line in
    its response, so that function itself isn't touched."""
    from hpga import operators as ops

    m = re.search(r"FOLD\s*:\s*(.+)", text, re.IGNORECASE)
    if not m:
        return None
    letters = re.findall(r"[SLRUDslrud]", m.group(1))
    if len(letters) != length:
        return None
    return [ops._CHAR_TO_MOVE[c.upper()] for c in letters]


# --- Consult (coordination, one call per circle) ----------------------------

_CONSULT_SYSTEM = (
    "You are the shared discussion for one group (circle) of agents "
    "folding the same protein chain on a 3D lattice as part of a genetic "
    "search. You see this circle's current folds and fitnesses only -- "
    "other circles are working independently and you cannot see their "
    "folds. Give brief, concrete guidance for what this circle's agents "
    "should try differently this round. Follow the requested output "
    "format exactly and output nothing else."
)


def _consult_prompt(circle_id: int, folds_fitnesses: list[tuple[str, float]], retry_hint: str) -> str:
    lines = "\n".join(f"  Agent {i}: fold={g}  fitness={f:.3f}" for i, (g, f) in enumerate(folds_fitnesses))
    return f"""Circle {circle_id}'s current agents:
{lines}

In one sentence, say what this circle should focus on differently this
round (e.g. which part of the chain to change, whether to make small or
large edits). Do not propose a fold yourself.

Respond with EXACTLY one line and nothing else:
NOTE: <your guidance, one sentence>
{retry_hint}"""


def _consult(circle_id: int, folds_fitnesses: list[tuple[str, float]], rng: random.Random) -> str:
    from hpga import operators as ops

    def build_prompt(retry_hint: str) -> str:
        return _consult_prompt(circle_id, folds_fitnesses, retry_hint)

    def parse(text: str):
        return _extract_note(text, "NOTE")

    def fallback():
        return ""

    note = ops._run_llm_op(
        op="consult", style="note", system=_CONSULT_SYSTEM, build_prompt=build_prompt,
        parse=parse, rng=rng, num_predict=96, fallback=fallback,
        retry_hint_text="\nIMPORTANT: respond with ONLY the NOTE line above.",
        extra_log_fields={"circle_id": circle_id, "call_kind": "coordination", "phase3_step": "consult"},
    )
    return note or ""


# --- Observe (coordination, one call per circle-agent slot) -----------------

_OBSERVE_SYSTEM = (
    "You write one short observation about a genetic search over 3D "
    "protein folds, for a shared board other agents and a coordinator will "
    "read. Write what happened and what it suggests -- not a fold, not a "
    "restated number, an actual inference (e.g. 'no improvement from "
    "changes near the end of the chain' or 'small edits near the middle "
    "helped, large ones did not'). Follow the requested output format "
    "exactly and output nothing else."
)


def _observe_prompt(old_fold: str, new_fold: str, old_fitness: float, new_fitness: float, retry_hint: str) -> str:
    return f"""Previous fold: {old_fold}  (fitness {old_fitness:.3f})
Current fold:  {new_fold}  (fitness {new_fitness:.3f})

Write one short observation about what this change suggests for future
search -- an inference from the numbers above, not a restatement of them.

Respond with EXACTLY one line and nothing else:
OBSERVATION: <your observation, one sentence>
{retry_hint}"""


def _observe(
    circle_id: int, agent_id: int, prior_genome: Genome, cur_genome: Genome,
    prior_fitness: float, cur_fitness: float, generation: int, rng: random.Random,
) -> None:
    from hpga import blackboard as bb
    from hpga import operators as ops

    old_str, new_str = ops._genome_to_str(prior_genome), ops._genome_to_str(cur_genome)

    def build_prompt(retry_hint: str) -> str:
        return _observe_prompt(old_str, new_str, prior_fitness, cur_fitness, retry_hint)

    def parse(text: str):
        return _extract_note(text, "OBSERVATION")

    def fallback():
        return None

    result = ops._run_llm_op(
        op="observe", style="note", system=_OBSERVE_SYSTEM, build_prompt=build_prompt,
        parse=parse, rng=rng, num_predict=96, fallback=fallback,
        retry_hint_text="\nIMPORTANT: respond with ONLY the OBSERVATION line above.",
        extra_log_fields={
            "circle_id": circle_id, "agent_id": agent_id, "generation": generation,
            "call_kind": "coordination", "phase3_step": "observe",
        },
    )
    if not result:
        return  # fell back: nothing grounded to write, skip the blackboard entry
    bb.append(
        generation=generation, author=f"circle{circle_id}-agent{agent_id}", circle_id=circle_id,
        type_="observation", content=result, tokens=_approx_tokens(result),
    )


# --- Central circle (coordination, global directive) -------------------------

_CENTRAL_SYSTEM = (
    "You coordinate several independent circles of agents folding the same "
    "protein chain on a 3D lattice as part of a genetic search. You see "
    "observations agents have written to a shared board -- not their "
    "folds. Decide ONE thing all circles should prioritise this round. "
    "Follow the requested output format exactly and output nothing else."
)


def _central_prompt(observations, retry_hint: str) -> str:
    lines = "\n".join(f"  (gen {e.generation}, {e.author}): {e.content}" for e in observations) or "  (none yet)"
    return f"""Recent observations from the shared board:
{lines}

In one sentence, say what every circle should prioritise this round (e.g.
explore more broadly, focus on a specific part of the chain, make smaller
edits). This applies to all circles equally -- you cannot assign different
things to different circles.

Respond with EXACTLY one line and nothing else:
DIRECTIVE: <your instruction, one sentence>
{retry_hint}"""


def _deterministic_directive(per_circle: int, slot_fitnesses: dict[int, float]) -> str:
    """Structured-signal-only baseline: no free-text reading, since there is
    no non-LLM way to interpret an observation's content -- see this
    module's docstring on why this isn't a clean A/B against the llm mode."""
    n_circles = _n_circles()
    deltas = []
    for c in range(n_circles):
        members = range(c * per_circle, (c + 1) * per_circle)
        cur = [slot_fitnesses[s] for s in members if slot_fitnesses.get(s) is not None]
        prior = [_last_fitness_by_slot[s] for s in members if s in _last_fitness_by_slot]
        if cur and prior:
            deltas.append((sum(cur) / len(cur)) - (sum(prior) / len(prior)))
    stalled = bool(deltas) and max(deltas) <= 0
    return (
        "Progress has stalled recently -- try larger, more different changes this round."
        if stalled else
        "Recent changes have been improving fitness -- keep making small, targeted changes."
    )


def _central_directive(
    generation: int, slot_fitnesses: dict[int, float], per_circle: int, rng: random.Random,
) -> str:
    from hpga import blackboard as bb
    from hpga import operators as ops

    mode = _central_mode()

    def fallback():
        return _deterministic_directive(per_circle, slot_fitnesses)

    if mode != "llm":
        text = fallback()
        author = "central-deterministic"
    else:
        observations = bb.read_live(types=("observation", "summary"))[-10:]

        def build_prompt(retry_hint: str) -> str:
            return _central_prompt(observations, retry_hint)

        def parse(text: str):
            return _extract_note(text, "DIRECTIVE")

        text = ops._run_llm_op(
            op="central_directive", style="note", system=_CENTRAL_SYSTEM, build_prompt=build_prompt,
            parse=parse, rng=rng, num_predict=96, fallback=fallback,
            retry_hint_text="\nIMPORTANT: respond with ONLY the DIRECTIVE line above.",
            extra_log_fields={"generation": generation, "call_kind": "coordination", "phase3_step": "central"},
        )
        author = "central"

    bb.append(
        generation=generation, author=author, circle_id=None, type_="directive",
        content=text, tokens=_approx_tokens(text),
    )
    return text


# --- Propose (useful, one call per circle-agent slot) ------------------------

_PROPOSE_SYSTEM = (
    "You are one of several agents folding the same protein chain on a 3D "
    "lattice, working in a small group (circle) as part of a genetic "
    "search. Each move is one letter: S (straight), L (turn left), R (turn "
    "right), U (turn up), or D (turn down), relative to the current "
    "heading and orientation. Follow the requested output format exactly "
    "and output nothing else."
)


def _propose_prompt(
    genome_str: str, length: int, directive: str, consult_note: str, observations, retry_hint: str,
) -> str:
    context = ""
    if directive:
        context += f"\nCoordinator's directive this round: {directive}"
    if consult_note:
        context += f"\nYour circle's guidance this round: {consult_note}"
    if observations:
        obs_lines = "\n".join(f"  - {e.content}" for e in observations)
        context += f"\nRecent observations from the shared board:\n{obs_lines}"
    return f"""Your current fold: {genome_str}{context}

Propose a new fold of length {length}, using only the letters S, L, R, U,
D. Take the guidance above into account if it's useful.

Respond with EXACTLY two lines and nothing else:
FOLD: <{length} letters from {{S,L,R,U,D}} separated by single spaces>
RATIONALE: <one short sentence: what you changed and why, in your own words>
{retry_hint}"""


def _propose(
    circle_id: int, agent_id: int, genome: Genome, length: int, directive: str,
    consult_note: str, observations, generation: int, rng: random.Random,
) -> Genome:
    from hpga import operators as ops

    genome_str = ops._genome_to_str(genome)
    obs_ids = [e.id for e in observations]
    obs_tokens = sum(e.tokens for e in observations)

    def build_prompt(retry_hint: str) -> str:
        return _propose_prompt(genome_str, length, directive, consult_note, observations, retry_hint)

    def parse(text: str):
        g = _extract_fold_line(text, length)
        if g is None:
            return None
        return (g, _extract_note(text, "RATIONALE") or "")

    def fallback():
        return (list(genome), "")

    new_genome, _rationale = ops._run_llm_op(
        op="propose", style="full", system=_PROPOSE_SYSTEM, build_prompt=build_prompt,
        parse=parse, rng=rng, num_predict=min(2048, max(96, 8 * length + 40)), fallback=fallback,
        retry_hint_text=(
            "\nIMPORTANT: respond with ONLY the FOLD and RATIONALE lines above, "
            "FOLD with exactly the requested number of letters."
        ),
        extra_log_fields={
            "circle_id": circle_id, "agent_id": agent_id, "generation": generation,
            "call_kind": "useful", "phase3_step": "propose",
            "blackboard_entries_shown": obs_ids, "blackboard_tokens_shown": obs_tokens,
            "directive_shown": bool(directive), "consult_note_shown": bool(consult_note),
        },
        # _rationale is not logged separately: it's already inside this call's
        # "response" field in the operator-call log (the RATIONALE line), so a
        # post-hoc analysis can read it from there without a redundant field.
    )
    return new_genome


# --- Entry point called from operators.next_generation() --------------------


def run_circles(
    population: Sequence[Genome],
    fitnesses: Sequence[float],
    pop_size: int,
    genome_length: int,
    generation: int,
    rng: random.Random,
) -> list[Genome]:
    """Returns exactly n_circles * agents_per_circle genomes, one per
    (circle, agent) slot in slot-index order, meant to be APPENDED to
    new_pop by the caller (operators.next_generation()) after its ordinary
    fill-to-pop_size loop -- additive, see module docstring. Only called
    when HPGA_OPERATOR_MODE=llm and HPGA_CIRCLES_ENABLED=1 (checked by the
    caller, not here).

    Per-generation order, and why: fitness for a slot's fold isn't known
    until the generation after it's proposed (island.py's DIS phase runs
    between this call and the next). So at generation g, population/
    fitnesses already hold the outcome of what THIS module proposed at
    g-1 -- observe() uses that plus the fitness stored from g-2 (this
    module's own _last_fitness_by_slot) to ground a real before/after,
    before central/consult/propose run for g. Observations therefore start
    at generation 2, not generation 0 (see reset_circle_state() and the
    None-guards below) -- generation 0 has no tail slots yet (fresh random
    genome per slot, no LLM call, mirroring agents.py's identical gen-0
    fallback), and generation 1 has a current fitness but no prior one to
    compare against yet.
    """
    from hpga import blackboard as bb
    from hpga import operators as ops

    n_circles = _n_circles()
    per_circle = _agents_per_circle()
    n_slots = n_circles * per_circle

    slot_folds: dict[int, Genome] = {}
    slot_fitnesses: dict[int, float | None] = {}
    for slot in range(n_slots):
        tail_idx = pop_size + slot
        if tail_idx < len(population):
            slot_folds[slot] = list(population[tail_idx])
            slot_fitnesses[slot] = fitnesses[tail_idx]
        else:
            slot_folds[slot] = ops.random_genome(genome_length, rng)
            slot_fitnesses[slot] = None

    # --- observe ---
    for slot in range(n_slots):
        prior_fitness = _last_fitness_by_slot.get(slot)
        prior_genome = _last_genome_by_slot.get(slot)
        cur_fitness = slot_fitnesses[slot]
        if prior_fitness is not None and cur_fitness is not None:
            circle_id, agent_id = divmod(slot, per_circle)
            _observe(circle_id, agent_id, prior_genome, slot_folds[slot], prior_fitness, cur_fitness, generation, rng)
        if cur_fitness is not None:
            _last_fitness_by_slot[slot] = cur_fitness
            _last_genome_by_slot[slot] = slot_folds[slot]

    # --- central directive (global, every HPGA_CENTRAL_INTERVAL generations) ---
    directive_text = ""
    if generation % _central_interval() == 0:
        directive_text = _central_directive(generation, slot_fitnesses, per_circle, rng)

    # --- consult (one call per circle with >1 agent) ---
    consult_notes: dict[int, str] = {}
    for circle_id in range(n_circles):
        members = list(range(circle_id * per_circle, (circle_id + 1) * per_circle))
        if len(members) > 1:
            pairs = [(ops._genome_to_str(slot_folds[s]), slot_fitnesses[s] or 0.0) for s in members]
            consult_notes[circle_id] = _consult(circle_id, pairs, rng)
        else:
            consult_notes[circle_id] = ""

    # --- curation (periodic, board maintenance) ---
    if generation > 0 and generation % _curation_interval() == 0:
        bb.curate(generation, rng)

    # --- propose (useful calls, one per slot) ---
    recent_observations = bb.read_live(types=("observation",))[-6:]
    offspring = []
    for slot in range(n_slots):
        circle_id, agent_id = divmod(slot, per_circle)
        genome = _propose(
            circle_id, agent_id, slot_folds[slot], genome_length,
            directive_text, consult_notes.get(circle_id, ""), recent_observations, generation, rng,
        )
        offspring.append(genome)
    return offspring
