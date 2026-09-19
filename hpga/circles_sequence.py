"""Circles on the SEQUENCE genome (variable-length amino-acid str, 20 letters), reached only when an
active GenomeModel has genome_type str. hpga/circles.py dispatches here from the top of run_circles();
its lattice code is untouched, so lattice circles behaviour is unchanged (experiments/
golden_lattice_circles.py reproduces it byte for byte).

Same architecture and per-generation order as circles.run_circles -- observe, central directive,
consult, curation, propose -- and the same additive contract (exactly n_circles * agents_per_circle
genomes returned, appended by next_generation after its ordinary fill; circle-agent i owns
population index pop_size + i from generation 1 on). Every genome-touching piece is redone for str
genomes:

  PROPOSALS ARE EDITS, NOT REWRITES. The lattice propose() asks for a whole new fold ("FOLD: <letters>"):
  the full-restatement format, which falls back 80-87% of the time on this genome (PHASE3_RESULTS.md
  9.1) and which the model answers by copying. Here a proposal is `k` position edits to the slot's
  CURRENT population member, in exactly the position-style format that measured 0% fallback
  (sequence_model._mutate_position_prompt / _extract_position_mutation -- the same text and parser mutate
  uses), with the directive / circle note / observations prepended as context. k = max(1, round(rate *
  length)); rate is HPGA_CIRCLES_SEQ_EDIT_RATE (default 0.05, the GA's mutation rate, so a proposal is
  the same size of change as one LLM mutation). No RATIONALE line: circles.propose() never used it (it
  was parsed and dropped), and the position-style format is what was measured.
  OBSERVATIONS STAY TEXT on the blackboard, exactly as in the lattice version. An observation prompt also
  names the positions that changed (a proposal differs from its base at only k positions, so the diff is
  exact and cheap) so a one-sentence inference has something concrete to refer to.

The prompts describe an amino-acid sequence being designed and give fitness numbers; they do not
name the target structure (the sequence-model prompts don't either).

Circles' central directive, curation and blackboard are genome-agnostic and reused as they are.
"""

import os
import random
from typing import Sequence

from hpga.agents import _approx_tokens

_SYSTEM_COMMON = (
    "Each residue is one letter from the 20 standard amino-acid codes "
    "(A C D E F G H I K L M N P Q R S T V W Y), and sequences are written as letters separated by single spaces. "
    "Follow the requested output format exactly and output nothing else."
)
_CONSULT_SYSTEM = (
    "You are the shared discussion for one group (circle) of agents designing protein sequences for a "
    "genetic algorithm. You see this circle's current sequences and their fitnesses only (higher is better) -- "
    "other circles work independently and you cannot see their sequences. Give brief, concrete guidance for "
    "what this circle's agents should try differently this round. " + _SYSTEM_COMMON
)
_OBSERVE_SYSTEM = (
    "You write one short observation about a genetic search over protein sequences, for a shared board other "
    "agents and a coordinator will read. Write what happened and what it suggests -- not a sequence, not a "
    "restated number, an actual inference (e.g. 'edits near the start of the chain did not help' or "
    "'changing a few residues in the middle raised fitness'). " + _SYSTEM_COMMON
)
_CENTRAL_SYSTEM = (
    "You coordinate several independent circles of agents designing the same protein sequence in a genetic "
    "search. You see observations agents have written to a shared board -- not their sequences. Decide ONE "
    "thing all circles should prioritise this round. " + _SYSTEM_COMMON
)
_PROPOSE_SYSTEM = (
    "You are one of several agents designing a protein sequence for a genetic algorithm, working in a small "
    "group (circle). You improve your current sequence by editing a few positions. " + _SYSTEM_COMMON
)
_PROPOSE_RETRY_HINT = (
    "\nIMPORTANT: your previous response did not match the required format, gave a number of lines other "
    "than the requested count, repeated a position, or gave a NEW letter equal to the current letter at that "
    "position (that is not a change). Respond with ONLY the requested labelled line(s)."
)


def _spaced(genome: str) -> str:
    from hpga import sequence_model as sm

    return sm._spaced(genome)


def edit_rate() -> float:
    return float(os.environ.get("HPGA_CIRCLES_SEQ_EDIT_RATE", "0.05"))


def n_edits(length: int) -> int:
    return max(1, round(edit_rate() * length))


def changed_positions(old: str, new: str) -> list[int]:
    return [i for i, (a, b) in enumerate(zip(old, new)) if a != b] if len(old) == len(new) else []


# --- consult ---------------------------------------------------------------

def _consult_prompt(circle_id: int, folds_fitnesses: list[tuple[str, float]], retry_hint: str) -> str:
    lines = "\n".join(f"  Agent {i}: sequence={g}  fitness={f:.3f}" for i, (g, f) in enumerate(folds_fitnesses))
    return f"""Circle {circle_id}'s current agents:
{lines}

In one sentence, say what this circle should focus on differently this
round (e.g. which part of the chain to change, whether to make small or
large edits). Do not propose a sequence yourself.

Respond with EXACTLY one line and nothing else:
NOTE: <your guidance, one sentence>
{retry_hint}"""


def consult(circle_id: int, folds_fitnesses: list[tuple[str, float]], rng: random.Random) -> str:
    from hpga import circles
    from hpga import operators as ops

    note = ops._run_llm_op(
        op="consult", style="note", system=_CONSULT_SYSTEM,
        build_prompt=lambda retry_hint: _consult_prompt(circle_id, folds_fitnesses, retry_hint),
        parse=lambda text: circles._extract_note(text, "NOTE"), rng=rng, num_predict=96, fallback=lambda: "",
        retry_hint_text="\nIMPORTANT: respond with ONLY the NOTE line above.",
        extra_log_fields={"circle_id": circle_id, "call_kind": "coordination", "phase3_step": "consult"},
    )
    return note or ""


# --- observe ---------------------------------------------------------------

def _observe_prompt(old: str, new: str, old_fitness: float, new_fitness: float, retry_hint: str) -> str:
    changed = changed_positions(old, new)
    where = f"\nPositions that changed (0-indexed): {', '.join(map(str, changed))}" if 0 < len(changed) <= 8 else ""
    return f"""Previous sequence: {_spaced(old)}  (fitness {old_fitness:.3f})
Current sequence:  {_spaced(new)}  (fitness {new_fitness:.3f}){where}

Write one short observation about what this change suggests for future
search -- an inference from the numbers above, not a restatement of them.

Respond with EXACTLY one line and nothing else:
OBSERVATION: <your observation, one sentence>
{retry_hint}"""


def observe(circle_id: int, agent_id: int, prior: str, cur: str, prior_fitness: float, cur_fitness: float,
            generation: int, rng: random.Random) -> None:
    from hpga import blackboard as bb
    from hpga import circles
    from hpga import operators as ops

    result = ops._run_llm_op(
        op="observe", style="note", system=_OBSERVE_SYSTEM,
        build_prompt=lambda retry_hint: _observe_prompt(prior, cur, prior_fitness, cur_fitness, retry_hint),
        parse=lambda text: circles._extract_note(text, "OBSERVATION"), rng=rng, num_predict=96, fallback=lambda: None,
        retry_hint_text="\nIMPORTANT: respond with ONLY the OBSERVATION line above.",
        extra_log_fields={"circle_id": circle_id, "agent_id": agent_id, "generation": generation,
                          "call_kind": "coordination", "phase3_step": "observe"},
    )
    if not result:
        return  # fell back: nothing grounded to write
    bb.append(generation=generation, author=f"circle{circle_id}-agent{agent_id}", circle_id=circle_id,
              type_="observation", content=result, tokens=_approx_tokens(result))


# --- central directive -----------------------------------------------------

def central_directive(generation: int, slot_fitnesses: dict, per_circle: int, rng: random.Random) -> str:
    from hpga import blackboard as bb
    from hpga import circles
    from hpga import operators as ops

    def fallback():
        return circles._deterministic_directive(per_circle, slot_fitnesses)

    if circles._central_mode() != "llm":
        text, author = fallback(), "central-deterministic"
    else:
        observations = bb.read_live(types=("observation", "summary"))[-10:]
        text = ops._run_llm_op(
            op="central_directive", style="note", system=_CENTRAL_SYSTEM,
            build_prompt=lambda retry_hint: circles._central_prompt(observations, retry_hint),
            parse=lambda t: circles._extract_note(t, "DIRECTIVE"), rng=rng, num_predict=96, fallback=fallback,
            retry_hint_text="\nIMPORTANT: respond with ONLY the DIRECTIVE line above.",
            extra_log_fields={"generation": generation, "call_kind": "coordination", "phase3_step": "central"},
        )
        author = "central"
    bb.append(generation=generation, author=author, circle_id=None, type_="directive", content=text,
              tokens=_approx_tokens(text))
    return text


# --- propose (position edits to an existing population member) ---------------

def _context_block(directive: str, consult_note: str, observations) -> str:
    context = ""
    if directive:
        context += f"Coordinator's directive this round: {directive}\n"
    if consult_note:
        context += f"Your circle's guidance this round: {consult_note}\n"
    if observations:
        obs_lines = "\n".join(f"  - {e.content}" for e in observations)
        context += f"Recent observations from the shared board:\n{obs_lines}\n"
    return context + "Take the guidance above into account if it is useful.\n\n" if context else ""


def propose(circle_id: int, agent_id: int, genome: str, directive: str, consult_note: str, observations,
            generation: int, rng: random.Random) -> str:
    """`genome` is the slot's current population member; the result is that member with k positions edited
    (or the member unchanged if the model never produced a valid edit -- the fallback, logged as such)."""
    from hpga import operators as ops
    from hpga import sequence_model as sm

    length = len(genome)
    k = n_edits(length)
    context = _context_block(directive, consult_note, observations)
    obs_ids = [e.id for e in observations]

    result = ops._run_llm_op(
        op="propose", style="position", system=_PROPOSE_SYSTEM,
        build_prompt=lambda retry_hint: context + sm._mutate_position_prompt(_spaced(genome), length, k, retry_hint),
        parse=lambda text: sm._extract_position_mutation(text, genome, k), rng=rng,
        num_predict=min(2048, max(32, 3 * length)), fallback=lambda: genome, retry_hint_text=_PROPOSE_RETRY_HINT,
        extra_log_fields={"circle_id": circle_id, "agent_id": agent_id, "generation": generation,
                          "call_kind": "useful", "phase3_step": "propose", "n_edits_requested": k,
                          "blackboard_entries_shown": obs_ids, "blackboard_tokens_shown": sum(e.tokens for e in observations),
                          "directive_shown": bool(directive), "consult_note_shown": bool(consult_note)},
    )
    return result


# --- entry point (called from circles.run_circles when a sequence GenomeModel is active) ----------

def run_circles(model, population: Sequence[str], fitnesses: Sequence[float], pop_size: int, generation: int,
                rng: random.Random) -> list[str]:
    """Same order as the lattice run_circles: observe -> central -> consult -> curate -> propose."""
    from hpga import blackboard as bb
    from hpga import circles

    n_circles, per_circle = circles._n_circles(), circles._agents_per_circle()
    n_slots = n_circles * per_circle

    slot_folds: dict[int, str] = {}
    slot_fitnesses: dict[int, float | None] = {}
    for slot in range(n_slots):
        tail_idx = pop_size + slot
        if tail_idx < len(population):
            slot_folds[slot], slot_fitnesses[slot] = model.copy(population[tail_idx]), fitnesses[tail_idx]
        else:  # generation 0: no tail slots yet -- a fresh random sequence per slot, no LLM call to make one
            slot_folds[slot], slot_fitnesses[slot] = model.random_genome(rng, None), None

    for slot in range(n_slots):  # observe
        prior_fitness, prior_genome = circles._last_fitness_by_slot.get(slot), circles._last_genome_by_slot.get(slot)
        cur_fitness = slot_fitnesses[slot]
        if prior_fitness is not None and cur_fitness is not None:
            circle_id, agent_id = divmod(slot, per_circle)
            observe(circle_id, agent_id, prior_genome, slot_folds[slot], prior_fitness, cur_fitness, generation, rng)
        if cur_fitness is not None:
            circles._last_fitness_by_slot[slot] = cur_fitness
            circles._last_genome_by_slot[slot] = slot_folds[slot]

    directive_text = ""
    if generation % circles._central_interval() == 0:
        directive_text = central_directive(generation, slot_fitnesses, per_circle, rng)

    consult_notes: dict[int, str] = {}
    for circle_id in range(n_circles):
        members = list(range(circle_id * per_circle, (circle_id + 1) * per_circle))
        if len(members) > 1:
            consult_notes[circle_id] = consult(circle_id, [(_spaced(slot_folds[s]), slot_fitnesses[s] or 0.0) for s in members], rng)
        else:
            consult_notes[circle_id] = ""

    if generation > 0 and generation % circles._curation_interval() == 0:
        bb.curate(generation, rng)

    recent_observations = bb.read_live(types=("observation",))[-6:]
    offspring = []
    for slot in range(n_slots):
        circle_id, agent_id = divmod(slot, per_circle)
        offspring.append(propose(circle_id, agent_id, slot_folds[slot], directive_text,
                                 consult_notes.get(circle_id, ""), recent_observations, generation, rng))
    return offspring
