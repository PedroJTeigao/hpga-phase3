"""Parallel agents on the SEQUENCE genome (variable-length amino-acid str, 20 letters), with an optional
private record of each agent's own past proposals ("memory"). Reached only when an active GenomeModel has
genome_type str: hpga/agents.py dispatches here from the top of run_agents(); its lattice code is untouched.

What an agent is here. The lattice agents have roles (explore = whole-fold rewrite, refine = position edits);
explore's whole-restatement format falls back 80-87% of the time on this genome (PHASE3_RESULTS.md 9.1), so it
is not ported and roles do not exist in sequence mode. An agent is a persistent population slot plus a record:
agent i owns population index pop_size + i from generation 1 on (the additive contract of agents.py), and each
generation it proposes `k` position edits to the CURRENT occupant of that slot -- literally the mutate/position
call (same system prompt, same parser, same retry hint, k = max(1, round(rate * length)); rate is
HPGA_AGENTS_SEQ_EDIT_RATE, default 0.05, the GA's mutation rate). Nothing is shared between agents: no peer
messages (agents.py's communication rounds are not ported), no blackboard, and an agent's prompt is built from
its own record only.

The record. For every proposal an agent makes whose base genome has a measured fitness, the module keeps
  {agent_id, generation, base_fitness, changes [(position, old letter, new letter)], fitness_after, improved}
where improved means fitness_after > base_fitness (strict). The fitness result is routed back to the proposer
through the slot itself: the genome an agent proposes at generation g is carried by Island into the next
population at the same tail index, so at generation g+1 population[pop_size + i] / fitnesses[pop_size + i] IS
agent i's proposal and its measured fitness (a cache hit if the proposal was a fallback and equals its base).
The pending entry stored at g (base genome, base fitness, changes) is resolved against it. If the tail slot
does not hold what the agent proposed, the entry is dropped and counted as misrouted (verify_agent_memory.py
asserts that count is 0). A generation-0 proposal has no record entry: its base is a fresh random genome that
is never evaluated, so there is no "before" fitness (same as circles). The first entry therefore exists from
generation 2, and a run of G evaluated populations yields G - 2 entries per agent.

Two switches, both read fresh on every call:
  HPGA_AGENTS_MEMORY=0 (default)  the record is KEPT AND LOGGED but never shown to the model -- the
                                  "agents without memory" arm, measured by exactly the same code
  HPGA_AGENTS_MEMORY=1            the last HPGA_AGENTS_MEMORY_WINDOW (default 8) entries with a valid edit, plus
                                  a one-line tally over all of them, are prepended to that agent's prompt
An entry for a fallback proposal (the model never produced a valid edit, the agent's genome stayed as it was)
is kept in the log with no_edit=true and is not shown to the model and not counted in improvement rates.

Nothing about the target structure is named in any prompt. Every proposal and every resolution is logged to
results/raw/agent_memory_<run_id>.jsonl (HPGA_AGENT_MEMORY_LOG_PATH overrides), and every LLM call goes
through operators._run_llm_op into the usual llm_operator_calls_<run_id>.jsonl with op="agent_propose",
agent_id, generation, n_edits_requested and how many record entries the prompt carried.
"""

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "raw"

# agent_id -> resolved entries, oldest first (dicts, see module docstring)
_records: dict[int, list[dict]] = {}
# agent_id -> the proposal made last generation, waiting for its fitness
_pending: dict[int, dict] = {}
_stats = {"proposals": 0, "resolved": 0, "no_edit": 0, "misrouted": 0}


def reset_state() -> None:
    """Called from agents.reset_agent_state(): records must not leak between runs in one process."""
    _records.clear()
    _pending.clear()
    for k in _stats:
        _stats[k] = 0


def get_stats() -> dict:
    return dict(_stats)


def get_records() -> dict[int, list[dict]]:
    return {i: [dict(e) for e in v] for i, v in _records.items()}


def memory_enabled() -> bool:
    return os.environ.get("HPGA_AGENTS_MEMORY", "0") == "1"


def memory_window() -> int:
    return int(os.environ.get("HPGA_AGENTS_MEMORY_WINDOW", "8"))


def edit_rate() -> float:
    return float(os.environ.get("HPGA_AGENTS_SEQ_EDIT_RATE", "0.05"))


def n_edits(length: int) -> int:
    return max(1, round(edit_rate() * length))


def _log_path() -> Path:
    override = os.environ.get("HPGA_AGENT_MEMORY_LOG_PATH")
    if override:
        return Path(override)
    from hpga import operators as ops

    return _RESULTS_DIR / f"agent_memory_{os.environ.get('HPGA_RUN_ID') or ops._DEFAULT_RUN_ID}.jsonl"


def _log(record: dict) -> None:
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), **record}, default=str) + "\n")


def changes_between(old: str, new: str) -> list[list]:
    """[[position, old letter, new letter], ...] for equal-length strings, else []."""
    return [[i, a, b] for i, (a, b) in enumerate(zip(old, new)) if a != b] if len(old) == len(new) else []


# --- the record as prompt text -------------------------------------------------------------------

def _outcome(before: float, after: float) -> str:
    return "better" if after > before else ("worse" if after < before else "same")


def _describe_changes(changes: list[list]) -> str:
    parts = [f"{p} {a}->{b}" for p, a, b in changes]
    return ("position " if len(parts) == 1 else "positions ") + ", ".join(parts)


def render_record(entries: Sequence[dict], window: int | None = None) -> str:
    """The context block for one agent's next prompt, from that agent's own entries only. Entries without a
    valid edit are skipped. Empty string when there is nothing to show (generation 0-2: no resolved entry yet)."""
    window = memory_window() if window is None else window
    valid = [e for e in entries if not e["no_edit"]]
    if not valid:
        return ""
    shown = valid[-window:]
    lines = "\n".join(
        f"  round {e['generation']}: {_describe_changes(e['changes'])}; "
        f"fitness {e['base_fitness']:.3f} -> {e['fitness_after']:.3f} ({_outcome(e['base_fitness'], e['fitness_after'])})"
        for e in shown
    )
    n_better = sum(1 for e in valid if e["improved"])
    return (
        "Your own record of the edits you proposed in earlier rounds (oldest first), with the fitness of "
        "the sequence before and after each one (higher is better):\n"
        f"{lines}\n"
        f"So far {n_better} of your {len(valid)} edits raised fitness.\n"
        "Take this into account if it is useful.\n\n"
    )


# --- one proposal ------------------------------------------------------------------------------------

def propose(agent_id: int, genome: str, entries: Sequence[dict], generation: int, rng: random.Random,
            show_record: bool | None = None) -> str:
    """`genome` is the agent's current slot occupant; the result is that genome with k positions edited (or
    unchanged if the model never produced a valid edit -- the fallback, logged as such). `entries` is THIS
    agent's record; it is shown only when memory is on."""
    from hpga import operators as ops
    from hpga import sequence_model as sm

    length = len(genome)
    k = n_edits(length)
    plan = sm.plan_llm_mutate("position", genome, k)  # the mutate/position call, unchanged
    show = memory_enabled() if show_record is None else show_record
    context = render_record(entries) if show else ""
    n_shown = len([e for e in entries if not e["no_edit"]][-memory_window():]) if context else 0

    return ops._run_llm_op(
        op="agent_propose", style="position", system=plan.system,
        build_prompt=lambda retry_hint: context + plan.build_prompt(retry_hint), parse=plan.parse, rng=rng,
        num_predict=plan.num_predict, fallback=lambda: genome, retry_hint_text=plan.retry_hint_text,
        extra_log_fields={"agent_id": agent_id, "generation": generation, "call_kind": "useful",
                          "n_edits_requested": k, "memory_enabled": bool(show), "record_entries_shown": n_shown},
    )


# --- entry point (called from agents.run_agents when a sequence GenomeModel is active) -----------------

def run_agents(model, population: Sequence[str], fitnesses: Sequence[float], pop_size: int, generation: int,
               rng: random.Random) -> list[str]:
    """Returns exactly HPGA_N_AGENTS genomes, agent-id order, to be APPENDED by next_generation (additive)."""
    from hpga import agents

    n = agents._n_agents()
    folds: dict[int, str] = {}
    fits: dict[int, float | None] = {}
    for i in range(n):
        tail_idx = pop_size + i
        if tail_idx < len(population):
            folds[i], fits[i] = model.copy(population[tail_idx]), fitnesses[tail_idx]
        else:  # generation 0: no tail slots yet -- a fresh random sequence per agent, unevaluated
            folds[i], fits[i] = model.random_genome(rng, None), None

    # route each agent's fitness result back to it: its last proposal now sits in its own tail slot
    for i in range(n):
        p = _pending.pop(i, None)
        if p is None or fits[i] is None:
            continue
        if folds[i] != p["proposal"]:
            _stats["misrouted"] += 1
            _log({"event": "misrouted", "agent_id": i, "generation": generation})
            continue
        entry = {
            "agent_id": i, "generation": p["generation"], "base_fitness": p["base_fitness"],
            "changes": p["changes"], "fitness_after": fits[i], "no_edit": not p["changes"],
            "improved": bool(p["changes"]) and fits[i] > p["base_fitness"],
        }
        _records.setdefault(i, []).append(entry)
        _stats["resolved"] += 1
        _stats["no_edit"] += int(entry["no_edit"])
        _log({"event": "resolved", "resolved_at_generation": generation,
              "record_size_after": len(_records[i]), **entry})

    offspring = []
    for i in range(n):
        own = _records.get(i, [])
        new = propose(i, folds[i], own, generation, rng)
        changes = changes_between(folds[i], new)
        _stats["proposals"] += 1
        _log({"event": "proposal", "agent_id": i, "generation": generation, "base": folds[i], "base_fitness": fits[i],
              "proposal": new, "changes": changes, "fallback": not changes, "memory_enabled": memory_enabled(),
              "record_entries_shown": len([e for e in own if not e["no_edit"]][-memory_window():]) if memory_enabled() else 0})
        if fits[i] is not None:
            _pending[i] = {"generation": generation, "base_fitness": fits[i], "proposal": new, "changes": changes}
        offspring.append(new)
    return offspring
