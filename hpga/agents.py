"""Agents as first-class entities inside one island, plus central-node
inter-agent communication and population-diversity instrumentation.

New architecture for Phase 2, not a fix: until now, crossover()/mutate() in
hpga/operators.py were stateless -- no entity had a role, an identity, or
state that persisted between calls. This module introduces `A` agents per
island, each with a stable id and a role ("explore" or "refine") that
determines its prompt, and a switchable central-node communication
protocol between them. No "recombine" role: probe_prompt_diversity.py
measured 0/10 outputs mechanically consistent with combining two parents,
so recombination stays deterministic in the master (operators.crossover/
_llm_crossover), unchanged by this module.

Deliberately additive and self-contained: hpga/island.py, worker.py,
instrumentation.py, and config.py are NOT touched. Everything here is
driven from hpga/operators.py's next_generation(), which already runs once
per generation with its call signature/site in island.py unchanged, plus a
handful of new env vars matching the existing env-var-driven config style
(HPGA_OPERATOR_MODE, HPGA_LLM_MODEL, etc.). Off by default
(HPGA_AGENTS_ENABLED unset).

Agent-to-population mapping: agents are ADDITIVE, appended after
next_generation()'s ordinary fill-to-pop_size loop rather than displacing
non-elite slots (an earlier version of this module displaced them, which
meant an agents-on run made fewer top-level LLM calls than the
agents-off baseline -- a call-budget confound on any fitness comparison
between them). Agent i (0-indexed) owns population index `pop_size + i`
in the list next_generation() returns from generation 1 onward -- a fixed
tail slot, not tracked via separate shadow state. Because Island carries
the returned population forward unchanged into the next generation's
fitness dispatch, population[pop_size + i] / fitnesses[pop_size + i] on
the *next* call to next_generation() is exactly agent i's prior fold and
its measured fitness. This is what gives agents persistent identity across
generations without island.py knowing agents exist, and without a second,
potentially drifting copy of "the population" kept in this module. Since
Island's own initial population (built in Island.__init__, before any
generation runs) has exactly pop_size individuals, agent tail slots don't
exist yet at generation 0 -- run_agents() falls back to a fresh random
genome per agent for that one generation only, after which the population
has grown to pop_size + A and stays there (next_generation()'s fill loop
always tops up to pop_size regardless of the population's current size,
so it doesn't grow further after that). Role assignment is
similarly stateless: it is a pure function of (agent_id, HPGA_AGENT_ROLES),
re-read fresh each call like every other env-var switch in this codebase,
not stored -- there is nothing that changes an agent's role mid-run.

Communication design (run_agents(), called once per generation when
enabled): every HPGA_COMM_INTERVAL generations (starting after generation
0, not at it), each agent's current fold is treated as a "send" to the
master, and the master "forwards" it individually to every OTHER agent
(A*(A-1) forward messages, not one broadcast blob) -- deliberately not
aggregated, so a later peer-to-peer topology change only has to remove the
master hop, not restructure what content moves. All A^2 messages/round (A
sends + A*(A-1) forwards) are logged individually to
results/raw/agent_messages_<run_id>.jsonl, tagged with island id, agent id
(sender/receiver), generation, an approximate token count for the payload,
and real (not simulated) latency for the in-process send/forward
bookkeeping -- there is no network hop in this single-process
implementation, so that latency is genuinely near-zero, not a stand-in for
a real one. Message "tokens" are a word-count proxy over the serialized
fold string (len(text.split())), NOT a real model tokenizer count -- these
aren't LLM calls, so there is no Ollama-reported token count to use;
documented here so it is never confused with the real tokens_in/tokens_out
already tracked for actual generate() calls in the operator-call log.

Each agent's own fold-generation call *is* a real LLM call (role-
conditioned explore/refine prompt, optionally including peer folds as
context on a comm-round generation) and reuses operators._run_llm_op /
_extract_labelled / _genome_to_str verbatim, so it goes through the exact
same retry/fallback/logging machinery as crossover/mutate, into the exact
same results/raw/llm_operator_calls_<run_id>.jsonl file, tagged with
op="agent_explore"/"agent_refine" plus island id, agent id, role, and
generation (via _run_llm_op's extra_log_fields parameter) so that log is
never missing attribution either.

operators.py imports this module at the top level; this module imports
operators lazily, inside the functions that need it, to break the
otherwise-circular import (operators.next_generation calls into here; here
calls back into operators' LLM plumbing).
"""

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

Genome = list[int]

ROLE_EXPLORE = "explore"
ROLE_REFINE = "refine"
_VALID_ROLES = (ROLE_EXPLORE, ROLE_REFINE)

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "raw"

_generation = 0  # 0-indexed, incremented once per next_generation() call


def reset_agent_state() -> None:
    """Call before a run whose generation counter you want isolated from
    any prior run in the same process -- mirrors
    operators.reset_operator_stats(). Nothing else in this module is
    stateful (role assignment is read fresh from env vars each call, not
    stored), so this is the only reset needed."""
    global _generation
    _generation = 0


def tick_generation() -> int:
    """Advance and return the 0-indexed generation counter. Called
    unconditionally, once, at the top of every next_generation() call --
    the only source of a generation number available to this module, since
    island.py's own gen_id loop variable is not passed through (island.py
    is not touched by this feature)."""
    global _generation
    g = _generation
    _generation += 1
    return g


def _n_agents() -> int:
    return int(os.environ.get("HPGA_N_AGENTS", "2"))


def _comm_interval() -> int:
    return int(os.environ.get("HPGA_COMM_INTERVAL", "5"))


def _island_id() -> int:
    return int(os.environ.get("HPGA_ISLAND_ID", "0"))


def _default_roles(n: int) -> list[str]:
    """Alternating explore/refine, explore first -- used when
    HPGA_AGENT_ROLES is unset or doesn't have exactly n valid entries."""
    return [ROLE_EXPLORE if i % 2 == 0 else ROLE_REFINE for i in range(n)]


def _resolve_roles(n: int) -> list[str]:
    """Comma-separated HPGA_AGENT_ROLES (e.g. 'explore,explore,refine'),
    read fresh -- must have exactly n entries, each 'explore' or 'refine',
    or the default alternating assignment is used instead."""
    raw = os.environ.get("HPGA_AGENT_ROLES")
    if raw:
        roles = [r.strip() for r in raw.split(",")]
        if len(roles) == n and all(r in _VALID_ROLES for r in roles):
            return roles
    return _default_roles(n)


def _run_id() -> str:
    from hpga import operators as ops
    return os.environ.get("HPGA_RUN_ID") or ops._DEFAULT_RUN_ID


# --- Population diversity ---------------------------------------------------


def mean_pairwise_hamming(population: Sequence[Genome]) -> float:
    n = len(population)
    if n < 2:
        return 0.0
    total = 0
    pairs = 0
    for i in range(n):
        gi = population[i]
        for j in range(i + 1, n):
            total += sum(a != b for a, b in zip(gi, population[j]))
            pairs += 1
    return total / pairs


def _diversity_log_path() -> Path:
    override = os.environ.get("HPGA_DIVERSITY_LOG_PATH")
    if override:
        return Path(override)
    return _RESULTS_DIR / f"diversity_{_run_id()}.jsonl"


def log_diversity(population: Sequence[Genome], generation: int) -> float:
    diversity = mean_pairwise_hamming(population)
    path = _diversity_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "island_id": _island_id(),
        "generation": generation,
        "pop_size": len(population),
        "mean_pairwise_hamming": diversity,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return diversity


# --- Communication -----------------------------------------------------------


def _approx_tokens(text: str) -> int:
    """Word-count proxy, NOT a real tokenizer -- see module docstring.
    Good enough to compare message sizes to each other and to track
    communication cost over a run; not comparable in an absolute sense to
    Ollama's own prompt_eval_count/eval_count for real generate() calls."""
    return len(text.split())


def _message_log_path() -> Path:
    override = os.environ.get("HPGA_AGENT_LOG_PATH")
    if override:
        return Path(override)
    return _RESULTS_DIR / f"agent_messages_{_run_id()}.jsonl"


def _log_message(record: dict) -> None:
    path = _message_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _run_comm_round(generation: int, agent_folds: dict[int, Genome]) -> dict[int, list[Genome]]:
    """Central-node round: every agent sends its current fold to the
    master, the master forwards each fold individually to every OTHER
    agent (not aggregated -- see module docstring: this keeps the content
    identical to what a peer-to-peer topology would carry, so topology is
    the only variable if/when that's added later). Returns {agent_id:
    [peer folds received]}. Logs every send/forward message plus one
    round-summary record."""
    from hpga import operators as ops

    n_messages = 0
    total_tokens = 0
    received: dict[int, list[Genome]] = {aid: [] for aid in agent_folds}

    for sender_id, genome in agent_folds.items():
        payload = ops._genome_to_str(genome)
        tokens = _approx_tokens(payload)

        t0 = time.perf_counter()
        # "send": agent -> master. In-process bookkeeping, no transport --
        # latency measured here is genuinely near-zero, not a network stand-in.
        latency = time.perf_counter() - t0
        _log_message({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "island_id": _island_id(), "generation": generation,
            "message_type": "send", "sender_agent_id": sender_id, "receiver": "master",
            "tokens_in": tokens, "tokens_out": tokens, "latency_s": latency,
            "payload": payload,
        })
        n_messages += 1
        total_tokens += tokens

        for receiver_id in agent_folds:
            if receiver_id == sender_id:
                continue
            t0 = time.perf_counter()
            # "forward": master -> agent, unaggregated (see module docstring).
            latency = time.perf_counter() - t0
            _log_message({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "island_id": _island_id(), "generation": generation,
                "message_type": "forward", "sender_agent_id": sender_id,
                "receiver": receiver_id,
                "tokens_in": tokens, "tokens_out": tokens, "latency_s": latency,
                "payload": payload,
            })
            n_messages += 1
            total_tokens += tokens
            received[receiver_id].append(genome)

    _log_message({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "island_id": _island_id(), "generation": generation,
        "message_type": "round_summary", "n_agents": len(agent_folds),
        "n_messages": n_messages, "total_tokens": total_tokens,
    })
    return received


# --- Agent fold-generation LLM calls -----------------------------------------


_EXPLORE_SYSTEM = (
    "You are one of several independent agents folding the same protein "
    "chain on a 3D lattice, each trying a different approach. Your role is "
    "EXPLORE: propose a fold that is substantially different from the one "
    "you are given, not a small variation of it. Each move is one letter: "
    "S (straight), L (turn left), R (turn right), U (turn up), or D (turn "
    "down), relative to the current heading and orientation. Follow the "
    "requested output format exactly and output nothing else."
)

_REFINE_SYSTEM = (
    "You are one of several independent agents folding the same protein "
    "chain on a 3D lattice, each trying a different approach. Your role is "
    "REFINE: improve the fold you are given with small, targeted changes, "
    "not a wholesale replacement. Each move is one letter: S (straight), L "
    "(turn left), R (turn right), U (turn up), or D (turn down), relative "
    "to the current heading and orientation. Follow the requested output "
    "format exactly and output nothing else."
)


def _peer_context(peer_folds: list[Genome]) -> str:
    from hpga import operators as ops
    if not peer_folds:
        return ""
    lines = "\n".join(f"  Peer {i + 1}: {ops._genome_to_str(g)}" for i, g in enumerate(peer_folds))
    return (
        f"\n\nOther agents' current folds, for context only -- you are not "
        f"required to match or avoid them:\n{lines}"
    )


def _explore_prompt(genome_str: str, length: int, peer_folds: list[Genome], retry_hint: str) -> str:
    return f"""Your current fold: {genome_str}{_peer_context(peer_folds)}

Propose a substantially different fold of length {length}, using only the
letters S, L, R, U, D. It should differ from your current fold in many
positions, not a small tweak.

Respond with EXACTLY one line and nothing else:
FOLD: <{length} letters from {{S,L,R,U,D}} separated by single spaces>
{retry_hint}"""


def _refine_min_changed() -> int:
    """Minimum number of positions a refine call must actually change,
    enforced by _extract_refine_position via reject-and-retry -- same
    purpose as operators.CROSSOVER_MIN_DIFF, sized for refine's much
    smaller intended edit (a handful of positions, not a whole genome).
    Starting value of 1 only closes off the exact fixed-point no-op
    documented in PHASE2_RESULTS.md Sec. 8.3, not intended to force a
    large edit."""
    return int(os.environ.get("HPGA_REFINE_MIN_CHANGED", "1"))


def _refine_position_prompt(
    genome_str: str, length: int, peer_folds: list[Genome], min_changed: int, retry_hint: str,
) -> str:
    """Position-only affordance fix for refine, mirroring
    operators._mutate_position_prompt (Sec. 4.3): the model names only the
    position(s) it wants to change and the new letter, never restates the
    fold, so copying the input back is not an expressible answer -- unlike
    mutate/position, the count is not fixed at exactly k; refine's own
    'small, targeted changes' framing means the model picks how many,
    subject only to the min_changed floor enforced by the parser."""
    return f"""Your current fold (0-indexed positions 0-{length - 1}): {genome_str}{_peer_context(peer_folds)}

Improve this fold with small, targeted changes: name only the position(s)
you want to change and the new letter for each -- most positions should
stay the same as your current fold. Change at least {min_changed}
position(s), but only as many as you judge actually improve the fold. The
new letter at each named position MUST differ from the letter currently
there (copying it back is not a change). Do not restate the whole fold.

Respond with one line per change and nothing else (at least {min_changed}
line(s)):
POSITION: <0-{length - 1}>, NEW: <S, L, R, U, or D>
{retry_hint}"""


def _extract_refine_position(text: str, base: Genome, length: int, min_changed: int) -> Genome | None:
    """Parse one or more 'POSITION: <n>, NEW: <letter>' lines and apply them
    to a copy of `base` -- same POSITION/NEW grammar as
    operators._extract_position_mutation, reused via ops._POSITION_PAIR, but
    with a floor (min_changed) instead of an exact count, since refine's
    edit size isn't fixed the way mutate's k is. Rejects: fewer than
    min_changed pairs, an out-of-range or repeated position, or a NEW letter
    equal to the letter already at that position (there is nothing left to
    restate in this format, so a same-letter pair is the only way left to
    fake a no-op)."""
    from hpga import operators as ops

    pairs = ops._POSITION_PAIR.findall(text)
    if len(pairs) < min_changed:
        return None
    result = list(base)
    seen: set[int] = set()
    for pos_str, letter in pairs:
        pos = int(pos_str)
        if pos < 0 or pos >= length or pos in seen:
            return None
        letter_code = ops._CHAR_TO_MOVE[letter.upper()]
        if letter_code == base[pos]:
            return None
        seen.add(pos)
        result[pos] = letter_code
    return result


def _refine_retry_hint(min_changed: int) -> str:
    return (
        f"\nIMPORTANT: your previous response did not match the required "
        f"format, gave fewer than {min_changed} line(s), repeated a "
        f"position, or gave a NEW letter equal to the current letter at "
        f"that position (that is not a change). Respond with ONLY the "
        f"labelled line(s) above, at least {min_changed} of them."
    )


_RETRY_HINT = (
    "\nIMPORTANT: your previous response did not match the required "
    "format. Respond with ONLY the FOLD line above, containing exactly "
    "the requested number of S/L/R/U/D letters."
)


def _agent_generate(
    agent_id: int, role: str, genome: Genome, peer_folds: list[Genome],
    length: int, generation: int, rng: random.Random,
) -> Genome:
    from hpga import operators as ops

    genome_str = ops._genome_to_str(genome)

    def fallback():
        return list(genome)  # no-op: keep this agent's current fold unchanged

    if role == ROLE_REFINE:
        # Position-only format (Sec. 8.3 fix): copying the input is not an
        # expressible answer here, same affordance idea as mutate/position
        # (Sec. 4.3). Explore is untouched below -- it already works.
        min_changed = _refine_min_changed()
        style = "position"
        system = _REFINE_SYSTEM
        num_predict = min(2048, max(24, 10 * min_changed))

        def build_prompt(retry_hint: str) -> str:
            return _refine_position_prompt(genome_str, length, peer_folds, min_changed, retry_hint)

        def parse(text: str):
            return _extract_refine_position(text, genome, length, min_changed)

        retry_hint_text = _refine_retry_hint(min_changed)
    else:
        style = "full"
        system = _EXPLORE_SYSTEM
        num_predict = min(2048, max(64, 8 * length))

        def build_prompt(retry_hint: str) -> str:
            return _explore_prompt(genome_str, length, peer_folds, retry_hint)

        def parse(text: str):
            return ops._extract_labelled(text, "FOLD", length)

        retry_hint_text = _RETRY_HINT

    result = ops._run_llm_op(
        op=f"agent_{role}", style=style, system=system, build_prompt=build_prompt,
        parse=parse, rng=rng, num_predict=num_predict, fallback=fallback,
        retry_hint_text=retry_hint_text,
        extra_log_fields={
            "island_id": _island_id(), "agent_id": agent_id, "role": role,
            "generation": generation,
        },
    )

    # Position-bias tracking, same purpose as PHASE2_RESULTS.md §7.6's
    # mutate/position position_counts: agents respond with a full fold
    # (style="full"), not named positions, so the "which position changed"
    # signal has to be derived by diffing the returned fold against the one
    # the agent was given, rather than read directly off the response. One
    # supplementary log line per call, in the same operator-call log, so
    # position-bias analysis doesn't need a second log file or to
    # reconstruct genomes from prompt text after the fact.
    changed = [i for i, (a, b) in enumerate(zip(genome, result)) if a != b]
    ops._log_call({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "op": "agent_fold_diff", "island_id": _island_id(), "agent_id": agent_id,
        "role": role, "generation": generation,
        "changed_positions": changed, "n_changed": len(changed), "length": length,
    })

    return result


# --- Entry point called from operators.next_generation() --------------------


def run_agents(
    population: Sequence[Genome],
    fitnesses: Sequence[float],
    pop_size: int,
    genome_length: int,
    generation: int,
    rng: random.Random,
) -> list[Genome]:
    """Returns exactly A genomes (A = HPGA_N_AGENTS), one per agent in
    agent-id order, meant to be APPENDED to new_pop by the caller
    (operators.next_generation()) after its ordinary fill-to-pop_size loop
    -- additive, not a slot carve-out (see module docstring). Only called
    when HPGA_OPERATOR_MODE=llm and HPGA_AGENTS_ENABLED=1 (checked by the
    caller, not here)."""
    n = _n_agents()
    roles = _resolve_roles(n)
    agent_folds = {}
    for i in range(n):
        tail_idx = pop_size + i
        if tail_idx < len(population):
            agent_folds[i] = list(population[tail_idx])
        else:
            # Generation 0 only: Island's initial population has exactly
            # pop_size individuals, no agent tail slots yet.
            from hpga import operators as ops
            agent_folds[i] = ops.random_genome(genome_length, rng)

    peer_by_agent: dict[int, list[Genome]] = {i: [] for i in range(n)}
    interval = _comm_interval()
    if generation > 0 and generation % interval == 0:
        peer_by_agent = _run_comm_round(generation, agent_folds)

    offspring = []
    for i in range(n):
        new_genome = _agent_generate(
            i, roles[i], agent_folds[i], peer_by_agent[i], genome_length, generation, rng,
        )
        offspring.append(new_genome)
    return offspring
