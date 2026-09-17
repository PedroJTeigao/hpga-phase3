"""Phase 3: shared blackboard for circle-based agents.

New architecture, not an extension of hpga/agents.py's role/communication
design. PHASE2_RESULTS.md sec 8 found that passing candidate folds between
agents carried no reasoning the receiver couldn't derive itself -- the
blackboard exists so agents exchange OBSERVATIONS instead: short reasoning
about the search ("changes near the end of the chain haven't helped"),
never a fold. hpga/circles.py is the only caller of this module; this file
owns the entry schema, the append-only log, and curation.

Append-only by design: curate() never deletes or rewrites a JSONL line.
Superseding an entry appends a tombstone record (live=False) for the old
id(s) plus a new type="summary" entry -- the full history stays on disk
even though the *live* working set (what circles.py actually reads) can
shrink. This is what makes the curation-on vs curation-off board-growth
comparison honest: board size depends only on whether curation ran, not on
how it happened to be implemented.

`tokens` on each entry is a word-count proxy (hpga.agents._approx_tokens,
reused as a utility, not as part of that module's architecture -- see
hpga/circles.py's docstring), used only for the board-size growth trace
below. The authoritative per-call tokens_in/tokens_out for every LLM call
this module or hpga/circles.py makes is already logged by
hpga.operators._run_llm_op into the usual
results/raw/llm_operator_calls_<run_id>.jsonl -- that log, not this one, is
what the "coordination calls per useful call" and "fitness gain per token"
measurements read.

Imports hpga.operators lazily inside functions (not at module level) to
avoid a circular import: operators.py imports hpga.circles at the top
level, circles.py imports this module at the top level, and both of those
import operators lazily -- the same pattern hpga/agents.py already uses
for the same reason.
"""

import json
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from hpga.agents import _approx_tokens

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "raw"

_lock = threading.Lock()  # guards _entries/_seq, mirrors operators._stats_lock
_entries: dict[str, "BlackboardEntry"] = {}  # id -> entry, insertion order = write order
_seq = 0


@dataclass
class BlackboardEntry:
    id: str
    generation: int
    author: str                 # "circle{c}-agent{a}" | "central" | "central-deterministic" | "curator"
    circle_id: int | None       # None for central/curator entries
    type: str                   # "observation" | "directive" | "summary"
    content: str
    tokens: int
    source_entry_ids: list[str] = field(default_factory=list)  # non-empty only for type="summary"
    live: bool = True


def reset_blackboard() -> None:
    """Call before a run whose blackboard state should be isolated from any
    prior run in the same process -- mirrors agents.reset_agent_state() /
    operators.reset_operator_stats()."""
    global _entries, _seq
    with _lock:
        _entries = {}
        _seq = 0


def _run_id() -> str:
    from hpga import operators as ops
    return os.environ.get("HPGA_RUN_ID") or ops._DEFAULT_RUN_ID


def _log_path() -> Path:
    override = os.environ.get("HPGA_BLACKBOARD_LOG_PATH")
    if override:
        return Path(override)
    return _RESULTS_DIR / f"blackboard_{_run_id()}.jsonl"


def _write_line(record: dict) -> None:
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


def _next_id() -> str:
    global _seq
    with _lock:
        _seq += 1
        return f"{_run_id()}-{_seq}"


def append(
    generation: int, author: str, circle_id: int | None, type_: str, content: str,
    tokens: int, source_entry_ids: list[str] | None = None,
) -> BlackboardEntry:
    """Writes one entry: updates the in-memory live index, appends one JSONL
    'entry' line, then appends a 'board_size' line recording the post-write
    live-entry count and live-token total -- the size trace the
    board-growth measurement reads directly, so it doesn't need to be
    reconstructed after the fact by replaying entry/tombstone events."""
    entry = BlackboardEntry(
        id=_next_id(), generation=generation, author=author, circle_id=circle_id,
        type=type_, content=content, tokens=tokens,
        source_entry_ids=list(source_entry_ids or []),
    )
    with _lock:
        _entries[entry.id] = entry
        live_count = sum(1 for e in _entries.values() if e.live)
        live_tokens = sum(e.tokens for e in _entries.values() if e.live)
    _write_line({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "entry", **asdict(entry),
    })
    _write_line({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "board_size", "generation": generation,
        "live_entries": live_count, "live_tokens": live_tokens,
    })
    return entry


def _mark_dead(entry_id: str, superseded_by: str | None, generation: int) -> None:
    """Also writes a 'board_size' line, same as append(): without this, a
    curate() call's LAST board_size snapshot (written by the summary
    entry's own append()) would transiently overstate live size by however
    many members that same call is about to drop -- append() runs before
    the mark_dead() calls for its own merge group, so the size it logs
    briefly double-counts the new summary AND the members it replaces."""
    with _lock:
        e = _entries.get(entry_id)
        if e is None:
            return
        e.live = False
        live_count = sum(1 for x in _entries.values() if x.live)
        live_tokens = sum(x.tokens for x in _entries.values() if x.live)
    _write_line({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "tombstone", "id": entry_id, "superseded_by": superseded_by,
    })
    _write_line({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "board_size", "generation": generation,
        "live_entries": live_count, "live_tokens": live_tokens,
    })


def read_live(types: tuple[str, ...] | None = None) -> list[BlackboardEntry]:
    """Entries visible to circles.py. No circle_id filtering: the whole
    point of the blackboard is that it's the cross-circle channel, so every
    circle sees every live entry regardless of which circle authored it --
    circle_id on an entry is attribution metadata, not an access filter.
    In-circle consultation (hpga.circles._consult) never touches this
    module at all, which is what keeps "circles can't see each other's
    consultation" true by construction rather than by convention."""
    with _lock:
        items = [e for e in _entries.values() if e.live]
    if types is not None:
        items = [e for e in items if e.type in types]
    return sorted(items, key=lambda e: e.generation)


# --- Curation -----------------------------------------------------------

_CURATION_SYSTEM = (
    "You maintain a shared notes board for a genetic search over 3D "
    "protein folds. Entries are short observations agents wrote about the "
    "search, never folds themselves. Decide which entries are still "
    "useful, which should be merged into one summary, and which are stale "
    "and can be dropped. Follow the requested output format exactly and "
    "output nothing else."
)

_CURATION_RETRY_HINT = (
    "\nIMPORTANT: your previous response did not match the required "
    "format. Every entry id you mention must be copied exactly as given "
    "above. Respond with DROP/MERGE lines for the entries you want to "
    "change (entries you don't mention are kept as-is), then one SUMMARY "
    "line per merge group."
)


def _curation_prompt(entries: list[BlackboardEntry], generation: int, retry_hint: str) -> str:
    lines = "\n".join(
        f"  {e.id} (gen {e.generation}, {e.author}, {e.type}): {e.content}"
        for e in entries
    )
    return f"""Current generation: {generation}
Board entries ({len(entries)} total):
{lines}

Decide what to do with these entries. Anything you don't mention is kept
unchanged -- only list entries you want to DROP or MERGE.

Respond with this format, zero or more DROP/MERGE lines, then zero or
more SUMMARY lines (one per merge group), and nothing else:
DROP: <id>, <id>, ...
MERGE <group letter> <id>, <id>, ...
SUMMARY <group letter>: <one-line summary of that group>
{retry_hint}"""


_DROP_LINE = re.compile(r"^DROP\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_MERGE_LINE = re.compile(r"^MERGE\s+(\S+)\s+(.+)$", re.IGNORECASE | re.MULTILINE)
_SUMMARY_LINE = re.compile(r"^SUMMARY\s+(\S+)\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def _extract_curation(text: str, valid_ids: set) -> dict | None:
    """Returns {'decisions': {id: 'keep'|'drop'|group_letter},
    'summaries': {group_letter: text}}, or None if nothing in the response
    names a valid id at all. Every id not explicitly mentioned defaults to
    'keep' -- a response that names nothing changes nothing, a safe
    no-op rather than a data-loss risk."""
    decisions: dict[str, str] = {vid: "keep" for vid in valid_ids}
    found_any = False

    for m in _DROP_LINE.finditer(text):
        for tok in m.group(1).split(","):
            tok = tok.strip().rstrip(".")
            if tok in decisions:
                decisions[tok] = "drop"
                found_any = True

    groups: dict[str, list[str]] = {}
    for m in _MERGE_LINE.finditer(text):
        group, ids_part = m.group(1).strip(":"), m.group(2)
        ids = [t.strip().rstrip(".") for t in ids_part.split(",")]
        ids = [t for t in ids if t in decisions]
        if ids:
            groups.setdefault(group, []).extend(ids)

    summaries: dict[str, str] = {}
    for m in _SUMMARY_LINE.finditer(text):
        group, summary = m.group(1).strip(":"), m.group(2).strip()
        if group in groups and summary:
            summaries[group] = summary

    for group, ids in groups.items():
        if group not in summaries or len(ids) < 2:
            continue
        for tok in ids:
            decisions[tok] = group
        found_any = True

    if not found_any:
        return None
    return {"decisions": decisions, "summaries": summaries}


def curate(generation: int, rng) -> dict:
    """One coordination LLM call, run by hpga.circles.run_circles() when
    generation % curation_interval == 0. Reads every live entry, applies
    KEEP/DROP/MERGE decisions to the in-memory index, and appends tombstone
    + summary records for whatever changed. On repeated parse failure the
    fallback is the identity -- leave the board untouched, logged
    explicitly by _run_llm_op's own fallback event -- never a silent drop,
    since losing entries silently would corrupt the exact curation-on vs
    curation-off comparison this phase measures."""
    from hpga import operators as ops

    live = read_live()
    if len(live) < 2:
        return {"applied": False, "reason": "fewer than 2 live entries", "n_live": len(live)}

    valid_ids = {e.id for e in live}
    num_predict = min(2048, max(128, 20 * len(live)))

    def build_prompt(retry_hint: str) -> str:
        return _curation_prompt(live, generation, retry_hint)

    def parse(text: str):
        return _extract_curation(text, valid_ids)

    def fallback():
        return None

    result = ops._run_llm_op(
        op="curate", style="curation", system=_CURATION_SYSTEM, build_prompt=build_prompt,
        parse=parse, rng=rng, num_predict=num_predict, fallback=fallback,
        retry_hint_text=_CURATION_RETRY_HINT,
        extra_log_fields={"generation": generation, "call_kind": "coordination", "phase3_step": "curate"},
    )
    if result is None:
        return {"applied": False, "reason": "no parseable decision", "n_live": len(live)}

    decisions, summaries = result["decisions"], result["summaries"]
    groups: dict[str, list[str]] = {}
    for entry_id, decision in decisions.items():
        if decision == "drop":
            _mark_dead(entry_id, superseded_by=None, generation=generation)
        elif decision != "keep":
            groups.setdefault(decision, []).append(entry_id)

    n_merged = 0
    for group_key, member_ids in groups.items():
        summary_text = summaries.get(group_key, "")
        if not summary_text:
            continue
        summary_entry = append(
            generation=generation, author="curator", circle_id=None, type_="summary",
            content=summary_text, tokens=_approx_tokens(summary_text), source_entry_ids=member_ids,
        )
        for mid in member_ids:
            _mark_dead(mid, superseded_by=summary_entry.id, generation=generation)
        n_merged += len(member_ids)

    n_dropped = sum(1 for d in decisions.values() if d == "drop")
    return {"applied": True, "n_live_before": len(live), "n_dropped": n_dropped, "n_merged": n_merged}
