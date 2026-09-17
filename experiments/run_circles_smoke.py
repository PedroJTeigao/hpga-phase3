"""Phase 3 first deliverable, second half: paired smoke run of the circle/
blackboard architecture (hpga/circles.py, hpga/blackboard.py) -- circles
disabled (today's plain LLM-operator baseline, HPGA_CIRCLES_ENABLED=0) vs
enabled (HPGA_N_CIRCLES=2, HPGA_AGENTS_PER_CIRCLE=2, HPGA_CENTRAL_MODE=llm,
HPGA_CURATION_INTERVAL=3).

Revision 2, after the first smoke run (pop_size=5, n_generations=6,
HPGA_CENTRAL_INTERVAL=1, HPGA_LLM_PROMPT_STYLE unset -> "full") surfaced
three problems, not fixed here silently -- each is a direct response to
that run's numbers:

1. Crossover under the "full" prompt style fell back to the deterministic
   operator 82-100% of the time in that run (same unresolved issue
   diagnosed against results/raw/llm_operator_calls_phase2pilot_llm_*
   before Phase 3 started). At that fallback rate, most children in BOTH
   arms came from the deterministic operator, which contaminates the
   diversity comparison -- the "off" arm was effectively running
   deterministic crossover at pop_size 5, not LLM crossover, so the
   diversity gap it showed against the "on" arm can't be attributed to
   circles. Fixed here by setting HPGA_LLM_PROMPT_STYLE="segment", the
   crossover format PHASE2_RESULTS.md sec 7.6.1 already measured at 0%
   fallback on this exact 3D/5-symbol alphabet (n=20, genome_length=18).
   Mutate is left on "full" (unset defaults to it) -- it measured 0%
   fallback in the first smoke run already; this change is scoped to the
   operator that was actually broken.
2. HPGA_CENTRAL_INTERVAL=1 fires a directive call every generation, which
   made "coordination calls per useful call" mostly a measurement of that
   interval choice, not of the architecture. Set to 5 here.
3. best_fitness_by_gen was flat the entire run in both arms, which said
   nothing either way about the architecture. Checked directly (deterministic
   GA, no LLM, free) before changing anything: at pop_size=5 the GA itself
   doesn't move off its starting fitness until ~generation 23 for this
   sequence/seed -- 6 or even 15 generations at pop_size=5 could never have
   shown movement, independent of circles. pop_size=8 (seed=0) moves by
   generation 1 and holds, deterministically confirmed, so raised POP_SIZE
   to 8 and N_GENERATIONS to 10 (cheaper than reaching gen ~23 at pop_size 5,
   and independently verified capable of showing movement) rather than
   guessing at a longer genome instead.

Computes the five measurements from the Phase 3 design doc:
coordination-calls-per-useful-call, tokens read vs. tokens the propose
call's own RATIONALE line reflects, board-size growth, diversity per
generation (reusing agents.log_diversity's existing file/metric, via
operators.next_generation()'s unconditional HPGA_LOG_DIVERSITY hook -- not
a Phase 3-specific mechanism), and fitness gain per token.

The "tokens reflected" number is reported as a FLOOR, not a measure: it's
mechanical keyword overlap between a shown blackboard entry's content and
the propose call's RATIONALE text. An agent that reads "the tail hasn't
helped" and moves elsewhere because of it scores zero overlap while having
used the entry correctly -- this only ever undercounts genuine use, never
overcounts it.

Revision 3, after revision 2's run (genome_length=8, "segment" crossover)
found the format itself was 100% parse-clean on both arms (0/156 invalid
attempts were parse failures) but overall crossover fallback stayed high
(92% off / 54% on) purely from _crossover_sufficiently_mixed rejections --
a real, distinct mechanism, not a leftover of the format bug:

1. Genuinely identical selected parents (28% off / 20% on of top-level
   crossover calls) make CROSSOVER_MIN_DIFF=2 impossible to satisfy under
   ANY prompt style -- not a model failure. This is more likely at short
   genome lengths, small pop_size, and as a population converges -- exactly
   what off-arm's faster-converging population produces more of, which is
   itself possibly a downstream effect of losing more crossovers to
   deterministic fallback in the first place (self-reinforcing). Whether
   that loop is a real mechanism the circles interrupt, or just noise, is
   what this revision is for.
2. Even with genuinely different parents, CROSSOVER_MIN_DIFF=2 was
   validated at genome_length=18 (PHASE2_RESULTS.md sec 7.6.1); at length 8
   each half of a 2-segment split has only 4 positions to accumulate 2+
   differences in, a tighter bar than the 9-position halves the threshold
   was actually measured against. SEQUENCE changed to genome_length=18 (via
   make_timing_sequence, matching sec 7.6.1's methodology) to remove this
   mismatch WITHOUT touching CROSSOVER_MIN_DIFF -- recalibrating the
   threshold for this run would make it a free parameter tuned toward the
   result, and lose the anchor to the one length it's actually validated at.
   POP_SIZE/N_GENERATIONS unchanged (deterministically re-confirmed to still
   move fitness at genome_length=18 before running anything real).

hpga/operators.py's _llm_crossover() now takes an optional `generation`
parameter and logs it plus `identical_parents` on every attempt -- added
specifically so mechanism 1 above is directly observable per generation per
arm (identical_parent_rate_by_generation() below), not just as an aggregate
count. Revision 3's result: a real, persistent gap (mean identical-parent
rate 0.54 off vs 0.20 on) but too coarse (3-4 crossover calls/generation) to
show a climb, and additionally confounded by mutate/full collapsing at
genome_length=18 (88-93% fallback, discovered while investigating revision
3 -- see PHASE2_RESULTS.md-to-be sec on the mutate curve). Not fixed in
revision 3 by design (mutate wasn't what that revision was testing).

Revision 4, after fixing mutate the same way it was fixed before (position
format, verified via experiments/probe_operator_compliance.py: 0% fallback
at both genome_length 8 and 18, vs. full's 0-16%/60-93% split -- three
independent measurements of that curve now). PROMPT_STYLE="best"
(hpga/operators.py) runs position for mutate and segment for crossover
simultaneously -- the two couldn't be selected together under the old
single-value env var. This is the run revision 3 was actually blocked on:
with both operators reliable, fallback in both arms should drop sharply,
producing more genuine (non-deterministic-fallback) crossover calls per
generation and a finer identical-parent trace than revision 3's 3-4
calls/generation could resolve.

Three more additions, all in response to what broke revision 3's first
attempt (an httpx.ReadTimeout after 120s, on this shared, unscheduled node):
- gpu_snapshot() (usage + compute-apps), taken immediately before the run
  starts and immediately after it ends, saved into the summary JSON --
  self-reports whether the GPU was actually free at both ends instead of
  requiring that to be reconstructed from memory after the fact.
- warm_up_model() -- one throwaway generate() call before timing starts, so
  a cold model load (confirmed to take up to ~270s on this deployment, from
  the mutate-curve probe's 14.47s-mean-at-length-8 anomaly) lands in setup
  time, not inside the first arm's measured latency.
- HPGA_LLM_TIMEOUT_S raised to 300s (was the 120s default) -- generous
  enough for a real stall to still be caught, not so tight that a slow-but-
  legitimate response gets mistaken for one.
"""

import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpga.config import HPGAConfig, make_timing_sequence
from hpga.island import Island
from hpga import operators as ops
from hpga import agents
from hpga import circles
from hpga import blackboard as bb

RESULTS_RAW = Path(__file__).resolve().parent.parent / "results" / "raw"

SEQUENCE = make_timing_sequence(length=20, seed=1)  # genome_length=18, matches PHASE2_RESULTS.md sec 7.6.1
POP_SIZE = 8       # deterministically confirmed to move by gen 1-2 at this size/genome_length
ELITISM = 1
N_WORKERS = 2
N_GENERATIONS = 10
N_CIRCLES = 2
AGENTS_PER_CIRCLE = 2
CENTRAL_MODE = "llm"
CENTRAL_INTERVAL = 5
CURATION_INTERVAL = 3
PROMPT_STYLE = "best"  # position (mutate) + segment (crossover) simultaneously; CROSSOVER_MIN_DIFF untouched
MODEL = "gemma4:12b"


def gpu_snapshot() -> dict:
    def _run(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as exc:
            return f"<nvidia-smi call failed: {exc!r}>"

    return {
        "usage": _run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader"]),
        "compute_apps": _run(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
                               "--format=csv,noheader"]) or "(none)",
    }


def warm_up_model() -> float:
    """One throwaway generate() call so a cold model load (confirmed up to
    ~270s on this deployment) lands here, not inside the first arm's timed
    measurement. Bypasses the operator-mode machinery entirely -- this is a
    raw _call_ollama, not a mutate/crossover call, so it needs no genome and
    isn't logged to the operator-call log."""
    t0 = time.perf_counter()
    ops._call_ollama("Say OK.", "You are a test.", 8, 0)
    return time.perf_counter() - t0


def run_arm(*, circles_enabled: bool, run_id: str) -> dict:
    os.environ["HPGA_OPERATOR_MODE"] = "llm"
    os.environ["HPGA_LLM_MODEL"] = MODEL
    os.environ["HPGA_LLM_PROMPT_STYLE"] = PROMPT_STYLE
    os.environ["HPGA_CIRCLES_ENABLED"] = "1" if circles_enabled else "0"
    os.environ["HPGA_AGENTS_ENABLED"] = "0"
    os.environ["HPGA_N_CIRCLES"] = str(N_CIRCLES)
    os.environ["HPGA_AGENTS_PER_CIRCLE"] = str(AGENTS_PER_CIRCLE)
    os.environ["HPGA_CENTRAL_MODE"] = CENTRAL_MODE
    os.environ["HPGA_CENTRAL_INTERVAL"] = str(CENTRAL_INTERVAL)
    os.environ["HPGA_CURATION_INTERVAL"] = str(CURATION_INTERVAL)
    os.environ["HPGA_ISLAND_ID"] = "0"
    os.environ["HPGA_LOG_DIVERSITY"] = "1"
    os.environ["HPGA_RUN_ID"] = run_id

    ops.reset_operator_stats()
    agents.reset_agent_state()
    circles.reset_circle_state()
    bb.reset_blackboard()

    cfg = HPGAConfig(
        sequence=SEQUENCE, pop_size=POP_SIZE, n_generations=N_GENERATIONS,
        n_workers=N_WORKERS, elitism=ELITISM, seed=0,
    )
    island = Island(cfg)

    t0 = time.perf_counter()
    recorder = island.run()
    wall_s = time.perf_counter() - t0

    return {
        "arm": "on" if circles_enabled else "off",
        "run_id": run_id,
        "wall_s": wall_s,
        "best_fitness_by_gen": recorder.best_fitness_by_gen,
        "operator_stats": ops.get_operator_stats(),
    }


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "with", "this",
    "that", "it", "its", "your", "you", "is", "are", "was", "were", "not",
    "for", "as", "at", "be", "have", "has", "from", "which", "one", "only",
    "more", "less", "than", "did", "do", "does", "will", "would", "near",
}


def _keywords(text: str) -> set:
    return {
        w.strip(".,;:").lower() for w in text.split()
        if len(w) > 3 and w.strip(".,;:").lower() not in _STOPWORDS
    }


def tokens_reflected_floor(ops_log: list[dict], bb_entries: dict) -> dict:
    """Floor, not a measure -- see module docstring. Only counts propose
    calls that were actually shown >=1 blackboard entry."""
    propose_calls = [r for r in ops_log if r.get("phase3_step") == "propose" and r.get("valid")]
    shown_calls = [r for r in propose_calls if r.get("blackboard_entries_shown")]
    if not shown_calls:
        return {"n_calls_with_entries_shown": 0}

    n_overlap = 0
    tokens_shown_total = 0
    for r in shown_calls:
        m = re.search(r"RATIONALE\s*:\s*(.+)", r.get("response", ""), re.IGNORECASE)
        rationale = m.group(1).strip() if m else ""
        rationale_kw = _keywords(rationale)
        shown_ids = r["blackboard_entries_shown"]
        shown_kw = set()
        for eid in shown_ids:
            entry = bb_entries.get(eid)
            if entry:
                shown_kw |= _keywords(entry["content"])
        tokens_shown_total += r.get("blackboard_tokens_shown", 0)
        if rationale_kw & shown_kw:
            n_overlap += 1

    return {
        "n_calls_with_entries_shown": len(shown_calls),
        "n_with_keyword_overlap": n_overlap,
        "overlap_floor_fraction": n_overlap / len(shown_calls),
        "total_blackboard_tokens_shown": tokens_shown_total,
        "mean_blackboard_tokens_shown_per_call": tokens_shown_total / len(shown_calls),
    }


def coordination_ratio(ops_log: list[dict]) -> dict:
    """Top-level calls only (attempt==0), not log lines: ops_log has one
    record per REQUEST, including retries, so counting every record with a
    given call_kind actually measures 'coordination requests per useful
    request', not 'coordination calls per useful call' -- the two diverge
    whenever either kind's retry rate isn't 0, which caught this metric out
    in revision 4 once propose stopped being perfectly 0%-fallback."""
    useful = [r for r in ops_log if r.get("call_kind") == "useful" and r.get("attempt") == 0]
    coordination = [r for r in ops_log if r.get("call_kind") == "coordination" and r.get("attempt") == 0]
    return {
        "n_useful_calls": len(useful),
        "n_coordination_calls": len(coordination),
        "coordination_per_useful": (len(coordination) / len(useful)) if useful else None,
    }


def board_growth_trace(bb_log: list[dict]) -> list[dict]:
    return [
        {"generation": r["generation"], "live_entries": r["live_entries"], "live_tokens": r["live_tokens"]}
        for r in bb_log if r.get("event") == "board_size"
    ]


def identical_parent_rate_by_generation(ops_log: list[dict]) -> list[dict]:
    """Per generation: fraction of top-level crossover calls (attempt=0,
    past the rate-gate skip) whose two selected parents were literally
    identical genomes -- the mechanism that makes CROSSOVER_MIN_DIFF
    unsatisfiable regardless of prompt style (see module docstring)."""
    by_gen: dict[int, list[bool]] = defaultdict(list)
    for r in ops_log:
        if r.get("op") == "crossover" and r.get("attempt") == 0 and "identical_parents" in r:
            gen = r.get("generation")
            if gen is not None:
                by_gen[gen].append(r["identical_parents"])
    return [
        {"generation": g, "n_calls": len(vals), "identical_rate": sum(vals) / len(vals)}
        for g, vals in sorted(by_gen.items())
    ]


def fitness_per_token(result: dict, ops_log: list[dict]) -> dict:
    fits = result["best_fitness_by_gen"]
    gain = fits[-1] - fits[0] if fits else 0.0
    total_tokens = sum(r.get("tokens_in", 0) + r.get("tokens_out", 0) for r in ops_log)
    return {
        "fitness_gain": gain,
        "total_tokens": total_tokens,
        "fitness_gain_per_1k_tokens": (gain / total_tokens * 1000) if total_tokens else None,
    }


def main() -> None:
    ts = int(time.time())
    off_run_id = f"circles_smoke_off_{ts}"
    on_run_id = f"circles_smoke_on_{ts}"

    warm_s = warm_up_model()
    print(f"model warm-up: {warm_s:.1f}s (excluded from measured latency)\n", flush=True)

    gpu_before = gpu_snapshot()
    print(f"gpu before: {gpu_before['usage']}  compute_apps: {gpu_before['compute_apps']}\n", flush=True)

    print(f"=== OFF arm (circles disabled, run_id={off_run_id}) ===", flush=True)
    off_result = run_arm(circles_enabled=False, run_id=off_run_id)
    print(f"wall_s={off_result['wall_s']:.1f}  best_fitness_by_gen={off_result['best_fitness_by_gen']}")
    print(f"operator_stats={off_result['operator_stats']}\n", flush=True)

    print(f"=== ON arm (circles enabled, run_id={on_run_id}) ===", flush=True)
    on_result = run_arm(circles_enabled=True, run_id=on_run_id)
    print(f"wall_s={on_result['wall_s']:.1f}  best_fitness_by_gen={on_result['best_fitness_by_gen']}")
    print(f"operator_stats={on_result['operator_stats']}\n", flush=True)

    off_ops_log = load_jsonl(RESULTS_RAW / f"llm_operator_calls_{off_run_id}.jsonl")
    on_ops_log = load_jsonl(RESULTS_RAW / f"llm_operator_calls_{on_run_id}.jsonl")
    on_bb_log = load_jsonl(RESULTS_RAW / f"blackboard_{on_run_id}.jsonl")
    on_bb_entries = {r["id"]: r for r in on_bb_log if r.get("event") == "entry"}

    off_div = load_jsonl(RESULTS_RAW / f"diversity_{off_run_id}.jsonl")
    on_div = load_jsonl(RESULTS_RAW / f"diversity_{on_run_id}.jsonl")
    print("=== 1. Population diversity per generation (mean pairwise Hamming) ===")
    print(f"  off: {[(r['generation'], round(r['mean_pairwise_hamming'], 2)) for r in off_div]}")
    print(f"  on:  {[(r['generation'], round(r['mean_pairwise_hamming'], 2)) for r in on_div]}\n")

    coord = coordination_ratio(on_ops_log)
    print("=== 2. Coordination calls per useful call (on arm only -- off arm has no circle calls) ===")
    print(f"  {coord}\n")

    reflected = tokens_reflected_floor(on_ops_log, on_bb_entries)
    print("=== 3. Tokens read from blackboard vs. tokens the propose call's RATIONALE reflects (FLOOR, not a measure) ===")
    print(f"  {reflected}\n")

    growth = board_growth_trace(on_bb_log)
    print("=== 4. Board-size growth over generations (on arm; curation fires at generation(s) "
          f"{[g for g in range(1, N_GENERATIONS) if g % CURATION_INTERVAL == 0]}) ===")
    print(f"  {growth}\n")

    off_fpt = fitness_per_token(off_result, off_ops_log)
    on_fpt = fitness_per_token(on_result, on_ops_log)
    print("=== 5. Fitness gain per token ===")
    print(f"  off: {off_fpt}")
    print(f"  on:  {on_fpt}\n")

    off_identical = identical_parent_rate_by_generation(off_ops_log)
    on_identical = identical_parent_rate_by_generation(on_ops_log)
    print("=== 6. Identical-parent rate per generation (crossover) -- the feedback-loop diagnostic ===")
    print(f"  off: {[(r['generation'], r['n_calls'], round(r['identical_rate'], 2)) for r in off_identical]}")
    print(f"  on:  {[(r['generation'], r['n_calls'], round(r['identical_rate'], 2)) for r in on_identical]}\n")

    gpu_after = gpu_snapshot()
    print(f"gpu after: {gpu_after['usage']}  compute_apps: {gpu_after['compute_apps']}\n", flush=True)

    summary = {
        "gpu_before": gpu_before, "gpu_after": gpu_after, "warm_up_s": warm_s,
        "off": off_result, "on": on_result,
        "off_diversity": off_div, "on_diversity": on_div,
        "coordination_ratio": coord,
        "tokens_reflected_floor": reflected,
        "board_growth": growth,
        "off_fitness_per_token": off_fpt,
        "on_fitness_per_token": on_fpt,
        "off_identical_parent_rate": off_identical,
        "on_identical_parent_rate": on_identical,
    }
    out_path = RESULTS_RAW / f"circles_smoke_summary_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
