"""Live compliance probe for the LLM genetic operators on the SEQUENCE genome
(variable-length amino-acid strings, 20-letter alphabet) -- the counterpart of
probe_operator_compliance.py, which does the same for the 5-symbol lattice
genome. Same four conditions (mutate/full, mutate/position, crossover/full,
crossover/segment) -- run here in priority order, the condition that matters
first so it is checkpointed before any later one can time out -- with the same
fresh-random-genome-per-call methodology, same GPU snapshots before/after, and
same output-JSON convention.
Fires real Ollama calls through the actual hpga.operators._llm_mutate /
_llm_crossover machinery, dispatched through an active SequenceGenomeModel:
this script calls genome_model.set_active(build_genome_model(config)) itself,
since the lattice probe (and every other existing script) doesn't, and the
operators refuse a str genome with no active model.

It never evaluates fitness, so ESMFold is not loaded: SequenceGenomeModel is
constructed but its GPU-heavy _get_fitness() is lazy and never reached.

DELIBERATE DIFFERENCES FROM probe_operator_compliance.py (recorded in every
output JSON under "differences_from_lattice_probe" so a gap between the two
models' results isn't misattributed to the alphabet):

  1. Segment prompt: the sequence "segment" prompt carries a worked example
     ('e.g. 0-40:1, 40-100:2') and uses percentage boundaries shared by
     consecutive segments, because parents differ in length; the lattice
     segment prompt has neither. A segment-compliance gap is alphabet +
     boundary convention + example. (See hpga/sequence_model.py's docstring.)
  2. One change per mutate call, by construction: mutation rate = 1/genome_len
     so k = round(rate * len) = 1. The lattice probe's 0.05 at length 18 also
     gave k = 1, which is what makes each call exactly one independent
     (position, letter) observation. 0.05 at length 63 would give k = 3
     correlated observations per call and break the chi-square below.
  3. Crossover rate is 1.0, not 0.9: every call is a live LLM call, so N calls
     is N samples. (The lattice probe's 0.9 makes ~10% of its calls rate-gate
     no-ops that divide into its fallback_rate.)
  4. Parents are variable-length (uniform over [MIN_LENGTH, MAX_LENGTH]);
     mutate uses one fixed length (--genome-len, default 63 = the 7UR7
     resolved core).
  5. Uniformity is computed on LLM-SUCCESSFUL calls only. A fallback call's
     output comes from the deterministic operator (uniform by construction)
     and would dilute exactly the bias this measures.
  6. Per-condition sample sizes differ (see DEFAULT_N) -- below. mutate/full
     and crossover/full are 30-call CONFIRMATION runs, not measurements: on
     the lattice, full-style mutate failed 150/150 and 88-93% at length 18, so
     the question is only whether that pattern holds at 20 letters. 30 calls
     bounds a zero-success result at <= ~10% success (rule of three) and would
     show a surprisingly high success rate immediately; if it does, that is a
     result to follow up with a real sample, and their letter/position tests
     here (on successes only, so tiny n) are descriptive at best.

WHY mutate/position DEFAULTS TO 400 CALLS, not 40-50. The 5-symbol uniformity
check (PHASE2_RESULTS.md sec. 4.4: chi-square of chosen positions, df=17,
n=40, 211.10) worked at n=40 because what it found was a GROSS collapse
(80% of mass on 2 of 18 cells). It would still detect that at 20 letters. What
does not carry over is (a) validity -- 40 calls over 20 letters is 2 expected
per cell, under the usual >=5 for the chi-square approximation -- and (b)
power for a moderate bias. Monte-Carlo power at alpha=.05 (20 cells; effect
size w, w=.25 ~ half the letters at 1.25x and half at 0.75x): n=50 -> 12%,
n=100 -> 24%, n=200 -> 52%, n=400 -> 89% (80% power needs ~350; ~170 for w=.35).
400 also gives 6.3 expected per cell for the position test at length 63
(96% power at w=.35), so both tests are valid. To stay independent of the
approximation the p-values here are Monte-Carlo against the simulated null,
valid at any n -- but validity is not power, and n_success is reported next to
every test so an underpowered "no bias found" is not read as "no bias".

Null hypothesis: on uniform-random genomes, a fair operator's NEW letter is
uniform over the 20 letters (P(new=x) = P(current != x) * 1/19 = 1/20 by
symmetry, given NEW must differ from current) and its position is uniform
over genome_len positions.

Model is whatever HPGA_LLM_MODEL is set to (read by operators.py at import).
Every call is also logged to the usual llm_operator_calls_<run_id>.jsonl.

SEGMENT-EXAMPLE ABLATION (--segment-example {current,shifted,absent}). The
first run found 83/100 segment cuts at 40, the value in the prompt's worked
example ('e.g. 0-40:1, 40-100:2'). This flag varies that example, one setting
per invocation, to see whether the modal cut follows it:
  current -- the example as sequence_model.py has it (prompts left untouched)
  shifted -- the same example with both boundary numbers 70
  absent  -- the example removed; the rest of the prose is unchanged
The example is varied wherever it occurs: in the segment prompt AND in the
retry hint, so a retry can't reintroduce the value being ablated. The variant
is applied by wrapping sm._crossover_segment_prompt / patching
sm._RETRY_HINT_SEGMENT from HERE; hpga/ is not edited, and omitting the flag
changes nothing. NOT varied, in any setting: the prose sentence "a boundary at
40 falls 40% of the way along Parent 1 and 40% ... Parent 2" and the
"SEGMENTS: <start>-<end>:<parent>" template line (no numbers). The first is a
second '40' anchor outside the worked example; the header records it under
"unvaried_anchors" so a modal of 40 under 'absent' isn't read as "not the
prompt". Only crossover/segment is affected, so the flag requires it in
--conditions; the output file gets a _segex_<setting> suffix automatically so
it can't overwrite the default one. Each result carries "segment_example_check"
(what the logged prompts actually contained) and "boundary_summary" (the full
cut distribution and every modal value, ties included).
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpga import genome_model
from hpga import operators as ops
from hpga import sequence_model as sm
from hpga.config import HPGAConfig

ALPHABET = sm.ALPHABET
GENOME_LEN = 63  # 7UR7's resolved core; must lie in [sm.MIN_LENGTH, sm.MAX_LENGTH]
CROSSOVER_RATE = 1.0

# Calls per condition (560 total before retries). mutate/position carries the
# uniformity analysis (see module docstring). crossover/segment is the one
# genuinely new format here -- percentage cuts for unequal lengths plus a
# worked example the lattice prompt lacks -- so it gets a real sample: N=100
# bounds a zero-failure fallback rate at <= 3% (rule of three) and a 50% rate
# at +-10pp. The two full-style conditions are 30-call confirmations (see
# module docstring, difference 6).
DEFAULT_N = {"mutate/position": 400, "crossover/segment": 100, "mutate/full": 30, "crossover/full": 30}

DIFFERENCES_FROM_LATTICE_PROBE = [
    "segment prompt has a worked example and percentage boundaries (deliberate; sequence_model.py docstring)",
    "mutation rate = 1/genome_len so k=1 exactly (one independent observation per call)",
    "crossover rate 1.0 (every call is a live LLM call)",
    "variable-length parents; mutate at one fixed genome length",
    "uniformity computed on LLM-successful calls only",
    "per-condition sample sizes (mutate/position=400, crossover/segment=100, full styles=30 confirmation runs)",
]


def gpu_snapshot() -> dict:
    # Same as probe_operator_compliance.gpu_snapshot; copied, not imported, so
    # a change to that script can't silently change this one.
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


# --- statistics (numpy only; scipy isn't in requirements.txt) ----------------

def chi2_uniform(counts: list[int]) -> tuple[float, float]:
    n, k = sum(counts), len(counts)
    e = n / k
    return sum((c - e) ** 2 / e for c in counts), e


def mc_p_value(stat: float, n: int, k: int, n_sim: int = 20000, seed: int = 0) -> float:
    """Monte-Carlo p-value of a uniform-null chi-square statistic: valid at any
    n, unlike the chi-square approximation (needs ~5 expected per cell)."""
    sims = np.random.default_rng(seed).multinomial(n, [1.0 / k] * k, size=n_sim)
    e = n / k
    sim_stats = ((sims - e) ** 2 / e).sum(axis=1)
    return float((1 + (sim_stats >= stat).sum()) / (1 + n_sim))


def uniformity(observed: Counter, cells: list, label: str) -> dict:
    counts = [observed.get(c, 0) for c in cells]
    n = sum(counts)
    if n == 0:
        return {"label": label, "n": 0, "note": "no successful calls"}
    stat, e = chi2_uniform(counts)
    ranked = sorted(counts, reverse=True)
    return {
        "label": label, "n": n, "n_cells": len(cells), "expected_per_cell": e,
        "chi2": stat, "df": len(cells) - 1,
        "mc_p_value": mc_p_value(stat, n, len(cells)),
        "asymptotic_chi2_valid": e >= 5,
        "n_distinct": sum(1 for c in counts if c),
        "top1_share": ranked[0] / n, "top3_share": sum(ranked[:3]) / n,
    }


# --- conditions --------------------------------------------------------------

def _failures() -> int:
    return ops.get_operator_stats()["n_failures"]


def run_mutate(style: str, n_calls: int, rng: random.Random, genome_len: int) -> dict:
    rate = 1.0 / genome_len
    assert max(1, round(rate * genome_len)) == 1, "k must be exactly 1 (see module docstring, difference 2)"
    os.environ["HPGA_LLM_PROMPT_STYLE"] = style
    ops.reset_operator_stats()
    positions, letters = Counter(), Counter()
    n_success = zero_diff = anomalies = 0
    for _ in range(n_calls):
        genome = sm.random_sequence(rng, genome_len)
        before = _failures()
        out = ops._llm_mutate(genome, rate, rng)
        diffs = [i for i, (a, b) in enumerate(zip(genome, out)) if a != b]
        zero_diff += (len(diffs) == 0)
        if _failures() > before:
            continue  # fell back: deterministic output, excluded from uniformity
        n_success += 1
        if len(diffs) != 1 or len(out) != len(genome):
            anomalies += 1  # can't happen if the parser is right; counted, not hidden
            continue
        positions[diffs[0]] += 1
        letters[out[diffs[0]]] += 1
    stats = ops.get_operator_stats()
    return {
        "op": "mutate", "style": style, "model": ops.LLM_MODEL, "n_calls": n_calls, "genome_len": genome_len,
        "stats": stats, "fallback_rate": stats["n_failures"] / n_calls,
        "requests_per_call": stats["n_llm_requests"] / n_calls,
        "n_success": n_success, "zero_diff_outputs": zero_diff, "success_anomalies": anomalies,
        "letter_counts": dict(sorted(letters.items())),
        "position_counts": dict(sorted(positions.items())),
        "letter_uniformity": uniformity(letters, list(ALPHABET), "NEW letter over 20"),
        "position_uniformity": uniformity(positions, list(range(genome_len)), f"position over {genome_len}"),
    }


_SEGMENT_TRIPLE = re.compile(r"(\d+)\s*-\s*(\d+)\s*:\s*([12])")


def _segment_records(path: Path, start_line: int, style: str) -> list[dict]:
    if not path.exists():
        return []
    recs = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            r = json.loads(line)
            if r.get("op") == "crossover" and r.get("prompt_style") == style and r.get("valid") is True:
                recs.append(r)
    return recs


def segment_structure(records: list[dict]) -> dict:
    """What the model actually declared, from the valid attempts in the log:
    the collapse check the 5-symbol study did by hand (PHASE2_RESULTS.md 4.4:
    always 2 segments, cut at the midpoint in 36/40)."""
    n_segments, cuts = Counter(), Counter()
    for r in records:
        m = re.search(r"SEGMENTS\s*:\s*(.*)", r.get("response", ""))
        triples = _SEGMENT_TRIPLE.findall(m.group(1)) if m else []
        segs = sorted((int(s), int(e), int(p)) for s, e, p in triples)
        n_segments[len(segs)] += 1
        for s, _, _ in segs[1:]:
            cuts[s] += 1
    total_cuts = sum(cuts.values())
    out = {"n_valid_declarations": len(records), "segments_per_call": dict(sorted(n_segments.items())),
           "cut_percent_counts": dict(sorted(cuts.items()))}
    if total_cuts:
        top, top_n = cuts.most_common(1)[0]
        out.update({
            "n_distinct_cut_values": len(cuts), "modal_cut": top, "modal_cut_share": top_n / total_cuts,
            "share_cuts_45_to_55": sum(c for v, c in cuts.items() if 45 <= v <= 55) / total_cuts,
            "single_cut_share": n_segments.get(2, 0) / max(1, len(records)),
        })
    return out


def run_crossover(style: str, n_calls: int, rng: random.Random, genome_len: int) -> dict:
    os.environ["HPGA_LLM_PROMPT_STYLE"] = style
    ops.reset_operator_stats()
    log_path = ops._log_path()
    start_line = sum(1 for _ in open(log_path, encoding="utf-8")) if log_path.exists() else 0
    echo = n_success = 0
    child_lens = []
    for _ in range(n_calls):
        p1, p2 = sm.random_sequence(rng), sm.random_sequence(rng)
        before = _failures()
        c1, c2 = ops._llm_crossover(p1, p2, CROSSOVER_RATE, rng)
        if _failures() > before:
            continue
        n_success += 1
        echo += (c1 in (p1, p2) or c2 in (p1, p2))
        child_lens += [len(c1), len(c2)]
    stats = ops.get_operator_stats()
    result = {
        "op": "crossover", "style": style, "model": ops.LLM_MODEL, "n_calls": n_calls,
        "stats": stats, "fallback_rate": stats["n_failures"] / n_calls,
        "requests_per_call": stats["n_llm_requests"] / n_calls,
        "n_success": n_success, "echo_outputs": echo,
        "child_length_min": min(child_lens) if child_lens else None,
        "child_length_max": max(child_lens) if child_lens else None,
    }
    if style == "segment":
        result["segment_structure"] = segment_structure(_segment_records(log_path, start_line, style))
    return result


# --- segment-example ablation (see module docstring) -------------------------

SEGMENT_EXAMPLE_SETTINGS = ("current", "shifted", "absent")
_EXAMPLE_TEXT = " (e.g. 0-40:1, 40-100:2)"  # as in sequence_model.py, with the leading space
_SHIFTED_EXAMPLE_TEXT = " (e.g. 0-70:1, 70-100:2)"
UNVARIED_ANCHORS = [
    'prose "a boundary at 40 falls 40% of the way along Parent 1 and 40% of the way along Parent 2" '
    "(present in all three settings)",
    'template line "SEGMENTS: <start>-<end>:<parent>, ..." (no numbers; all three settings)',
]


def _swap_example(text: str, setting: str, where: str) -> str:
    if text.count(_EXAMPLE_TEXT) != 1:  # sequence_model.py changed under us: refuse rather than mis-ablate
        raise SystemExit(f"expected exactly one {_EXAMPLE_TEXT!r} in the {where}, found {text.count(_EXAMPLE_TEXT)}")
    return text.replace(_EXAMPLE_TEXT, _SHIFTED_EXAMPLE_TEXT if setting == "shifted" else "")


def apply_segment_example(setting: str) -> None:
    """Rewrite the segment prompt's worked example, everywhere it occurs, for
    this process. 'current' patches nothing."""
    if setting == "current":
        return
    orig_prompt = sm._crossover_segment_prompt
    sm._RETRY_HINT_SEGMENT = _swap_example(sm._RETRY_HINT_SEGMENT, setting, "retry hint")  # read at plan time

    def varied_prompt(p1_str, p2_str, n1, n2, retry_hint):
        # retry_hint is the already-varied constant above, so only the body's example is left to swap
        return _swap_example(orig_prompt(p1_str, p2_str, n1, n2, retry_hint), setting, "segment prompt")

    sm._crossover_segment_prompt = varied_prompt


def _count_lines(path: Path) -> int:
    return sum(1 for _ in open(path, encoding="utf-8")) if path.exists() else 0


def segment_example_check(path: Path, start_line: int) -> dict:
    """What the segment prompts actually sent contained, from the call log: the
    proof the setting took effect, counted over EVERY logged attempt."""
    n = orig = shifted = any_eg = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            r = json.loads(line)
            if r.get("op") != "crossover" or r.get("prompt_style") != "segment" or "prompt" not in r:
                continue
            n += 1
            orig += "0-40:1, 40-100:2" in r["prompt"]
            shifted += "0-70:1, 70-100:2" in r["prompt"]
            any_eg += "e.g." in r["prompt"]
    return {"n_logged_attempts": n, "with_0-40_example": orig, "with_0-70_example": shifted, "with_any_e.g.": any_eg}


def boundary_summary(structure: dict) -> dict:
    """The cut distribution from segment_structure(), plus EVERY modal value
    (the structure's own modal_cut silently takes the first of a tie)."""
    cuts = {int(k): v for k, v in structure["cut_percent_counts"].items()}
    total = sum(cuts.values())
    if not total:
        return {"n_cuts": 0}
    top = max(cuts.values())
    modal = sorted(v for v, c in cuts.items() if c == top)
    return {"n_cuts": total, "n_valid_declarations": structure["n_valid_declarations"],
            "cut_percent_counts": dict(sorted(cuts.items())), "modal_values": modal,
            "modal_count": top, "modal_share_of_cuts": top / total}


CONDITIONS = [  # priority order == default run order
    ("mutate/position", run_mutate, "position"),
    ("crossover/segment", run_crossover, "segment"),
    ("mutate/full", run_mutate, "full"),
    ("crossover/full", run_crossover, "full"),
]


def _print_result(label: str, r: dict) -> None:
    s = r["stats"]
    lat = f"{s['mean_latency_s']:.3f}" if s["mean_latency_s"] is not None else "n/a"
    print(f"  fallback_rate={r['fallback_rate']:.3f}  requests_per_call={r['requests_per_call']:.2f}  "
          f"mean_latency_s={lat}  n_retries={s['n_retries']}  n_success={r['n_success']}/{r['n_calls']}")
    if r["op"] == "mutate":
        print(f"  zero_diff_outputs={r['zero_diff_outputs']}/{r['n_calls']}  anomalies={r['success_anomalies']}")
        for key in ("letter_uniformity", "position_uniformity"):
            u = r[key]
            if u["n"] == 0:
                print(f"  {u['label']}: no successful calls")
                continue
            print(f"  {u['label']}: n={u['n']} chi2={u['chi2']:.1f} (df={u['df']}) MC p={u['mc_p_value']:.4f} "
                  f"distinct={u['n_distinct']}/{u['n_cells']} top1={u['top1_share']:.2f} top3={u['top3_share']:.2f} "
                  f"exp/cell={u['expected_per_cell']:.1f}{'' if u['asymptotic_chi2_valid'] else ' (asymptotic chi2 INVALID)'}")
    else:
        print(f"  echo_outputs={r['echo_outputs']}/{r['n_success']} successful  "
              f"child_length=[{r['child_length_min']}, {r['child_length_max']}]")
        if "segment_structure" in r:
            print(f"  segment_structure={r['segment_structure']}")
        if "boundary_summary" in r:
            print(f"  segment_example={r['segment_example']}  check={r['segment_example_check']}")
            print(f"  boundary_summary={r['boundary_summary']}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-calls", type=int, default=None,
                        help=f"Calls per condition, overriding all defaults {DEFAULT_N}.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditions", type=str, default=None,
                        help="Comma-separated conditions, RUN IN THE ORDER GIVEN (default: all, in this order): "
                             + ", ".join(c[0] for c in CONDITIONS))
    parser.add_argument("--genome-len", type=int, default=GENOME_LEN,
                        help="Mutate genome length (default 63); must be within [MIN_LENGTH, MAX_LENGTH].")
    parser.add_argument("--out-suffix", type=str, default="")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Default: results/raw. (The operator call log location is HPGA_LLM_LOG_PATH / "
                             "HPGA_RUN_ID, as everywhere else.)")
    parser.add_argument("--segment-example", choices=SEGMENT_EXAMPLE_SETTINGS, default=None,
                        help="Ablate the worked example in the crossover/segment prompt (see module docstring). "
                             "Omitted: prompts untouched and output identical to before.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and exit without contacting Ollama.")
    args = parser.parse_args(argv)

    conditions = CONDITIONS
    if args.conditions:
        by_label = {c[0]: c for c in CONDITIONS}
        labels = args.conditions.split(",")
        missing = [l for l in labels if l not in by_label]
        if missing:
            raise SystemExit(f"Unknown condition label(s): {sorted(missing)}")
        if len(set(labels)) != len(labels):
            raise SystemExit(f"Duplicate condition label(s) in --conditions: {labels}")
        conditions = [by_label[l] for l in labels]  # run in the order given
    n_for = {label: (args.n_calls if args.n_calls is not None else DEFAULT_N[label]) for label, _, _ in conditions}
    if not (sm.MIN_LENGTH <= args.genome_len <= sm.MAX_LENGTH):
        raise SystemExit(f"--genome-len {args.genome_len} outside [{sm.MIN_LENGTH}, {sm.MAX_LENGTH}]")

    if args.segment_example is not None:
        if "crossover/segment" not in n_for:
            raise SystemExit("--segment-example only affects crossover/segment; include it in --conditions")
        args.out_suffix += f"_segex_{args.segment_example}"

    print(f"model={ops.LLM_MODEL} host={ops.LLM_HOST} genome_len={args.genome_len} seed={args.seed} "
          f"length_bounds=[{sm.MIN_LENGTH},{sm.MAX_LENGTH}]")
    print(f"plan: {n_for}  (total {sum(n_for.values())} calls, before retries)")
    if args.segment_example is not None:
        print(f"segment example: {args.segment_example}")
    if args.dry_run:
        return

    genome_model.set_active(genome_model.build_genome_model(HPGAConfig(genome_model="sequence")))

    if args.segment_example is not None:
        apply_segment_example(args.segment_example)
    rng = random.Random(args.seed)
    gpu_before = gpu_snapshot()
    print(f"gpu before: {gpu_before['usage']}  compute_apps: {gpu_before['compute_apps']}\n")

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent.parent / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = ops.LLM_MODEL.replace(":", "_")
    out_path = out_dir / f"sequence_operator_compliance_{tag}_len{args.genome_len}{args.out_suffix}.json"

    results = []
    header = {
        "differences_from_lattice_probe": DIFFERENCES_FROM_LATTICE_PROBE, "seed": args.seed,
        "n_calls": n_for, "genome_len": args.genome_len, "length_bounds": [sm.MIN_LENGTH, sm.MAX_LENGTH],
        "model": ops.LLM_MODEL, "temperature": ops.LLM_TEMPERATURE, "gpu_before": gpu_before,
    }
    if args.segment_example is not None:
        header.update({"segment_example": args.segment_example, "unvaried_anchors": UNVARIED_ANCHORS})
    for label, fn, style in conditions:
        print(f"=== {label} (n={n_for[label]}) ===")
        ablating = args.segment_example is not None and label == "crossover/segment"
        start_line = _count_lines(ops._log_path()) if ablating else 0
        r = fn(style, n_for[label], rng, args.genome_len)
        if ablating:
            r["segment_example"] = args.segment_example
            r["segment_example_check"] = segment_example_check(ops._log_path(), start_line)
            r["boundary_summary"] = boundary_summary(r["segment_structure"])
        results.append(r)
        _print_result(label, r)
        print()
        # Checkpoint after each condition: a shared-GPU timeout in a later one
        # (see probe_operator_compliance.py) shouldn't cost the finished ones.
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({**header, "conditions": results, "complete": False}, f, indent=2, default=str)

    gpu_after = gpu_snapshot()
    print(f"gpu after: {gpu_after['usage']}  compute_apps: {gpu_after['compute_apps']}\n")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({**header, "conditions": results, "gpu_after": gpu_after, "complete": True}, f, indent=2, default=str)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
