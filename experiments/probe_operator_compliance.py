"""Live compliance probe for the LLM genetic operators, generalized across
models. Fires real Ollama calls through the actual `hpga.operators._llm_mutate`
/ `_llm_crossover` retry/fallback machinery (no monkeypatching, unlike
`test_operator_failure_path.py`) and reports, per (operator, prompt style)
condition: how often the LLM path succeeds vs. falls back to the
deterministic operator, retries/call, mean latency, and -- for the
affordance-fixed styles ('position' for mutate, 'segment' for crossover) --
the distribution of which positions/segment-counts the model actually chose,
to check for the position/midpoint collapse documented in
PHASE2_RESULTS.md Sec. 4.4.

Four conditions, run in this order: mutate/full, mutate/position,
crossover/full, crossover/segment. Model is whatever HPGA_LLM_MODEL is set to
in the environment (read by hpga/operators.py at import time) -- this script
does not set it itself, so results are clearly attributable to whichever
model the caller configured.

Fresh random genome/parents per call (not fixed), matching the live-sample
methodology in PHASE2_RESULTS.md Sec. 4.2-4.4. Every call is logged to the
usual llm_operator_calls_<run_id>.jsonl via the existing _log_call path.

GPU snapshot (nvidia-smi memory/util + compute-apps), taken immediately
before the first condition and immediately after the last, saved into the
output JSON alongside the actual measurements -- added after a run on this
shared, unscheduled node timed out (httpx.ReadTimeout) with other users'
sessions active. This doesn't prevent contention; it makes the run's own
result self-report whether the GPU was actually free at both ends, so a
timing anomaly doesn't have to be diagnosed after the fact from memory.
"""

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpga import operators as ops
from hpga.hp_model import MOVES


def gpu_snapshot() -> dict:
    def _run(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as exc:  # nvidia-smi missing/erroring shouldn't crash the probe itself
            return f"<nvidia-smi call failed: {exc!r}>"

    return {
        "usage": _run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader"]),
        "compute_apps": _run(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
                               "--format=csv,noheader"]) or "(none)",
    }

GENOME_LEN = 18  # default; overridden by --genome-len, kept as the module default for direct-import callers
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.9


def random_genome(length: int, rng: random.Random) -> list[int]:
    return [rng.choice(MOVES) for _ in range(length)]


def run_mutate(style: str, n_calls: int, rng: random.Random, genome_len: int = GENOME_LEN) -> dict:
    import os
    os.environ["HPGA_LLM_PROMPT_STYLE"] = style
    ops.reset_operator_stats()
    changed_positions = Counter()
    n_diffs = []
    for _ in range(n_calls):
        genome = random_genome(genome_len, rng)
        out = ops._llm_mutate(genome, MUTATION_RATE, rng)
        diffs = [i for i, (a, b) in enumerate(zip(genome, out)) if a != b]
        n_diffs.append(len(diffs))
        changed_positions.update(diffs)
    stats = ops.get_operator_stats()
    return {
        "op": "mutate", "style": style, "model": ops.LLM_MODEL, "n_calls": n_calls,
        "genome_len": genome_len, "stats": stats,
        "fallback_rate": stats["n_failures"] / n_calls,
        "requests_per_call": stats["n_llm_requests"] / n_calls,
        "zero_diff_outputs": sum(1 for d in n_diffs if d == 0),
        "position_counts": dict(sorted(changed_positions.items())),
        "n_positions_touched": len(changed_positions),
    }


def run_crossover(style: str, n_calls: int, rng: random.Random, genome_len: int = GENOME_LEN) -> dict:
    import os
    os.environ["HPGA_LLM_PROMPT_STYLE"] = style
    ops.reset_operator_stats()
    echo_count = 0
    for _ in range(n_calls):
        p1 = random_genome(genome_len, rng)
        p2 = random_genome(genome_len, rng)
        c1, c2 = ops._llm_crossover(p1, p2, CROSSOVER_RATE, rng)
        if c1 in (p1, p2) or c2 in (p1, p2):
            echo_count += 1
    stats = ops.get_operator_stats()
    return {
        "op": "crossover", "style": style, "model": ops.LLM_MODEL, "n_calls": n_calls,
        "genome_len": genome_len, "stats": stats,
        "fallback_rate": stats["n_failures"] / n_calls,
        "requests_per_call": stats["n_llm_requests"] / n_calls,
        "echo_outputs": echo_count,
    }


ALL_CONDITIONS = [
    ("mutate/full", run_mutate, "full"),
    ("mutate/position", run_mutate, "position"),
    ("crossover/full", run_crossover, "full"),
    ("crossover/segment", run_crossover, "segment"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-calls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--conditions", type=str, default=None,
        help="Comma-separated subset of condition labels (e.g. "
             "'mutate/position,crossover/segment') to run instead of all four. "
             "Labels must match exactly: mutate/full, mutate/position, "
             "crossover/full, crossover/segment.",
    )
    parser.add_argument(
        "--out-suffix", type=str, default="",
        help="Appended to the output filename before .json (e.g. '_n20'), so "
             "a run at a different --n-calls doesn't silently overwrite a "
             "previous run's result file for the same model.",
    )
    parser.add_argument(
        "--genome-len", type=int, default=GENOME_LEN,
        help="Genome length to probe at (default 18). Added to build a "
             "fallback-rate-vs-length curve per style, after 'full' was "
             "found to degrade sharply between length 8 and 18 while "
             "'position'/'segment' were only ever validated at 18.",
    )
    args = parser.parse_args()

    conditions = ALL_CONDITIONS
    if args.conditions:
        wanted = set(args.conditions.split(","))
        conditions = [c for c in ALL_CONDITIONS if c[0] in wanted]
        missing = wanted - {c[0] for c in conditions}
        if missing:
            raise SystemExit(f"Unknown condition label(s): {sorted(missing)}")

    rng = random.Random(args.seed)
    print(f"model={ops.LLM_MODEL} host={ops.LLM_HOST} n_calls_per_condition={args.n_calls} "
          f"genome_len={args.genome_len} conditions={[c[0] for c in conditions]}\n")

    gpu_before = gpu_snapshot()
    print(f"gpu before: {gpu_before['usage']}  compute_apps: {gpu_before['compute_apps']}\n")

    results = []
    for label, fn, style in conditions:
        print(f"=== {label} (genome_len={args.genome_len}) ===")
        r = fn(style, args.n_calls, rng, genome_len=args.genome_len)
        results.append(r)
        s = r["stats"]
        print(f"  fallback_rate={r['fallback_rate']:.2f}  requests_per_call={r['requests_per_call']:.2f}  "
              f"mean_latency_s={s['mean_latency_s']:.3f}  n_retries={s['n_retries']}")
        if r["op"] == "mutate":
            print(f"  zero_diff_outputs={r['zero_diff_outputs']}/{r['n_calls']}  "
                  f"positions_touched={r['n_positions_touched']}/{args.genome_len}  "
                  f"position_counts={r['position_counts']}")
        else:
            print(f"  echo_outputs={r['echo_outputs']}/{r['n_calls']}")
        print()

    gpu_after = gpu_snapshot()
    print(f"gpu after: {gpu_after['usage']}  compute_apps: {gpu_after['compute_apps']}\n")

    tag = ops.LLM_MODEL.replace(":", "_")
    out_path = (Path(__file__).resolve().parent.parent / "results" / "raw"
                / f"operator_compliance_{tag}_len{args.genome_len}{args.out_suffix}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {"gpu_before": gpu_before, "gpu_after": gpu_after, "conditions": results}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
