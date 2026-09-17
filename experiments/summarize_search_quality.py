"""Read results/raw/search_quality_comparison.jsonl (written incrementally,
one line per finalized (seed, mode) run by run_search_quality_comparison.py)
and print the per-seed and aggregate comparison. Safe to run before all
(seed, mode) combinations have finished -- reports whatever has landed so far.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpga.config import BENCHMARK_OPTIMAL_ENERGY, BENCHMARK_SEQUENCE

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "raw"
IN_JSONL = RESULTS_DIR / "search_quality_comparison.jsonl"
SEEDS = [0, 1, 2]


def gen_reached(best_by_gen: list[float], target: float) -> int | None:
    for i, v in enumerate(best_by_gen):
        if v >= target:
            return i
    return None


def main() -> None:
    target_contacts = -BENCHMARK_OPTIMAL_ENERGY
    print(f"sequence={BENCHMARK_SEQUENCE}  target={target_contacts} H-H contacts (E*={BENCHMARK_OPTIMAL_ENERGY})\n")

    if not IN_JSONL.exists():
        print(f"no results yet at {IN_JSONL}")
        return

    by_key: dict[tuple[int, str], dict] = {}
    for line in open(IN_JSONL, encoding="utf-8"):
        r = json.loads(line)
        by_key[(r["seed"], r["mode"])] = r

    print(f"{len(by_key)}/{len(SEEDS) * 2} (seed, mode) combinations finalized\n")

    for seed in SEEDS:
        det = by_key.get((seed, "deterministic"))
        llm = by_key.get((seed, "llm"))
        if det is None and llm is None:
            print(f"seed={seed}: no data yet")
            continue
        print(f"seed={seed}")
        for mode, r in (("deterministic", det), ("llm", llm)):
            if r is None:
                print(f"  {mode:<13}: not finalized yet")
                continue
            reached = gen_reached(r["best_fitness_by_gen"], target_contacts)
            reached_str = f"gen {reached}" if reached is not None else "not reached"
            extra = ""
            if mode == "llm" and r.get("operator_stats"):
                st = r["operator_stats"]
                extra = (f"  (n_calls={st['n_llm_calls']} n_retries={st['n_retries']} "
                         f"n_failures={st['n_failures']} total_tokens_out={st['total_tokens_out']})")
            print(f"  {mode:<13}: best_final={r['best_fitness_final']:.0f}  "
                  f"reached_optimum={reached_str}  gap={target_contacts - r['best_fitness_final']:.0f}"
                  f"  wall_s={r['wall_elapsed_s']:.1f}{extra}")
        if det is not None and llm is not None:
            diff = llm["best_fitness_final"] - det["best_fitness_final"]
            verdict = "LLM better" if diff > 0 else ("deterministic better" if diff < 0 else "tie")
            print(f"  --> diff={diff:+.0f}  ({verdict})")
        print()

    print("=== aggregate ===")
    for mode in ("deterministic", "llm"):
        rows = [r for (s, m), r in by_key.items() if m == mode]
        if not rows:
            continue
        finals = [r["best_fitness_final"] for r in rows]
        n_reached = sum(1 for r in rows if gen_reached(r["best_fitness_by_gen"], target_contacts) is not None)
        print(f"  {mode:<13}: n={len(rows)}  finals={finals}  reached_optimum={n_reached}/{len(rows)}")


if __name__ == "__main__":
    main()
