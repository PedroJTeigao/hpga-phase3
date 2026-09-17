"""Deterministic (no LLM, free) sweep to find a (genome_length, pop_size,
n_generations) configuration where the diversity-vs-fitness comparison in
PHASE3_RESULTS.md sec 5/6's first "next step" is actually testable.

The anchor circles run (genome_length=18, pop_size=8, n_generations=10)
plateaus by generation 3-4 in both arms -- "diversity didn't help" there is
indistinguishable from "neither arm needed it," and at that (length,
pop_size) the fold space is small enough to cover without any strategy, so
there are no real local optima to get trapped in for diversity to rescue.

This script looks for a configuration where a MODEST population (pop_size
close to the anchor's 8, to keep the eventual LLM comparison's call budget
comparable) genuinely stalls below what a much larger reference population
reaches over a long, free, deterministic horizon -- proof the landscape has
exploitable structure -- while also checking where in a SHORT generation
budget (the one the LLM run will actually afford) the modest population is
still climbing rather than already flat. Both checks use only
HPGA_OPERATOR_MODE=deterministic (unset, the default) -- zero LLM calls,
zero cost, same as revision 1 of run_circles_smoke.py confirming pop_size/
n_generations would show movement before spending any LLM budget on it.

Output: results/raw/diversity_sweep_<ts>.json (full per-generation fitness
and diversity traces for every (length, pop_size, seed) tried) plus a
stdout summary table.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HPGA_OPERATOR_MODE", "deterministic")
os.environ["HPGA_LOG_DIVERSITY"] = "1"

from hpga.config import HPGAConfig, make_timing_sequence
from hpga.island import Island

RESULTS_RAW = Path(__file__).resolve().parent.parent / "results" / "raw"

CANDIDATE_LENGTHS = [18, 24, 30, 36, 42]  # genome_length; sequence length = this + 2
SMALL_POP = 8       # matches the anchor run -- what we'd actually afford under LLM operators
BIG_POP = 64        # cheap reference ceiling; deterministic, so size is free
LONG_GENERATIONS = 150   # long horizon to find where SMALL_POP actually plateaus
SEEDS = [0, 1, 2]
SHORT_BUDGET_CHECKPOINTS = [10, 15, 20, 25]  # candidate LLM-run generation counts to evaluate


def run_one(*, seq_length: int, pop_size: int, n_generations: int, seed: int, run_id: str) -> dict:
    os.environ["HPGA_RUN_ID"] = run_id
    sequence = make_timing_sequence(length=seq_length, seed=1)  # fixed target, matches anchor's convention
    cfg = HPGAConfig(sequence=sequence, pop_size=pop_size, n_generations=n_generations,
                      n_workers=2, elitism=max(1, pop_size // 8), seed=seed)
    island = Island(cfg)
    t0 = time.perf_counter()
    recorder = island.run()
    wall_s = time.perf_counter() - t0

    div_path = RESULTS_RAW / f"diversity_{run_id}.jsonl"
    diversity_by_gen = []
    if div_path.exists():
        diversity_by_gen = [json.loads(l)["mean_pairwise_hamming"] for l in div_path.read_text().splitlines() if l.strip()]
        div_path.unlink()  # sweep-internal scratch; final config's real run will regenerate cleanly

    return {
        "genome_length": seq_length - 2,
        "pop_size": pop_size,
        "n_generations": n_generations,
        "seed": seed,
        "wall_s": wall_s,
        "best_fitness_by_gen": recorder.best_fitness_by_gen,
        "diversity_by_gen": diversity_by_gen,
    }


def plateau_gen(fits: list[float], window: int = 20) -> int:
    """First generation g such that fits[g:] never improves on fits[g] --
    i.e. the run's value has already reached its own final plateau by g."""
    final = fits[-1]
    for g in range(len(fits)):
        if fits[g] >= final:
            return g
    return len(fits) - 1


def main() -> None:
    ts = int(time.time())
    results = []
    t_start = time.perf_counter()

    for L in CANDIDATE_LENGTHS:
        seq_len = L + 2
        print(f"=== genome_length={L} ===", flush=True)

        for seed in SEEDS:
            r = run_one(seq_length=seq_len, pop_size=SMALL_POP, n_generations=LONG_GENERATIONS,
                        seed=seed, run_id=f"sweep_small_L{L}_s{seed}_{ts}")
            results.append(r)
            pg = plateau_gen(r["best_fitness_by_gen"])
            print(f"  small pop={SMALL_POP} seed={seed}: final={r['best_fitness_by_gen'][-1]:.1f} "
                  f"plateau_gen={pg}  div[0]={r['diversity_by_gen'][0]:.2f} "
                  f"div[-1]={r['diversity_by_gen'][-1]:.2f}  wall={r['wall_s']:.2f}s", flush=True)

        for seed in SEEDS[:2]:
            r = run_one(seq_length=seq_len, pop_size=BIG_POP, n_generations=LONG_GENERATIONS,
                        seed=seed, run_id=f"sweep_big_L{L}_s{seed}_{ts}")
            results.append(r)
            print(f"  BIG  pop={BIG_POP} seed={seed}: final={r['best_fitness_by_gen'][-1]:.1f}  "
                  f"wall={r['wall_s']:.2f}s", flush=True)

    print(f"\ntotal sweep wall time: {time.perf_counter() - t_start:.1f}s\n")

    out_path = RESULTS_RAW / f"diversity_sweep_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")

    # --- summary table -----------------------------------------------
    print("\n=== SUMMARY ===")
    print(f"{'L':>4} {'small_final(mean)':>18} {'small_plateau_gen(mean)':>24} "
          f"{'big_final(mean)':>16} {'gap':>6}")
    for L in CANDIDATE_LENGTHS:
        small = [r for r in results if r["genome_length"] == L and r["pop_size"] == SMALL_POP]
        big = [r for r in results if r["genome_length"] == L and r["pop_size"] == BIG_POP]
        small_final = sum(r["best_fitness_by_gen"][-1] for r in small) / len(small)
        small_plateau = sum(plateau_gen(r["best_fitness_by_gen"]) for r in small) / len(small)
        big_final = sum(r["best_fitness_by_gen"][-1] for r in big) / len(big)
        gap = big_final - small_final
        print(f"{L:>4} {small_final:>18.2f} {small_plateau:>24.1f} {big_final:>16.2f} {gap:>6.2f}")

        for budget in SHORT_BUDGET_CHECKPOINTS:
            if budget <= LONG_GENERATIONS:
                at_budget = sum(r["best_fitness_by_gen"][budget - 1] for r in small) / len(small)
                still_climbing = small_plateau > budget
                print(f"       at n_generations={budget}: mean_fitness={at_budget:.2f}  "
                      f"(own eventual plateau at gen {small_plateau:.1f}) "
                      f"-> {'STILL CLIMBING' if still_climbing else 'ALREADY FLAT'}")


if __name__ == "__main__":
    main()
