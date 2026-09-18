"""Estimates the cache hit rate a hash-cache in front of the ESMFold oracle
would see in a real search, before building one.

ESMFold is deterministic -- identical sequences fold to identical structures.
An evolutionary search revisits candidates: elitism carries the same best
genome forward unevaluated-but-re-evaluated generation after generation (this
harness's Island._dispatch_and_collect re-evaluates the FULL population every
generation, including unchanged elites -- confirmed by reading island.py, not
assumed), and as a population converges (diversity collapsing is already
well-documented for this harness, e.g. PHASE3_RESULTS.md sec 2/6/7),
crossover and mutation increasingly regenerate genomes already seen.

Genome content (HP-lattice move symbols vs. amino acids) doesn't matter for
this measurement -- a revisit is a property of the GA's population dynamics
(selection, elitism, convergence), not of what the symbols mean. Reuses this
project's existing deterministic GA harness at the same pop_size/generations/
genome_length as the circles diversity-fitness experiments
(PHASE3_RESULTS.md sec 6-7) as the direct proxy for what a real ESMFold-backed
search would revisit under identical population mechanics.

Method: monkeypatch Island._dispatch_and_collect (the one place every
generation's full population passes through before fitness evaluation) to
log every genome evaluated, in call order. Process that log as a real
hash-cache would: maintain a running "already seen" set, count a hit when a
genome was seen before, a miss when it wasn't. 3 seeds, matching this
project's now-standard minimum.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hpga.island as island_mod
from hpga.config import HPGAConfig, make_timing_sequence
from hpga.island import Island

POP_SIZE = 8
N_GENERATIONS = 15
GENOME_LENGTH = 24  # sequence length 26, matches the circles divfit config
ELITISM = 1
SEEDS = [0, 1, 2]

_eval_log: list[tuple[int, tuple]] = []
_orig_dispatch = island_mod.Island._dispatch_and_collect


def _patched_dispatch(self, population, gen_id, gen_record):
    for genome in population:
        _eval_log.append((gen_id, tuple(genome)))
    return _orig_dispatch(self, population, gen_id, gen_record)


island_mod.Island._dispatch_and_collect = _patched_dispatch


def run_one(seed: int) -> dict:
    global _eval_log
    _eval_log = []

    sequence = make_timing_sequence(length=GENOME_LENGTH + 2, seed=1)
    cfg = HPGAConfig(sequence=sequence, pop_size=POP_SIZE, n_generations=N_GENERATIONS,
                      n_workers=2, elitism=ELITISM, seed=seed)
    island = Island(cfg)
    island.run()

    seen: set[tuple] = set()
    hits = 0
    misses = 0
    hits_by_gen: dict[int, int] = {}
    calls_by_gen: dict[int, int] = {}
    for gen_id, genome in _eval_log:
        calls_by_gen[gen_id] = calls_by_gen.get(gen_id, 0) + 1
        if genome in seen:
            hits += 1
            hits_by_gen[gen_id] = hits_by_gen.get(gen_id, 0) + 1
        else:
            misses += 1
            seen.add(genome)

    total = hits + misses
    return {
        "seed": seed,
        "total_evaluations": total,
        "unique_genomes": len(seen),
        "cache_hits": hits,
        "cache_misses": misses,
        "hit_rate": hits / total if total else 0.0,
        "hits_by_gen": hits_by_gen,
        "calls_by_gen": calls_by_gen,
    }


def main():
    print(f"pop_size={POP_SIZE}  n_generations={N_GENERATIONS}  genome_length={GENOME_LENGTH}  "
          f"elitism={ELITISM}  seeds={SEEDS}\n", flush=True)

    results = []
    for seed in SEEDS:
        r = run_one(seed)
        results.append(r)
        print(f"seed={seed}: total={r['total_evaluations']}  unique={r['unique_genomes']}  "
              f"hits={r['cache_hits']}  misses={r['cache_misses']}  hit_rate={r['hit_rate']:.1%}",
              flush=True)
        by_gen_str = ", ".join(f"g{g}={r['hits_by_gen'].get(g, 0)}/{r['calls_by_gen'][g]}"
                                for g in sorted(r["calls_by_gen"]))
        print(f"  hits/calls by generation: {by_gen_str}\n", flush=True)

    mean_total = sum(r["total_evaluations"] for r in results) / len(results)
    mean_hit_rate = sum(r["hit_rate"] for r in results) / len(results)
    print(f"=== across {len(SEEDS)} seeds ===")
    print(f"mean total evaluations/run: {mean_total:.1f}")
    print(f"mean cache hit rate: {mean_hit_rate:.1%}")

    import json
    out = {"config": {"pop_size": POP_SIZE, "n_generations": N_GENERATIONS,
                       "genome_length": GENOME_LENGTH, "elitism": ELITISM, "seeds": SEEDS},
           "results": results, "mean_total_evaluations": mean_total, "mean_hit_rate": mean_hit_rate}
    out_path = Path(__file__).resolve().parent.parent / "results" / "raw" / "genome_revisit_rate.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
