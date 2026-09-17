"""Review round 1, lower-priority item: the original sweep was one run per N,
so "saturates at N=8" had no error bars. This repeats the *identical*
configuration (same sequence, same GA seed=0) 3 times per N -- deliberately
not varying the GA seed, since the question here is pure execution-timing
noise (OS scheduling, core contention) at fixed workload, not GA-outcome
variance (already covered by run_benchmark_validation.py). Each repeat's
speedup is computed against that repeat's *own* N=1 run (paired), not a
shared baseline, so cross-run baseline drift doesn't leak into the speedup
numbers.

Writes to results/raw/sweep_repeats.json / sweep_repeats_generations.csv,
separate from the original single-run sweep files so nothing already
reported gets overwritten or silently changed by rerun noise.
"""

import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpga.config import HPGAConfig, make_timing_sequence
from hpga.island import Island

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "raw"
RUNS_JSON = RESULTS_DIR / "sweep_repeats.json"

WORKER_COUNTS = [1, 2, 4, 8, 12, 16, 32]
N_REPEATS = 3
POP_SIZE = 256
N_GENERATIONS = 40
SEED = 0
SEQ_LENGTH = 300


def main() -> None:
    sequence = make_timing_sequence(length=SEQ_LENGTH, seed=SEED)
    print(f"pop_size={POP_SIZE}  n_generations={N_GENERATIONS}  seed={SEED}  repeats={N_REPEATS}\n")

    all_runs = []
    for repeat in range(N_REPEATS):
        for n in WORKER_COUNTS:
            print(f"repeat={repeat} N={n:>2} ...")
            cfg = HPGAConfig(sequence=sequence, pop_size=POP_SIZE, n_generations=N_GENERATIONS, n_workers=n, seed=SEED)
            island = Island(cfg)
            recorder = island.run()
            summary = recorder.summary()
            summary["repeat"] = repeat
            all_runs.append(summary)
            print(f"  wall={summary['wall_time_s']:.4f}s  dis={summary['dis_total_s']:.4f}s  "
                  f"t_calc={summary['t_calc']['mean']*1e3:.4f}ms  dis_ga={summary['dis_ga_ratio']:.3f}")

    with open(RUNS_JSON, "w") as f:
        json.dump(all_runs, f, indent=2, default=str)
    print(f"\nwrote {len(all_runs)} runs to {RUNS_JSON}")

    # Paired-by-repeat aggregation: each repeat's speedup uses that repeat's own N=1.
    by_repeat = {}
    for r in all_runs:
        by_repeat.setdefault(r["repeat"], {})[r["n_workers"]] = r

    per_n = {n: {"dis_speedup": [], "overall_speedup": [], "t_calc_mean": [], "dis_ga_ratio": []} for n in WORKER_COUNTS}
    for repeat, runs_by_n in by_repeat.items():
        base = runs_by_n[1]
        for n in WORKER_COUNTS:
            r = runs_by_n[n]
            per_n[n]["dis_speedup"].append(base["dis_total_s"] / r["dis_total_s"])
            per_n[n]["overall_speedup"].append(base["wall_time_s"] / r["wall_time_s"])
            per_n[n]["t_calc_mean"].append(r["t_calc"]["mean"])
            per_n[n]["dis_ga_ratio"].append(r["dis_ga_ratio"])

    print(f"\n{'N':>4} {'DIS speedup':>18} {'overall speedup':>18} {'T_calc (ms)':>16} {'DIS/GA ratio':>16}")
    for n in WORKER_COUNTS:
        d = per_n[n]
        def fmt(key, scale=1.0, prec=3):
            vals = [v * scale for v in d[key]]
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return f"{m:.{prec}f}±{s:.{prec}f}"
        print(f"{n:>4} {fmt('dis_speedup'):>18} {fmt('overall_speedup'):>18} "
              f"{fmt('t_calc_mean', 1e3):>16} {fmt('dis_ga_ratio'):>16}")


if __name__ == "__main__":
    main()
