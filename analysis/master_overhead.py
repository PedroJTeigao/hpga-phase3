"""Effective per-individual master serial cost (review finding #1).

The original T_interval (island.py: gaps between consecutive dispatch_timestamps,
i.e. the queue.put() call itself) only captures ~1us of enqueue cost. It misses
the rest of the master's serial per-individual work: pickling the genome,
unpickling the result dict, appending to bookkeeping structures, etc. At N=1
there is no worker/master overlap, so the DIS-phase wall time for a generation
is exactly (sum of per-individual master serial overhead) + (sum of T_calc)
for that generation -- subtracting the measured T_calc sum out of the measured
DIS-phase time leaves the master's true serial overhead per individual
directly, computed from data already recorded by run_sweep.py (dis_time_s and
the t_calc distribution), no rerun needed.

This is done per n_workers (not just N=1) so the effective master overhead can
also be checked for its own N-dependence (does master-side serialisation cost
degrade under core contention the way T_calc does?).
"""

import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.model import network_capacity

ROOT = Path(__file__).resolve().parent.parent
GEN_CSV = ROOT / "results" / "raw" / "sweep_generations.csv"
RUNS_JSON = ROOT / "results" / "raw" / "sweep_runs.json"
OUT_CSV = ROOT / "results" / "raw" / "master_overhead.csv"


def load_n1_original_stats() -> tuple[float, float]:
    """Returns (t_calc_mean_s, t_interval_original_mean_s) for N=1, read from
    sweep_runs.json rather than hardcoded, so this stays correct if the sweep
    is ever rerun."""
    with open(RUNS_JSON) as f:
        runs = json.load(f)
    run1 = next(r for r in runs if r["n_workers"] == 1)
    return run1["t_calc"]["mean"], run1["t_interval"]["mean"]


def load_per_generation_overhead() -> dict[int, list[float]]:
    by_n: dict[int, list[float]] = {}
    with open(GEN_CSV, newline="") as f:
        for row in csv.DictReader(f):
            n_workers = int(row["n_workers"])
            dis_time = float(row["dis_time_s"])
            t_calc_mean = float(row["t_calc_mean_s"])
            t_calc_n = int(row["t_calc_n"])
            t_calc_sum = t_calc_mean * t_calc_n
            master_overhead_total = dis_time - t_calc_sum
            per_individual = master_overhead_total / t_calc_n
            by_n.setdefault(n_workers, []).append(per_individual)
    return by_n


def effective_t_interval_n1() -> float:
    """T_interval effective (N=1), for reuse by plots.py without recomputing
    the subtraction inline."""
    by_n = load_per_generation_overhead()
    return statistics.mean(by_n[1])


def main() -> None:
    by_n = load_per_generation_overhead()

    rows = []
    for n in sorted(by_n):
        vals = by_n[n]
        mean = statistics.mean(vals)
        stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
        rows.append({
            "n_workers": n,
            "master_overhead_per_individual_mean_s": mean,
            "master_overhead_per_individual_stdev_s": stdev,
            "n_generations": len(vals),
        })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # C recomputed at N=1: T_calc / effective T_interval (master serial cost).
    n1 = by_n[1]
    t_interval_effective_mean = statistics.mean(n1)

    t_calc_mean_n1, t_interval_original_mean_n1 = load_n1_original_stats()

    c_naive = network_capacity(t_calc_mean_n1, t_interval_original_mean_n1)
    c_effective = network_capacity(t_calc_mean_n1, t_interval_effective_mean)

    print("Effective per-individual master serial cost (T_interval, redefined), by N:")
    print(f"{'N':>4} {'mean (ms)':>12} {'stdev (ms)':>12} {'n_gens':>8}")
    for r in rows:
        print(f"{r['n_workers']:>4} {r['master_overhead_per_individual_mean_s']*1e3:>12.4f} "
              f"{r['master_overhead_per_individual_stdev_s']*1e3:>12.4f} {r['n_generations']:>8}")

    print(f"\nT_calc(N=1) mean = {t_calc_mean_n1*1e3:.4f} ms")
    print(f"T_interval original (N=1, enqueue-call gap only) = {t_interval_original_mean_n1*1e3:.6f} ms")
    print(f"T_interval effective (N=1, total master serial cost/individual) = {t_interval_effective_mean*1e3:.4f} ms")
    print(f"\nC (Eq. 3), original T_interval  = {c_naive:.1f}")
    print(f"C (Eq. 3), effective T_interval = {c_effective:.3f}")
    print(
        "\nNote: only the N=1 row is a valid master-serial-overhead estimate. "
        "For N>1 workers overlap compute with dispatch/collect, so DIS-phase "
        "time is no longer (sum of per-individual overhead) + (sum of T_calc); "
        "the subtraction goes negative there because parallelism makes DIS "
        "faster than the serial sum of T_calc, not because master overhead "
        "shrinks. The N>1 rows are reported for transparency, not reused as "
        "an overhead measurement."
    )
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
