"""Full worker-count sweep for the Phase 1 HPGA timing harness.

Runs one Island per worker count in WORKER_COUNTS, capturing raw
per-generation timing records (T_calc, T_interval, T_turnaround, DIS/GA
phase time, per-worker busy time) plus a per-run summary. Writes only to
results/raw/ — analysis/model.py and analysis/plots.py read from there and
never re-run anything.

Worker counts above the machine's core count (reported below) are run
anyway but flagged oversubscribed=True in the output, per the instruction
not to silently oversubscribe.
"""

import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpga.config import HPGAConfig, make_timing_sequence
from hpga.island import Island

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "raw"
GENERATIONS_CSV = RESULTS_DIR / "sweep_generations.csv"
RUNS_JSON = RESULTS_DIR / "sweep_runs.json"

# 12 added explicitly as the machine's actual core count (a clean
# at-capacity data point); 16 and 32 kept as deliberate oversubscription.
WORKER_COUNTS = [1, 2, 4, 8, 12, 16, 32]
POP_SIZE = 256
N_GENERATIONS = 40
SEED = 0
SEQ_LENGTH = 300


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": None, "median": None, "stdev": None}
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
    }


def generation_records(n_workers: int, recorder) -> list[dict]:
    records = []
    for g in recorder.generations:
        ts = sorted(g.dispatch_timestamps)
        t_interval = [b - a for a, b in zip(ts, ts[1:])]
        ar = sorted(g.arrival_timestamps)
        t_turnaround = [b - a for a, b in zip(ar, ar[1:])]

        tc, ti, tt = _stats(g.t_calc_list), _stats(t_interval), _stats(t_turnaround)

        records.append({
            "n_workers": n_workers,
            "gen_id": g.gen_id,
            "dis_time_s": g.dis_time,
            "ga_time_s": g.ga_time,
            "t_calc_mean_s": tc["mean"], "t_calc_median_s": tc["median"], "t_calc_stdev_s": tc["stdev"], "t_calc_n": tc["n"],
            "t_interval_mean_s": ti["mean"], "t_interval_median_s": ti["median"], "t_interval_stdev_s": ti["stdev"], "t_interval_n": ti["n"],
            "t_turnaround_mean_s": tt["mean"], "t_turnaround_median_s": tt["median"], "t_turnaround_stdev_s": tt["stdev"], "t_turnaround_n": tt["n"],
            "per_worker_calc_time_json": json.dumps(g.per_worker_calc_time),
        })
    return records


def write_generations_csv(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(GENERATIONS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _physical_core_count() -> int | None:
    """os.cpu_count() returns logical processors, which silently conflates
    real cores with hyperthreading — review round 1 caught this reading a
    12-logical machine as "12 cores" when it's 6 physical. psutil is the only
    reliable cross-platform way to get the physical count; degrade to None
    (reported as unknown, not guessed) if it isn't installed."""
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except ImportError:
        return None


def main() -> None:
    cpu_count = os.cpu_count() or 1
    physical_cpu_count = _physical_core_count()
    physical_str = str(physical_cpu_count) if physical_cpu_count is not None else "unknown (pip install psutil)"
    print(f"machine core count: {cpu_count} logical, {physical_str} physical")

    sequence = make_timing_sequence(length=SEQ_LENGTH, seed=SEED)
    print(f"timing sequence length={len(sequence)}  genome_length={len(sequence) - 2}")
    print(f"pop_size={POP_SIZE}  n_generations={N_GENERATIONS}  seed={SEED}\n")

    all_gen_rows: list[dict] = []
    run_summaries: list[dict] = []

    for n in WORKER_COUNTS:
        oversubscribed = n > cpu_count
        oversubscribed_physical = (n > physical_cpu_count) if physical_cpu_count is not None else False
        tag = "  [OVERSUBSCRIBED logical]" if oversubscribed else (
            "  [>physical cores]" if oversubscribed_physical else "")
        print(f"running N={n:>2}{tag} ...")

        cfg = HPGAConfig(sequence=sequence, pop_size=POP_SIZE, n_generations=N_GENERATIONS, n_workers=n, seed=SEED)
        island = Island(cfg)
        recorder = island.run()

        all_gen_rows.extend(generation_records(n, recorder))

        summary = recorder.summary()
        summary["cpu_count"] = cpu_count
        summary["physical_cpu_count"] = physical_cpu_count
        summary["oversubscribed"] = oversubscribed
        summary["oversubscribed_physical"] = (n > physical_cpu_count) if physical_cpu_count is not None else None
        run_summaries.append(summary)

        print(f"  wall={summary['wall_time_s']:.4f}s  "
              f"T_calc(mean)={summary['t_calc']['mean'] * 1e3:.4f}ms  "
              f"T_interval(mean)={summary['t_interval']['mean'] * 1e3:.4f}ms  "
              f"T_turnaround(mean)={summary['t_turnaround']['mean'] * 1e3:.4f}ms  "
              f"DIS/GA={summary['dis_ga_ratio']:.2f}  "
              f"mean_util={summary['mean_worker_utilisation']:.3f}")

    write_generations_csv(all_gen_rows)
    with open(RUNS_JSON, "w") as f:
        json.dump(run_summaries, f, indent=2, default=str)

    base_wall = run_summaries[0]["wall_time_s"]
    print("\nspeedup / efficiency relative to N=1:")
    for s in run_summaries:
        speedup = base_wall / s["wall_time_s"]
        print(f"  N={s['n_workers']:>2}  speedup={speedup:.3f}  efficiency={speedup / s['n_workers']:.3f}")

    print(f"\nwrote {len(all_gen_rows)} per-generation records to {GENERATIONS_CSV}")
    print(f"wrote {len(run_summaries)} run summaries to {RUNS_JSON}")


if __name__ == "__main__":
    main()
