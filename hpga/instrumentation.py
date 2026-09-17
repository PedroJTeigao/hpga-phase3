"""Timing/metrics collection for one island run.

Vocabulary matches the plan's §6 table:
  T_calc        wall-clock time of one fitness evaluation (measured on the worker)
  T_interval    time for the master to dispatch one individual (gap between
                consecutive master-side dispatch calls, i.e. injection-channel cost)
  T_turnaround  master's inter-arrival time between ready individuals (gap
                between consecutive results arriving back at the master)
  DIS/GA ratio  per-generation: time in dispatch+evaluate phase vs time in
                selection/crossover/mutation phase
  utilisation   fraction of total run wall-clock time each worker spent
                actually computing (sum of its T_calc / run wall-clock)
"""

import csv
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GenerationRecord:
    gen_id: int
    dis_time: float
    ga_time: float
    dispatch_timestamps: list[float] = field(default_factory=list)
    arrival_timestamps: list[float] = field(default_factory=list)
    per_worker_calc_time: dict[int, float] = field(default_factory=dict)
    t_calc_list: list[float] = field(default_factory=list)


@dataclass
class RunRecorder:
    n_workers: int
    config_summary: dict
    run_start: float = field(default_factory=time.perf_counter)
    run_end: float | None = None
    startup_time: float = 0.0
    shutdown_time: float = 0.0
    generations: list[GenerationRecord] = field(default_factory=list)
    best_fitness_by_gen: list[float] = field(default_factory=list)

    def finish(self) -> None:
        self.run_end = time.perf_counter()

    @property
    def wall_time(self) -> float:
        end = self.run_end if self.run_end is not None else time.perf_counter()
        return end - self.run_start

    def summary(self) -> dict:
        t_calc_all: list[float] = []
        t_interval_all: list[float] = []
        t_turnaround_all: list[float] = []
        dis_total = 0.0
        ga_total = 0.0
        worker_busy = {w: 0.0 for w in range(self.n_workers)}

        for g in self.generations:
            dis_total += g.dis_time
            ga_total += g.ga_time
            ts = sorted(g.dispatch_timestamps)
            t_interval_all.extend(b - a for a, b in zip(ts, ts[1:]))
            ar = sorted(g.arrival_timestamps)
            t_turnaround_all.extend(b - a for a, b in zip(ar, ar[1:]))
            for w, busy in g.per_worker_calc_time.items():
                worker_busy[w] = worker_busy.get(w, 0.0) + busy
            t_calc_all.extend(g.t_calc_list)

        wall = self.wall_time
        utilisation = {w: (busy / wall if wall > 0 else 0.0) for w, busy in worker_busy.items()}

        def stats(xs: list[float]) -> dict:
            if not xs:
                return {"n": 0, "mean": None, "median": None, "stdev": None, "min": None, "max": None}
            return {
                "n": len(xs),
                "mean": statistics.mean(xs),
                "median": statistics.median(xs),
                "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
                "min": min(xs),
                "max": max(xs),
            }

        return {
            "config": self.config_summary,
            "n_workers": self.n_workers,
            "wall_time_s": wall,
            "startup_time_s": self.startup_time,
            "shutdown_time_s": self.shutdown_time,
            "dis_total_s": dis_total,
            "ga_total_s": ga_total,
            "dis_ga_ratio": (dis_total / ga_total) if ga_total > 0 else None,
            "t_calc": stats(t_calc_all),
            "t_interval": stats(t_interval_all),
            "t_turnaround": stats(t_turnaround_all),
            "worker_busy_time_s": worker_busy,
            "worker_utilisation": utilisation,
            "mean_worker_utilisation": statistics.mean(utilisation.values()) if utilisation else None,
            "best_fitness_final": self.best_fitness_by_gen[-1] if self.best_fitness_by_gen else None,
            "best_fitness_by_gen": self.best_fitness_by_gen,
        }


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def append_sweep_row_csv(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
