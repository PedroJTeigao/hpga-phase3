"""Quick sanity check: single island, N in {1,2,4}, few generations.

Run before the full sweep to confirm the harness produces sane numbers
(T_calc/T_interval/T_turnaround all positive and in a plausible range,
speedup increases with N, DIS/GA ratio computed, worker utilisation in [0,1]).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpga.config import HPGAConfig, make_timing_sequence
from hpga.island import Island


def run_one(n_workers: int, sequence: str) -> dict:
    cfg = HPGAConfig(
        sequence=sequence,
        pop_size=256,
        n_generations=25,
        n_workers=n_workers,
        seed=0,
    )
    island = Island(cfg)
    recorder = island.run()
    return recorder.summary()


def main() -> None:
    sequence = make_timing_sequence(length=300, seed=0)
    print(f"timing sequence length={len(sequence)}\n")

    results = {}
    for n in (1, 2, 4):
        s = run_one(n, sequence)
        results[n] = s
        t_calc = s["t_calc"]["mean"]
        t_int = s["t_interval"]["mean"]
        t_turn = s["t_turnaround"]["mean"]
        print(f"N={n:>2}  wall={s['wall_time_s']:.4f}s  "
              f"T_calc(mean)={t_calc*1e3:.4f}ms  "
              f"T_interval(mean)={t_int*1e3 if t_int else float('nan'):.4f}ms  "
              f"T_turnaround(mean)={t_turn*1e3 if t_turn else float('nan'):.4f}ms  "
              f"DIS/GA={s['dis_ga_ratio']:.2f}  "
              f"mean_util={s['mean_worker_utilisation']:.3f}  "
              f"best_fitness={s['best_fitness_final']}")

    base_wall = results[1]["wall_time_s"]
    print("\nspeedup relative to N=1:")
    for n in (1, 2, 4):
        speedup = base_wall / results[n]["wall_time_s"]
        print(f"  N={n:>2}  speedup={speedup:.3f}")


if __name__ == "__main__":
    main()
