"""Phase 2 item 3: deliberately small N-sweep under LLM operators, same
pop_size/n_generations as the pilot (run_phase2_pilot.py) -- only n_workers
varies.

This is NOT a test of Xue et al.'s Eq. 3/7 (already established in Phase 1
as a DIS-phase-speedup-ceiling model, not an overall-speedup model) and
finding a flat line here is not "the model predicted this and was right."
The flatness is architectural, not predicted: n_workers only exists inside
the DIS phase (fitness dispatch), and under LLM operators the GA phase
(sequential LLM calls for crossover/mutation) is ~99.95% of wall time in the
pilot (ga_total_s/wall_time_s) -- so N is cut off from essentially all of
wall time by construction, before any timing data is collected. The Amdahl
bookkeeping below (predicted_wall/predicted_speedup) is a mechanical
consequence of that cutoff, included to show the observed numbers land where
the architecture says they must, not as a hypothesis being tested:

    predicted_wall(N) = ga_total_s(N=1) + dis_total_s(N=1) / N
    predicted_speedup(N) = wall(N=1) / predicted_wall(N)

compared against the observed_speedup(N) = wall(N=1) / wall(N) from actually
running each N.

Deliberately NOT run at Phase 1's pop_size=256/40-generation scale -- the
whole point is that this architectural cutoff shows up at small scale
already, and demonstrating it doesn't need hours of LLM calls.

Also runs the deterministic operators at the same small scale for
reference/contrast (fast, seconds not minutes) -- see PHASE2_RESULTS.md for
why this deterministic arm is NOT usable for a deterministic-vs-LLM scaling
comparison (dispatch overhead dominates at pop_size=8; use Phase 1's numbers
for that comparison instead).
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpga.config import HPGAConfig, make_timing_sequence
from hpga.island import Island
from hpga import operators as ops

POP_SIZE = 8
N_GENERATIONS = 3
SEED = 0
SEQ_LENGTH = 20
N_VALUES = [1, 2, 4, 8]


def run_one(mode: str, n_workers: int, sequence: str) -> dict:
    os.environ["HPGA_OPERATOR_MODE"] = mode
    os.environ["HPGA_RUN_ID"] = f"phase2_nsweep_{mode}_N{n_workers}_{int(time.time())}"
    ops.reset_operator_stats()

    cfg = HPGAConfig(
        sequence=sequence, pop_size=POP_SIZE, n_generations=N_GENERATIONS,
        n_workers=n_workers, seed=SEED,
    )
    island = Island(cfg)
    recorder = island.run()
    summary = recorder.summary()
    summary["operator_stats"] = ops.get_operator_stats()
    summary["operator_mode"] = mode
    return summary


def sweep(mode: str, sequence: str) -> dict[int, dict]:
    results = {}
    for n in N_VALUES:
        print(f"  [{mode}] N={n} ...")
        t0 = time.perf_counter()
        s = run_one(mode, n, sequence)
        print(f"    wall={s['wall_time_s']:.4f}s  ga_total={s['ga_total_s']:.4f}s  "
              f"dis_total={s['dis_total_s']:.4f}s  (took {time.perf_counter()-t0:.2f}s)")
        results[n] = s
    return results


def report(mode: str, results: dict[int, dict]) -> None:
    base = results[N_VALUES[0]]
    base_wall = base["wall_time_s"]
    base_ga = base["ga_total_s"]
    base_dis = base["dis_total_s"]

    print(f"\n=== {mode}: predicted vs. observed overall speedup (paired against N={N_VALUES[0]}) ===")
    print(f"  N=1 baseline: wall={base_wall:.4f}s  ga_total={base_ga:.4f}s  dis_total={base_dis:.4f}s  "
          f"serial_fraction={base_ga/base_wall:.4f}")
    print(f"\n  {'N':>4} {'observed wall (s)':>20} {'observed speedup':>18} "
          f"{'predicted wall (s)':>20} {'predicted speedup':>18}")
    for n in N_VALUES:
        r = results[n]
        obs_wall = r["wall_time_s"]
        obs_speedup = base_wall / obs_wall if obs_wall else float("nan")
        pred_wall = base_ga + base_dis / n
        pred_speedup = base_wall / pred_wall if pred_wall else float("nan")
        print(f"  {n:>4} {obs_wall:>20.4f} {obs_speedup:>18.4f} {pred_wall:>20.4f} {pred_speedup:>18.4f}")


def main() -> None:
    sequence = make_timing_sequence(length=SEQ_LENGTH, seed=SEED)
    print(f"genome_length={len(sequence)-2}  pop_size={POP_SIZE}  n_generations={N_GENERATIONS}  "
          f"seed={SEED}  N_values={N_VALUES}\n")

    print("=== deterministic sweep (reference/contrast) ===")
    det_results = sweep("deterministic", sequence)
    report("deterministic", det_results)

    print("\n=== llm sweep ===")
    llm_results = sweep("llm", sequence)
    report("llm", llm_results)


if __name__ == "__main__":
    main()
