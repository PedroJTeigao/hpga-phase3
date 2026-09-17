"""Stage 2 (DIBM analogue): measure GA-phase wall time as a function of the
concurrent operator-dispatch degree P (HPGA_GA_DISPATCH_P in hpga/operators.py),
holding pop_size/n_generations/n_workers/sequence fixed across P. Each P is
repeated (--repeats, default 3) with a different run seed per repeat -- same
target sequence, different population/operator-call RNG stream each time --
so timing variance (session-level LLM latency drift, contention noise) shows
up as spread rather than being silently absorbed into a single-run number.

Purpose is narrowly TIMING, not operator quality. Whatever model is
configured (HPGA_LLM_MODEL) is exercised purely as an issue-and-time
instrument here -- responses do not need to be usable, only issued and
timed. Do not read the resulting ga_total_s numbers as an endorsement of the
configured model's operator competence; see PHASE2_RESULTS.md's Stage 2
section for the disqualification note on that axis specifically.

The paper's Eq. 11 predicts capacity/speedup scale by P (the master borrows
P injection channels instead of serialising through one). This script
reports observed ga_total_s speedup at each P (mean +/- sd across repeats)
against that ideal-P prediction, run_one() being the same paired-run pattern
as run_phase2_pilot.py.
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpga.config import HPGAConfig, make_timing_sequence
from hpga.island import Island
from hpga import operators as ops

N_WORKERS = 1
SEQUENCE_SEED = 0  # fixes the target protein sequence across all P and all repeats
SEQ_LENGTH = 20
P_VALUES = [1, 2, 4, 8]
DEFAULT_ELITISM = 2  # hpga/config.py's HPGAConfig default; used only to report n_units/generation


def run_one(p: int, sequence: str, pop_size: int, n_generations: int, run_seed: int) -> dict:
    os.environ["HPGA_OPERATOR_MODE"] = "llm"
    os.environ["HPGA_GA_DISPATCH_P"] = str(p)
    os.environ["HPGA_RUN_ID"] = f"phase2_dispatch_pop{pop_size}_P{p}_seed{run_seed}_{int(time.time())}"
    ops.reset_operator_stats()

    cfg = HPGAConfig(
        sequence=sequence, pop_size=pop_size, n_generations=n_generations,
        n_workers=N_WORKERS, seed=run_seed,
    )
    island = Island(cfg)
    t0 = time.perf_counter()
    recorder = island.run()
    wall = time.perf_counter() - t0
    summary = recorder.summary()
    summary["measured_wall_s"] = wall
    summary["operator_stats"] = ops.get_operator_stats()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop-size", type=int, default=8)
    parser.add_argument("--n-generations", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    pop_size, n_generations, repeats = args.pop_size, args.n_generations, args.repeats

    n_units = -(-(pop_size - DEFAULT_ELITISM) // 2)
    sequence = make_timing_sequence(length=SEQ_LENGTH, seed=SEQUENCE_SEED)
    model = os.environ.get("HPGA_LLM_MODEL", "gemma4:12b")
    print(f"model={model}  genome_length={len(sequence) - 2}  pop_size={pop_size}  "
          f"n_generations={n_generations}  n_workers={N_WORKERS}  repeats={repeats}  "
          f"reproduction_units/generation={n_units}")
    print("Timing-only run: response validity/compliance is not evaluated here.\n")

    per_p_times: dict[int, list[float]] = {}
    for p in P_VALUES:
        print(f"=== P={p} ===")
        times = []
        for r in range(repeats):
            s = run_one(p, sequence, pop_size, n_generations, run_seed=r)
            times.append(s["ga_total_s"])
            st = s["operator_stats"]
            print(f"  rep={r}  ga_total_s={s['ga_total_s']:.3f}  wall_time_s={s['wall_time_s']:.3f}  "
                  f"n_llm_calls={st['n_llm_calls']}  n_failures={st['n_failures']}")
        per_p_times[p] = times
        mean = statistics.mean(times)
        sd = statistics.stdev(times) if len(times) > 1 else 0.0
        print(f"  -> mean={mean:.3f}s  sd={sd:.3f}s\n")

    base_mean = statistics.mean(per_p_times[P_VALUES[0]])
    print(f"{'P':>3}  {'ga_total_s (mean+/-sd)':>26}  {'observed_speedup':>18}  {'ideal_speedup(=P)':>18}")
    for p in P_VALUES:
        times = per_p_times[p]
        mean = statistics.mean(times)
        sd = statistics.stdev(times) if len(times) > 1 else 0.0
        observed = base_mean / mean if mean else float("nan")
        print(f"{p:>3}  {mean:>10.3f} +/- {sd:<10.3f}  {observed:>18.3f}  {p:>18.3f}")


if __name__ == "__main__":
    main()
