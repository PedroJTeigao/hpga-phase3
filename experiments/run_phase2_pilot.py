"""Phase 2 first deliverable: one short paired run, deterministic operators
vs LLM operators, everything else identical (same sequence, seed, pop_size,
n_generations, n_workers). Reports T_calc, T_turnaround, GA phase time,
DIS/GA ratio, and the resulting serial fraction (GA time / wall time) for
both, so the shift can be read off directly before committing to a full
sweep. Does not touch island.py/worker.py/instrumentation.py or any existing
experiment script -- only imports them.

Writes one prompt/response JSONL log per run to
results/raw/llm_operator_calls_<run_id>.jsonl (deterministic runs write no
log, since no LLM calls happen).
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
N_WORKERS = 4
SEED = 0
SEQ_LENGTH = 20


def run_one(mode: str, sequence: str) -> dict:
    os.environ["HPGA_OPERATOR_MODE"] = mode
    os.environ["HPGA_RUN_ID"] = f"phase2pilot_{mode}_{int(time.time())}"
    ops.reset_operator_stats()

    cfg = HPGAConfig(
        sequence=sequence, pop_size=POP_SIZE, n_generations=N_GENERATIONS,
        n_workers=N_WORKERS, seed=SEED,
    )
    island = Island(cfg)
    recorder = island.run()
    summary = recorder.summary()
    summary["operator_stats"] = ops.get_operator_stats()
    summary["operator_mode"] = mode
    return summary


def serial_fraction(s: dict) -> float | None:
    return s["ga_total_s"] / s["wall_time_s"] if s["wall_time_s"] else None


def print_summary(s: dict) -> None:
    t_calc = s["t_calc"]["mean"]
    t_turn = s["t_turnaround"]["mean"]
    print(f"  wall_time={s['wall_time_s']:.4f}s  dis_total={s['dis_total_s']:.4f}s  ga_total={s['ga_total_s']:.4f}s")
    print(f"  T_calc(mean)={t_calc*1e3:.4f}ms  "
          f"T_turnaround(mean)={t_turn*1e3 if t_turn else float('nan'):.4f}ms")
    print(f"  DIS/GA ratio={s['dis_ga_ratio']:.4f}  serial_fraction(GA/wall)={serial_fraction(s):.4f}")
    print(f"  best_fitness_final={s['best_fitness_final']}")
    st = s.get("operator_stats") or {}
    if st.get("n_llm_calls"):
        print(f"  operator_stats: n_llm_calls={st['n_llm_calls']} n_llm_requests={st['n_llm_requests']} "
              f"n_retries={st['n_retries']} n_failures={st['n_failures']} n_skipped_no_op={st['n_skipped_no_op']}")
        if st.get("mean_latency_s") is not None:
            print(f"                  mean_latency={st['mean_latency_s']:.3f}s  "
                  f"tokens_in={st['total_tokens_in']}  tokens_out={st['total_tokens_out']}")


def compare(det: dict, llm: dict) -> None:
    rows = [
        ("T_calc mean (ms)", det["t_calc"]["mean"] * 1e3, llm["t_calc"]["mean"] * 1e3),
        ("T_turnaround mean (ms)", (det["t_turnaround"]["mean"] or 0) * 1e3, (llm["t_turnaround"]["mean"] or 0) * 1e3),
        ("GA phase time (s)", det["ga_total_s"], llm["ga_total_s"]),
        ("DIS phase time (s)", det["dis_total_s"], llm["dis_total_s"]),
        ("wall time (s)", det["wall_time_s"], llm["wall_time_s"]),
        ("DIS/GA ratio", det["dis_ga_ratio"], llm["dis_ga_ratio"]),
        ("serial fraction (GA/wall)", serial_fraction(det), serial_fraction(llm)),
    ]
    print(f"\n  {'metric':<28}{'deterministic':>16}{'llm':>16}")
    for name, d, l in rows:
        print(f"  {name:<28}{d:>16.4f}{l:>16.4f}")


def main() -> None:
    sequence = make_timing_sequence(length=SEQ_LENGTH, seed=SEED)
    print(f"genome_length={len(sequence) - 2}  pop_size={POP_SIZE}  "
          f"n_generations={N_GENERATIONS}  n_workers={N_WORKERS}  seed={SEED}\n")

    print("=== deterministic ===")
    t0 = time.perf_counter()
    det = run_one("deterministic", sequence)
    print(f"(took {time.perf_counter() - t0:.2f}s wall)")
    print_summary(det)

    print("\n=== llm ===")
    t0 = time.perf_counter()
    llm = run_one("llm", sequence)
    print(f"(took {time.perf_counter() - t0:.2f}s wall)")
    print_summary(llm)

    compare(det, llm)


if __name__ == "__main__":
    main()
