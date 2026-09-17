"""Phase 2 item 2: does a diff-style operator prompt (model emits only
changed positions, not the full genome) reduce output tokens and, via the
latency model fit in probe_latency_vs_tokens.py (latency_s ~= 1.73 +
0.0054*tokens_in + 0.339*tokens_out), reduce latency proportionally?

Runs the same paired-comparison shape as run_phase2_pilot.py, but both arms
use HPGA_OPERATOR_MODE=llm; only HPGA_LLM_PROMPT_STYLE varies (full vs diff).
Same sequence/seed/pop_size/n_generations/n_workers as the pilot, so this is
directly comparable to the pilot's own per-op numbers.

Reports, per op (crossover/mutate): mean tokens_in, mean tokens_out, mean
latency_s, and validity rate (parses correctly without falling back to the
deterministic operator) -- diff style asks more of the model's arithmetic
(positions are integers, not just letters) and could plausibly fail more
often, which would matter as much as the raw speed number.
"""

import os
import sys
import time
from collections import defaultdict
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

# Fit from experiments/probe_latency_vs_tokens.py's saved data
# (results/raw/latency_vs_tokens.jsonl), refit at the top of main() rather
# than hardcoded so this script stays correct if that data is regenerated.


def fit_latency_model() -> tuple[float, float, float]:
    import json
    import numpy as np

    path = Path(__file__).resolve().parent.parent / "results" / "raw" / "latency_vs_tokens.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    X = np.array([[1.0, r["tokens_in"], r["tokens_out"]] for r in rows])
    y = np.array([r["latency_s"] for r in rows])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def run_one(style: str, sequence: str) -> dict:
    os.environ["HPGA_OPERATOR_MODE"] = "llm"
    os.environ["HPGA_LLM_PROMPT_STYLE"] = style
    os.environ["HPGA_RUN_ID"] = f"phase2_diffstyle_{style}_{int(time.time())}"
    ops.reset_operator_stats()

    cfg = HPGAConfig(
        sequence=sequence, pop_size=POP_SIZE, n_generations=N_GENERATIONS,
        n_workers=N_WORKERS, seed=SEED,
    )
    island = Island(cfg)
    recorder = island.run()
    summary = recorder.summary()
    summary["operator_stats"] = ops.get_operator_stats()
    summary["style"] = style
    summary["run_id"] = os.environ["HPGA_RUN_ID"]
    return summary


def per_op_stats(run_id: str) -> dict:
    import json

    path = Path(__file__).resolve().parent.parent / "results" / "raw" / f"llm_operator_calls_{run_id}.jsonl"
    by_op = defaultdict(list)
    n_valid = defaultdict(int)
    n_total = defaultdict(int)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "latency_s" not in r:
            continue
        by_op[r["op"]].append(r)
        n_total[r["op"]] += 1
        if r.get("valid"):
            n_valid[r["op"]] += 1

    out = {}
    for op, rows in by_op.items():
        tin = [r["tokens_in"] for r in rows]
        tout = [r["tokens_out"] for r in rows]
        lat = [r["latency_s"] for r in rows]
        out[op] = {
            "n_requests": len(rows),
            "mean_tokens_in": sum(tin) / len(tin),
            "mean_tokens_out": sum(tout) / len(tout),
            "mean_latency_s": sum(lat) / len(lat),
            "valid_rate": n_valid[op] / n_total[op] if n_total[op] else float("nan"),
        }
    return out


def main() -> None:
    a, b, c = fit_latency_model()
    print(f"latency model (from probe_latency_vs_tokens.py data): "
          f"latency_s ~= {a:.3f} + {b*1e3:.3f}ms*tokens_in + {c*1e3:.3f}ms*tokens_out\n")

    sequence = make_timing_sequence(length=SEQ_LENGTH, seed=SEED)
    print(f"genome_length={len(sequence) - 2}  pop_size={POP_SIZE}  "
          f"n_generations={N_GENERATIONS}  n_workers={N_WORKERS}  seed={SEED}\n")

    results = {}
    op_stats = {}
    for style in ("full", "diff"):
        print(f"=== style={style} ===")
        t0 = time.perf_counter()
        s = run_one(style, sequence)
        wall = time.perf_counter() - t0
        print(f"(took {wall:.2f}s wall)  ga_total={s['ga_total_s']:.4f}s  best_fitness_final={s['best_fitness_final']}")
        results[style] = s
        op_stats[style] = per_op_stats(s["run_id"])
        for op, st in op_stats[style].items():
            pred = a + b * st["mean_tokens_in"] + c * st["mean_tokens_out"]
            print(f"  {op:<10} n={st['n_requests']:>3}  mean_tokens_in={st['mean_tokens_in']:.1f}  "
                  f"mean_tokens_out={st['mean_tokens_out']:.1f}  mean_latency_s={st['mean_latency_s']:.3f}  "
                  f"model_predicted={pred:.3f}  valid_rate={st['valid_rate']:.2f}")
        print()

    print("=== full vs diff, per op ===")
    print(f"  {'op':<10}{'tokens_out(full)':>18}{'tokens_out(diff)':>18}{'reduction':>12}"
          f"{'latency(full)':>16}{'latency(diff)':>16}{'reduction':>12}")
    for op in sorted(set(op_stats["full"]) | set(op_stats["diff"])):
        f_st, d_st = op_stats["full"].get(op), op_stats["diff"].get(op)
        if not f_st or not d_st:
            continue
        tout_red = 1 - d_st["mean_tokens_out"] / f_st["mean_tokens_out"]
        lat_red = 1 - d_st["mean_latency_s"] / f_st["mean_latency_s"]
        print(f"  {op:<10}{f_st['mean_tokens_out']:>18.1f}{d_st['mean_tokens_out']:>18.1f}{tout_red:>12.1%}"
              f"{f_st['mean_latency_s']:>16.3f}{d_st['mean_latency_s']:>16.3f}{lat_red:>12.1%}")

    print(f"\n  GA phase time: full={results['full']['ga_total_s']:.3f}s  "
          f"diff={results['diff']['ga_total_s']:.3f}s  "
          f"reduction={1 - results['diff']['ga_total_s']/results['full']['ga_total_s']:.1%}")


if __name__ == "__main__":
    main()
