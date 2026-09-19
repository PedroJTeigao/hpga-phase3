"""Smoke run: the GA loop on the SEQUENCE GenomeModel, deterministic operators, scored by
the TM fitness (esmfold/tm_fitness.py). In-process, one warm predictor -- NOT Island.

Why not Island: Island builds a lattice population and evaluates in a pool of worker
processes whose target (worker.py) hard-wires hp_model.evaluate_fitness. The ESMFold
predictor is ~13.7GB and fits on the GPU once, so it cannot be replicated per worker;
sequence mode needs a single in-process fitness object, which Island's structure does
not offer without editing island.py / worker.py. This driver is Island.run's loop with
the worker pool replaced by direct calls to model.evaluate_fitness (GenomeModel ->
sequence_model.evaluate_fitness -> the process-wide TMFitness), and n_generations meaning
what it means there: that many EVALUATED populations (generation 0 is the random start),
with next_generation run between them -- so n_generations-1 breeding steps here (Island
also runs one after the last evaluation and discards it).

Per generation it records best / mean / min fitness, the best genome, its length, and
the edit distance from the best genome to the reference sequence; the raw JSON also keeps
every population with its fitnesses. Wall time is split into fitness (the evaluation
loops), breeding (next_generation) and the remainder (bookkeeping/printing); the ESMFold
load is timed separately and excluded. Call 1 includes the predictor's cold-start cost.

Refuses to run if any process is using the GPU, or if the operator mode / length bounds
are not deterministic / [30, 80].
"""

import argparse
import json
import logging
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ["HPGA_OPERATOR_MODE"] = "deterministic"  # before anything reads it; no LLM calls in this task

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import transformers  # noqa: E402

from esmfold import tm_fitness as tf  # noqa: E402
from hpga import genome_model  # noqa: E402
from hpga import operators as ops  # noqa: E402
from hpga import sequence_model as sm  # noqa: E402
from hpga.config import HPGAConfig  # noqa: E402


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as exc:
        return f"<call failed: {exc!r}>"


def gpu_snapshot() -> dict:
    return {
        "gpu": _run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                     "--format=csv,noheader"]),
        "compute_apps": _run(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
                              "--format=csv,noheader"]) or "(none)",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop-size", type=int, default=8)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(ROOT / "results" / "raw" / "sequence_ga_smoke_seed0.json"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if (sm.MIN_LENGTH, sm.MAX_LENGTH) != (30, 80):
        raise SystemExit(f"length bounds are {(sm.MIN_LENGTH, sm.MAX_LENGTH)}, expected (30, 80)")
    gpu_before = gpu_snapshot()
    print(f"gpu before: {gpu_before}")
    if gpu_before["compute_apps"] != "(none)":
        raise SystemExit("GPU is in use by another process; not running")

    cfg = HPGAConfig(genome_model="sequence", pop_size=args.pop_size, n_generations=args.generations, seed=args.seed)
    model = genome_model.build_genome_model(cfg)
    genome_model.set_active(model)
    ref = tf.reference_sequence()

    t0 = time.perf_counter()
    fit = sm._get_fitness()  # the shared warm TMFitness (ESMFold loaded here, once)
    model_load_s = time.perf_counter() - t0
    assert fit is tf.get_default(), "sequence fitness is not the process-wide TMFitness"
    print(f"model load: {model_load_s:.1f}s\n")

    rng = random.Random(cfg.seed)
    population = ops.random_population(cfg.pop_size, None, rng)  # variable lengths over [30, 80]
    records, fitness_s_total, breed_s_total, seen = [], 0.0, 0.0, set()
    loop_start = time.perf_counter()
    for gen in range(cfg.n_generations):
        t = time.perf_counter()
        fits = [model.evaluate_fitness(g) for g in population]  # GenomeModel -> sequence_model -> TMFitness
        fit_s = time.perf_counter() - t
        fitness_s_total += fit_s
        seen.update(population)

        b = max(range(len(population)), key=fits.__getitem__)
        rec = {
            "generation": gen, "best_fitness": fits[b], "mean_fitness": statistics.mean(fits), "min_fitness": min(fits),
            "best_genome": population[b], "best_length": len(population[b]),
            "best_edit_distance_to_reference": sm.edit_distance(population[b], ref),
            "population_lengths": [len(g) for g in population], "fitness_s": fit_s, "breed_s": None,
            "population": population, "fitnesses": fits,
        }
        if gen < cfg.n_generations - 1:
            t = time.perf_counter()
            population = ops.next_generation(population, fits, cfg.pop_size, cfg.tournament_k, cfg.crossover_rate,
                                             cfg.mutation_rate, cfg.elitism, rng)
            rec["breed_s"] = time.perf_counter() - t
            breed_s_total += rec["breed_s"]
        records.append(rec)
        print(f"gen {gen:>2}  best={rec['best_fitness']:.5f}  mean={rec['mean_fitness']:.5f}  min={rec['min_fitness']:.5f}  "
              f"best_len={rec['best_length']:>2}  edit_dist_to_ref={rec['best_edit_distance_to_reference']:>2}  "
              f"fitness_s={fit_s:.2f}", flush=True)
    wall = time.perf_counter() - loop_start

    llm_requests = ops.get_operator_stats()["n_llm_requests"]
    bests = [r["best_fitness"] for r in records]
    final = records[-1]
    summary = {
        "random_start_best": bests[0], "random_start_mean": records[0]["mean_fitness"],
        "final_best": final["best_fitness"], "final_mean": final["mean_fitness"],
        "final_best_genome": final["best_genome"], "final_best_length": final["best_length"],
        "final_best_edit_distance_to_reference": final["best_edit_distance_to_reference"],
        "best_by_generation": bests, "best_never_decreased": all(y >= x for x, y in zip(bests, bests[1:])),
        "wall_s": wall, "fitness_s": fitness_s_total, "breed_s": breed_s_total,
        "other_s": wall - fitness_s_total - breed_s_total, "fitness_share_of_wall": fitness_s_total / wall,
        "n_evaluations": cfg.pop_size * cfg.n_generations, "n_distinct_genomes_evaluated": len(seen),
        "mean_fitness_s_per_evaluation": fitness_s_total / (cfg.pop_size * cfg.n_generations),
        "n_llm_requests": llm_requests,
    }
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "final_best_genome"}, indent=2))
    print(f"final best genome ({summary['final_best_length']} aa): {summary['final_best_genome']}")
    print(f"reference sequence ({len(ref)} aa):               {ref}")
    assert llm_requests == 0, "an LLM request was made; this smoke run must be deterministic-operators only"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gpu_after = gpu_snapshot()
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"pop_size": cfg.pop_size, "n_generations": cfg.n_generations, "seed": cfg.seed,
                       "crossover_rate": cfg.crossover_rate, "mutation_rate": cfg.mutation_rate,
                       "tournament_k": cfg.tournament_k, "elitism": cfg.elitism, "operator_mode": "deterministic",
                       "length_bounds": [sm.MIN_LENGTH, sm.MAX_LENGTH], "initial_lengths": "uniform over the bounds"},
            "env": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                    "tmalign_bin": str(tf.TMALIGN_BIN), "reference_pdb": str(tf.REFERENCE_PDB.relative_to(ROOT)),
                    "reference_sequence": ref, "git_head": _run(["git", "-C", str(ROOT), "rev-parse", "HEAD"])},
            "model_load_s": model_load_s, "gpu_before": gpu_before, "gpu_after": gpu_after,
            "summary": summary, "generations": records,
        }, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
