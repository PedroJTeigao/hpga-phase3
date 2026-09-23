"""Move-class experiment: does a different class of mutation move pay more per evaluation than single-position
substitution? No LLM anywhere -- HPGA_OPERATOR_MODE=deterministic, so the move class is the one thing that varies.

Everything is arm B of run_sequence_ga_comparison.py (which this imports as `base` and whose run_ga loop is mirrored
line for line): pop 16, 20 evaluated populations (19 breeding steps), crossover 0.9 (sequence_model.crossover, single
proportional cut), tournament 3, elitism 2, length bounds [30, 80], Random(seed) drives everything, same generation-0
population per seed, fitness = TM-score of the ESMFold fold against 7UR7 through the same cache-aware Evaluator. The
only change: the active model's deterministic_mutate is replaced, for this run, by a wrapper that applies the arm's
move class (hpga/move_class.py) and records the move. Nothing in hpga/ other than the new move_class.py is touched.

  B   the existing operator (per-site substitution p=0.05, ~3 substitutions per child on ~60 residues). A REFERENCE
      row, not S1: when results/raw/sequence_ga_cmp_B_seed<n>.json exists the run must reproduce it exactly
      (distinct count, best-so-far curve, every generation's population and fitnesses) -- the check that the wrapper
      plumbing changes nothing.
  S1  exactly one single-position substitution per child     S2  one k-position substitution (k in 2-5) per child
  S3  one segment replacement (span 5-15) per child          S4  one indel (delete or insert 1-5) per child

Move payoff needs the fitness of each move's BASE (the post-crossover child), which the GA never folds unless it is an
unchanged parent. After the GA finishes, every base (and any child) not already in the run's cache is folded in a
SEPARATE measurement cache that never enters selection, the distinct-evaluation count or the best-so-far curve, and
is written to results/raw/move_class_bases_<arm>_seed<n>.json -- never into the run's own file. Inertness is asserted
twice per run: (1) a digest of the GA's outputs taken before any measurement fold must equal the digest after;
(2) the GA is replayed from the seed with move recording off and fitness served ONLY from the run's own cache (a
genome it never folded raises), and the replay's digest must equal the run's.

  (launch with PYTHONHASHSEED=0, as the earlier drivers)
  run --arms B S1 S2 S3 S4 --seeds 0 1 2      one result file per (arm, seed), written atomically; finished runs skipped
"""

import argparse
import hashlib
import json
import logging
import os
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_sequence_ga_comparison as base  # noqa: E402
import torch  # noqa: E402

from esmfold import tm_fitness as tf  # noqa: E402
from hpga import genome_model, move_class as mc, operators as ops  # noqa: E402
from hpga import sequence_model as sm  # noqa: E402
from hpga.config import HPGAConfig  # noqa: E402

log = logging.getLogger("moveclass")


class NoResidency:
    swap_s, n_swaps = 0.0, 0

    def to_fold_phase(self): ...


class CacheOnlyModel:
    """Replay fitness: served from a finished run's cache only; a genome the run never folded is a divergence."""

    def __init__(self, cache: dict):
        self.cache = cache

    def evaluate_fitness(self, g: str) -> float:
        if g not in self.cache:
            raise RuntimeError(f"replay diverged: genome never folded by the run ({g[:20]}...)")
        return self.cache[g]


def ga_loop(arm: str, seed: int, cfg: HPGAConfig, model, fit_model, ref: str, res, record_moves: bool, partial=None):
    """The GA of base.run_ga(llm=False), with model.deterministic_mutate replaced by the arm's move. `fit_model`
    supplies evaluate_fitness to the Evaluator (the real model, or a CacheOnlyModel for the replay)."""
    ev = base.Evaluator(fit_model)
    move = mc.MOVES[arm]
    moves, step = [], {"gen": 0}

    def mutate(genome, rate, rng):
        child, desc = move(genome, rate, rng)
        if record_moves:
            moves.append({"step": step["gen"], "base": genome, "child": child, **desc})
        return child

    model.deterministic_mutate = mutate  # instance attribute: this model object only, for this loop only
    rng = random.Random(seed)
    population = ops.random_population(cfg.pop_size, None, rng)
    gens, breed_s = [], 0.0
    try:
        for gen in range(cfg.n_generations):
            if gen == 0:
                res.to_fold_phase()
            ev.tag = gen
            n_before, hits_before = len(ev.cache), ev.hits
            fits = [ev.score(g) for g in population]
            b = max(range(len(population)), key=fits.__getitem__)
            rg = {"generation": gen, "best_fitness": fits[b], "mean_fitness": statistics.mean(fits),
                  "min_fitness": min(fits), "best_genome": population[b], "best_length": len(population[b]),
                  "best_edit_distance_to_reference": sm.edit_distance(population[b], ref),
                  "mean_pairwise_edit_distance": base.mean_pairwise_edit(population),
                  "n_distinct_in_population": len(set(population)), "mean_length": statistics.mean(map(len, population)),
                  "lengths": sorted(map(len, population)),
                  "new_distinct_evaluations": len(ev.cache) - n_before, "cache_hits": int(ev.hits - hits_before),
                  "distinct_evaluations_so_far": len(ev.cache), "best_so_far": ev.best,
                  "population": population, "fitnesses": fits, "breed_s": None}
            if gen < cfg.n_generations - 1:
                step["gen"] = gen
                t = time.perf_counter()
                population = ops.next_generation(population, fits, cfg.pop_size, cfg.tournament_k, cfg.crossover_rate,
                                                 cfg.mutation_rate, cfg.elitism, rng)
                rg["breed_s"] = time.perf_counter() - t
                breed_s += rg["breed_s"]
            gens.append(rg)
            if partial is not None:
                log.info("%s seed %d gen %2d best=%.4f mean=%.4f distinct=%d hits=%d len=%.1f", arm, seed, gen,
                         rg["best_fitness"], rg["mean_fitness"], len(ev.cache), int(ev.hits), rg["mean_length"])
                partial(gens, ev)
    finally:
        del model.deterministic_mutate
    return ev, gens, moves, breed_s


def digest(ev, gens) -> dict:
    """What 'measurement changed nothing' has to mean: distinct count, the best-so-far curve and its generation tags,
    and every generation's population and fitnesses, byte for byte (JSON with repr floats)."""
    parts = {"n_distinct": len(ev.cache), "curve": ev.curve, "curve_gen": ev.curve_gen,
             "populations": [g["population"] for g in gens], "fitnesses": [g["fitnesses"] for g in gens]}
    return {k: hashlib.sha256(json.dumps(v).encode()).hexdigest() for k, v in parts.items()}


def reference_match(seed: int, ev, gens) -> dict | None:
    """Arm B only: compare against the published arm-B run of run_sequence_ga_comparison.py, if there is one."""
    p = base.RAW / f"sequence_ga_cmp_B_seed{seed}.json"
    if not p.exists():
        return None
    ref = json.load(open(p))
    mine = json.loads(json.dumps({"curve": ev.curve, "curve_gen": ev.curve_gen,
                                  "pop": [g["population"] for g in gens], "fit": [g["fitnesses"] for g in gens]}))
    checks = {"n_distinct": len(ev.cache) == ref["summary"]["n_distinct_evaluations"],
              "best_so_far_curve": mine["curve"] == ref["best_so_far_by_distinct_evaluation"],
              "curve_generation_tags": mine["curve_gen"] == ref["generation_of_distinct_evaluation"],
              "populations": mine["pop"] == [g["population"] for g in ref["generations"]],
              "fitnesses": mine["fit"] == [g["fitnesses"] for g in ref["generations"]]}
    return {"reference_file": base.rel(p), "checks": checks, "identical": all(checks.values())}


def measure(moves: list, run_cache: dict, model) -> tuple[list, dict]:
    """Fitness of every move's base and child. Run-cache values are free lookups; anything else is folded here, into
    a local cache that is returned and never shared with the run's Evaluator."""
    meas, fold_s = {}, 0.0

    def f(g):
        nonlocal fold_s
        if g in run_cache:
            return run_cache[g], "run_cache"
        if g not in meas:
            t = time.perf_counter()
            meas[g] = model.evaluate_fitness(g)
            fold_s += time.perf_counter() - t
        return meas[g], "measurement_fold"

    out = []
    for m in moves:
        bf, bsrc = f(m["base"])
        cf, csrc = f(m["child"])
        out.append({**m, "base_fitness": bf, "base_source": bsrc, "child_fitness": cf, "child_source": csrc,
                    "no_op": m["base"] == m["child"], "improved": cf > bf, "gain": cf - bf})
    return out, {"n_measurement_folds": len(meas), "measurement_fold_s": fold_s}


def run_arm(arm: str, seed: int, args, path: Path) -> dict:
    rec = base.base_record(arm, seed, args)
    base.clean_env()
    os.environ["HPGA_OPERATOR_MODE"] = "deterministic"
    cfg = HPGAConfig(genome_model="sequence", pop_size=args.pop_size, n_generations=args.generations, seed=seed)
    model = genome_model.build_genome_model(cfg)
    genome_model.set_active(model)
    ref = tf.reference_sequence()
    res = base.Residency()
    rec["config"].update({"move_class": arm, "move_class_spec": mc.__doc__.split("\n\n")[1],
                          "mutation_rate_used": arm == "B", "llm": "none (deterministic operators only)"})
    partial_path = path.with_suffix(".partial.json")

    def partial(gens, ev):
        base.atomic_write_json(partial_path, {**rec, "generations": gens, "best_so_far_by_distinct_evaluation": ev.curve})

    t_start = time.perf_counter()
    ev, gens, moves, breed_s = ga_loop(arm, seed, cfg, model, model, ref, res, record_moves=True, partial=partial)
    wall = time.perf_counter() - t_start
    before = digest(ev, gens)

    # --- measurement: after the GA has finished, separate cache, separate file
    run_cache_snapshot = dict(ev.cache)
    measured, mstats = measure(moves, run_cache_snapshot, model)
    after = digest(ev, gens)
    if after != before or ev.cache != run_cache_snapshot:
        raise RuntimeError(f"measurement changed the run: {before} vs {after}")

    # --- replay without move recording, fitness from the run's own cache only
    replay_ev, replay_gens, _, _ = ga_loop(arm, seed, cfg, model, CacheOnlyModel(run_cache_snapshot), ref, NoResidency(),
                                           record_moves=False)
    replay = digest(replay_ev, replay_gens)
    if replay != before:
        raise RuntimeError(f"replay without measurement differs: {before} vs {replay}")

    ref_match = reference_match(seed, ev, gens) if arm == "B" else None
    if ref_match is not None and not ref_match["identical"]:
        raise RuntimeError(f"B does not reproduce {ref_match['reference_file']}: {ref_match['checks']}")

    bases_path = Path(args.out_dir) / f"move_class_bases_{arm}_seed{seed}.json"
    base.atomic_write_json(bases_path, {
        "arm": arm, "seed": seed, "run_file": base.rel(path),
        "note": "measurement only: these folds never entered the run's selection, distinct count or best-so-far curve",
        **mstats, "ga_fold_s": ev.fitness_s,
        "measurement_share_of_fold_time": mstats["measurement_fold_s"] / (mstats["measurement_fold_s"] + ev.fitness_s),
        "moves": measured})
    extra = {"breed_s_total": breed_s, "n_swaps": res.n_swaps, "swap_integrity_checks": [],
             "n_population_evaluations": cfg.pop_size * cfg.n_generations, "n_moves": len(moves),
             "move_bases_file": base.rel(bases_path),
             "inertness": {"digest_before_measurement": before, "digest_after_measurement_identical": True,
                           "replay_without_measurement_identical": True},
             "reference_match": ref_match}
    rec.update({"complete": True, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "generations": gens,
                "best_so_far_by_distinct_evaluation": ev.curve, "generation_of_distinct_evaluation": ev.curve_gen,
                "summary": base.finish_summary(ev, ref, wall, res, 0.0, extra)})
    partial_path.unlink(missing_ok=True)
    log.info("%s seed %d done: best=%.4f distinct=%d moves=%d measurement folds=%d (%.0fs) ref_match=%s", arm, seed,
             ev.best, len(ev.cache), len(moves), mstats["n_measurement_folds"], mstats["measurement_fold_s"],
             None if ref_match is None else ref_match["identical"])
    return rec


def out_path(args, arm, seed) -> Path:
    return Path(args.out_dir) / f"move_class_{arm}_seed{seed}.json"


def cmd_run(args) -> int:
    plan = [(arm, seed) for seed in args.seeds for arm in args.arms]
    status_path = Path(args.out_dir) / "move_class_status.json"
    status = {"plan": plan, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "done": [], "failed": [], "skipped": [],
              "finished": False, "exit_reason": None}
    base.atomic_write_json(status_path, status)
    log.info("plan: %s", plan)
    code = 0
    try:
        base.wait_gpu_not_used_by_others()
        base.unload_ollama()
        sm._get_fitness()
        for arm, seed in plan:
            path = out_path(args, arm, seed)
            if path.exists() and json.load(open(path)).get("complete"):
                status["skipped"].append({"arm": arm, "seed": seed, "reason": "result file already complete"})
                continue
            base.check_disk()
            base.wait_gpu_not_used_by_others()
            for attempt in (1, 2):
                t0 = time.time()
                try:
                    log.info("=== %s seed %d attempt %d ===", arm, seed, attempt)
                    rec = run_arm(arm, seed, args, path)
                    rec["attempt"] = attempt
                    base.atomic_write_json(path, rec)
                    status["done"].append({"arm": arm, "seed": seed, "attempt": attempt, "seconds": time.time() - t0,
                                           "final_best": rec["summary"]["final_best"]})
                    break
                except (base.GpuBusy, base.DiskLow):
                    raise
                except Exception:  # noqa: BLE001
                    err = traceback.format_exc()
                    log.error("%s seed %d attempt %d crashed:\n%s", arm, seed, attempt, err)
                    torch.cuda.empty_cache()
                    if attempt == 2 or "does not reproduce" in err or "changed the run" in err or "differs" in err:
                        path.with_suffix(".error.txt").write_text(err)
                        status["failed"].append({"arm": arm, "seed": seed, "error": err.strip().splitlines()[-1]})
                        break  # a verification failure is not transient: do not retry it
            base.atomic_write_json(status_path, status)
        status["exit_reason"] = "all planned runs attempted"
    except base.GpuBusy as e:
        status["exit_reason"], code = f"GPU_BUSY_TIMEOUT: {e}", 3
    except base.DiskLow as e:
        status["exit_reason"], code = f"DISK_LOW: {e}", 4
    status["finished"], status["ended"] = True, time.strftime("%Y-%m-%dT%H:%M:%S%z")
    base.atomic_write_json(status_path, status)
    log.info("finished: %s", status["exit_reason"])
    return code or (5 if status["failed"] else 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_cmd", choices=["run"])
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", required=True, choices=list(mc.MOVE_CLASSES))
    ap.add_argument("--pop-size", type=int, default=16)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--out-dir", type=str, default=str(base.RAW))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for noisy in ("httpx", "esmfold.tm_fitness", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if (sm.MIN_LENGTH, sm.MAX_LENGTH) != (30, 80):
        raise SystemExit(f"length bounds are {(sm.MIN_LENGTH, sm.MAX_LENGTH)}, expected (30, 80)")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("launch with PYTHONHASHSEED=0")
    sys.exit(cmd_run(args))


if __name__ == "__main__":
    main()
