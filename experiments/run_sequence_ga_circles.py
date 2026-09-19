"""Arms D and E of the sequence-mode comparison (A random, B GA, C GA+LLM are in
run_sequence_ga_comparison.py, unchanged; this script imports its helpers).

  D  GA + LLM operators + circles   (hpga/circles_sequence.py; proposals are position edits)
  E  GA + LLM operators + random immigrants -- the control: the same number of new genomes injected per
     breeding step as D injects (n_circles * agents_per_circle = 4), drawn the way random_population draws
     them (model.random_genome(rng, None)), no model involved in producing them

Everything else is identical to arm C: pop 16, 20 evaluated populations (generation 0 = random start, so 19
breeding steps), crossover 0.9, mutation 0.05, tournament 3, elitism 2, mutate=position / crossover=segment,
gemma4:12b at temperature 0.7, length bounds [30, 80], same seeds, same generation-0 population per seed.
Circle configuration = the Phase 3 anchor run's: 2 circles x 2 agents, LLM central directive every 5
generations, curation every 3 (run_circles_smoke.py). Both arms ADD their 4 genomes to the population after the
ordinary fill (so the evaluated population is 20 from generation 1), the additive contract circles already has.

  (launch with PYTHONHASHSEED=0: the blackboard's curation iterates a set of ids, so tombstone order otherwise varies by process)
  run --seeds 0 1 2 3 4     for each seed: D then E; one result file per (arm, seed), written atomically;
                            each run retried once, then skipped and logged; finished runs skipped on restart.

Differences from arm C's accounting, on purpose: (1) after moving ESMFold to CPU the driver sends one tiny request
to load the Ollama model and counts that time as SWAP time, so llm_s is time in actual calls only (in C the reload
was inside the first LLM call of each breeding step); (2) LLM time and stats are tallied per operator by wrapping
operators._run_llm_op (mutate, crossover, consult, observe, central_directive, curate, propose all go through it).
"""

import argparse
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
from hpga import agents, blackboard as bb, circles, genome_model, operators as ops, sequence_model as sm  # noqa: E402
from hpga.config import HPGAConfig  # noqa: E402

log = logging.getLogger("seqcircles")
N_CIRCLES, PER_CIRCLE, CENTRAL_INTERVAL, CURATION_INTERVAL = 2, 2, 5, 3
N_SLOTS = N_CIRCLES * PER_CIRCLE
STAT_KEYS = base.STAT_KEYS


class Residency(base.Residency):
    """base.Residency plus an explicit Ollama load after moving ESMFold to CPU, timed as swap time."""

    def __init__(self):
        super().__init__()
        self.ollama_load_s = 0.0

    def to_llm_phase(self) -> None:
        super().to_llm_phase()
        t = time.perf_counter()
        try:
            ops._get_client().generate(model=ops.LLM_MODEL, prompt="Reply with the single word OK.", stream=False, think=False,
                                       options={"num_predict": 4, "temperature": 0}, keep_alive=ops.LLM_KEEP_ALIVE)
        except Exception as exc:  # noqa: BLE001 -- a genuinely dead server shows up at the first real call
            log.warning("ollama warm-up failed: %r", exc)
        dt = time.perf_counter() - t
        self.swap_s += dt
        self.ollama_load_s += dt


class OpTally:
    def __init__(self):
        self.d = {}

    def add(self, op, wall, before, after):
        t = self.d.setdefault(op, {"calls": 0, "wall_s": 0.0, **{k: 0 for k in STAT_KEYS}})
        t["calls"] += 1
        t["wall_s"] += wall
        for k in STAT_KEYS:
            t[k] += (after[k] or 0) - (before[k] or 0)

    def summary(self):
        out = {}
        for op, t in self.d.items():
            n = t["n_llm_calls"]
            out[op] = {**t, "fallback_rate": (t["n_failures"] / n) if n else None, "requests_per_call": (t["n_llm_requests"] / n) if n else None}
        return out


_ORIG = {}


def install(tally: OpTally) -> None:
    """Per-operator style (mutate=position, crossover=segment) and a tally around every operators._run_llm_op call."""
    _ORIG.setdefault("run", ops._run_llm_op)
    _ORIG.setdefault("crossover", ops._llm_crossover)
    _ORIG.setdefault("mutate", ops._llm_mutate)
    orig_run = _ORIG["run"]

    def run_wrapped(*args, **kwargs):
        before, t0 = ops.get_operator_stats(), time.perf_counter()
        out = orig_run(*args, **kwargs)
        tally.add(kwargs.get("op"), time.perf_counter() - t0, before, ops.get_operator_stats())
        return out

    def style(op):
        orig = _ORIG[op]

        def f(*a, **k):
            os.environ["HPGA_LLM_PROMPT_STYLE"] = base.STYLES[op]
            return orig(*a, **k)

        return f

    ops._run_llm_op, ops._llm_crossover, ops._llm_mutate = run_wrapped, style("crossover"), style("mutate")


def uninstall() -> None:
    if _ORIG:
        ops._run_llm_op, ops._llm_crossover, ops._llm_mutate = _ORIG["run"], _ORIG["crossover"], _ORIG["mutate"]
    os.environ.pop("HPGA_LLM_PROMPT_STYLE", None)


def run_arm(arm: str, seed: int, args) -> dict:
    assert arm in ("D", "E")
    rec = base.base_record(arm, seed, args)
    base.clean_env()
    os.environ.update({"HPGA_OPERATOR_MODE": "llm", "HPGA_AGENTS_ENABLED": "0"})
    if arm == "D":
        os.environ.update({"HPGA_CIRCLES_ENABLED": "1", "HPGA_N_CIRCLES": str(N_CIRCLES), "HPGA_AGENTS_PER_CIRCLE": str(PER_CIRCLE),
                           "HPGA_CENTRAL_MODE": "llm", "HPGA_CENTRAL_INTERVAL": str(CENTRAL_INTERVAL),
                           "HPGA_CURATION_INTERVAL": str(CURATION_INTERVAL)})
    else:
        os.environ["HPGA_CIRCLES_ENABLED"] = "0"
    cfg = HPGAConfig(genome_model="sequence", pop_size=args.pop_size, n_generations=args.generations, seed=seed)
    model = genome_model.build_genome_model(cfg)
    genome_model.set_active(model)
    ref = tf.reference_sequence()
    res, tally, ev = Residency(), OpTally(), base.Evaluator(model)
    os.environ["HPGA_RUN_ID"] = f"seqcmp_{arm}_seed{seed}_{int(time.time())}"
    ops.reset_operator_stats(); agents.reset_agent_state(); circles.reset_circle_state(); bb.reset_blackboard()
    install(tally)
    rec["llm_call_log"] = base.rel(ops._log_path())
    if arm == "D":
        rec["blackboard_log"] = base.rel(bb._log_path())
    rec["config"].update({"arm_description": "GA + LLM operators + circles" if arm == "D" else "GA + LLM operators + random immigrants (control)",
                          "injected_per_breeding_step": N_SLOTS, "llm_styles": base.STYLES,
                          "circles": {"n_circles": N_CIRCLES, "agents_per_circle": PER_CIRCLE, "central_mode": "llm",
                                      "central_interval": CENTRAL_INTERVAL, "curation_interval": CURATION_INTERVAL,
                                      "proposal_format": "position edits to an existing population member",
                                      "edit_rate": 0.05} if arm == "D" else None})
    rng = random.Random(seed)
    population = ops.random_population(cfg.pop_size, None, rng)
    gens, integrity, breed_s = [], [], 0.0
    t_start = time.perf_counter()
    try:
        for gen in range(cfg.n_generations):
            res.to_fold_phase()
            if gen in base.SWAP_CHECK_GENS and ev.cache:
                g0 = next(iter(ev.cache))
                again = model.evaluate_fitness(g0)
                integrity.append({"generation": gen, "before": ev.cache[g0], "after": again, "identical": again == ev.cache[g0]})
                if again != ev.cache[g0]:
                    raise RuntimeError(f"ESMFold CPU/GPU swap changed a fitness: {ev.cache[g0]} -> {again}")
            ev.tag = gen
            n_before, hits_before = len(ev.cache), ev.hits
            fits = [ev.score(g) for g in population]
            b = max(range(len(population)), key=fits.__getitem__)
            rg = {"generation": gen, "n_population": len(population), "best_fitness": fits[b], "mean_fitness": statistics.mean(fits),
                  "min_fitness": min(fits), "best_genome": population[b], "best_length": len(population[b]),
                  "best_edit_distance_to_reference": sm.edit_distance(population[b], ref),
                  "mean_pairwise_edit_distance": base.mean_pairwise_edit(population), "n_distinct_in_population": len(set(population)),
                  "mean_length": statistics.mean(map(len, population)), "new_distinct_evaluations": len(ev.cache) - n_before,
                  "cache_hits": int(ev.hits - hits_before), "distinct_evaluations_so_far": len(ev.cache), "best_so_far": ev.best,
                  "population": population, "fitnesses": fits, "breed_s": None}
            if gen < cfg.n_generations - 1:
                res.to_llm_phase()
                t = time.perf_counter()
                population = ops.next_generation(population, fits, cfg.pop_size, cfg.tournament_k, cfg.crossover_rate,
                                                 cfg.mutation_rate, cfg.elitism, rng)
                if arm == "E":  # the control: N_SLOTS random immigrants, drawn as random_population draws them
                    population = population + [model.random_genome(rng, None) for _ in range(N_SLOTS)]
                rg["breed_s"] = time.perf_counter() - t
                breed_s += rg["breed_s"]
            gens.append(rg)
            log.info("%s seed %d gen %2d n=%d best=%.4f mean=%.4f distinct=%d hits=%d div=%.1f", arm, seed, gen, len(gens[-1]["population"]),
                     rg["best_fitness"], rg["mean_fitness"], len(ev.cache), int(ev.hits), rg["mean_pairwise_edit_distance"])
    finally:
        uninstall()
    wall = time.perf_counter() - t_start
    ops_summary = tally.summary()
    llm_s = sum(t["wall_s"] for t in ops_summary.values())
    st = ops.get_operator_stats()
    n_calls = sum(t["n_llm_calls"] for t in ops_summary.values())
    extra = {"breed_s_total": breed_s, "n_swaps": res.n_swaps, "ollama_load_s": res.ollama_load_s, "swap_integrity_checks": integrity,
             "n_population_evaluations": sum(g["n_population"] for g in gens), "operators": ops_summary,
             "operator_stats_total": {k: st[k] for k in STAT_KEYS}, "n_llm_calls_total": n_calls,
             "fallback_rate_all_llm_calls": (sum(t["n_failures"] for t in ops_summary.values()) / n_calls) if n_calls else None,
             "requests_per_call_all_llm_calls": (sum(t["n_llm_requests"] for t in ops_summary.values()) / n_calls) if n_calls else None,
             "injected_total": N_SLOTS * (cfg.n_generations - 1)}
    if arm == "D":
        extra["blackboard"] = {"entries_written": len(bb._entries), "live_at_end": len(bb.read_live()),
                               "by_type": {t: sum(1 for e in bb._entries.values() if e.type == t) for t in ("observation", "directive", "summary")}}
    rec.update({"complete": True, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "generations": gens,
                "best_so_far_by_distinct_evaluation": ev.curve, "generation_of_distinct_evaluation": ev.curve_gen,
                "summary": base.finish_summary(ev, ref, wall, res, llm_s, extra)})
    return rec


def out_path(args, arm, seed) -> Path:
    return Path(args.out_dir) / f"sequence_ga_cmp_{arm}_seed{seed}.json"


def cmd_run(args) -> int:
    plan = [(arm, seed) for seed in args.seeds for arm in args.arms]
    status_path = Path(args.out_dir) / "sequence_ga_circles_status.json"
    status = {"plan": plan, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "done": [], "failed": [], "skipped": [], "finished": False, "exit_reason": None}
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
                    rec = run_arm(arm, seed, args)
                    rec["attempt"] = attempt
                    base.atomic_write_json(path, rec)
                    status["done"].append({"arm": arm, "seed": seed, "attempt": attempt, "seconds": time.time() - t0, "final_best": rec["summary"]["final_best"]})
                    break
                except (base.GpuBusy, base.DiskLow):
                    raise
                except Exception:  # noqa: BLE001
                    err = traceback.format_exc()
                    log.error("%s seed %d attempt %d crashed:\n%s", arm, seed, attempt, err)
                    uninstall()
                    try:
                        torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001
                        pass
                    if attempt == 2:
                        path.with_suffix(".error.txt").write_text(err)
                        status["failed"].append({"arm": arm, "seed": seed, "error": err.strip().splitlines()[-1]})
            base.atomic_write_json(status_path, status)
        status["exit_reason"] = "all planned runs attempted"
    except base.GpuBusy as e:
        status["exit_reason"], code = f"GPU_BUSY_TIMEOUT: {e}", 3
    except base.DiskLow as e:
        status["exit_reason"], code = f"DISK_LOW: {e}", 4
    status["finished"], status["ended"] = True, time.strftime("%Y-%m-%dT%H:%M:%S%z")
    base.atomic_write_json(status_path, status)
    log.info("finished: %s", status["exit_reason"])
    return code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_cmd", choices=["run"])
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", default=["D", "E"], choices=["D", "E"])
    ap.add_argument("--pop-size", type=int, default=16)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--out-dir", type=str, default=str(base.RAW))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for noisy in ("httpx", "esmfold.tm_fitness", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if (sm.MIN_LENGTH, sm.MAX_LENGTH) != (30, 80):
        raise SystemExit(f"length bounds are {(sm.MIN_LENGTH, sm.MAX_LENGTH)}, expected (30, 80)")
    sys.exit(cmd_run(args))


if __name__ == "__main__":
    main()
