"""Fitness-aware-prompt experiment: does telling the LLM operator the genome's
fitness change what it proposes, and does that change the search?

Arm F is arm C of run_sequence_ga_comparison.py with ONE difference: each
operator prompt carries a labelled fitness line (hpga/sequence_model_fitness.py).
Same pop 16, 20 evaluated populations (19 breeding steps), crossover 0.9,
tournament 3, elitism 2, length bounds [30, 80], Random(seed) driving
everything, same generation-0 population per seed, same fitness (TM-score of
the ESMFold fold against 7UR7) through the same cache-aware Evaluator, same
prompt styles (mutate 'position', crossover 'segment'), same model and
temperature. Everything here imports run_sequence_ga_comparison as `base` and
run_move_class as `mcd` rather than restating it, so the comparison arms and
the off-budget measurement are the published code, not a copy.

  B   the published deterministic arm, run through THIS driver as a control.
      When results/raw/sequence_ga_cmp_B_seed<n>.json exists the run must
      reproduce it exactly -- distinct count, best-so-far curve and its
      generation tags, and every generation's population and fitnesses. This
      is the check that the shared edits to hpga/operators.py (the two
      optional `fitness` kwargs and next_generation's pass-through) and the
      new genome_model value changed nothing. B uses genome_model="sequence",
      byte-identical to how the published run was configured.
  F   the fitness-aware arm: genome_model="sequence_fitness",
      HPGA_OPERATOR_MODE=llm.

WHAT IS INSTRUMENTED, and why each one is here rather than derived later:

  * invalid-output rate and fallback-to-deterministic rate: already counted by
    operators.py's stats (n_llm_requests vs n_llm_calls, n_retries, n_failures)
    and per-attempt in the JSONL call log's `valid` field. Reported per operator
    via base.OpTally, same keys as arm C so the two are directly comparable.
  * tokens in/out per call: operators.py's total_tokens_in/out, per operator.
  * which fitness case each call was: `fitness_basis` ('own' vs 'inherited'),
    `n_parents_shown` and `fitness_values_shown`, merged into every call-log
    record by the operator entry points. The two mutate cases are different
    treatments, so an analysis that pooled them would be measuring a mixture.
  * per-call payoff, OFF the fold budget: the fitness of every mutate call's
    BASE (the post-crossover child) is what decides whether a proposal helped,
    and the GA never folds it -- it folds the mutated child one generation
    later. So after the GA finishes, every base not already in the run's cache
    is folded in a SEPARATE measurement cache that never enters selection, the
    distinct-evaluation count or the best-so-far curve, and is written to
    results/raw/fitness_prompt_bases_<arm>_seed<n>.json -- never into the run's
    own file. Cost is reported as fold time (measurement_fold_s) and as a share
    of total fold time, not GPU time.

  Inertness is asserted twice per run, as in run_move_class.py:
    (1) a digest of the GA's outputs taken before any measurement fold must
        equal the digest after, and the run's cache must be unchanged;
    (2) the GA is replayed from the seed with recording off and fitness served
        ONLY from the run's own cache (a genome it never folded raises), and
        the replay's digest must equal the run's. For arm F the replay also
        substitutes the RECORDED operator outputs instead of calling Ollama --
        an LLM run is not reproducible by re-prompting, so the replay proves
        the trajectory follows from (seed + recorded outputs) and that the
        measurement folds did not perturb it.

  (launch with PYTHONHASHSEED=0, as the earlier drivers)
  check-llm --seeds 0              one generation of arm F against the live
                                   model: the gate before a real launch. Prints
                                   one prompt as sent, the fitness log fields as
                                   they landed, the invalid-output and fallback
                                   rates, and mutate_basis_own_share. Its call
                                   log goes to --out-dir, not results/raw.
  run --arms B F --seeds 0 1 2     one result file per (arm, seed), written
                                   atomically; finished runs are skipped
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

import run_move_class as mcd  # noqa: E402  -- digest / measure / CacheOnlyModel / NoResidency
import run_sequence_ga_comparison as base  # noqa: E402
import torch  # noqa: E402

from esmfold import tm_fitness as tf  # noqa: E402
from hpga import genome_model, operators as ops  # noqa: E402
from hpga import sequence_model as sm  # noqa: E402
from hpga import sequence_model_fitness as smf  # noqa: E402
from hpga.config import HPGAConfig  # noqa: E402

log = logging.getLogger("fitprompt")

ARMS = ("B", "F")
GENOME_MODEL = {"B": "sequence", "F": "sequence_fitness"}
USES_LLM = {"B": False, "F": True}


# ------------------------------------------------------------------ wrappers

class Recorder:
    """Every operator call's inputs, outputs and fitness-line case."""

    def __init__(self):
        self.mutate, self.crossover = [], []
        self.step = {"gen": 0}

    def note_mutate(self, genome, out, fitness) -> None:
        self.mutate.append({"step": self.step["gen"], "base": genome, "child": out,
                            "changed": genome != out, **_fitness_log(fitness)})

    def note_crossover(self, p1, p2, out, fitness) -> None:
        c1, c2 = out
        self.crossover.append({"step": self.step["gen"], "parent1": p1, "parent2": p2,
                               "child1": c1, "child2": c2,
                               "child1_is_parent": c1 in (p1, p2), "child2_is_parent": c2 in (p1, p2),
                               **_fitness_log(fitness)})

    def replay_queues(self) -> dict:
        """Recorded outputs in call order, for the arm-F replay."""
        return {"mutate": [m["child"] for m in self.mutate],
                "crossover": [(c["child1"], c["child2"]) for c in self.crossover]}


def _fitness_log(fitness) -> dict:
    return fitness.log_fields() if fitness is not None else {"fitness_shown": False}


_ORIG = {}


def install_wrappers(tally: base.OpTally, rec: Recorder) -> None:
    """base.install_llm_wrappers plus recording. Sets the per-operator prompt
    style exactly as arm C does (one HPGA_LLM_PROMPT_STYLE value cannot name
    both styles for the sequence model), delegates unchanged, tallies the stats
    deltas and wall time, and records the call."""
    _ORIG.setdefault("crossover", ops._llm_crossover)
    _ORIG.setdefault("mutate", ops._llm_mutate)

    def make(op):
        orig = _ORIG[op]

        def wrapped(*args, **kwargs):
            os.environ["HPGA_LLM_PROMPT_STYLE"] = base.STYLES[op]
            before = ops.get_operator_stats()
            t0 = time.perf_counter()
            out = orig(*args, **kwargs)
            wall = time.perf_counter() - t0
            after = ops.get_operator_stats()
            t = tally.d[op]
            t["wrapper_calls"] += 1
            t["wall_s"] += wall
            for k in base.STAT_KEYS:
                t[k] += (after[k] or 0) - (before[k] or 0)
            if op == "mutate":
                rec.note_mutate(args[0], out, kwargs.get("fitness"))
            else:
                rec.note_crossover(args[0], args[1], out, kwargs.get("fitness"))
            return out

        return wrapped

    ops._llm_crossover, ops._llm_mutate = make("crossover"), make("mutate")


def install_replay_wrappers(queues: dict) -> None:
    """Arm F's replay: serve the recorded operator outputs in call order
    instead of prompting Ollama. Running out of recorded outputs, or being
    asked for one in a different order, is a divergence."""
    _ORIG.setdefault("crossover", ops._llm_crossover)
    _ORIG.setdefault("mutate", ops._llm_mutate)
    cursor = {"mutate": 0, "crossover": 0}

    def make(op):
        def wrapped(*args, **kwargs):
            i = cursor[op]
            if i >= len(queues[op]):
                raise RuntimeError(f"replay diverged: more {op} calls than the run recorded ({i})")
            cursor[op] = i + 1
            return queues[op][i]

        return wrapped

    ops._llm_crossover, ops._llm_mutate = make("crossover"), make("mutate")


def uninstall_wrappers() -> None:
    if _ORIG:
        ops._llm_crossover, ops._llm_mutate = _ORIG["crossover"], _ORIG["mutate"]
    os.environ.pop("HPGA_LLM_PROMPT_STYLE", None)


# ---------------------------------------------------------------------- loop

def ga_loop(arm, seed, cfg, model, fit_model, ref, res, llm, rec=None, partial=None):
    """base.run_ga's loop. `rec` records operator calls (None on the replay);
    `fit_model` supplies evaluate_fitness (the real model, or mcd.CacheOnlyModel)."""
    ev = base.Evaluator(fit_model)
    rng = random.Random(seed)
    population = ops.random_population(cfg.pop_size, None, rng)
    gens, integrity, breed_s = [], [], 0.0
    for gen in range(cfg.n_generations):
        if llm or gen == 0:
            res.to_fold_phase()
            if llm and gen in base.SWAP_CHECK_GENS and ev.cache and fit_model is model:
                g0 = next(iter(ev.cache))
                again = model.evaluate_fitness(g0)  # not counted: an integrity re-fold after the swap
                integrity.append({"generation": gen, "before": ev.cache[g0], "after": again,
                                  "identical": again == ev.cache[g0]})
                if again != ev.cache[g0]:
                    raise RuntimeError(f"ESMFold CPU/GPU swap changed a fitness: {ev.cache[g0]} -> {again}")
        ev.tag = gen
        n_before, hits_before = len(ev.cache), ev.hits
        fits = [ev.score(g) for g in population]
        b = max(range(len(population)), key=fits.__getitem__)
        rg = {"generation": gen, "best_fitness": fits[b], "mean_fitness": statistics.mean(fits),
              "min_fitness": min(fits), "best_genome": population[b], "best_length": len(population[b]),
              "best_edit_distance_to_reference": sm.edit_distance(population[b], ref),
              "mean_pairwise_edit_distance": base.mean_pairwise_edit(population),
              "n_distinct_in_population": len(set(population)), "mean_length": statistics.mean(map(len, population)),
              "new_distinct_evaluations": len(ev.cache) - n_before, "cache_hits": int(ev.hits - hits_before),
              "distinct_evaluations_so_far": len(ev.cache), "best_so_far": ev.best,
              "population": population, "fitnesses": fits, "breed_s": None}
        if gen < cfg.n_generations - 1:
            if llm:
                res.to_llm_phase()
            if rec is not None:
                rec.step["gen"] = gen
            t = time.perf_counter()
            population = ops.next_generation(population, fits, cfg.pop_size, cfg.tournament_k, cfg.crossover_rate,
                                             cfg.mutation_rate, cfg.elitism, rng)
            rg["breed_s"] = time.perf_counter() - t
            breed_s += rg["breed_s"]
        gens.append(rg)
        if partial is not None:
            log.info("%s seed %d gen %2d best=%.4f mean=%.4f distinct=%d hits=%d div=%.1f", arm, seed, gen,
                     rg["best_fitness"], rg["mean_fitness"], len(ev.cache), int(ev.hits),
                     rg["mean_pairwise_edit_distance"])
            partial(gens, ev)
    return ev, gens, breed_s, integrity


def reference_match(arm: str, seed: int, ev, gens) -> dict | None:
    """Arm B only: byte-identity against the published arm-B run."""
    if arm != "B":
        return None
    return mcd.reference_match(seed, ev, gens)


def fitness_line_summary(rec: Recorder) -> dict:
    """How the arm's treatment actually landed: the two mutate cases are
    different prompts, so their counts are part of the result, not a detail."""
    basis = [m.get("fitness_basis") for m in rec.mutate if m.get("fitness_shown")]
    n = len(rec.mutate)
    return {
        "n_mutate_calls": n,
        "n_crossover_calls": len(rec.crossover),
        "mutate_fitness_shown": sum(1 for m in rec.mutate if m.get("fitness_shown")),
        "mutate_basis_own": basis.count("own"),
        "mutate_basis_inherited": basis.count("inherited"),
        "mutate_basis_own_share": (basis.count("own") / n) if n else None,
        "crossover_fitness_shown": sum(1 for c in rec.crossover if c.get("fitness_shown")),
        "fitness_decimals": smf._DP,
    }


# ----------------------------------------------------------------------- run

def run_arm(arm: str, seed: int, args, path: Path) -> dict:
    llm = USES_LLM[arm]
    rec_out = base.base_record(arm, seed, args)
    base.clean_env()
    os.environ["HPGA_OPERATOR_MODE"] = "llm" if llm else "deterministic"
    cfg = HPGAConfig(genome_model=GENOME_MODEL[arm], pop_size=args.pop_size,
                     n_generations=args.generations, seed=seed)
    model = genome_model.build_genome_model(cfg)
    genome_model.set_active(model)
    ref = tf.reference_sequence()
    res, tally, rec = base.Residency(), base.OpTally(), Recorder()
    rec_out["config"].update({
        "genome_model": cfg.genome_model,
        "llm_styles": base.STYLES if llm else None,
        "fitness_prompt": ("labelled fitness line per sequence; mutate 'Fitness of this sequence' when the "
                           "child equals a parent (exact) else 'Parent fitness scores' (lineage); crossover "
                           "'Parent N fitness' (exact)") if llm else None,
        "fitness_decimals": smf._DP if llm else None,
        "reference_arm": "reproduces results/raw/sequence_ga_cmp_B_seed<n>.json exactly" if arm == "B" else None,
    })
    if llm:
        os.environ["HPGA_RUN_ID"] = f"fitprompt_{arm}_seed{seed}_{int(time.time())}"
        ops.reset_operator_stats()
        install_wrappers(tally, rec)
        rec_out["llm_call_log"] = base.rel(ops._log_path())

    partial_path = path.with_suffix(".partial.json")

    def partial(gens, ev):
        base.atomic_write_json(partial_path, {**rec_out, "generations": gens,
                                             "best_so_far_by_distinct_evaluation": ev.curve})

    t_start = time.perf_counter()
    try:
        ev, gens, breed_s, integrity = ga_loop(arm, seed, cfg, model, model, ref, res, llm, rec, partial)
    finally:
        if llm:
            uninstall_wrappers()
    wall = time.perf_counter() - t_start
    before = mcd.digest(ev, gens)

    # --- off-budget measurement: after the GA, separate cache, separate file
    run_cache_snapshot = dict(ev.cache)
    measured, mstats = mcd.measure(rec.mutate, run_cache_snapshot, model)
    after = mcd.digest(ev, gens)
    if after != before or ev.cache != run_cache_snapshot:
        raise RuntimeError(f"measurement changed the run: {before} vs {after}")

    # --- replay: recording off, fitness from the run's own cache only
    if llm:
        install_replay_wrappers(rec.replay_queues())
    try:
        replay_ev, replay_gens, _, _ = ga_loop(arm, seed, cfg, model, mcd.CacheOnlyModel(run_cache_snapshot),
                                               ref, mcd.NoResidency(), llm=False)
    finally:
        if llm:
            uninstall_wrappers()
    if mcd.digest(replay_ev, replay_gens) != before:
        raise RuntimeError(f"replay without measurement differs: {before} vs {mcd.digest(replay_ev, replay_gens)}")

    ref_match = reference_match(arm, seed, ev, gens)
    if ref_match is not None and not ref_match["identical"]:
        raise RuntimeError(f"{arm} does not reproduce {ref_match['reference_file']}: {ref_match['checks']}")

    bases_path = Path(args.out_dir) / f"fitness_prompt_bases_{arm}_seed{seed}.json"
    base.atomic_write_json(bases_path, {
        "arm": arm, "seed": seed, "run_file": base.rel(path),
        "note": ("measurement only: these folds never entered the run's selection, distinct count or "
                 "best-so-far curve"),
        **mstats, "ga_fold_s": ev.fitness_s,
        "measurement_share_of_fold_time": mstats["measurement_fold_s"] / (mstats["measurement_fold_s"] + ev.fitness_s),
        "mutate_calls": measured, "crossover_calls": rec.crossover})

    llm_s = sum(t["wall_s"] for t in tally.d.values())
    extra = {"breed_s_total": breed_s, "n_swaps": res.n_swaps, "swap_integrity_checks": integrity,
             "n_population_evaluations": cfg.pop_size * cfg.n_generations,
             "bases_file": base.rel(bases_path),
             "fitness_line": fitness_line_summary(rec) if llm else None,
             "inertness": {"digest_before_measurement": before, "digest_after_measurement_identical": True,
                           "replay_identical": True,
                           "replay_kind": "recorded operator outputs" if llm else "from seed"},
             "reference_match": ref_match}
    if llm:
        extra["operators"] = tally.summary()
        st = ops.get_operator_stats()
        extra["operator_stats_total"] = {k: st[k] for k in base.STAT_KEYS}
        n_calls = sum(t["n_llm_calls"] for t in tally.d.values())
        extra["fallback_rate_all_llm_calls"] = (
            sum(t["n_failures"] for t in tally.d.values()) / n_calls) if n_calls else None
        extra["requests_per_call_all_llm_calls"] = (
            sum(t["n_llm_requests"] for t in tally.d.values()) / n_calls) if n_calls else None
        extra["invalid_output_rate_all_attempts"] = (
            st["n_retries"] / st["n_llm_requests"]) if st["n_llm_requests"] else None
        extra["tokens_per_call"] = {
            "in": st["total_tokens_in"] / n_calls if n_calls else None,
            "out": st["total_tokens_out"] / n_calls if n_calls else None}
    rec_out.update({"complete": True, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "generations": gens,
                    "best_so_far_by_distinct_evaluation": ev.curve,
                    "generation_of_distinct_evaluation": ev.curve_gen,
                    "summary": base.finish_summary(ev, ref, wall, res, llm_s, extra)})
    partial_path.unlink(missing_ok=True)
    log.info("%s seed %d done: best=%.4f distinct=%d measurement folds=%d (%.0fs) ref_match=%s",
             arm, seed, ev.best, len(ev.cache), mstats["n_measurement_folds"], mstats["measurement_fold_s"],
             None if ref_match is None else ref_match["identical"])
    return rec_out


def out_path(args, arm: str, seed: int) -> Path:
    return Path(args.out_dir) / f"fitness_prompt_{arm}_seed{seed}.json"


def cmd_check_llm(args) -> int:
    """One generation of arm F against the LIVE model -- the gate before a real
    launch, the counterpart of run_sequence_ga_comparison.py's check-llm for arm
    C. Reports one prompt exactly as sent, the fitness log fields as they landed,
    the per-operator invalid-output and fallback rates, and the share of mutate
    calls that got the exact ('own') label rather than the lineage one. Writes
    its call log to --out-dir, not results/raw, so a gate run leaves no artefact
    among the published logs."""
    seed = args.seeds[0]
    rec = base.base_record("F", seed, args)
    rec["arm"], rec["purpose"] = "F-llm-check", "one generation, fitness-aware LLM operators"
    base.clean_env()
    base.wait_gpu_not_used_by_others()
    base.unload_ollama()
    os.environ["HPGA_OPERATOR_MODE"] = "llm"
    os.environ["HPGA_RUN_ID"] = f"fitprompt_llmcheck_seed{seed}_{int(time.time())}"
    log_path = Path(args.out_dir) / f"llm_operator_calls_{os.environ['HPGA_RUN_ID']}.jsonl"
    os.environ["HPGA_LLM_LOG_PATH"] = str(log_path)
    cfg = HPGAConfig(genome_model="sequence_fitness", pop_size=args.pop_size,
                     n_generations=args.generations, seed=seed)
    model = genome_model.build_genome_model(cfg)
    genome_model.set_active(model)
    ref = tf.reference_sequence()
    res, tally, rc, ev = base.Residency(), base.OpTally(), Recorder(), base.Evaluator(model)
    ops.reset_operator_stats()
    install_wrappers(tally, rc)
    try:
        rng = random.Random(seed)
        population = ops.random_population(cfg.pop_size, None, rng)
        res.to_fold_phase()
        t0 = time.perf_counter()
        fits = [ev.score(g) for g in population]
        fold_s = time.perf_counter() - t0
        res.to_llm_phase()
        t0 = time.perf_counter()
        ops.next_generation(population, fits, cfg.pop_size, cfg.tournament_k, cfg.crossover_rate,
                            cfg.mutation_rate, cfg.elitism, rng)
        breed_s = time.perf_counter() - t0
    finally:
        uninstall_wrappers()
        os.environ.pop("HPGA_LLM_LOG_PATH", None)

    st = ops.get_operator_stats()
    line = fitness_line_summary(rc)
    rec.update({"complete": True, "fold_s": fold_s, "breed_s": breed_s,
                "operators": tally.summary(), "operator_stats_total": {k: st[k] for k in base.STAT_KEYS},
                "fitness_line": line, "call_log": str(log_path),
                "invalid_output_rate_all_attempts": (st["n_retries"] / st["n_llm_requests"])
                if st["n_llm_requests"] else None,
                "fallback_rate_all_llm_calls": (st["n_failures"] / st["n_llm_calls"])
                if st["n_llm_calls"] else None})
    base.atomic_write_json(Path(args.out_dir) / f"fitness_prompt_llm_check_seed{seed}.json", rec)

    records = [json.loads(ln) for ln in open(log_path)]
    print(f"\n{'=' * 78}\nONE PROMPT EXACTLY AS SENT TO {ops.LLM_MODEL}\n{'=' * 78}")
    shown = next((r for r in records if r.get("op") == "mutate"
                  and r.get("fitness_basis") == "inherited" and "prompt" in r), None) \
        or next(r for r in records if r.get("op") == "mutate" and "prompt" in r)
    # _run_llm_op's log record does not include the system prompt; it is the
    # unchanged sequence_model.py constant, printed here so what is shown is the
    # complete pair actually sent.
    print(f"--- system:\n{sm._MUTATE_SYSTEM}")
    print(f"--- prompt:\n{shown['prompt']}")
    print(f"--- response:\n{shown.get('response')!r}")
    print(f"\n{'=' * 78}\nLOG FIELDS AS THEY LANDED (one mutate, one crossover)\n{'=' * 78}")
    for op in ("mutate", "crossover"):
        r = next((x for x in records if x.get("op") == op and "prompt" in x), None)
        if r:
            print(f"{op}: " + json.dumps({k: v for k, v in r.items()
                                          if k not in ("prompt", "response", "system")}, indent=1))
    print(f"\n{'=' * 78}\nRATES FOR THIS GENERATION\n{'=' * 78}")
    for op, t in tally.summary().items():
        print(f"  {op:10s} calls={t['n_llm_calls']:3d} requests={t['n_llm_requests']:3d} "
              f"retries={t['n_retries']:3d} fallbacks={t['n_failures']:3d} "
              f"fallback_rate={t['fallback_rate']} tokens_in/call="
              f"{t['total_tokens_in'] / t['n_llm_calls']:.0f} tokens_out/call="
              f"{t['total_tokens_out'] / t['n_llm_calls']:.0f}" if t['n_llm_calls'] else f"  {op}: no calls")
    print(f"  invalid-output rate (retries/requests): {rec['invalid_output_rate_all_attempts']}")
    print(f"  fallback-to-deterministic rate:         {rec['fallback_rate_all_llm_calls']}")
    print(f"  mutate_basis_own_share:                 {line['mutate_basis_own_share']} "
          f"({line['mutate_basis_own']} own / {line['mutate_basis_inherited']} inherited "
          f"of {line['n_mutate_calls']})")
    print(f"  fold_s={fold_s:.0f}  breed_s={breed_s:.0f}  call log: {log_path}")
    return 0


def cmd_run(args) -> int:
    plan = [(arm, seed) for seed in args.seeds for arm in args.arms]
    status_path = Path(args.out_dir) / "fitness_prompt_status.json"
    status = {"plan": plan, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "done": [], "failed": [],
              "skipped": [], "finished": False, "exit_reason": None}
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
                    status["done"].append({"arm": arm, "seed": seed, "attempt": attempt,
                                           "seconds": time.time() - t0,
                                           "final_best": rec["summary"]["final_best"]})
                    break
                except (base.GpuBusy, base.DiskLow):
                    raise
                except Exception:  # noqa: BLE001
                    err = traceback.format_exc()
                    log.error("%s seed %d attempt %d crashed:\n%s", arm, seed, attempt, err)
                    torch.cuda.empty_cache()
                    if attempt == 2 or any(s in err for s in
                                           ("does not reproduce", "changed the run", "replay diverged",
                                            "replay without measurement differs")):
                        path.with_suffix(".error.txt").write_text(err)
                        status["failed"].append({"arm": arm, "seed": seed,
                                                 "error": err.strip().splitlines()[-1]})
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
    ap.add_argument("run_cmd", choices=["run", "check-llm"])
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", default=["F"], choices=list(ARMS))
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
    # HF_HOME must point at the /scratch cache that already holds ESMFold's 8.4GB
    # checkpoint. A non-interactive shell (nohup, cron, ssh without a login
    # shell) does not source ~/.bashrc and so does not inherit it; transformers
    # then silently re-downloads into ~/.cache/huggingface on AFS, fills the AFS
    # quota, and fails with a corrupt-checkpoint error from the truncated file
    # -- 20 minutes after launch, with the real cause three layers down a torch
    # traceback. Checked here instead, before anything loads.
    hf_home = os.environ.get("HF_HOME", "")
    if not hf_home.startswith("/scratch/"):
        raise SystemExit(
            f"HF_HOME={hf_home!r} is not under /scratch -- export HF_HOME=/scratch/pcanaste/cache/huggingface "
            f"(and XDG_CACHE_HOME / TORCH_HOME, see ~/.bashrc) before launching, or ESMFold will be "
            f"re-downloaded into the AFS home directory and overrun its quota"
        )
    sys.exit(cmd_check_llm(args) if args.run_cmd == "check-llm" else cmd_run(args))


if __name__ == "__main__":
    main()
