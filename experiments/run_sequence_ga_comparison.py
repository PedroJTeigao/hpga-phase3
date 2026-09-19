"""Sequence-mode search comparison: A random search, B GA with deterministic operators,
C GA with LLM operators (mutate=position, crossover=segment). Fitness = TM-score vs 7UR7
(esmfold/tm_fitness.py). In-process, one warm predictor -- see run_sequence_ga_smoke.py
for why this does not go through Island.

Subcommands
  check-llm   one generation, LLM operators, per-operator fallback rate (the gate for arm C)
  run         every (arm, seed) in sequence; one result file per (arm, seed), written
              atomically the moment that run finishes, so a crash loses at most one run.
              Order: for each seed, B then C then A (A runs up to the largest distinct-evaluation
              count of B/C for that seed). Finished runs are skipped on restart.

Conventions (decisions, also recorded in every result file):
  * n_generations counts EVALUATED populations, as Island.run does: generation 0 is the random
    start, so G generations = G evaluations of a population and G-1 breeding steps.
  * "distinct evaluations" = distinct genome strings folded in this run. Fitness is
    deterministic, so a genome seen again is served from a per-run cache (a "cache hit"), which
    costs no fold and does not advance the distinct count. The cache is per run, never shared, so
    wall times are comparable across arms.
  * Arm A draws genomes with model.random_genome(rng, None) from Random(seed) -- exactly how
    random_population draws them, so A's first pop_size draws ARE the GA arms' generation 0.
  * Arm C: mutate style 'position' and crossover style 'segment' are set per call by wrapping
    ops._llm_mutate / ops._llm_crossover (one HPGA_LLM_PROMPT_STYLE value cannot name both for the
    sequence model: 'best' is not implemented there). Nothing in hpga/ is changed for this.
  * GPU: ESMFold (~13.7GB fp32) and the Ollama model cannot both be resident on the 15GB T4. Arm C
    alternates: before folding it unloads Ollama (keep_alive=0) and moves ESMFold to the GPU; before
    the LLM calls it moves ESMFold to CPU RAM. That swap time is reported separately, and the
    Ollama reload lands in the first LLM call of each breeding step (counted as LLM time). Every few
    swaps a cached genome is re-folded and must give the identical number.
  * Wall time is split: fitness (folds, cache hits excluded), LLM (time inside the two operator
    wrappers, retries and Ollama reloads included), swap (GPU residency changes), everything else.
  * GPU held by anyone but this process or this user's own Ollama: wait, re-check every 10 minutes,
    give up after 2 hours (exit code 3). Free disk < 2GB: stop launching (exit code 4).
"""

import argparse
import json
import logging
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import transformers  # noqa: E402

from esmfold import tm_fitness as tf  # noqa: E402
from hpga import genome_model  # noqa: E402
from hpga import operators as ops  # noqa: E402
from hpga import sequence_model as sm  # noqa: E402
from hpga.config import HPGAConfig  # noqa: E402

log = logging.getLogger("seqcmp")
RAW = ROOT / "results" / "raw"
OWN_OLLAMA_PREFIX = "/scratch/pcanaste/ollama"
GPU_WAIT_POLL_S, GPU_WAIT_MAX_S = 600, 7200
MIN_FREE_DISK_GB = 2.0
SWAP_CHECK_GENS = (1, 6, 11, 16)
STYLES = {"mutate": "position", "crossover": "segment"}
STAT_KEYS = ("n_llm_calls", "n_llm_requests", "n_retries", "n_failures", "n_skipped_no_op",
             "total_latency_s", "total_tokens_in", "total_tokens_out")


class GpuBusy(RuntimeError):
    pass


class DiskLow(RuntimeError):
    pass


# ----------------------------------------------------------------- environment

def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<failed: {exc!r}>"


def gpu_apps() -> list[tuple[int, float, str]]:
    rows = []
    out = _sh(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader,nounits"])
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            try:
                rows.append((int(parts[0]), float(parts[1]), ",".join(parts[2:])))
            except ValueError:
                pass
    return rows


def gpu_used_mib() -> float:
    try:
        return float(_sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).splitlines()[0])
    except (ValueError, IndexError):
        return float("nan")


def _is_own(app) -> bool:
    return app[0] == os.getpid() or app[2].startswith(OWN_OLLAMA_PREFIX)


def wait_gpu_not_used_by_others() -> None:
    """Block while the GPU is held by something that is neither this process nor this user's own
    Ollama. Never touches the other process. Raises GpuBusy after GPU_WAIT_MAX_S."""
    waited = 0
    while True:
        apps = gpu_apps()
        others = [a for a in apps if not _is_own(a)]
        own_mib = sum(a[1] for a in apps if _is_own(a))
        if not any(a[0] == os.getpid() for a in apps) and torch.cuda.is_initialized():
            own_mib += torch.cuda.memory_reserved() / 2**20 + 500  # this process, if nvidia-smi doesn't list it
        unexplained = gpu_used_mib() - own_mib
        if not others and not unexplained > 1500:
            return
        log.warning("GPU held by others (apps=%s, unexplained %.0f MiB); waited %ds of %ds", others, unexplained,
                    waited, GPU_WAIT_MAX_S)
        if waited >= GPU_WAIT_MAX_S:
            raise GpuBusy(f"GPU still in use by others after {waited}s: {others}")
        time.sleep(GPU_WAIT_POLL_S)
        waited += GPU_WAIT_POLL_S


def check_disk() -> None:
    free_gb = shutil.disk_usage(RAW).free / 1e9
    if free_gb < MIN_FREE_DISK_GB:
        raise DiskLow(f"only {free_gb:.1f} GB free")


def unload_ollama(timeout_s: float = 180) -> None:
    """Ask this user's Ollama to drop the model (keep_alive=0) and wait for its runner to leave the
    GPU. A documented API call, not a process kill; Ollama reloads on the next request."""
    try:
        ops._get_client().generate(model=ops.LLM_MODEL, prompt="", keep_alive=0)
    except Exception as exc:  # noqa: BLE001 -- server down/no model loaded: nothing to unload
        log.info("ollama unload request: %r", exc)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if not any(a[2].startswith(OWN_OLLAMA_PREFIX) for a in gpu_apps()):
            return
        time.sleep(2)
    log.warning("Ollama runner still on the GPU after %.0fs", timeout_s)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


# -------------------------------------------------------------- GPU residency

class Residency:
    """Which of ESMFold / Ollama holds the GPU. Only arm C ever swaps."""

    def __init__(self):
        self.swap_s = 0.0
        self.n_swaps = 0

    @staticmethod
    def esm():
        return sm._get_fitness().predictor.model

    def to_fold_phase(self) -> None:
        t = time.perf_counter()
        unload_ollama()
        m = self.esm()
        if next(m.parameters()).device.type != "cuda":
            m.cuda()
            torch.cuda.synchronize()
        self.swap_s += time.perf_counter() - t
        self.n_swaps += 1

    def to_llm_phase(self) -> None:
        t = time.perf_counter()
        m = self.esm()
        if next(m.parameters()).device.type == "cuda":
            m.cpu()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        self.swap_s += time.perf_counter() - t
        self.n_swaps += 1
        used = gpu_used_mib()
        if used > 3000:
            log.warning("GPU still holds %.0f MiB after moving ESMFold to CPU", used)


# ------------------------------------------------------------------ evaluation

class Evaluator:
    """Per-run fitness cache keyed on the genome string. Counts distinct evaluations (folds) and cache
    hits separately and keeps best-so-far after every distinct evaluation."""

    def __init__(self, model):
        self.model, self.cache = model, {}
        self.hits = self.fitness_s = 0.0
        self.best, self.best_genome = float("-inf"), None
        self.curve = []  # curve[i] = best-so-far after i+1 distinct evaluations
        self.curve_gen = []  # generation (or draw index) in which distinct evaluation i happened
        self.tag = 0

    def score(self, genome: str) -> float:
        if genome in self.cache:
            self.hits += 1
            return self.cache[genome]
        t = time.perf_counter()
        f = self.model.evaluate_fitness(genome)
        self.fitness_s += time.perf_counter() - t
        self.cache[genome] = f
        if f > self.best:
            self.best, self.best_genome = f, genome
        self.curve.append(self.best)
        self.curve_gen.append(self.tag)
        return f


def mean_pairwise_edit(pop: list[str]) -> float:
    d = [sm.edit_distance(a, b) for i, a in enumerate(pop) for b in pop[i + 1:]]
    return statistics.mean(d) if d else 0.0


# -------------------------------------------------------------- LLM operators

class OpTally:
    def __init__(self):
        self.d = {op: {"wrapper_calls": 0, "wall_s": 0.0, **{k: 0 for k in STAT_KEYS}} for op in STYLES}

    def summary(self) -> dict:
        out = {}
        for op, t in self.d.items():
            n = t["n_llm_calls"]
            out[op] = {**t, "style": STYLES[op],
                       "fallback_rate": (t["n_failures"] / n) if n else None,
                       "requests_per_call": (t["n_llm_requests"] / n) if n else None,
                       "gated_off_share": (t["n_skipped_no_op"] / t["wrapper_calls"]) if t["wrapper_calls"] else None}
        return out


_ORIG = {}


def install_llm_wrappers(tally: OpTally) -> None:
    """Wrap ops._llm_crossover / ops._llm_mutate (next_generation resolves them as module globals at call
    time): set this operator's prompt style, delegate unchanged, and tally the stats deltas and wall time."""
    _ORIG.setdefault("crossover", ops._llm_crossover)
    _ORIG.setdefault("mutate", ops._llm_mutate)

    def make(op):
        orig = _ORIG[op]

        def wrapped(*args, **kwargs):
            os.environ["HPGA_LLM_PROMPT_STYLE"] = STYLES[op]
            before = ops.get_operator_stats()
            t0 = time.perf_counter()
            out = orig(*args, **kwargs)
            wall = time.perf_counter() - t0
            after = ops.get_operator_stats()
            t = tally.d[op]
            t["wrapper_calls"] += 1
            t["wall_s"] += wall
            for k in STAT_KEYS:
                t[k] += (after[k] or 0) - (before[k] or 0)
            return out

        return wrapped

    ops._llm_crossover, ops._llm_mutate = make("crossover"), make("mutate")


def uninstall_llm_wrappers() -> None:
    if _ORIG:
        ops._llm_crossover, ops._llm_mutate = _ORIG["crossover"], _ORIG["mutate"]
    os.environ.pop("HPGA_LLM_PROMPT_STYLE", None)


def clean_env() -> None:
    for var in ("HPGA_AGENTS_ENABLED", "HPGA_CIRCLES_ENABLED", "HPGA_LOG_DIVERSITY", "HPGA_GA_DISPATCH_P",
                "HPGA_LLM_PROMPT_STYLE"):
        os.environ.pop(var, None)


# ---------------------------------------------------------------------- runs

def base_record(arm: str, seed: int, args) -> dict:
    return {
        "arm": arm, "seed": seed, "complete": False, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {"pop_size": args.pop_size, "n_generations": args.generations, "crossover_rate": 0.9,
                   "mutation_rate": 0.05, "tournament_k": 3, "elitism": 2,
                   "length_bounds": [sm.MIN_LENGTH, sm.MAX_LENGTH], "llm_model": ops.LLM_MODEL,
                   "llm_temperature": ops.LLM_TEMPERATURE, "llm_styles": STYLES if arm == "C" else None,
                   "generations_meaning": "evaluated populations (generation 0 = random start), as Island.run"},
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "transformers": transformers.__version__, "tmalign_bin": str(tf.TMALIGN_BIN),
                "git_head": _sh(["git", "-C", str(ROOT), "rev-parse", "HEAD"])},
    }


def finish_summary(ev: Evaluator, ref: str, wall: float, res: Residency | None, llm_s: float, extra: dict) -> dict:
    swap_s = res.swap_s if res else 0.0
    return {
        "final_best": ev.best, "final_best_genome": ev.best_genome, "final_best_length": len(ev.best_genome),
        "final_best_edit_distance_to_reference": sm.edit_distance(ev.best_genome, ref),
        "n_distinct_evaluations": len(ev.cache), "n_cache_hits": int(ev.hits),
        "wall_s": wall, "fitness_s": ev.fitness_s, "llm_s": llm_s, "swap_s": swap_s,
        "other_s": wall - ev.fitness_s - llm_s - swap_s,
        "mean_fitness_s_per_distinct_evaluation": ev.fitness_s / max(1, len(ev.cache)),
        **extra,
    }


def run_ga(arm: str, seed: int, args, llm: bool) -> dict:
    rec = base_record(arm, seed, args)
    clean_env()
    os.environ["HPGA_OPERATOR_MODE"] = "llm" if llm else "deterministic"
    cfg = HPGAConfig(genome_model="sequence", pop_size=args.pop_size, n_generations=args.generations, seed=seed)
    model = genome_model.build_genome_model(cfg)
    genome_model.set_active(model)
    ref = tf.reference_sequence()
    res, tally, ev = Residency(), OpTally(), Evaluator(model)
    if llm:
        os.environ["HPGA_RUN_ID"] = f"seqcmp_C_seed{seed}_{int(time.time())}"
        ops.reset_operator_stats()
        install_llm_wrappers(tally)
        rec["llm_call_log"] = rel(ops._log_path())
    rng = random.Random(seed)
    population = ops.random_population(cfg.pop_size, None, rng)
    gens, integrity, breed_s = [], [], 0.0
    t_start = time.perf_counter()
    try:
        for gen in range(cfg.n_generations):
            if llm or gen == 0:
                res.to_fold_phase()
                if llm and gen in SWAP_CHECK_GENS and ev.cache:
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
            rec_gen = {
                "generation": gen, "best_fitness": fits[b], "mean_fitness": statistics.mean(fits),
                "min_fitness": min(fits), "best_genome": population[b], "best_length": len(population[b]),
                "best_edit_distance_to_reference": sm.edit_distance(population[b], ref),
                "mean_pairwise_edit_distance": mean_pairwise_edit(population),
                "n_distinct_in_population": len(set(population)), "mean_length": statistics.mean(map(len, population)),
                "new_distinct_evaluations": len(ev.cache) - n_before, "cache_hits": int(ev.hits - hits_before),
                "distinct_evaluations_so_far": len(ev.cache), "best_so_far": ev.best,
                "population": population, "fitnesses": fits, "breed_s": None,
            }
            if gen < cfg.n_generations - 1:
                if llm:
                    res.to_llm_phase()
                t = time.perf_counter()
                population = ops.next_generation(population, fits, cfg.pop_size, cfg.tournament_k, cfg.crossover_rate,
                                                 cfg.mutation_rate, cfg.elitism, rng)
                rec_gen["breed_s"] = time.perf_counter() - t
                breed_s += rec_gen["breed_s"]
            gens.append(rec_gen)
            log.info("%s seed %d gen %2d best=%.4f mean=%.4f distinct=%d hits=%d div=%.1f", arm, seed, gen,
                     rec_gen["best_fitness"], rec_gen["mean_fitness"], len(ev.cache), int(ev.hits),
                     rec_gen["mean_pairwise_edit_distance"])
    finally:
        if llm:
            uninstall_llm_wrappers()
    wall = time.perf_counter() - t_start
    llm_s = sum(t["wall_s"] for t in tally.d.values())
    extra = {"breed_s_total": breed_s, "n_swaps": res.n_swaps, "swap_integrity_checks": integrity,
             "n_population_evaluations": cfg.pop_size * cfg.n_generations}
    if llm:
        extra["operators"] = tally.summary()
        st = ops.get_operator_stats()
        extra["operator_stats_total"] = {k: st[k] for k in STAT_KEYS}
        n_calls = sum(t["n_llm_calls"] for t in tally.d.values())
        extra["fallback_rate_all_llm_calls"] = (sum(t["n_failures"] for t in tally.d.values()) / n_calls) if n_calls else None
        extra["requests_per_call_all_llm_calls"] = (sum(t["n_llm_requests"] for t in tally.d.values()) / n_calls) if n_calls else None
    rec.update({"complete": True, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "generations": gens,
                "best_so_far_by_distinct_evaluation": ev.curve, "generation_of_distinct_evaluation": ev.curve_gen,
                "summary": finish_summary(ev, ref, wall, res, llm_s, extra)})
    return rec


def run_random(seed: int, n_target: int, args) -> dict:
    rec = base_record("A", seed, args)
    clean_env()
    os.environ["HPGA_OPERATOR_MODE"] = "deterministic"
    cfg = HPGAConfig(genome_model="sequence", pop_size=args.pop_size, n_generations=args.generations, seed=seed)
    model = genome_model.build_genome_model(cfg)
    genome_model.set_active(model)
    ref = tf.reference_sequence()
    res, ev, rng = Residency(), Evaluator(model), random.Random(seed)
    t_start, draws = time.perf_counter(), 0
    res.to_fold_phase()  # inside the timed window, so swap_s is a component of wall_s as in the GA arms
    while len(ev.cache) < n_target:
        ev.tag = draws
        ev.score(model.random_genome(rng, None))
        draws += 1
        if draws % 50 == 0:
            log.info("A seed %d draws=%d distinct=%d best=%.4f", seed, draws, len(ev.cache), ev.best)
    wall = time.perf_counter() - t_start
    rec.update({"complete": True, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_target_distinct": n_target,
                "best_so_far_by_distinct_evaluation": ev.curve, "generation_of_distinct_evaluation": ev.curve_gen,
                "summary": finish_summary(ev, ref, wall, res, 0.0, {"n_draws": draws,
                                                                    "note": "draws == the GA arms' generation-0 population for the first pop_size draws"})})
    return rec


# ------------------------------------------------------------------ commands

def out_path(args, arm: str, seed: int) -> Path:
    return Path(args.out_dir) / f"sequence_ga_cmp_{arm}_seed{seed}.json"


def distinct_target(args, seed: int) -> int:
    counts = []
    for arm in ("B", "C"):
        p = out_path(args, arm, seed)
        if p.exists():
            d = json.load(open(p))
            if d.get("complete"):
                counts.append(d["summary"]["n_distinct_evaluations"])
    return max(counts) if counts else args.pop_size * args.generations  # both missing: the most a GA could use


def cmd_check_llm(args) -> None:
    rec = base_record("C", args.seed, args)
    rec["arm"], rec["purpose"] = "C-llm-check", "one generation, LLM operators; per-operator fallback rate"
    clean_env()
    wait_gpu_not_used_by_others()
    unload_ollama()
    os.environ["HPGA_OPERATOR_MODE"] = "llm"
    os.environ["HPGA_RUN_ID"] = f"seqcmp_llmcheck_seed{args.seed}_{int(time.time())}"
    cfg = HPGAConfig(genome_model="sequence", pop_size=args.pop_size, seed=args.seed)
    model = genome_model.build_genome_model(cfg)
    genome_model.set_active(model)
    ref = tf.reference_sequence()
    res, tally, ev = Residency(), OpTally(), Evaluator(model)
    ops.reset_operator_stats()
    install_llm_wrappers(tally)
    rec["llm_call_log"] = rel(ops._log_path())
    rng = random.Random(args.seed)
    population = ops.random_population(cfg.pop_size, None, rng)
    t0 = time.perf_counter()
    res.to_fold_phase()
    fits = [ev.score(g) for g in population]
    t_fold = time.perf_counter() - t0
    res.to_llm_phase()
    t1 = time.perf_counter()
    new_pop = ops.next_generation(population, fits, cfg.pop_size, cfg.tournament_k, cfg.crossover_rate,
                                  cfg.mutation_rate, cfg.elitism, rng)
    t_breed = time.perf_counter() - t1
    uninstall_llm_wrappers()
    res.to_fold_phase()  # prove the swap back works and leaves the fitness unchanged
    g0 = population[0]
    again = model.evaluate_fitness(g0)
    ops_summary = tally.summary()
    rates = {op: s["fallback_rate"] for op, s in ops_summary.items()}
    skip_c = any(r is not None and r > 0.20 for r in rates.values())
    lo, hi = sm.MIN_LENGTH, sm.MAX_LENGTH
    rec.update({
        "complete": True, "operators": ops_summary, "fallback_rate_by_operator": rates,
        "rule": "skip arm C if any operator's fallback_rate (failures / LLM calls, rate-gated crossovers excluded) > 0.20",
        "skip_arm_C": skip_c, "fold_phase_s": t_fold, "breed_s": t_breed, "swap_s": res.swap_s,
        "children_valid": len(new_pop) == cfg.pop_size and all(isinstance(g, str) and lo <= len(g) <= hi for g in new_pop),
        "swap_integrity": {"before": fits[0], "after": again, "identical": again == fits[0]},
        "population_in": population, "fitnesses_in": fits, "population_out": new_pop,
        "mean_pairwise_edit_in": mean_pairwise_edit(population), "mean_pairwise_edit_out": mean_pairwise_edit(new_pop),
        "operator_stats_total": {k: ops.get_operator_stats()[k] for k in STAT_KEYS},
        "reference_length": len(ref),
    })
    atomic_write_json(Path(args.out_dir) / "sequence_ga_llm_check.json", rec)
    print(json.dumps({k: rec[k] for k in ("fallback_rate_by_operator", "skip_arm_C", "fold_phase_s", "breed_s", "swap_s",
                                          "children_valid", "swap_integrity")}, indent=1))
    for op, s in ops_summary.items():
        print(op, {k: s[k] for k in ("style", "wrapper_calls", "n_llm_calls", "n_llm_requests", "n_retries", "n_failures",
                                     "n_skipped_no_op", "fallback_rate", "requests_per_call", "wall_s")})


def write_status(args, status: dict) -> None:
    atomic_write_json(Path(args.out_dir) / "sequence_ga_comparison_status.json", status)


def cmd_run(args) -> int:
    plan = []
    for seed in args.seeds:
        for arm in ("B", "C", "A"):
            if arm in args.arms and not (arm == "C" and args.skip_c):
                plan.append((arm, seed))
    status = {"plan": plan, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "done": [], "failed": [], "skipped": [],
              "finished": False, "exit_reason": None}
    write_status(args, status)
    log.info("plan: %s", plan)
    exit_code = 0
    try:
        wait_gpu_not_used_by_others()
        unload_ollama()  # whatever Ollama still holds from earlier work; ESMFold needs the card
        sm._get_fitness()  # load ESMFold once; kept warm for every run below
        for arm, seed in plan:
            path = out_path(args, arm, seed)
            if path.exists() and json.load(open(path)).get("complete"):
                status["skipped"].append({"arm": arm, "seed": seed, "reason": "result file already complete"})
                continue
            check_disk()
            wait_gpu_not_used_by_others()
            for attempt in (1, 2):
                t0 = time.time()
                try:
                    log.info("=== %s seed %d attempt %d ===", arm, seed, attempt)
                    if arm == "A":
                        rec = run_random(seed, distinct_target(args, seed), args)
                    else:
                        rec = run_ga(arm, seed, args, llm=(arm == "C"))
                    rec["attempt"] = attempt
                    atomic_write_json(path, rec)
                    status["done"].append({"arm": arm, "seed": seed, "attempt": attempt, "seconds": time.time() - t0,
                                           "final_best": rec["summary"]["final_best"]})
                    break
                except (GpuBusy, DiskLow):
                    raise
                except Exception:  # noqa: BLE001
                    err = traceback.format_exc()
                    log.error("%s seed %d attempt %d crashed:\n%s", arm, seed, attempt, err)
                    uninstall_llm_wrappers()
                    try:
                        torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001
                        pass
                    if attempt == 2:
                        (path.with_suffix(".error.txt")).write_text(err)
                        status["failed"].append({"arm": arm, "seed": seed, "error": err.strip().splitlines()[-1]})
            write_status(args, status)
        status["exit_reason"] = "all planned runs attempted"
    except GpuBusy as e:
        status["exit_reason"], exit_code = f"GPU_BUSY_TIMEOUT: {e}", 3
    except DiskLow as e:
        status["exit_reason"], exit_code = f"DISK_LOW: {e}", 4
    status["finished"] = True
    status["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_status(args, status)
    log.info("finished: %s", status["exit_reason"])
    return exit_code


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check-llm", "run"):
        p = sub.add_parser(name)
        p.add_argument("--pop-size", type=int, default=16)
        p.add_argument("--generations", type=int, default=20)
        p.add_argument("--out-dir", type=str, default=str(RAW))
    sub.choices["check-llm"].add_argument("--seed", type=int, default=0)
    sub.choices["run"].add_argument("--seeds", type=int, nargs="+", required=True)
    sub.choices["run"].add_argument("--arms", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    sub.choices["run"].add_argument("--skip-c", action="store_true", help="omit arm C (LLM gate failed)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for noisy in ("httpx", "esmfold.tm_fitness", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if (sm.MIN_LENGTH, sm.MAX_LENGTH) != (30, 80):
        raise SystemExit(f"length bounds are {(sm.MIN_LENGTH, sm.MAX_LENGTH)}, expected (30, 80)")
    if args.cmd == "check-llm":
        wait_gpu_not_used_by_others()
        unload_ollama()
        sm._get_fitness()
        cmd_check_llm(args)
    else:
        sys.exit(cmd_run(args))


if __name__ == "__main__":
    main()
