"""Stage 3 of the agent-memory question: does giving each agent a private record of its own past proposals change the
search? Three arms, sequence mode, everything else identical to arms C/D/E of the five-arm comparison
(run_sequence_ga_circles.py, which this imports from): pop 16, 20 evaluated populations (generation 0 = random start, so
19 breeding steps), crossover 0.9, mutation 0.05, tournament 3, elitism 2, mutate=position / crossover=segment,
gemma4:12b at temperature 0.7, length bounds [30, 80], same seeds, same generation-0 population per seed.

  M1  GA + LLM operators + 2 agents, no memory   (HPGA_AGENTS_ENABLED=1, HPGA_AGENTS_MEMORY=0: the record is kept and
                                                  logged, never shown)
  M2  GA + LLM operators + 2 agents, with memory (HPGA_AGENTS_MEMORY=1: each agent's own last 8 entries + tally in its prompt)
  M3  GA + LLM operators + 2 random immigrants   -- the control, drawn as arm E draws them (model.random_genome(rng, None))

M1 and M2 differ in exactly one thing, whether the record block is in the agent's prompt; the agents' proposals are the
mutate/position call on their own tail slot (hpga/agents_sequence.py). All three ADD 2 genomes to the population after the
ordinary fill (so the evaluated population is 18 from generation 1), the additive contract agents already have; the budget is
read at the common distinct-fold count, exactly as for D and E (see summarize_agent_memory.py).

  (launch with PYTHONHASHSEED=0 for parity with the earlier drivers)
  run --seeds 0 1 2     for each seed: M1, M2, M3; one result file per (arm, seed), written atomically; each run retried
                        once, then skipped and logged; finished runs skipped on restart. A partial file with every
                        generation completed so far is rewritten after each generation (progress and post-mortem, not resume).
  run --stub            no GPU, no Ollama: stubbed model and fitness, small population -- checks the pipeline end to end.
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

import run_sequence_ga_circles as circ  # noqa: E402  (imports run_sequence_ga_comparison as base, torch, esmfold)
import torch  # noqa: E402

base = circ.base
from esmfold import tm_fitness as tf  # noqa: E402
from hpga import agents, agents_sequence as aseq, blackboard as bb, circles, genome_model, operators as ops  # noqa: E402
from hpga import sequence_model as sm  # noqa: E402
from hpga.config import HPGAConfig  # noqa: E402

log = logging.getLogger("agmem")
N_AGENTS = 2
ARMS = ("M1", "M2", "M3")
DESCRIPTION = {"M1": "GA + LLM operators + 2 agents, no memory", "M2": "GA + LLM operators + 2 agents, private record shown",
               "M3": "GA + LLM operators + 2 random immigrants (control)"}


class NoResidency:
    """--stub: nothing to swap."""
    swap_s, n_swaps, ollama_load_s = 0.0, 0, 0.0

    def to_fold_phase(self): ...
    def to_llm_phase(self): ...


def stub_call(prompt, system, num_predict, seed):
    import hashlib
    import re

    r = random.Random(int(hashlib.sha256(f"{seed}|{prompt[-300:]}".encode()).hexdigest()[:8], 16))
    bad = r.random() < 0.05
    if "SEGMENTS:" in prompt:
        text = "SEGMENTS: junk" if bad else "SEGMENTS: 0-40:1, 40-100:2"
    else:
        k = int(re.search(r"Change exactly (\d+)", prompt).group(1))
        n = int(re.search(r"positions 0-(\d+)\)", prompt).group(1)) + 1
        cur = "".join(re.search(r"Sequence \(0-indexed positions 0-\d+\): ([A-Z ]+)", prompt).group(1).split())
        pos = r.sample(range(n), k)
        text = "junk" if bad else "\n".join(f"POSITION: {p}, NEW: {r.choice([c for c in sm.ALPHABET if c != cur[p]])}" for p in pos)
    return text, 0.01, len(prompt) // 4, len(text) // 4


def stub_fitness(g: str) -> float:
    return sum((i % 7 + 1) * (sm.ALPHABET.index(c) + 1) for i, c in enumerate(g)) % 1000 / 1000.0


def run_arm(arm: str, seed: int, args, partial_path: Path | None = None) -> dict:
    assert arm in ARMS
    rec = base.base_record(arm, seed, args)
    base.clean_env()
    for v in ("HPGA_AGENTS_MEMORY", "HPGA_N_AGENTS", "HPGA_AGENT_MEMORY_LOG_PATH", "HPGA_AGENTS_SEQ_EDIT_RATE", "HPGA_AGENTS_MEMORY_WINDOW"):
        os.environ.pop(v, None)
    os.environ.update({"HPGA_OPERATOR_MODE": "llm", "HPGA_CIRCLES_ENABLED": "0"})
    if arm in ("M1", "M2"):
        os.environ.update({"HPGA_AGENTS_ENABLED": "1", "HPGA_N_AGENTS": str(N_AGENTS),
                           "HPGA_AGENTS_MEMORY": "1" if arm == "M2" else "0"})
    else:
        os.environ["HPGA_AGENTS_ENABLED"] = "0"
    cfg = HPGAConfig(genome_model="sequence", pop_size=args.pop_size, n_generations=args.generations, seed=seed)
    model = genome_model.build_genome_model(cfg)
    genome_model.set_active(model)
    ref = tf.reference_sequence() if not args.stub else "A" * 63
    res, tally, ev = (NoResidency() if args.stub else circ.Residency()), circ.OpTally(), base.Evaluator(model)
    os.environ["HPGA_RUN_ID"] = f"agmem_{arm}_seed{seed}_{int(time.time())}"
    if args.stub:  # keep every log out of results/raw
        os.environ["HPGA_LLM_LOG_PATH"] = str(Path(args.out_dir) / f"llm_{os.environ['HPGA_RUN_ID']}.jsonl")
        os.environ["HPGA_AGENT_MEMORY_LOG_PATH"] = str(Path(args.out_dir) / f"mem_{os.environ['HPGA_RUN_ID']}.jsonl")
    ops.reset_operator_stats(); agents.reset_agent_state(); circles.reset_circle_state(); bb.reset_blackboard()
    circ.install(tally)
    rec["llm_call_log"] = base.rel(ops._log_path())
    rec["agent_memory_log"] = base.rel(aseq._log_path()) if arm in ("M1", "M2") else None
    rec["config"].update({"arm_description": DESCRIPTION[arm], "injected_per_breeding_step": N_AGENTS, "llm_styles": base.STYLES,
                          "agents": {"n_agents": N_AGENTS, "memory": arm == "M2", "record_window": 8, "edit_rate": 0.05,
                                     "proposal_format": "position edits to the agent's own tail slot (the mutate/position call)"}
                          if arm in ("M1", "M2") else None})
    rng = random.Random(seed)
    population = ops.random_population(cfg.pop_size, None, rng)
    gens, integrity, breed_s = [], [], 0.0
    t_start = time.perf_counter()
    try:
        for gen in range(cfg.n_generations):
            res.to_fold_phase()
            if not args.stub and gen in base.SWAP_CHECK_GENS and ev.cache:
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
                if arm == "M3":  # the control: N_AGENTS random immigrants, drawn as random_population draws them
                    population = population + [model.random_genome(rng, None) for _ in range(N_AGENTS)]
                rg["breed_s"] = time.perf_counter() - t
                breed_s += rg["breed_s"]
            gens.append(rg)
            log.info("%s seed %d gen %2d n=%d best=%.4f mean=%.4f distinct=%d hits=%d div=%.1f", arm, seed, gen, len(gens[-1]["population"]),
                     rg["best_fitness"], rg["mean_fitness"], len(ev.cache), int(ev.hits), rg["mean_pairwise_edit_distance"])
            if partial_path is not None:
                base.atomic_write_json(partial_path, {**rec, "complete": False, "generations": gens, "best_so_far_by_distinct_evaluation": ev.curve})
    finally:
        circ.uninstall()
    wall = time.perf_counter() - t_start
    ops_summary = tally.summary()
    llm_s = sum(t["wall_s"] for t in ops_summary.values())
    st = ops.get_operator_stats()
    n_calls = sum(t["n_llm_calls"] for t in ops_summary.values())
    extra = {"breed_s_total": breed_s, "n_swaps": res.n_swaps, "ollama_load_s": res.ollama_load_s, "swap_integrity_checks": integrity,
             "n_population_evaluations": sum(g["n_population"] for g in gens), "operators": ops_summary,
             "operator_stats_total": {k: st[k] for k in base.STAT_KEYS}, "n_llm_calls_total": n_calls,
             "fallback_rate_all_llm_calls": (sum(t["n_failures"] for t in ops_summary.values()) / n_calls) if n_calls else None,
             "requests_per_call_all_llm_calls": (sum(t["n_llm_requests"] for t in ops_summary.values()) / n_calls) if n_calls else None,
             "injected_total": N_AGENTS * (cfg.n_generations - 1)}
    if arm in ("M1", "M2"):
        events = []
        p = aseq._log_path()
        if p.exists():
            events = [json.loads(line) for line in open(p, encoding="utf-8")]
        extra["agent_state_stats"] = aseq.get_stats()
        extra["agent_events"] = events  # every proposal and every resolved record entry, timestamps included
    rec.update({"complete": True, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "generations": gens,
                "best_so_far_by_distinct_evaluation": ev.curve, "generation_of_distinct_evaluation": ev.curve_gen,
                "summary": base.finish_summary(ev, ref, wall, res, llm_s, extra)})
    return rec


def out_path(args, arm, seed) -> Path:
    return Path(args.out_dir) / f"agent_memory_{arm}_seed{seed}.json"


def cmd_run(args) -> int:
    plan = [(arm, seed) for seed in args.seeds for arm in args.arms]
    status_path = Path(args.out_dir) / "agent_memory_status.json"
    status = {"plan": plan, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "done": [], "failed": [], "skipped": [], "finished": False, "exit_reason": None}
    base.atomic_write_json(status_path, status)
    log.info("plan: %s", plan)
    code = 0
    try:
        if not args.stub:
            base.wait_gpu_not_used_by_others()
            base.unload_ollama()
            sm._get_fitness()
        for arm, seed in plan:
            path = out_path(args, arm, seed)
            if path.exists() and json.load(open(path)).get("complete"):
                status["skipped"].append({"arm": arm, "seed": seed, "reason": "result file already complete"})
                continue
            base.check_disk()
            if not args.stub:
                base.wait_gpu_not_used_by_others()
            for attempt in (1, 2):
                t0 = time.time()
                try:
                    log.info("=== %s seed %d attempt %d ===", arm, seed, attempt)
                    rec = run_arm(arm, seed, args, partial_path=path.with_suffix(".partial.json"))
                    rec["attempt"] = attempt
                    base.atomic_write_json(path, rec)
                    path.with_suffix(".partial.json").unlink(missing_ok=True)
                    status["done"].append({"arm": arm, "seed": seed, "attempt": attempt, "seconds": time.time() - t0, "final_best": rec["summary"]["final_best"]})
                    break
                except (base.GpuBusy, base.DiskLow):
                    raise
                except Exception:  # noqa: BLE001
                    err = traceback.format_exc()
                    log.error("%s seed %d attempt %d crashed:\n%s", arm, seed, attempt, err)
                    circ.uninstall()
                    if not args.stub:
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
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--pop-size", type=int, default=16)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--out-dir", type=str, default=str(base.RAW))
    ap.add_argument("--stub", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for noisy in ("httpx", "esmfold.tm_fitness", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if (sm.MIN_LENGTH, sm.MAX_LENGTH) != (30, 80):
        raise SystemExit(f"length bounds are {(sm.MIN_LENGTH, sm.MAX_LENGTH)}, expected (30, 80)")
    if args.stub:
        ops._call_ollama = stub_call
        sm.evaluate_fitness = stub_fitness
        if args.out_dir == str(base.RAW):
            raise SystemExit("--stub needs --out-dir somewhere outside results/raw")
    sys.exit(cmd_run(args))


if __name__ == "__main__":
    main()
