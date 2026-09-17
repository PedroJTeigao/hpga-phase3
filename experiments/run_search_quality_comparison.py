"""Search-quality comparison: does the LLM operator find better folds than
the deterministic operator on a real, known-optimum benchmark -- as opposed
to the timing-only comparisons elsewhere in Phase 2, which use a synthetic
sequence chosen for a non-trivial fitness-eval cost, not for meaningful
fitness values.

This is a HYPOTHESIS TEST, not a foregone conclusion: the reasoning that the
HP-lattice genome (a flat S/L/R string) has no structure for an LLM to
exploit is plausible but unverified prior to this script. Report whatever
the numbers say.

Uses the real Unger & Moult (1993) n=20 benchmark (E*=-9), same sequence as
Phase 1's run_benchmark_validation.py, so numbers are comparable in kind
(not in scale -- this runs at pop_size=8, Phase 1's real-benchmark run used
pop_size=256/n_generations=500/5 seeds; LLM cost makes that scale
infeasible here, see PHASE2_RESULTS.md for the budget worked out before
this ran).

Both arms use IDENTICAL seeds, pop_size, n_generations, n_workers, and
sequence -- only HPGA_OPERATOR_MODE differs. Caveat (stated up front, not
a bug): identical seeds guarantee an identical starting population and
identical generation-0 selection only. The LLM operator path consumes the
shared rng differently (one getrandbits(31) per attempt/retry) than the
deterministic path (per-gene draws), so the two arms' rng streams diverge
after generation 0's operators run; nothing downstream of that is a
lockstep-identical trajectory. This is a property of the existing harness
(hpga/operators.py, frozen except for the mode switch), not something this
script works around.

CHECKPOINTED / CHUNKED BY DESIGN. Two live runs of the single-shot version
of this script were killed by the environment after ~50-65 continuous
minutes (no error, no hang -- steady progress right up to the last logged
call, no reboot, Ollama healthy afterwards) -- an external constraint on
long-lived background processes, not a bug in the GA loop. A full 25-
generation LLM run at pop_size=8 takes ~62 minutes, i.e. squarely in the
kill window. Rather than fight that, this script processes a bounded
number of generations per invocation (--chunk-gens, default 8, ~14 min)
and checkpoints {population, rng state, accumulated fitness/timing/token
series} to results/raw/search_quality_checkpoints/ after every chunk, via
atomic write-then-rename so a kill mid-write can't corrupt the checkpoint.
The next invocation of the same --seed/--mode resumes from exactly where
the last one left off. Worst-case loss from an unexpected kill is now one
chunk (~14 min), not a whole seed's run (~62 min) or an entire sweep.

Usage: call once per (seed, mode) combination, repeatedly, until it reports
"done" -- see the orchestration loop this was driven from (not itself
scripted here, since each invocation needs the prior one's exit status
observed before deciding whether to launch another chunk).

    python run_search_quality_comparison.py --seed 0 --mode deterministic
    python run_search_quality_comparison.py --seed 0 --mode llm
    python run_search_quality_comparison.py --seed 0 --mode llm   # (repeat until done)

Finalized (seed, mode) results -- once generations_done reaches
N_GENERATIONS_TOTAL -- are appended to results/raw/search_quality_comparison.jsonl,
one JSON object per line, the same schema the original single-shot version
produced (see module-level ROW_SCHEMA_FIELDS in the source for the field
list); nothing here changes that downstream schema.
"""

import argparse
import bisect
import json
import os
import pickle
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpga.config import BENCHMARK_OPTIMAL_ENERGY, BENCHMARK_SEQUENCE, HPGAConfig
from hpga.island import Island
from hpga import operators as ops

POP_SIZE = 8
N_GENERATIONS_TOTAL = 25
N_WORKERS = 4
SEEDS = [0, 1, 2]
DEFAULT_CHUNK_GENS = 8  # ~8 * 106.2s/gen (llm mode) =~ 14 min, well under the observed ~50-65 min kill window

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "raw"
CHECKPOINT_DIR = RESULTS_DIR / "search_quality_checkpoints"
OUT_JSONL = RESULTS_DIR / "search_quality_comparison.jsonl"


def checkpoint_path(seed: int, mode: str) -> Path:
    return CHECKPOINT_DIR / f"ckpt_{mode}_seed{seed}.pkl"


def load_checkpoint(seed: int, mode: str) -> dict | None:
    path = checkpoint_path(seed, mode)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_checkpoint(seed: int, mode: str, ckpt: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(seed, mode)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(ckpt, f)
    tmp.replace(path)  # atomic on both POSIX and Windows -- a kill mid-write leaves the old checkpoint intact


def _log_path_for(run_id: str) -> Path:
    return RESULTS_DIR / f"llm_operator_calls_{run_id}.jsonl"


def _cumulative_tokens_by_gen(log_path: Path, gen_end_wall: list[datetime]) -> list[int]:
    """Bucket the per-call JSONL log's tokens_out by wall-clock time against
    each generation's end time, using ALL logged attempts (including failed
    retries) -- a retried call still spent real output tokens, and that's a
    real incurred cost, not one that should be hidden because it didn't
    parse. Chunk-oblivious: gen_end_wall spans the whole multi-chunk run in
    absolute UTC, and the log file is one continuous file across all chunks
    of a (seed, mode) run (same HPGA_RUN_ID reused every chunk)."""
    if not log_path.exists():
        return [0] * len(gen_end_wall)

    events: list[tuple[datetime, int]] = []
    for line in open(log_path, encoding="utf-8"):
        rec = json.loads(line)
        if "tokens_out" not in rec:
            continue
        ts = datetime.fromisoformat(rec["timestamp"])
        events.append((ts, rec["tokens_out"]))
    events.sort(key=lambda e: e[0])

    timestamps = [e[0] for e in events]
    cum = [0]
    for _, tout in events:
        cum.append(cum[-1] + tout)

    result = []
    for end in gen_end_wall:
        idx = bisect.bisect_right(timestamps, end)
        result.append(cum[idx])
    return result


def _stats_from_log(log_path: Path) -> dict:
    """Cumulative operator-call stats computed directly from the full
    (possibly multi-chunk) log file, rather than hpga.operators' in-memory
    OperatorCallStats -- that object resets on every process restart, so it
    can only ever reflect the current chunk. The log file is the one thing
    that survives every chunk boundary, so it's the single source of truth
    here."""
    n_calls = n_retries = n_failures = 0
    total_latency = 0.0
    total_tokens_in = total_tokens_out = 0
    if not log_path.exists():
        return {"n_llm_calls": 0, "n_retries": 0, "n_failures": 0,
                "total_latency_s": 0.0, "total_tokens_in": 0, "total_tokens_out": 0}
    for line in open(log_path, encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("attempt") == 0:
            n_calls += 1
        elif "attempt" in rec:
            n_retries += 1
        if rec.get("event") == "fallback_to_deterministic":
            n_failures += 1
        if "latency_s" in rec:
            total_latency += rec["latency_s"]
            total_tokens_in += rec.get("tokens_in", 0)
            total_tokens_out += rec.get("tokens_out", 0)
    return {"n_llm_calls": n_calls, "n_retries": n_retries, "n_failures": n_failures,
            "total_latency_s": total_latency, "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out}


def process_chunk(seed: int, mode: str, chunk_gens: int, sequence: str) -> tuple[int, str, bool]:
    """Run up to `chunk_gens` more generations for (seed, mode), resuming
    from disk if a checkpoint exists. Returns (generations_done, run_id,
    is_finalized)."""
    ckpt = load_checkpoint(seed, mode)
    if ckpt is None:
        rng = random.Random(seed)
        population = None
        best_fitness_by_gen: list[float] = []
        cum_wall_time: list[float] = []
        gen_end_wall_iso: list[str] = []
        generations_done = 0
        run_id = f"searchquality_{mode}_seed{seed}_{int(time.time())}"
    else:
        rng = random.Random()
        rng.setstate(ckpt["rng_state"])
        population = ckpt["population"]
        best_fitness_by_gen = ckpt["best_fitness_by_gen"]
        cum_wall_time = ckpt["cumulative_wall_time_s_by_gen"]
        gen_end_wall_iso = ckpt["gen_end_wall_iso"]
        generations_done = ckpt["generations_done"]
        run_id = ckpt["run_id"]

    remaining = N_GENERATIONS_TOTAL - generations_done
    this_chunk = min(chunk_gens, remaining)
    if this_chunk <= 0:
        return generations_done, run_id, True

    os.environ["HPGA_OPERATOR_MODE"] = mode
    os.environ["HPGA_RUN_ID"] = run_id  # same run_id every chunk -> one continuous log file

    cfg = HPGAConfig(sequence=sequence, pop_size=POP_SIZE, n_generations=this_chunk,
                      n_workers=N_WORKERS, seed=seed)
    if population is None:
        island = Island(cfg, rng=rng)
    else:
        # Island.__init__ unconditionally calls random_population(cfg, rng)
        # before we get control back -- if we passed the real resumed `rng`
        # here, that call would burn genome_length*pop_size draws from it on
        # a population we're about to discard, corrupting rng continuity
        # across the chunk boundary. Feed it a throwaway rng instead, then
        # swap in the real one afterwards, so the resumed rng's state is
        # untouched until next_generation() actually uses it.
        island = Island(cfg, rng=random.Random(0))
        island.population = population
        island.rng = rng

    t0_wall = datetime.now(timezone.utc)
    recorder = island.run()

    best_fitness_by_gen = best_fitness_by_gen + recorder.best_fitness_by_gen

    prev_total = cum_wall_time[-1] if cum_wall_time else 0.0
    running = prev_total + recorder.startup_time  # each chunk repays worker-spawn cost; real, if small
    running_local = recorder.startup_time
    for g in recorder.generations:
        running += g.dis_time + g.ga_time
        running_local += g.dis_time + g.ga_time
        cum_wall_time.append(running)
        gen_end_wall_iso.append((t0_wall + timedelta(seconds=running_local)).isoformat())

    generations_done += this_chunk
    ckpt_out = {
        "generations_done": generations_done,
        "population": island.population,
        "rng_state": rng.getstate(),
        "best_fitness_by_gen": best_fitness_by_gen,
        "cumulative_wall_time_s_by_gen": cum_wall_time,
        "gen_end_wall_iso": gen_end_wall_iso,
        "run_id": run_id,
    }
    save_checkpoint(seed, mode, ckpt_out)

    done = generations_done >= N_GENERATIONS_TOTAL
    if done:
        finalize(seed, mode, ckpt_out)
    return generations_done, run_id, done


def finalize(seed: int, mode: str, ckpt: dict) -> None:
    run_id = ckpt["run_id"]
    log_path = _log_path_for(run_id)
    gen_end_wall = [datetime.fromisoformat(s) for s in ckpt["gen_end_wall_iso"]]
    cum_tokens = _cumulative_tokens_by_gen(log_path, gen_end_wall) if mode == "llm" else [0] * len(gen_end_wall)
    op_stats = _stats_from_log(log_path) if mode == "llm" else None

    r = {
        "mode": mode,
        "seed": seed,
        "run_id": run_id,
        "wall_elapsed_s": ckpt["cumulative_wall_time_s_by_gen"][-1] if ckpt["cumulative_wall_time_s_by_gen"] else 0.0,
        "best_fitness_by_gen": ckpt["best_fitness_by_gen"],
        "cumulative_wall_time_s_by_gen": ckpt["cumulative_wall_time_s_by_gen"],
        "cumulative_tokens_out_by_gen": cum_tokens,
        "best_fitness_final": ckpt["best_fitness_by_gen"][-1] if ckpt["best_fitness_by_gen"] else None,
        "operator_stats": op_stats,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(r, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(f"  FINALIZED seed={seed} mode={mode}: best_fitness_final={r['best_fitness_final']:.0f}  "
          f"appended to {OUT_JSONL}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True, choices=SEEDS)
    p.add_argument("--mode", type=str, required=True, choices=("deterministic", "llm"))
    p.add_argument("--chunk-gens", type=int, default=DEFAULT_CHUNK_GENS)
    args = p.parse_args()

    target_contacts = -BENCHMARK_OPTIMAL_ENERGY
    print(f"seed={args.seed}  mode={args.mode}  chunk_gens={args.chunk_gens}  "
          f"pop_size={POP_SIZE}  n_generations_total={N_GENERATIONS_TOTAL}  n_workers={N_WORKERS}", flush=True)

    t0 = time.perf_counter()
    generations_done, run_id, done = process_chunk(args.seed, args.mode, args.chunk_gens, BENCHMARK_SEQUENCE)
    took = time.perf_counter() - t0

    print(f"  chunk took {took:.1f}s  generations_done={generations_done}/{N_GENERATIONS_TOTAL}  "
          f"run_id={run_id}  done={done}", flush=True)
    if not done:
        print(f"  NOT DONE -- rerun with the same --seed {args.seed} --mode {args.mode} to continue.", flush=True)


if __name__ == "__main__":
    main()
