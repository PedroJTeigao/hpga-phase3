"""Phase 3, second deliverable: does the diversity gap (PHASE3_RESULTS.md
sec 2) convert into fitness when there's actually room for it to matter?

The anchor run (run_circles_smoke.py, genome_length=18, pop_size=8,
n_generations=10) plateaus by generation 3-4 in both arms -- "diversity
didn't help" there is indistinguishable from "neither arm needed it," and
at that (length, pop_size) the fold space is small enough to cover without
strategy, so there's nothing for extra diversity to rescue.

experiments/sweep_diversity_config.py (deterministic, zero LLM cost) swept
genome_length x pop_size x n_generations looking for a configuration where
a modest population (pop_size close to the anchor's 8, to keep LLM call
budget comparable) genuinely stalls below what a much bigger reference
population reaches over a long free horizon (real local-optima trapping,
not just "hasn't finished yet"), while also still visibly climbing at a
generation count affordable under LLM operators. genome_length=24 (seq
length 26) was picked: pop=8 settles ~4.17 fitness points below pop=64's
ceiling over 150 generations (plateau not reached until generation ~54 on
average), and is still climbing at every checkpoint from generation 10
through 25 (4.33 -> 4.67 -> 5.00 -> 6.33), unlike genome_length=18 where
pop=8 was already flat by generation 4.

Caveat carried over from that sweep, not re-litigated here: the sweep used
the plain deterministic crossover()/mutate() operators as a free proxy for
landscape difficulty. The real "off" arm below still runs
HPGA_OPERATOR_MODE=llm (circles just disabled) -- PROMPT_STYLE="best"'s
position/segment operators, which PHASE2_RESULTS.md sec 4.4 already showed
are biased (80% of mutations on positions 0-1, crossover splits at the
exact midpoint) rather than uniform-random. This script is what tests
whether the sweep's landscape-difficulty proxy actually predicts real
behavior under those operators -- it isn't assumed to.

Three seeds (not one, unlike the anchor) specifically because every fitness
number in PHASE3_RESULTS.md sec 5 rested on a single seed -- named there as
exactly the weakness this run should not repeat.

Architecture unchanged from the anchor run: PROMPT_STYLE="best" (position
mutate + segment crossover), N_CIRCLES=2, AGENTS_PER_CIRCLE=2,
CENTRAL_MODE=llm, CENTRAL_INTERVAL=5, CURATION_INTERVAL=3. Only the
sequence (genome_length 18 -> 24) and seed count (1 -> 3) changed -- this
is a harder-landscape replication of the same comparison, not a redesign.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpga.config import HPGAConfig, make_timing_sequence
from hpga.island import Island
from hpga import operators as ops
from hpga import agents
from hpga import circles
from hpga import blackboard as bb

RESULTS_RAW = Path(__file__).resolve().parent.parent / "results" / "raw"

SEQUENCE = make_timing_sequence(length=26, seed=1)  # genome_length=24, per sweep_diversity_config.py
POP_SIZE = 8
ELITISM = 1
N_WORKERS = 2
N_GENERATIONS = int(os.environ.get("HPGA_DIVFIT_N_GENERATIONS", "10"))
N_CIRCLES = 2
AGENTS_PER_CIRCLE = 2
CENTRAL_MODE = "llm"
CENTRAL_INTERVAL = 5
CURATION_INTERVAL = 3
PROMPT_STYLE = "best"
MODEL = "gemma4:12b"
SEEDS = [0, 1, 2]


def gpu_snapshot() -> dict:
    def _run(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as exc:
            return f"<nvidia-smi call failed: {exc!r}>"

    return {
        "usage": _run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader"]),
        "compute_apps": _run(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
                               "--format=csv,noheader"]) or "(none)",
    }


def _known_pids(compute_apps: str) -> set[str]:
    """First field (pid) of each nvidia-smi compute-apps line, '' if none."""
    pids = set()
    for line in compute_apps.splitlines():
        line = line.strip()
        if not line or line == "(none)":
            continue
        pid = line.split(",")[0].strip()
        if pid:
            pids.add(pid)
    return pids


def check_contamination(baseline_pids: set[str], snap: dict, label: str) -> str | None:
    """Flags (doesn't block) any compute-apps pid not present in the
    baseline snapshot taken before this script's own model warm-up --
    i.e. a second GPU consumer appeared after this run started. Returns a
    warning string if so, else None."""
    current = _known_pids(snap["compute_apps"])
    unexpected = current - baseline_pids
    if unexpected:
        msg = (f"!!! GPU CONTAMINATION WARNING at {label}: pid(s) {unexpected} "
               f"not present in the pre-run baseline -- another process is now "
               f"using this GPU. compute_apps={snap['compute_apps']!r}. "
               f"Fitness/diversity data is unaffected; wall-clock timing from "
               f"here on is suspect.")
        print(msg, flush=True)
        return msg
    return None


def warm_up_model() -> float:
    t0 = time.perf_counter()
    ops._call_ollama("Say OK.", "You are a test.", 8, 0)
    return time.perf_counter() - t0


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_arm(*, circles_enabled: bool, seed: int, run_id: str) -> dict:
    os.environ["HPGA_OPERATOR_MODE"] = "llm"
    os.environ["HPGA_LLM_MODEL"] = MODEL
    os.environ["HPGA_LLM_PROMPT_STYLE"] = PROMPT_STYLE
    os.environ["HPGA_CIRCLES_ENABLED"] = "1" if circles_enabled else "0"
    os.environ["HPGA_AGENTS_ENABLED"] = "0"
    os.environ["HPGA_N_CIRCLES"] = str(N_CIRCLES)
    os.environ["HPGA_AGENTS_PER_CIRCLE"] = str(AGENTS_PER_CIRCLE)
    os.environ["HPGA_CENTRAL_MODE"] = CENTRAL_MODE
    os.environ["HPGA_CENTRAL_INTERVAL"] = str(CENTRAL_INTERVAL)
    os.environ["HPGA_CURATION_INTERVAL"] = str(CURATION_INTERVAL)
    os.environ["HPGA_ISLAND_ID"] = "0"
    os.environ["HPGA_LOG_DIVERSITY"] = "1"
    os.environ["HPGA_LLM_TIMEOUT_S"] = "300"
    os.environ["HPGA_RUN_ID"] = run_id

    ops.reset_operator_stats()
    agents.reset_agent_state()
    circles.reset_circle_state()
    bb.reset_blackboard()

    cfg = HPGAConfig(sequence=SEQUENCE, pop_size=POP_SIZE, n_generations=N_GENERATIONS,
                      n_workers=N_WORKERS, elitism=ELITISM, seed=seed)
    island = Island(cfg)

    t0 = time.perf_counter()
    recorder = island.run()
    wall_s = time.perf_counter() - t0

    ops_log = load_jsonl(RESULTS_RAW / f"llm_operator_calls_{run_id}.jsonl")
    div = load_jsonl(RESULTS_RAW / f"diversity_{run_id}.jsonl")
    fits = recorder.best_fitness_by_gen
    gain = fits[-1] - fits[0] if fits else 0.0
    total_tokens = sum(r.get("tokens_in", 0) + r.get("tokens_out", 0) for r in ops_log)

    return {
        "arm": "on" if circles_enabled else "off",
        "seed": seed,
        "run_id": run_id,
        "wall_s": wall_s,
        "best_fitness_by_gen": fits,
        "diversity_by_gen": [r["mean_pairwise_hamming"] for r in div],
        "operator_stats": ops.get_operator_stats(),
        "fitness_gain": gain,
        "total_tokens": total_tokens,
        "fitness_gain_per_1k_tokens": (gain / total_tokens * 1000) if total_tokens else None,
    }


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    ts = int(time.time())
    print(f"=== run_circles_diversity_fitness  ts={ts}  genome_length={len(SEQUENCE)-2}  "
          f"pop_size={POP_SIZE}  n_generations={N_GENERATIONS}  seeds={SEEDS} ===", flush=True)

    gpu_before = gpu_snapshot()
    print(f"gpu before: {gpu_before['usage']}  compute_apps: {gpu_before['compute_apps']}", flush=True)

    warm_s = warm_up_model()
    print(f"model warm-up: {warm_s:.1f}s (excluded from measured latency)\n", flush=True)

    # Baseline for contamination checks is taken AFTER warm-up, not before:
    # warm_up_model() is what spawns/loads this run's own Ollama llama-server
    # process, so a pre-warm-up baseline would flag that pid as "unexpected"
    # against itself at every subsequent checkpoint -- caught in testing when
    # it fired on the very first check.
    gpu_after_warmup = gpu_snapshot()
    baseline_pids = _known_pids(gpu_after_warmup["compute_apps"])
    print(f"gpu after warm-up (contamination baseline): {gpu_after_warmup['usage']}  "
          f"compute_apps: {gpu_after_warmup['compute_apps']}\n", flush=True)

    all_results = {"off": [], "on": []}
    gpu_snapshots: list[dict] = [{"label": "before", **gpu_before},
                                 {"label": "after_warmup_baseline", **gpu_after_warmup}]
    contamination_warnings: list[str] = []
    t_start = time.perf_counter()

    for seed in SEEDS:
        for circles_enabled in (False, True):
            arm = "on" if circles_enabled else "off"
            run_id = f"circles_divfit_{arm}_s{seed}_{ts}"
            print(f"--- seed={seed} arm={arm} run_id={run_id} ---", flush=True)

            pre = gpu_snapshot()
            w = check_contamination(baseline_pids, pre, f"before seed={seed} arm={arm}")
            if w:
                contamination_warnings.append(w)
            gpu_snapshots.append({"label": f"before_seed{seed}_{arm}", **pre})

            r = run_arm(circles_enabled=circles_enabled, seed=seed, run_id=run_id)
            all_results[arm].append(r)
            print(f"    wall_s={r['wall_s']:.1f}  best_fitness_by_gen={r['best_fitness_by_gen']}  "
                  f"fitness_gain_per_1k_tokens={r['fitness_gain_per_1k_tokens']}", flush=True)

            post = gpu_snapshot()
            w = check_contamination(baseline_pids, post, f"after seed={seed} arm={arm}")
            if w:
                contamination_warnings.append(w)
            gpu_snapshots.append({"label": f"after_seed{seed}_{arm}", **post})

            elapsed = time.perf_counter() - t_start
            print(f"    cumulative elapsed: {elapsed/60:.1f} min\n", flush=True)

    gpu_after = gpu_snapshot()
    print(f"gpu after: {gpu_after['usage']}  compute_apps: {gpu_after['compute_apps']}", flush=True)
    w = check_contamination(baseline_pids, gpu_after, "final")
    if w:
        contamination_warnings.append(w)

    total_wall = time.perf_counter() - t_start
    print(f"\ntotal wall time (both arms, all seeds): {total_wall/60:.1f} min", flush=True)
    if contamination_warnings:
        print(f"\n!!! {len(contamination_warnings)} GPU contamination warning(s) during this run -- "
              f"see gpu_snapshots/contamination_warnings in the summary JSON. Wall-clock timing "
              f"numbers from the affected windows should be treated as suspect. !!!\n", flush=True)
    else:
        print("\nNo GPU contamination detected at any checkpoint -- this run's own Ollama "
              "process was the only compute-apps entry throughout.\n", flush=True)

    # --- aggregate summary --------------------------------------------
    summary = {
        "ts": ts,
        "config": {
            "sequence": SEQUENCE, "genome_length": len(SEQUENCE) - 2, "pop_size": POP_SIZE,
            "n_generations": N_GENERATIONS, "seeds": SEEDS, "n_circles": N_CIRCLES,
            "agents_per_circle": AGENTS_PER_CIRCLE, "prompt_style": PROMPT_STYLE,
        },
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "gpu_snapshots": gpu_snapshots,
        "contamination_warnings": contamination_warnings,
        "warm_up_s": warm_s,
        "total_wall_s": total_wall,
        "off": all_results["off"],
        "on": all_results["on"],
    }

    for arm in ("off", "on"):
        finals = [r["best_fitness_by_gen"][-1] for r in all_results[arm]]
        gains_per_1k = [r["fitness_gain_per_1k_tokens"] for r in all_results[arm] if r["fitness_gain_per_1k_tokens"] is not None]
        div_final = [r["diversity_by_gen"][-1] for r in all_results[arm] if r["diversity_by_gen"]]
        print(f"\n=== {arm} summary across {len(all_results[arm])} seeds ===")
        print(f"  final fitness: {finals}  mean={mean(finals):.2f}")
        print(f"  final diversity: {[round(d,2) for d in div_final]}  mean={mean(div_final):.2f}")
        print(f"  fitness_gain_per_1k_tokens: {[round(g,4) for g in gains_per_1k]}  mean={mean(gains_per_1k):.4f}")
        summary[f"{arm}_final_fitness_mean"] = mean(finals)
        summary[f"{arm}_final_diversity_mean"] = mean(div_final)
        summary[f"{arm}_fitness_gain_per_1k_tokens_mean"] = mean(gains_per_1k)

    out_path = RESULTS_RAW / f"circles_diversity_fitness_summary_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
