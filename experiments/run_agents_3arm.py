"""Phase 2 agents/communication feature, 3-arm run: isolates whether
diversity preservation comes from the explore role's design (propose
something different) or from communication itself, and whether fitness can
move at all at a slightly harder scale than the first (2-arm) smoke run.

Arms (additive agent design -- see hpga/agents.py and operators.py's
next_generation()):
  1. off:            agents disabled entirely (today's baseline, unchanged).
  2. on, no comm:     agents enabled, HPGA_COMM_INTERVAL set past
                      n_generations so it never fires -- isolates the
                      explore/refine roles' own effect on diversity from
                      communication's effect.
  3. on, comm:        agents enabled, communication fires at generations
                      2 and 4 (comm_interval=2, n_generations=6).

Scale: pop_size=5, elitism=1, A=2 (1 explore + 1 refine), gemma4:12b,
sequence = make_timing_sequence(length=14, seed=1) -> genome_length=12, a
random (not benchmark/known-optimum) H/P string -- appropriate here since
this is a relative across-arm comparison, not benchmark validation.
Expectation set going in: fitness may still not move much in 6 generations
at this scale; if it doesn't, cost-per-fitness-point is reported as
undefined, not forced via a proxy. The diversity isolation and token
accounting are the point of this run.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpga.config import HPGAConfig, make_timing_sequence
from hpga.island import Island
from hpga import operators as ops
from hpga import agents

RESULTS_RAW = Path(__file__).resolve().parent.parent / "results" / "raw"

SEQUENCE = make_timing_sequence(length=14, seed=1)  # genome_length = 12
POP_SIZE = 5
ELITISM = 1
N_WORKERS = 2
N_GENERATIONS = 6
N_AGENTS = 2
COMM_INTERVAL_ON = 2
COMM_INTERVAL_OFF = 100  # > N_GENERATIONS: never fires
MODEL = "gemma4:12b"


def run_arm(*, agents_enabled: bool, comm_interval: int, run_id: str) -> dict:
    os.environ["HPGA_OPERATOR_MODE"] = "llm"
    os.environ["HPGA_LLM_MODEL"] = MODEL
    os.environ["HPGA_AGENTS_ENABLED"] = "1" if agents_enabled else "0"
    os.environ["HPGA_N_AGENTS"] = str(N_AGENTS)
    os.environ["HPGA_COMM_INTERVAL"] = str(comm_interval)
    os.environ["HPGA_ISLAND_ID"] = "0"
    os.environ["HPGA_LOG_DIVERSITY"] = "1"
    os.environ["HPGA_RUN_ID"] = run_id

    ops.reset_operator_stats()
    agents.reset_agent_state()

    cfg = HPGAConfig(
        sequence=SEQUENCE, pop_size=POP_SIZE, n_generations=N_GENERATIONS,
        n_workers=N_WORKERS, elitism=ELITISM, seed=0,
    )
    island = Island(cfg)

    t0 = time.perf_counter()
    recorder = island.run()
    wall_s = time.perf_counter() - t0

    return {
        "run_id": run_id, "wall_s": wall_s,
        "best_fitness_by_gen": recorder.best_fitness_by_gen,
        "operator_stats": ops.get_operator_stats(),
    }


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def position_bias(run_id: str) -> dict:
    """Aggregate agent_fold_diff records by role: how often each genome
    position changed, across all agent calls in this run."""
    recs = [r for r in load_jsonl(RESULTS_RAW / f"llm_operator_calls_{run_id}.jsonl")
            if r.get("op") == "agent_fold_diff"]
    by_role: dict[str, dict] = {}
    for r in recs:
        role = r["role"]
        d = by_role.setdefault(role, {"n_calls": 0, "position_counts": {}, "n_changed_list": []})
        d["n_calls"] += 1
        d["n_changed_list"].append(r["n_changed"])
        for p in r["changed_positions"]:
            d["position_counts"][p] = d["position_counts"].get(p, 0) + 1
    return by_role


def tokens_by_comm(run_id: str, comm_gens: set) -> dict:
    recs = [r for r in load_jsonl(RESULTS_RAW / f"llm_operator_calls_{run_id}.jsonl")
            if r.get("op", "").startswith("agent_") and r.get("op") != "agent_fold_diff"
            and "tokens_in" in r]
    comm = [r["tokens_in"] for r in recs if r["generation"] in comm_gens]
    noncomm = [r["tokens_in"] for r in recs if r["generation"] not in comm_gens]
    return {
        "comm_gen_tokens_in": comm, "noncomm_gen_tokens_in": noncomm,
        "comm_mean": sum(comm) / len(comm) if comm else None,
        "noncomm_mean": sum(noncomm) / len(noncomm) if noncomm else None,
    }


def main() -> None:
    ts = int(time.time())
    ids = {
        "off": f"agents3_off_{ts}",
        "nocomm": f"agents3_nocomm_{ts}",
        "comm": f"agents3_comm_{ts}",
    }
    print(f"sequence={SEQUENCE} (length={len(SEQUENCE)}, genome_length={len(SEQUENCE)-2})\n", flush=True)

    results = {}
    print(f"=== off (agents disabled), run_id={ids['off']} ===", flush=True)
    results["off"] = run_arm(agents_enabled=False, comm_interval=COMM_INTERVAL_ON, run_id=ids["off"])
    print(f"wall_s={results['off']['wall_s']:.1f}  best_fitness_by_gen={results['off']['best_fitness_by_gen']}")
    print(f"operator_stats={results['off']['operator_stats']}\n", flush=True)

    print(f"=== on, no comm, run_id={ids['nocomm']} ===", flush=True)
    results["nocomm"] = run_arm(agents_enabled=True, comm_interval=COMM_INTERVAL_OFF, run_id=ids["nocomm"])
    print(f"wall_s={results['nocomm']['wall_s']:.1f}  best_fitness_by_gen={results['nocomm']['best_fitness_by_gen']}")
    print(f"operator_stats={results['nocomm']['operator_stats']}\n", flush=True)

    print(f"=== on, comm every {COMM_INTERVAL_ON} gens, run_id={ids['comm']} ===", flush=True)
    results["comm"] = run_arm(agents_enabled=True, comm_interval=COMM_INTERVAL_ON, run_id=ids["comm"])
    print(f"wall_s={results['comm']['wall_s']:.1f}  best_fitness_by_gen={results['comm']['best_fitness_by_gen']}")
    print(f"operator_stats={results['comm']['operator_stats']}\n", flush=True)

    # --- Fitness gained / cost-per-fitness-point ---
    print("=== Fitness gained (final - gen0 best) and cost-per-point ===")
    for arm in ("off", "nocomm", "comm"):
        bf = results[arm]["best_fitness_by_gen"]
        gained = bf[-1] - bf[0]
        stats = results[arm]["operator_stats"]
        total_tokens = stats["total_tokens_in"] + stats["total_tokens_out"]
        if gained > 0:
            print(f"  {arm}: gained={gained:.4f}  total_tokens={total_tokens}  "
                  f"cost_per_point={total_tokens / gained:.1f} tokens/point")
        else:
            print(f"  {arm}: gained={gained:.4f} (<=0) -> cost-per-point UNDEFINED, not forced via proxy "
                  f"(total_tokens={total_tokens}, for reference only)")
    print()

    # --- Diversity traces ---
    print("=== Diversity trace (mean pairwise Hamming) ===")
    for arm in ("off", "nocomm", "comm"):
        div = load_jsonl(RESULTS_RAW / f"diversity_{ids[arm]}.jsonl")
        print(f"  {arm}: {[(r['generation'], round(r['mean_pairwise_hamming'], 2)) for r in div]}")
    print()

    # --- Position-bias distribution per arm, per role ---
    print("=== Position-bias distribution (agent fold changes vs input, by role) ===")
    for arm in ("nocomm", "comm"):
        by_role = position_bias(ids[arm])
        print(f"  {arm}:")
        for role, d in by_role.items():
            print(f"    {role}: n_calls={d['n_calls']}  n_changed per call={d['n_changed_list']}  "
                  f"position_counts={dict(sorted(d['position_counts'].items()))}")
    print()

    # --- tokens_in: comm-gen vs non-comm-gen agent calls, per arm ---
    comm_gens = {2, 4}
    print(f"=== tokens_in: comm generations {sorted(comm_gens)} vs non-comm generations, per arm ===")
    for arm in ("nocomm", "comm"):
        tb = tokens_by_comm(ids[arm], comm_gens)
        print(f"  {arm}: comm-gen tokens_in={tb['comm_gen_tokens_in']} (mean={tb['comm_mean']})")
        print(f"  {arm}: non-comm-gen tokens_in={tb['noncomm_gen_tokens_in']} (mean={tb['noncomm_mean']})")
    print()

    summary = {"results": results, "ids": ids}
    out_path = RESULTS_RAW / f"agents_3arm_summary_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
