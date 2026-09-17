"""Phase 2 next step: first paired run of the agent/communication feature
(hpga/agents.py) -- communication off (existing behaviour, agents disabled
entirely) vs on (HPGA_AGENTS_ENABLED=1). Smoke scale, not a search-quality
result: A=2 agents (1 explore, 1 refine), pop_size=5, elitism=1,
n_generations=5, genome_length=8, comm_interval=2, gemma4:12b.

Scale chosen so the "on" arm's agent slots (elitism + A = 1 + 2 = 3 of 5)
leave the fewest possible non-agent slots (2) needing the ordinary
crossover/mutate loop, while the "off" arm (agents disabled) fills all 4
non-elite slots that way -- this is why the two arms make a different
number of top-level LLM calls (15 vs 30), not just a different mechanism;
see PHASE2_RESULTS.md for the caveat this implies about reading the
fitness traces.

Communication token cost is measured, not estimated: agent calls on a
comm-round generation have peer folds appended to their prompt, so the
real Ollama-reported tokens_in for those calls (already logged per call in
the operator-call log) is the actual cost of communication, compared
against tokens_in on non-comm generations of the same run.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpga.config import HPGAConfig
from hpga.island import Island
from hpga import operators as ops
from hpga import agents

RESULTS_RAW = Path(__file__).resolve().parent.parent / "results" / "raw"

SEQUENCE = "HPHPPHHPHP"  # length 10 -> genome_length = 8
POP_SIZE = 5
ELITISM = 1
N_WORKERS = 2
N_GENERATIONS = 5
N_AGENTS = 2
COMM_INTERVAL = 2
MODEL = "gemma4:12b"


def run_arm(*, agents_enabled: bool, run_id: str) -> dict:
    os.environ["HPGA_OPERATOR_MODE"] = "llm"
    os.environ["HPGA_LLM_MODEL"] = MODEL
    os.environ["HPGA_AGENTS_ENABLED"] = "1" if agents_enabled else "0"
    os.environ["HPGA_N_AGENTS"] = str(N_AGENTS)
    os.environ["HPGA_COMM_INTERVAL"] = str(COMM_INTERVAL)
    os.environ["HPGA_ISLAND_ID"] = "0"
    os.environ["HPGA_LOG_DIVERSITY"] = "1"
    os.environ["HPGA_RUN_ID"] = run_id

    ops.reset_operator_stats()
    agents.reset_agent_state()

    cfg = HPGAConfig(
        sequence=SEQUENCE,
        pop_size=POP_SIZE,
        n_generations=N_GENERATIONS,
        n_workers=N_WORKERS,
        elitism=ELITISM,
        seed=0,
    )
    island = Island(cfg)

    t0 = time.perf_counter()
    recorder = island.run()
    wall_s = time.perf_counter() - t0

    return {
        "arm": "on" if agents_enabled else "off",
        "run_id": run_id,
        "wall_s": wall_s,
        "best_fitness_by_gen": recorder.best_fitness_by_gen,
        "operator_stats": ops.get_operator_stats(),
    }


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ts = int(time.time())
    off_run_id = f"agents_smoke_off_{ts}"
    on_run_id = f"agents_smoke_on_{ts}"

    print(f"=== OFF arm (agents disabled, run_id={off_run_id}) ===", flush=True)
    off_result = run_arm(agents_enabled=False, run_id=off_run_id)
    print(f"wall_s={off_result['wall_s']:.1f}  best_fitness_by_gen={off_result['best_fitness_by_gen']}")
    print(f"operator_stats={off_result['operator_stats']}\n", flush=True)

    print(f"=== ON arm (agents enabled, run_id={on_run_id}) ===", flush=True)
    on_result = run_arm(agents_enabled=True, run_id=on_run_id)
    print(f"wall_s={on_result['wall_s']:.1f}  best_fitness_by_gen={on_result['best_fitness_by_gen']}")
    print(f"operator_stats={on_result['operator_stats']}\n", flush=True)

    # --- Diversity traces ---
    off_div = load_jsonl(RESULTS_RAW / f"diversity_{off_run_id}.jsonl")
    on_div = load_jsonl(RESULTS_RAW / f"diversity_{on_run_id}.jsonl")
    print("=== Diversity trace (mean pairwise Hamming) ===")
    print(f"  off: {[(r['generation'], r['mean_pairwise_hamming']) for r in off_div]}")
    print(f"  on:  {[(r['generation'], r['mean_pairwise_hamming']) for r in on_div]}\n")

    # --- Agent messages (on arm only) ---
    agent_msgs = load_jsonl(RESULTS_RAW / f"agent_messages_{on_run_id}.jsonl")
    sends = [r for r in agent_msgs if r.get("message_type") == "send"]
    forwards = [r for r in agent_msgs if r.get("message_type") == "forward"]
    summaries = [r for r in agent_msgs if r.get("message_type") == "round_summary"]
    print("=== Agent communication (word-count proxy tokens) ===")
    print(f"  total messages: {len(agent_msgs)}  (send={len(sends)} forward={len(forwards)} round_summary={len(summaries)})")
    for s in summaries:
        print(f"  round @ gen={s['generation']}: n_messages={s['n_messages']} total_tokens={s['total_tokens']}")
    if sends:
        print(f"  per-message tokens (proxy, send+forward combined): "
              f"mean={sum(r['tokens_out'] for r in sends+forwards)/len(sends+forwards):.1f}  "
              f"min={min(r['tokens_out'] for r in sends+forwards)}  max={max(r['tokens_out'] for r in sends+forwards)}")
    print()

    # --- Real tokens_in: comm-generation vs non-comm-generation agent calls ---
    on_ops = load_jsonl(RESULTS_RAW / f"llm_operator_calls_{on_run_id}.jsonl")
    agent_calls = [r for r in on_ops if r.get("op", "").startswith("agent_") and "tokens_in" in r]
    comm_gens = {g for g in range(N_GENERATIONS) if g > 0 and g % COMM_INTERVAL == 0}
    comm_calls = [r for r in agent_calls if r["generation"] in comm_gens]
    noncomm_calls = [r for r in agent_calls if r["generation"] not in comm_gens]
    print("=== Real tokens_in: agent calls on comm generations vs non-comm generations ===")
    print(f"  comm generations: {sorted(comm_gens)}")
    if comm_calls:
        print(f"  comm-gen agent calls (n={len(comm_calls)}): mean tokens_in="
              f"{sum(r['tokens_in'] for r in comm_calls)/len(comm_calls):.1f}  "
              f"values={[r['tokens_in'] for r in comm_calls]}")
    else:
        print("  comm-gen agent calls: NONE")
    if noncomm_calls:
        print(f"  non-comm-gen agent calls (n={len(noncomm_calls)}): mean tokens_in="
              f"{sum(r['tokens_in'] for r in noncomm_calls)/len(noncomm_calls):.1f}  "
              f"values={[r['tokens_in'] for r in noncomm_calls]}")
    else:
        print("  non-comm-gen agent calls: NONE")

    summary = {
        "off": off_result, "on": on_result,
        "off_diversity": off_div, "on_diversity": on_div,
        "n_agent_messages": len(agent_msgs),
        "comm_gen_agent_calls": comm_calls,
        "noncomm_gen_agent_calls": noncomm_calls,
    }
    out_path = RESULTS_RAW / f"agents_smoke_summary_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
