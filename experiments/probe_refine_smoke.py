"""Isolated smoke check for the refine-agent fixed-point fix (PHASE2_RESULTS.md
Sec. 8.3): fires real Ollama calls directly through
`hpga.agents._agent_generate(role="refine", ...)` -- not through a full GA
run -- on a fresh random genome each call, and reports the no-op rate plus
the actual prompt/response pairs, so the fix can be checked in isolation
before spending a full 3-arm run on it.

Explore is not touched by the fix and is not exercised here.

Fresh random genome per call (not reused/fed forward), matching
probe_operator_compliance.py's methodology and deliberately NOT reproducing
Sec. 8.3's feed-forward setup (each call's output becomes the next call's
input) -- that feedback loop is exactly what turned one echo into a
permanent one, and isn't needed to check whether a single call can still
produce a no-op under the new format.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
os.environ.setdefault("HPGA_OPERATOR_MODE", "llm")
os.environ.setdefault("HPGA_LLM_MODEL", "gemma4:12b")

from hpga import agents
from hpga import operators as ops

GENOME_LEN = 12  # matches run_agents_3arm.py's genome_length (sequence length 14, 2 pinned)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-calls", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_id = f"refine_smoke_{int(__import__('time').time())}"
    os.environ["HPGA_RUN_ID"] = run_id
    ops.reset_operator_stats()
    agents.reset_agent_state()

    rng = random.Random(args.seed)
    print(f"model={ops.LLM_MODEL} host={ops.LLM_HOST} n_calls={args.n_calls} "
          f"genome_length={GENOME_LEN} run_id={run_id}\n")

    no_op_count = 0
    n_changed_list = []
    for i in range(args.n_calls):
        genome = ops.random_genome(GENOME_LEN, rng)
        out = agents._agent_generate(
            agent_id=0, role=agents.ROLE_REFINE, genome=genome, peer_folds=[],
            length=GENOME_LEN, generation=i, rng=rng,
        )
        changed = [j for j, (a, b) in enumerate(zip(genome, out)) if a != b]
        n_changed_list.append(len(changed))
        if len(changed) == 0:
            no_op_count += 1
        print(f"call {i}: in ={ops._genome_to_str(genome)}")
        print(f"       out={ops._genome_to_str(out)}  n_changed={len(changed)}  positions={changed}")

    stats = ops.get_operator_stats()
    print(f"\n=== summary ===")
    print(f"no_op_count={no_op_count}/{args.n_calls}")
    print(f"n_changed per call: {n_changed_list}")
    print(f"fallback_to_deterministic (exhausted retries)={stats['n_failures']}/{args.n_calls}")
    print(f"requests_per_call={stats['n_llm_requests'] / args.n_calls:.2f}")
    print(f"mean_latency_s={stats['mean_latency_s']:.2f}")

    # --- Actual prompt/response pairs, straight from the call log ---
    log_path = Path(__file__).resolve().parent.parent / "results" / "raw" / f"llm_operator_calls_{run_id}.jsonl"
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    refine_records = [r for r in records if r.get("op") == "agent_refine" and "response" in r]
    print(f"\n=== prompt/response pairs (n={len(refine_records)}) ===")
    for r in refine_records:
        print(f"--- attempt={r['attempt']} valid={r['valid']} latency_s={r['latency_s']:.2f} ---")
        print(f"PROMPT:\n{r['prompt']}")
        print(f"RESPONSE:\n{r['response']}\n")

    out_path = Path(__file__).resolve().parent.parent / "results" / "raw" / f"refine_smoke_summary_{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id, "n_calls": args.n_calls, "no_op_count": no_op_count,
            "n_changed_list": n_changed_list, "stats": stats,
        }, f, indent=2, default=str)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
