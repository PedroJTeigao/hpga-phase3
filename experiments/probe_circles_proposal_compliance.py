"""Live compliance probe for circle PROPOSALS on the sequence genome (hpga/circles_sequence.propose): 50 calls
on fresh random genomes (uniform length over [30, 80]), real Ollama, real prompts and parser. Gate for
running any circles GA: fallback rate above 20% -> stop.

Context (directive / circle note / observations) comes from a fixed pool of plausible one-sentence texts
(the GA generates them with the model; here they are fixed so the probe isolates the proposal format).
ESMFold is never loaded. Output: results/raw/circles_sequence_proposal_compliance.json + the usual call log.
Fallback rate = calls that fell back to the unchanged genome / calls (every one of the 50 reaches the LLM).
"""
import json, os, random, statistics, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["HPGA_OPERATOR_MODE"] = "llm"
os.environ["HPGA_RUN_ID"] = f"circles_probe_{int(time.time())}"

from hpga import circles_sequence as cs, genome_model as gm, operators as ops, sequence_model as sm  # noqa: E402
from hpga.config import HPGAConfig  # noqa: E402

N_CALLS, GATE = 50, 0.20
DIRECTIVES = ["", "Progress has stalled recently -- try larger, more different changes this round.",
              "Recent changes have been improving fitness -- keep making small, targeted changes.",
              "Explore more broadly by changing residues in different regions of the chain."]
NOTES = ["", "Focus on the middle of the chain and make smaller edits.", "Try changing a few residues near the start and end.",
         "Prefer replacing residues with chemically different ones."]
OBS = ["Small edits near the middle of the chain raised fitness.", "Changes near the end of the chain did not help.",
       "Replacing several residues at once lowered fitness compared with single edits.",
       "Edits in the first third of the chain gave the largest gains so far.",
       "No improvement from swapping residues for chemically similar ones.", "Larger edits were worse than small ones this round."]


class Entry:  # the two fields propose() reads from a blackboard entry
    def __init__(self, i, content):
        self.id, self.content, self.tokens = f"probe-{i}", content, len(content) // 4


def main():
    gm.set_active(gm.build_genome_model(HPGAConfig(genome_model="sequence")))
    model = gm.current()
    ops.reset_operator_stats()
    rng, rows = random.Random(0), []
    for i in range(N_CALLS):
        genome = model.random_genome(rng, None)
        obs = [Entry(j, t) for j, t in enumerate(rng.sample(OBS, rng.choice([0, 1, 2, 3, 4])))]
        before = ops.get_operator_stats(); t0 = time.perf_counter()
        out = cs.propose(i % 2, i % 3, genome, rng.choice(DIRECTIVES), rng.choice(NOTES), obs, 0, rng)
        wall = time.perf_counter() - t0; after = ops.get_operator_stats()
        fell = after["n_failures"] - before["n_failures"]
        k = cs.n_edits(len(genome)); changed = cs.changed_positions(genome, out)
        rows.append({"call": i, "length": len(genome), "k_requested": k, "n_context_observations": len(obs), "fell_back": bool(fell),
                     "requests": after["n_llm_requests"] - before["n_llm_requests"], "retries": after["n_retries"] - before["n_retries"],
                     "changed_positions": changed, "valid_edit": len(changed) == k and not fell, "wall_s": wall,
                     "new_letters": [out[p] for p in changed]})
        print(f"call {i:>2} len={len(genome)} k={k} obs={len(obs)} fell_back={bool(fell)} requests={rows[-1]['requests']} changed={changed} {wall:.1f}s", flush=True)
    n_fb = sum(r["fell_back"] for r in rows)
    st = ops.get_operator_stats()
    pos, let = Counter(p for r in rows for p in r["changed_positions"]), Counter(x for r in rows for x in r["new_letters"])
    summary = {"n_calls": N_CALLS, "n_fell_back": n_fb, "fallback_rate": n_fb / N_CALLS, "gate": GATE, "gate_passed": n_fb / N_CALLS <= GATE,
               "requests_per_call": st["n_llm_requests"] / N_CALLS, "n_retries": st["n_retries"], "valid_edits": sum(r["valid_edit"] for r in rows),
               "distinct_positions_edited": len(pos), "distinct_new_letters": len(let), "mean_wall_s": statistics.mean(r["wall_s"] for r in rows),
               "tokens_in": st["total_tokens_in"], "tokens_out": st["total_tokens_out"], "model": ops.LLM_MODEL, "temperature": ops.LLM_TEMPERATURE,
               "call_log": str(ops._log_path().relative_to(ROOT)), "context_note": "directive/note/observations drawn from a fixed pool, not model-generated"}
    out = ROOT / "results" / "raw" / "circles_sequence_proposal_compliance.json"
    out.write_text(json.dumps({"summary": summary, "calls": rows}, indent=1))
    print(json.dumps(summary, indent=1)); print("GATE PASSED" if summary["gate_passed"] else "GATE FAILED: fallback above 20% -- STOP, do not run stage 2")
    sys.exit(0 if summary["gate_passed"] else 2)

if __name__ == "__main__":
    main()
