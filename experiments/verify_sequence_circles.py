"""Checks for circles on the SEQUENCE genome (hpga/circles_sequence.py), with a stubbed model: no GPU, no
Ollama. Lattice circles are guarded separately, byte for byte, by golden_lattice_circles.py.

  1  shape          each next_generation returns pop_size + n_slots str genomes, in the length bounds
  2  edits          from generation 1 each tail slot is its previous occupant with exactly k positions changed
                    (or unchanged if the model produced no valid edit -- the logged fallback)
  3  format         NO prompt asks for a whole sequence (no FOLD / CHILD / MUTATED restatement), every
                    proposal prompt is the position-style format, and context is prepended when there is any
  4  observations   observation prompts name the changed positions; the blackboard fills; curation runs
  5  fallback       a model that never answers validly leaves every proposal equal to its base, without error
  6  determinism    same seed, same result
  7  loud failures  agents / diversity logging still raise; circles with deterministic operators raises
Exit status 0 only if every check passes.
"""

import hashlib
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if os.environ.get("PYTHONHASHSEED") != "0":  # blackboard curation iterates a set of ids
    os.execve(sys.executable, [sys.executable] + sys.argv, {**os.environ, "PYTHONHASHSEED": "0"})

import tempfile  # noqa: E402

from hpga import agents, blackboard as bb, circles, circles_sequence as cs, genome_model as gm  # noqa: E402
from hpga import operators as ops  # noqa: E402
from hpga import sequence_model as sm  # noqa: E402
from hpga.config import HPGAConfig  # noqa: E402

FAILS: list[str] = []
POP, N_SLOTS, GENS = 8, 4, 8


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail and not ok else ''}")
    if not ok:
        FAILS.append(label)


def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def make_stub(all_bad_propose=False):
    calls = []

    def stub(prompt, system, num_predict, seed):
        r = random.Random(_h(seed, prompt[:300], len(prompt)))
        bad = r.random() < 0.15
        calls.append(prompt)
        if "OBSERVATION:" in prompt:
            text = "no observation" if bad else f"OBSERVATION: those edits suggest changes near position {r.randrange(30)} matter"
        elif "NOTE:" in prompt:
            text = "nothing" if bad else "NOTE: try smaller edits in the middle of the chain"
        elif "DIRECTIVE:" in prompt:
            text = "nothing" if bad else "DIRECTIVE: prioritise small targeted edits"
        elif "DROP:" in prompt and "MERGE" in prompt:
            ids = re.findall(r"^\s+(\S+-\d+) \(gen", prompt, re.MULTILINE)
            text = "keep all" if bad or len(ids) < 2 else f"MERGE A {ids[0]}, {ids[1]}\nSUMMARY A: merged"
        elif "SEGMENTS:" in prompt:
            text = "SEGMENTS: junk" if bad else "SEGMENTS: 0-40:1, 40-100:2"
        elif "POSITION:" in prompt and "Change exactly" in prompt:
            k = int(re.search(r"Change exactly (\d+)", prompt).group(1))
            n = int(re.search(r"positions 0-(\d+)\)", prompt).group(1)) + 1
            cur = "".join(re.search(r"Sequence \(0-indexed positions 0-\d+\): ([A-Z ]+)", prompt).group(1).split())
            pos = r.sample(range(n), k)
            lines = [f"POSITION: {p}, NEW: {r.choice([c for c in sm.ALPHABET if c != cur[p]])}" for p in pos]
            is_proposal = "Take the guidance above" in prompt  # only circle proposals carry a context block
            text = "unparseable" if (bad or (all_bad_propose and is_proposal)) else "\n".join(lines)
        else:
            text = "unrecognised"
        return text, 0.1, len(prompt) // 4, len(text) // 4

    return stub, calls


def setup(circles_on=True, mode="llm", central="llm", tmp=None, seed=0):
    for k in ("HPGA_AGENTS_ENABLED", "HPGA_LOG_DIVERSITY", "HPGA_GA_DISPATCH_P", "HPGA_CIRCLES_SEQ_EDIT_RATE"):
        os.environ.pop(k, None)
    os.environ.update({
        "HPGA_OPERATOR_MODE": mode, "HPGA_CIRCLES_ENABLED": "1" if circles_on else "0", "HPGA_N_CIRCLES": "2",
        "HPGA_AGENTS_PER_CIRCLE": "2", "HPGA_CENTRAL_MODE": central, "HPGA_CENTRAL_INTERVAL": "2",
        "HPGA_CURATION_INTERVAL": "3", "HPGA_RUN_ID": f"vsc-{seed}", "HPGA_LLM_MAX_RETRIES": "2",
        "HPGA_LLM_LOG_PATH": str(Path(tmp) / f"llm{seed}.jsonl"), "HPGA_BLACKBOARD_LOG_PATH": str(Path(tmp) / f"bb{seed}.jsonl"),
    })
    gm.set_active(gm.build_genome_model(HPGAConfig(genome_model="sequence")))
    ops.reset_operator_stats(); agents.reset_agent_state(); circles.reset_circle_state(); bb.reset_blackboard()


def style_wrappers():
    o_x, o_m = ops._llm_crossover, ops._llm_mutate

    def x(*a, **k):
        os.environ["HPGA_LLM_PROMPT_STYLE"] = "segment"
        return o_x(*a, **k)

    def m(*a, **k):
        os.environ["HPGA_LLM_PROMPT_STYLE"] = "position"
        return o_m(*a, **k)

    ops._llm_crossover, ops._llm_mutate = x, m
    return lambda: (setattr(ops, "_llm_crossover", o_x), setattr(ops, "_llm_mutate", o_m))


def stub_fitness(g: str) -> float:
    return (g.count("A") + 0.01 * g.count("L")) / len(g)


def run(seed=0, all_bad=False, tmp=None):
    setup(tmp=tmp, seed=seed)
    stub, calls = make_stub(all_bad_propose=all_bad)
    real, undo = ops._call_ollama, style_wrappers()
    ops._call_ollama = stub
    try:
        rng = random.Random(seed)
        pop = ops.random_population(POP, None, rng)
        history = []
        for gen in range(GENS):
            fits = [stub_fitness(g) for g in pop]
            new = ops.next_generation(pop, fits, POP, 3, 0.9, 0.05, 2, rng)
            history.append({"in": list(pop), "out": list(new)})
            pop = new
    finally:
        ops._call_ollama = real
        undo()
    return history, calls


def main() -> None:
    lo, hi = sm.MIN_LENGTH, sm.MAX_LENGTH
    with tempfile.TemporaryDirectory(prefix="vsc_") as tmp:
        print("1-4. normal run (stubbed model, ~15% malformed replies)")
        hist, calls = run(seed=0, tmp=tmp)
        check("every generation returns pop_size + n_slots str genomes",
              all(len(h["out"]) == POP + N_SLOTS and all(isinstance(g, str) for g in h["out"]) for h in hist))
        check(f"every genome inside [{lo}, {hi}]", all(lo <= len(g) <= hi for h in hist for g in h["out"]))
        edits_ok, fallbacks, valid_edits = True, 0, 0
        for h in hist[1:]:  # from the second call the input holds the previous proposals in its tail
            for slot in range(N_SLOTS):
                base, prop = h["in"][POP + slot], h["out"][POP + slot]
                d = cs.changed_positions(base, prop)
                k = cs.n_edits(len(base))
                if prop == base:
                    fallbacks += 1
                else:
                    valid_edits += 1
                    edits_ok &= len(prop) == len(base) and len(d) == k
        check("each proposal is its slot's previous member with exactly k positions changed, or unchanged (fallback)", edits_ok,
              f"valid {valid_edits}, unchanged {fallbacks}")
        check("some proposals were valid edits", valid_edits >= 10, f"only {valid_edits}")
        prop_prompts = [c for c in calls if "Change exactly" in c and "Take the guidance above" in c]
        mut_prompts = [c for c in calls if "Change exactly" in c and "Take the guidance above" not in c]
        check("proposal prompts exist, carry the directive / guidance / observations, and use the position-style format",
              len(prop_prompts) >= 20 and all("POSITION: <0-" in c and "NEW: <one letter from" in c for c in prop_prompts),
              f"{len(prop_prompts)} proposal prompts")
        check("no prompt asks the model to write a whole sequence (no FOLD / CHILD1 / MUTATED / RATIONALE restatement)",
              not any(("FOLD:" in c or "CHILD1:" in c or "MUTATED:" in c or "RATIONALE:" in c) for c in calls))
        check("prompts never name the target structure ('7UR7' or the whole word 'target'; 'targeted' in generated text is fine)",
              not any(re.search(r"\btarget\b", c.lower()) or "7UR7" in c for c in calls))
        obs_prompts = [c for c in calls if "OBSERVATION:" in c]
        check("observation prompts exist and name the changed positions", len(obs_prompts) >= 10 and
              sum("Positions that changed" in c for c in obs_prompts) >= 0.8 * len(obs_prompts), f"{len(obs_prompts)}")
        check("the blackboard holds observations and directives", len(bb.read_live(types=("observation",))) > 0 and
              any(e.type == "directive" for e in bb._entries.values()))
        check("curation ran (an LLM call with DROP/MERGE format)", any("DROP:" in c and "MERGE" in c for c in calls))
        check("mutate and crossover in this run use position / segment (no full-restatement prompts)",
              len(mut_prompts) > 0 and any("SEGMENTS:" in c for c in calls))

        print("5. fallback: a model that never answers a proposal validly")
        hist2, _ = run(seed=1, all_bad=True, tmp=tmp)
        same = all(h["out"][POP + s] == h["in"][POP + s] for h in hist2[1:] for s in range(N_SLOTS))
        check("every proposal falls back to its unchanged base, no exception", same)
        import json
        events = [json.loads(l) for l in open(Path(tmp) / "llm1.jsonl")]
        check("the fallbacks are logged as propose fallback events",
              sum(e.get("event") == "fallback_to_deterministic" and e.get("op") == "propose" for e in events) >= 20)

        print("6. determinism")
        a, _ = run(seed=2, tmp=tmp)
        b, _ = run(seed=2, tmp=tmp)
        check("same seed -> identical populations at every generation", [h["out"] for h in a] == [h["out"] for h in b])

    print("7. loud failures")
    os.environ.pop("HPGA_LLM_PROMPT_STYLE", None)
    pop = ["ACDEFGHIKLMNPQRSTVWY" * 2] * 12
    fit = [float(i) for i in range(12)]
    for var in ("HPGA_AGENTS_ENABLED", "HPGA_LOG_DIVERSITY"):
        with tempfile.TemporaryDirectory() as t2:
            setup(tmp=t2); os.environ[var] = "1"
            try:
                ops.next_generation(pop, fit, POP, 3, 0.9, 0.05, 2, random.Random(0)); check(f"{var}=1 raises", False)
            except RuntimeError as e:
                check(f"{var}=1 still raises in sequence mode", var in str(e))
    with tempfile.TemporaryDirectory() as t2:
        setup(tmp=t2, mode="deterministic")
        try:
            ops.next_generation(pop, fit, POP, 3, 0.9, 0.05, 2, random.Random(0)); check("circles + deterministic operators raises", False)
        except RuntimeError as e:
            check("circles + deterministic operators raises (a set flag must not be silently ignored)", "HPGA_CIRCLES_ENABLED" in str(e))
    gm.set_active(None)
    print(f"\n{'ALL CHECKS PASSED' if not FAILS else str(len(FAILS)) + ' CHECK(S) FAILED: ' + '; '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
