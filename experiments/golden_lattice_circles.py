"""Golden run of LATTICE circles with a stubbed model: the guard for porting circles.py to the
sequence genome. Lattice behaviour must not change by so much as a byte.

  python experiments/golden_lattice_circles.py record   # write experiments/golden/lattice_circles_golden.json
  python experiments/golden_lattice_circles.py check    # re-run, compare to the golden BYTE FOR BYTE, exit 1 on any difference

What runs: operators.next_generation with HPGA_OPERATOR_MODE=llm and HPGA_CIRCLES_ENABLED=1 on lattice
(list[int]) genomes over several generations, for two configurations (LLM central directive with
curation; deterministic central directive), with operators._call_ollama replaced by a stub. The stub's reply
is a pure function of (seed, prompt), never of call order, so a change to RNG draw order, prompt text,
num_predict, retry behaviour, parsing, blackboard writes or log fields shows up. Some replies are
deliberately malformed, so retry-then-succeed and retry-then-fall-back paths in propose / observe / consult /
central / curate are all exercised.

What is compared (canonical JSON, sorted keys): every population after every generation, the RNG state after
every generation, the agents generation counter, every (prompt, system, num_predict, seed) the stub saw, every
LLM-call log record and every blackboard log record (timestamps removed), the operator stats, the circles
per-slot history, the live blackboard entries. Nothing time-dependent is compared.
"""

import hashlib
import json
import os
import random
import re
import sys
import tempfile
from pathlib import Path

# blackboard._extract_curation iterates a set of string ids, so tombstone order depends on the per-process
# hash seed. That is existing lattice behaviour we must not change, so the golden pins the seed instead.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable] + sys.argv, {**os.environ, "PYTHONHASHSEED": "0"})

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
GOLDEN = ROOT / "experiments" / "golden" / "lattice_circles_golden.json"

os.environ["HPGA_LLM_MAX_RETRIES"] = "2"

from hpga import agents, blackboard as bb, circles, genome_model  # noqa: E402
from hpga import operators as ops  # noqa: E402

LETTERS = "SLRUD"

CASES = [
    dict(name="llm_central_curation", seed=3, pop=8, length=12, gens=9, central="llm", central_interval=2,
         curation_interval=3, n_circles=2, per_circle=2, style="best"),
    dict(name="deterministic_central", seed=11, pop=6, length=9, gens=7, central="deterministic", central_interval=1,
         curation_interval=2, n_circles=2, per_circle=2, style="best"),
]


def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


def stub_reply(prompt: str, system: str, seed: int) -> str:
    """Pure function of (prompt, seed). Roughly a fifth of replies are malformed."""
    r = random.Random(_h(seed, prompt[:200], len(prompt)))
    # higher malformation rates for the circles calls so their retry-then-fallback paths are exercised too
    p_bad = 0.4 if any(m in prompt for m in ("RATIONALE:", "OBSERVATION:", "NOTE:", "DIRECTIVE:")) else (0.3 if "DROP:" in prompt else 0.2)
    bad = r.random() < p_bad
    if "FOLD:" in prompt and "RATIONALE:" in prompt:  # propose (lattice full-restatement format)
        n = int(re.search(r"Propose a new fold of length (\d+)", prompt).group(1))
        letters = " ".join(r.choice(LETTERS) for _ in range(n - (1 if bad else 0)))
        return f"FOLD: {letters}\nRATIONALE: changed some turns near position {r.randrange(n)}"
    if "OBSERVATION:" in prompt:
        return "no observation here" if bad else f"OBSERVATION: change #{r.randrange(50)} suggests smaller edits near the middle help"
    if "NOTE:" in prompt:
        return "nothing" if bad else f"NOTE: focus on region {r.randrange(9)} of the chain and make smaller edits"
    if "DIRECTIVE:" in prompt:
        return "nothing" if bad else f"DIRECTIVE: all circles prioritise option {r.randrange(9)} this round"
    if "DROP:" in prompt and "MERGE" in prompt:  # curation
        ids = re.findall(r"^\s+(\S+-\d+) \(gen", prompt, re.MULTILINE)
        if bad or len(ids) < 2:
            return "keep everything"
        if r.random() < 0.5:
            return f"DROP: {ids[0]}"
        return f"MERGE A {ids[0]}, {ids[1]}\nSUMMARY A: merged note {r.randrange(99)}"
    if "POSITION:" in prompt and "Change exactly" in prompt:  # mutate, position style
        k = int(re.search(r"Change exactly (\d+)", prompt).group(1))
        n = int(re.search(r"POSITION: <0-(\d+)>", prompt).group(1)) + 1
        pos = r.sample(range(n), min(k, n))
        lines = [f"POSITION: {p}, NEW: {r.choice(LETTERS)}" for p in pos]
        return "\n".join(lines[:-1] if bad and len(lines) > 1 else lines)
    if "SEGMENTS:" in prompt:  # crossover, segment style (inclusive position ranges on the lattice)
        n = int(re.search(r"ending at (\d+)", prompt).group(1)) + 1
        c = r.randrange(1, max(2, n - 2))
        return "SEGMENTS: nonsense" if bad else f"SEGMENTS: 0-{c}:1, {c + 1}-{n - 1}:2"
    return "unrecognised"


def fitness_of(genome) -> float:
    return float(sum((i + 1) * g for i, g in enumerate(genome)) % 11)  # pure function of the genome, with ties


def run_case(case: dict, workdir: Path) -> dict:
    env = {
        "HPGA_OPERATOR_MODE": "llm", "HPGA_CIRCLES_ENABLED": "1", "HPGA_AGENTS_ENABLED": "0",
        "HPGA_N_CIRCLES": str(case["n_circles"]), "HPGA_AGENTS_PER_CIRCLE": str(case["per_circle"]),
        "HPGA_CENTRAL_MODE": case["central"], "HPGA_CENTRAL_INTERVAL": str(case["central_interval"]),
        "HPGA_CURATION_INTERVAL": str(case["curation_interval"]), "HPGA_LLM_PROMPT_STYLE": case["style"],
        "HPGA_RUN_ID": f"golden-{case['name']}", "HPGA_LLM_LOG_PATH": str(workdir / f"{case['name']}_llm.jsonl"),
        "HPGA_BLACKBOARD_LOG_PATH": str(workdir / f"{case['name']}_bb.jsonl"),
    }
    for k in ("HPGA_LOG_DIVERSITY", "HPGA_GA_DISPATCH_P"):
        os.environ.pop(k, None)
    os.environ.update(env)
    genome_model.set_active(None)
    ops.reset_operator_stats()
    agents.reset_agent_state()
    circles.reset_circle_state()
    bb.reset_blackboard()
    seen = []

    def stub(prompt, system, num_predict, seed):
        text = stub_reply(prompt, system, seed)
        seen.append({"prompt": prompt, "system": system, "num_predict": num_predict, "seed": seed})
        return text, 0.25, len(prompt) // 4, len(text) // 4

    real = ops._call_ollama
    ops._call_ollama = stub
    try:
        rng = random.Random(case["seed"])
        pop = ops.random_population(case["pop"], case["length"], rng)
        steps = []
        for gen in range(case["gens"]):
            fits = [fitness_of(g) for g in pop]
            pop = ops.next_generation(pop, fits, case["pop"], 3, 0.9, 0.05, 2, rng)
            steps.append({"generation": gen, "population": [list(g) for g in pop], "fitnesses_in": fits,
                          "rng_state": repr(rng.getstate()), "agents_generation": agents._generation})
    finally:
        ops._call_ollama = real

    def strip_ts(path: Path):
        out = []
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            rec.pop("timestamp", None)
            out.append(rec)
        return out

    stats = ops.get_operator_stats()
    return {
        "case": case, "steps": steps, "stub_calls": seen,
        "llm_log": strip_ts(Path(env["HPGA_LLM_LOG_PATH"])), "blackboard_log": strip_ts(Path(env["HPGA_BLACKBOARD_LOG_PATH"])),
        "operator_stats": {k: v for k, v in stats.items()},
        "circles_last_fitness_by_slot": {str(k): v for k, v in circles._last_fitness_by_slot.items()},
        "circles_last_genome_by_slot": {str(k): list(v) for k, v in circles._last_genome_by_slot.items()},
        "live_blackboard": [vars(e) for e in bb.read_live()],
    }


def produce() -> str:
    with tempfile.TemporaryDirectory(prefix="golden_circles_") as d:
        traces = [run_case(c, Path(d)) for c in CASES]
    return json.dumps(traces, sort_keys=True, indent=1)


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    text = produce()
    n_calls = sum(len(json.loads(text)[i]["stub_calls"]) for i in range(len(CASES)))
    print(f"produced {len(text)} bytes, sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}, {n_calls} stubbed model calls")
    if mode == "record":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(text)
        print(f"recorded {GOLDEN.relative_to(ROOT)}")
        return 0
    golden = GOLDEN.read_text()
    if text == golden:
        print("GOLDEN REPRODUCED BYTE FOR BYTE")
        return 0
    print("GOLDEN DIFFERS")
    a, b = golden.splitlines(), text.splitlines()
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            print(f"first difference at line {i + 1}:\n  golden: {x[:200]}\n  now:    {y[:200]}")
            break
    else:
        print(f"line counts differ: golden {len(a)} vs now {len(b)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
