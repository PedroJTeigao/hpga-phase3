"""Guard for the LATTICE agents (hpga/agents.py) while the sequence agents are added: the same scenario is run
against a baseline git ref's tree and against the working tree, with a stubbed model, and the two outputs must be
identical byte for byte.

  python experiments/golden_lattice_agents.py check [--baseline REF]   # default REF a4967b0 (main when this work began)
  python experiments/golden_lattice_agents.py dump --tree DIR          # canonical JSON for the tree rooted at DIR

What runs (lattice list[int] genomes, HPGA_OPERATOR_MODE=llm, HPGA_AGENTS_ENABLED=1, operators._call_ollama stubbed):
two configurations -- 2 agents, default explore/refine roles, communication rounds every 2 generations; and 3 agents
(explore,explore,refine), communication every 3, with HPGA_LOG_DIVERSITY=1 and HPGA_AGENTS_MEMORY=1 set (a flag the
lattice code must ignore). The stub's reply is a pure function of (seed, prompt) and roughly a fifth of replies are
malformed, so the explore full-fold parser, the refine position parser, and their retry / fallback paths all run.

Compared (timestamps and latencies removed): every population after every generation, the RNG state after every
generation, the agents generation counter, every (prompt, system, num_predict, seed) the stub saw, every LLM-call log
record, every agent message record, every diversity record, the operator stats.
"""

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LETTERS = "SLRUD"

CASES = [
    dict(name="two_agents_comm2", seed=5, pop=8, length=12, gens=9, n_agents=2, roles="", comm=2, div=False, mem=False),
    dict(name="three_agents_div_mem", seed=17, pop=6, length=9, gens=8, n_agents=3, roles="explore,explore,refine", comm=3,
         div=True, mem=True),
]


def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


def stub_reply(prompt: str, seed: int) -> str:
    r = random.Random(_h(seed, prompt[:200], len(prompt)))
    bad = r.random() < 0.2
    if "FOLD:" in prompt:  # agents' explore
        n = int(re.search(r"fold of length (\d+)", prompt).group(1))
        return "FOLD: " + " ".join(r.choice(LETTERS) for _ in range(n - (1 if bad else 0)))
    if "POSITION:" in prompt and "Change at least" in prompt:  # agents' refine
        n = int(re.search(r"0-(\d+)\):", prompt).group(1)) + 1
        cnt = r.choice([1, 2, 3])
        lines = [f"POSITION: {p}, NEW: {r.choice(LETTERS)}" for p in r.sample(range(n), cnt)]
        return "no changes" if bad else "\n".join(lines)
    if "POSITION:" in prompt and "Change exactly" in prompt:  # mutate, position style
        k = int(re.search(r"Change exactly (\d+)", prompt).group(1))
        n = int(re.search(r"POSITION: <0-(\d+)>", prompt).group(1)) + 1
        lines = [f"POSITION: {p}, NEW: {r.choice(LETTERS)}" for p in r.sample(range(n), min(k, n))]
        return "\n".join(lines[:-1] if bad and len(lines) > 1 else lines)
    if "SEGMENTS:" in prompt:  # crossover, segment style
        n = int(re.search(r"ending at (\d+)", prompt).group(1)) + 1
        c = r.randrange(1, max(2, n - 2))
        return "SEGMENTS: nonsense" if bad else f"SEGMENTS: 0-{c}:1, {c + 1}-{n - 1}:2"
    return "unrecognised"


def fitness_of(genome) -> float:
    return float(sum((i + 1) * g for i, g in enumerate(genome)) % 11)


def dump(tree: Path) -> str:
    os.environ["PYTHONHASHSEED"] = "0"
    sys.path.insert(0, str(tree))
    from hpga import agents, genome_model, operators as ops  # noqa: E402

    assert Path(agents.__file__).resolve().is_relative_to(tree.resolve()), agents.__file__
    traces = []
    with tempfile.TemporaryDirectory(prefix="golden_agents_") as d:
        for case in CASES:
            w = Path(d)
            env = {
                "HPGA_OPERATOR_MODE": "llm", "HPGA_AGENTS_ENABLED": "1", "HPGA_CIRCLES_ENABLED": "0",
                "HPGA_N_AGENTS": str(case["n_agents"]), "HPGA_COMM_INTERVAL": str(case["comm"]),
                "HPGA_LLM_PROMPT_STYLE": "best", "HPGA_RUN_ID": f"golden-{case['name']}", "HPGA_LLM_MAX_RETRIES": "2",
                "HPGA_LLM_LOG_PATH": str(w / f"{case['name']}_llm.jsonl"),
                "HPGA_AGENT_LOG_PATH": str(w / f"{case['name']}_msg.jsonl"),
                "HPGA_DIVERSITY_LOG_PATH": str(w / f"{case['name']}_div.jsonl"),
            }
            for k in ("HPGA_LOG_DIVERSITY", "HPGA_GA_DISPATCH_P", "HPGA_AGENT_ROLES", "HPGA_AGENTS_MEMORY"):
                os.environ.pop(k, None)
            os.environ.update(env)
            if case["roles"]:
                os.environ["HPGA_AGENT_ROLES"] = case["roles"]
            if case["div"]:
                os.environ["HPGA_LOG_DIVERSITY"] = "1"
            if case["mem"]:
                os.environ["HPGA_AGENTS_MEMORY"] = "1"
            genome_model.set_active(None)
            ops.reset_operator_stats()
            agents.reset_agent_state()
            seen = []

            def stub(prompt, system, num_predict, seed):
                text = stub_reply(prompt, seed)
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

            def jl(path, drop=("timestamp", "latency_s")):
                out = []
                if Path(path).exists():
                    for line in open(path, encoding="utf-8"):
                        rec = json.loads(line)
                        for k in drop:
                            rec.pop(k, None)
                        out.append(rec)
                return out

            traces.append({"case": case, "steps": steps, "stub_calls": seen, "llm_log": jl(env["HPGA_LLM_LOG_PATH"]),
                           "messages": jl(env["HPGA_AGENT_LOG_PATH"]), "diversity": jl(env["HPGA_DIVERSITY_LOG_PATH"]),
                           "operator_stats": {k: v for k, v in ops.get_operator_stats().items() if "latenc" not in k}})
    return json.dumps(traces, sort_keys=True, indent=1)


def run_tree(tree: Path) -> str:
    return subprocess.run([sys.executable, str(Path(__file__).resolve()), "dump", "--tree", str(tree)], check=True,
                          capture_output=True, text=True, cwd=str(tree), env={**os.environ, "PYTHONHASHSEED": "0"}).stdout


def check(ref: str) -> int:
    with tempfile.TemporaryDirectory(prefix="agents_baseline_") as d:
        tar = subprocess.run(["git", "-C", str(ROOT), "archive", ref, "hpga"], check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", d], input=tar, check=True)
        before = run_tree(Path(d))
    after = run_tree(ROOT)
    n_calls = sum(len(t["stub_calls"]) for t in json.loads(after))
    print(f"baseline {ref}: {len(before)} bytes sha256 {hashlib.sha256(before.encode()).hexdigest()[:16]}; "
          f"working tree: {len(after)} bytes sha256 {hashlib.sha256(after.encode()).hexdigest()[:16]}; {n_calls} stubbed model calls")
    if before == after:
        print("LATTICE AGENTS IDENTICAL TO BASELINE, BYTE FOR BYTE")
        return 0
    a, b = before.splitlines(), after.splitlines()
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            print(f"first difference at line {i + 1}:\n  baseline: {x[:200]}\n  now:      {y[:200]}")
            break
    else:
        print(f"line counts differ: baseline {len(a)} vs now {len(b)}")
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "dump"])
    ap.add_argument("--baseline", default="a4967b0")
    ap.add_argument("--tree", default=None)
    a = ap.parse_args()
    if a.mode == "dump":
        sys.stdout.write(dump(Path(a.tree).resolve()))
    else:
        sys.exit(check(a.baseline))
