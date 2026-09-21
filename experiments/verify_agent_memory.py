"""Checks for agents (and their optional private record) on the SEQUENCE genome, hpga/agents_sequence.py, with a
stubbed model: no GPU, no Ollama. The lattice agents are guarded separately, against a baseline git ref, by
golden_lattice_agents.py.

  1  shape          each next_generation returns pop_size + n_agents str genomes, every one inside the length bounds
  2  edits          from generation 1 each agent's proposal is its tail slot's previous occupant with exactly k
                    positions changed, or unchanged (the logged fallback)
  3  routing        every record entry pairs an agent's own base/proposal with THAT proposal's measured fitness;
                    misrouted == 0; entries per agent == generations - 2; improved == (after > before)
  4  memory off     agent prompts are byte for byte the plain mutate/position prompt (no record block at all)
  5  memory on      agent prompt == render_record(this agent's entries so far) + the plain prompt, from the
                    generation where the first entry exists (2); the numbers shown are that agent's own
  6  privacy        no fitness pair shown to agent i appears only in another agent's record (3 agents, so a
                    leak between any two would show)
  7  fallback       a model that never answers a proposal validly: proposals equal their base, entries are
                    no_edit, never shown, never counted as improved; no exception
  8  determinism    same seed, same populations and same record log
  9  reset          agents.reset_agent_state() empties the records
  10 loud failures  agents with deterministic operators raises; diversity logging still raises; prompts never name
                    the target structure
Exit status 0 only if every check passes.
"""

import hashlib
import json
import os
import random
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hpga import agents, agents_sequence as aseq, genome_model as gm  # noqa: E402
from hpga import operators as ops  # noqa: E402
from hpga import sequence_model as sm  # noqa: E402
from hpga.config import HPGAConfig  # noqa: E402

FAILS: list[str] = []
POP, GENS = 8, 8


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail and not ok else ''}")
    if not ok:
        FAILS.append(label)


def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def stub_fitness(g: str) -> float:
    # depends on letters AND their positions, so two different genomes almost never tie
    return sum((i % 7 + 1) * (sm.ALPHABET.index(c) + 1) for i, c in enumerate(g)) % 1000 / 1000.0


def make_stub(all_bad=False):
    calls = []

    def stub(prompt, system, num_predict, seed):
        r = random.Random(_h(seed, prompt[:400], len(prompt)))
        bad = all_bad or r.random() < 0.60  # high, so retry-then-fallback is exercised
        calls.append(prompt)
        if "SEGMENTS:" in prompt:
            text = "SEGMENTS: junk" if bad else "SEGMENTS: 0-40:1, 40-100:2"
        elif "POSITION:" in prompt and "Change exactly" in prompt:
            k = int(re.search(r"Change exactly (\d+)", prompt).group(1))
            n = int(re.search(r"positions 0-(\d+)\)", prompt).group(1)) + 1
            cur = "".join(re.search(r"Sequence \(0-indexed positions 0-\d+\): ([A-Z ]+)", prompt).group(1).split())
            pos = r.sample(range(n), k)
            text = "unparseable" if bad else "\n".join(
                f"POSITION: {p}, NEW: {r.choice([c for c in sm.ALPHABET if c != cur[p]])}" for p in pos)
        else:
            text = "unrecognised"
        return text, 0.1, len(prompt) // 4, len(text) // 4

    return stub, calls


def setup(tmp: str, memory: bool, n_agents: int = 2, mode: str = "llm", seed: int = 0, tag: str = ""):
    for k in ("HPGA_CIRCLES_ENABLED", "HPGA_LOG_DIVERSITY", "HPGA_GA_DISPATCH_P", "HPGA_AGENTS_SEQ_EDIT_RATE",
              "HPGA_AGENTS_MEMORY_WINDOW", "HPGA_LLM_PROMPT_STYLE"):
        os.environ.pop(k, None)
    os.environ.update({
        "HPGA_OPERATOR_MODE": mode, "HPGA_AGENTS_ENABLED": "1", "HPGA_N_AGENTS": str(n_agents),
        "HPGA_AGENTS_MEMORY": "1" if memory else "0", "HPGA_RUN_ID": f"vam-{seed}{tag}", "HPGA_LLM_MAX_RETRIES": "2",
        "HPGA_LLM_LOG_PATH": str(Path(tmp) / f"llm{seed}{tag}.jsonl"),
        "HPGA_AGENT_MEMORY_LOG_PATH": str(Path(tmp) / f"mem{seed}{tag}.jsonl"),
    })
    gm.set_active(gm.build_genome_model(HPGAConfig(genome_model="sequence")))
    ops.reset_operator_stats()
    agents.reset_agent_state()


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


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")] if Path(path).exists() else []


def run(tmp, memory, n_agents=2, seed=0, all_bad=False, tag=""):
    setup(tmp, memory, n_agents, seed=seed, tag=tag)
    stub, calls = make_stub(all_bad)
    real, undo = ops._call_ollama, style_wrappers()
    ops._call_ollama = stub
    history = []
    try:
        rng = random.Random(seed)
        pop = ops.random_population(POP, None, rng)
        for _ in range(GENS):
            fits = [stub_fitness(g) for g in pop]
            new = ops.next_generation(pop, fits, POP, 3, 0.9, 0.05, 2, rng)
            history.append({"in": list(pop), "fits": fits, "out": list(new)})
            pop = new
    finally:
        ops._call_ollama = real
        undo()
    mem = read_jsonl(os.environ["HPGA_AGENT_MEMORY_LOG_PATH"])
    llm = read_jsonl(os.environ["HPGA_LLM_LOG_PATH"])
    return {"history": history, "mem": mem, "llm": llm, "records": aseq.get_records(), "stats": aseq.get_stats(),
            "calls": calls}


def plain_prompt(genome: str) -> str:
    return sm.plan_llm_mutate("position", genome, aseq.n_edits(len(genome))).build_prompt("")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        n_agents = 3
        print("1-3. shape, edits, routing (memory on, 3 agents, ~60% malformed replies)")
        R = run(tmp, memory=True, n_agents=n_agents, seed=0)
        H = R["history"]
        check("every generation returns pop_size + n_agents genomes",
              all(len(h["out"]) == POP + n_agents and all(isinstance(g, str) for g in h["out"]) for h in H))
        check("every genome inside [30, 80]", all(sm.MIN_LENGTH <= len(g) <= sm.MAX_LENGTH for h in H for g in h["out"]))
        props = [e for e in R["mem"] if e["event"] == "proposal"]
        ok_edit = True
        for e in props:
            k = aseq.n_edits(len(e["base"]))
            d = len(e["changes"])
            ok_edit &= (d == k) or (d == 0 and e["proposal"] == e["base"])
        check("each proposal is its base with exactly k positions changed, or unchanged (fallback)", ok_edit)
        check("some proposals were valid edits and some fell back", any(not e["fallback"] for e in props) and any(e["fallback"] for e in props))
        # a proposal's base from generation 1 on is the tail slot of the population handed in
        tail_ok = all(
            e["base"] == H[e["generation"]]["in"][POP + e["agent_id"]] for e in props if e["generation"] >= 1)
        check("from generation 1 an agent's base is its own tail slot (population[pop_size + i])", tail_ok)
        tail_out_ok = all(
            e["proposal"] == H[e["generation"]]["out"][POP + e["agent_id"]] for e in props)
        check("an agent's proposal is what next_generation appends at its own tail index", tail_out_ok)

        res = [e for e in R["mem"] if e["event"] == "resolved"]
        by_prop = {(e["agent_id"], e["generation"]): e for e in props}
        route_ok = True
        for e in res:
            p = by_prop[(e["agent_id"], e["generation"])]
            route_ok &= (e["fitness_after"] == stub_fitness(p["proposal"]) and e["base_fitness"] == stub_fitness(p["base"])
                         and e["improved"] == (e["fitness_after"] > e["base_fitness"] and not e["no_edit"]))
        check("every entry pairs the agent's own base and proposal with their measured fitnesses", route_ok and len(res) > 0)
        check("misrouted == 0", R["stats"]["misrouted"] == 0)
        per_agent = {i: len(R["records"].get(i, [])) for i in range(n_agents)}
        check(f"entries per agent == generations - 2 ({GENS - 2})", all(v == GENS - 2 for v in per_agent.values()), str(per_agent))
        check("a generation-0 proposal has no entry (its base was never evaluated)",
              all(e["generation"] >= 1 for e in res))

        print("4. memory off")
        Roff = run(tmp, memory=False, n_agents=n_agents, seed=0, tag="off")
        ag = [c for c in Roff["llm"] if c.get("op") == "agent_propose" and "prompt" in c]
        check("agent calls exist", len(ag) > 0)
        ok = all("Your own record" not in c["prompt"] and not c["memory_enabled"] and c["record_entries_shown"] == 0 for c in ag)
        check("no agent prompt carries a record block", ok)
        base_of = {(e["agent_id"], e["generation"]): e["base"] for e in Roff["mem"] if e["event"] == "proposal"}
        check("memory-off prompt is byte for byte the plain mutate/position prompt on the agent's base",
              all(c["prompt"] == plain_prompt(base_of[(c["agent_id"], c["generation"])]) for c in ag if c["attempt"] == 0))
        check("the record is still kept and logged when memory is off",
              Roff["stats"]["resolved"] == R["stats"]["resolved"] or Roff["stats"]["resolved"] > 0)

        print("5. memory on")
        ag = [c for c in R["llm"] if c.get("op") == "agent_propose" and "prompt" in c and c["attempt"] == 0]
        entries_before = {}  # (agent, generation) -> entries resolved before that proposal
        seen = {i: [] for i in range(n_agents)}
        for gen in range(GENS):
            for i in range(n_agents):
                seen[i] += [e for e in res if e["agent_id"] == i and e["resolved_at_generation"] == gen]
                entries_before[(i, gen)] = [dict(e) for e in seen[i]]
        exp_ok, block_from, count_ok = True, [], True
        for c in ag:
            i, g = c["agent_id"], c["generation"]
            ents = entries_before[(i, g)]
            base = [e for e in R["mem"] if e["event"] == "proposal" and e["agent_id"] == i and e["generation"] == g][0]["base"]
            exp_ok &= c["prompt"] == aseq.render_record(ents) + plain_prompt(base)
            if "Your own record" in c["prompt"]:
                block_from.append(g)
            count_ok &= c["record_entries_shown"] == len([e for e in ents if not e["no_edit"]][-aseq.memory_window():])
        check("prompt == render_record(this agent's entries so far) + plain prompt, every agent call", exp_ok)
        check("the record block first appears at generation 2, never before", block_from and min(block_from) == 2)
        check("record_entries_shown in the call log matches what was shown", count_ok)
        lastc = [c for c in ag if c["generation"] == GENS - 1][0]
        n_lines = len(re.findall(r"^  round \d+:", lastc["prompt"], re.MULTILINE))
        n_valid = len([e for e in entries_before[(lastc["agent_id"], GENS - 1)] if not e["no_edit"]])
        m = re.search(r"So far (\d+) of your (\d+) edits", lastc["prompt"])
        check(f"the window caps the shown entries at {aseq.memory_window()}; the tally covers all valid entries",
              n_lines == min(aseq.memory_window(), n_valid) and m is not None and int(m.group(2)) == n_valid,
              f"lines={n_lines} valid={n_valid} tally={m.group(0) if m else None}")

        print("6. privacy (3 agents)")
        leak = 0
        for c in ag:
            i = c["agent_id"]
            mine = {(e["generation"], f"{e['base_fitness']:.3f}", f"{e['fitness_after']:.3f}") for e in R["records"][i]}
            for g, a, b in re.findall(r"round (\d+): .*?fitness ([\d.]+) -> ([\d.]+)", c["prompt"]):
                if (int(g), a, b) not in mine:
                    leak += 1
        check("every entry shown to agent i is in agent i's own record", leak == 0, f"{leak} foreign lines")

        print("7. fallback")
        Rbad = run(tmp, memory=True, n_agents=2, seed=1, all_bad=True, tag="bad")
        pr = [e for e in Rbad["mem"] if e["event"] == "proposal"]
        check("every proposal falls back to its unchanged base, no exception", all(e["proposal"] == e["base"] and e["fallback"] for e in pr))
        rs = [e for e in Rbad["mem"] if e["event"] == "resolved"]
        check("entries exist, all no_edit and not improved", rs and all(e["no_edit"] and not e["improved"] for e in rs))
        agp = [c for c in Rbad["llm"] if c.get("op") == "agent_propose" and "prompt" in c]
        check("a no_edit entry is never shown to the model", all("Your own record" not in c["prompt"] for c in agp))
        check("the fallbacks are logged as agent_propose fallback events",
              any(c.get("event") == "fallback_to_deterministic" and c.get("op") == "agent_propose" for c in Rbad["llm"]))

        print("8. determinism, 9. reset")
        R2 = run(tmp, memory=True, n_agents=n_agents, seed=0, tag="again")
        strip = lambda m: [{k: v for k, v in e.items() if k != "timestamp"} for e in m]  # noqa: E731
        check("same seed, same populations", [h["out"] for h in R2["history"]] == [h["out"] for h in R["history"]])
        check("same seed, same record log", strip(R2["mem"]) == strip(R["mem"]))
        agents.reset_agent_state()
        check("reset_agent_state empties the records", aseq.get_records() == {} and aseq.get_stats()["proposals"] == 0)

        print("10. loud failures")
        pop = ["ACDEFGHIKLMNPQRSTVWY" * 2] * 12
        fit = [float(i) for i in range(12)]
        setup(tmp, memory=True, mode="deterministic", tag="det")
        try:
            ops.next_generation(pop, fit, POP, 3, 0.9, 0.05, 2, random.Random(0))
            check("agents + deterministic operators raises", False)
        except RuntimeError as e:
            check("agents + deterministic operators raises (a set flag must not be silently ignored)", "HPGA_AGENTS_ENABLED" in str(e))
        setup(tmp, memory=False, tag="div")
        os.environ["HPGA_LOG_DIVERSITY"] = "1"
        try:
            ops.next_generation(pop, fit, POP, 3, 0.9, 0.05, 2, random.Random(0))
            check("diversity logging still raises in sequence mode", False)
        except RuntimeError as e:
            check("diversity logging still raises in sequence mode", "HPGA_LOG_DIVERSITY" in str(e))
        bad = [c for c in R["llm"] + Roff["llm"] if "prompt" in c and re.search(r"7UR7|\btarget\b", c["prompt"] + c.get("system", ""), re.IGNORECASE)]
        check("prompts never name the target structure", not bad)
        gm.set_active(None)

    print(f"\n{'ALL CHECKS PASSED' if not FAILS else str(len(FAILS)) + ' CHECK(S) FAILED: ' + '; '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
