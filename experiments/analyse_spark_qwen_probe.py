"""Re-derive every number in results/SPARK_QWEN35_PROBE.md from the raw files in results/raw/spark_qwen/ (no GPU, no LLM).

The two directories there were produced on a DGX Spark against a vLLM OpenAI-compatible endpoint serving
nvidia/Qwen3.6-35B-A3B-NVFP4, by a client that is not in this repository. This script therefore also checks the logs
against the committed code: every logged prompt (retries included) is rebuilt with hpga/sequence_model*.py, every
`valid` flag is recomputed with the committed parsers, and the request-seed stream is replayed. The historical
summarizers (probe_mutate_oldfield.py / probe_mutate_numbered.py summarize) are not used: they iterate a fixed model list.

Scope as pre-registered for the numbered probe: first attempts; all-attempts numbers are printed alongside.

  python experiments/analyse_spark_qwen_probe.py [--out-md results/SPARK_QWEN35_PROBE_TABLES.md]
"""

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hpga import genome_model, sequence_model as sm, sequence_model_numbered as nb, sequence_model_oldfield as of  # noqa: E402
from hpga.config import HPGAConfig  # noqa: E402

RAW = ROOT / "results" / "raw" / "spark_qwen"
GATE, NUM = RAW / "oldfield_gate_spark_qwen", RAW / "numbered_probe_spark_qwen"
TAG = "nvidia_Qwen3.6-35B-A3B-NVFP4"
CELLS = {"C1_old": GATE, "C1_new": GATE, "C2_old": GATE, "C2_new": GATE, "C3_old": GATE, "C3_new": GATE,
         "C1_drift": NUM, "C2_num": NUM, "C2_numold": NUM, "C3_num": NUM, "C3_numold": NUM}
LABEL = {"C1_old": "spaced/base", "C1_new": "spaced/OLD", "C2_old": "spaced/base", "C2_new": "spaced/OLD",
         "C3_old": "spaced/base", "C3_new": "spaced/OLD", "C1_drift": "spaced/base (drift)",
         "C2_num": "numbered/base", "C2_numold": "numbered/OLD", "C3_num": "numbered/base", "C3_numold": "numbered/OLD"}
_SEQ = re.compile(r"positions 0-\d+\): ((?:\d+:)?[A-Z](?: (?:\d+:)?[A-Z])*)\n")
_PN = re.compile(r"POSITION\s*:\s*(\d+)\s*,\s*NEW\s*:\s*([A-Za-z])", re.I)
_PON = re.compile(r"POSITION\s*:\s*(\d+)\s*,\s*OLD\s*:\s*([A-Za-z])\s*,\s*NEW\s*:\s*([A-Za-z])", re.I)


def old_slot(cell: str) -> bool:
    return cell.endswith("_new") or cell.endswith("numold")


def genome_of(prompt: str) -> str:
    return "".join(t.split(":")[-1] for t in _SEQ.search(prompt).group(1).split())


def plan_for(cell: str, g: str, k: int):
    if cell in ("C1_old", "C2_old", "C3_old", "C1_drift"):
        return sm.plan_llm_mutate("position", g, k)
    if cell in ("C1_new", "C2_new", "C3_new"):
        return of.plan_llm_mutate_oldfield(g, k)
    return nb.plan_llm_mutate_numbered(g, k, old_slot=old_slot(cell))


def band(p: int) -> str:
    return "0-9" if p < 10 else "10-19" if p < 20 else "20+"


def checks(cell: str, recs: list[dict], c2_inputs, c3_inputs, model) -> dict:
    att = [r for r in recs if "response" in r]
    fb = {r["input_index"] for r in recs if r.get("event") == "fallback_to_deterministic"}
    by = collections.defaultdict(list)
    for r in att:
        by[r["input_index"]].append(r)
    structure = all([x["attempt"] for x in a] == list(range(len(a))) and not any(x["valid"] for x in a[:-1])
                    and ((not a[-1]["valid"]) == (i in fb)) and (a[-1]["valid"] or len(a) == 3) for i, a in by.items())
    valid_mismatch = prompt_mismatch = 0
    for r in att:
        g, k = genome_of(r["prompt"]), r["k"]
        plan = plan_for(cell, g, k)
        valid_mismatch += (plan.parse(r["response"]) is not None) != r["valid"]
        prompt_mismatch += plan.build_prompt("" if r["attempt"] == 0 else plan.retry_hint_text) != r["prompt"]
    inputs_ok = None
    if cell[:2] in ("C2", "C3"):
        ref = c2_inputs if cell[:2] == "C2" else c3_inputs
        inputs_ok = all((genome_of(a[0]["prompt"]), a[0]["k"]) == ref[i] for i, a in by.items())
    rng, seeds_ok = random.Random(0), True  # the drivers' request-seed stream, fallbacks replayed
    for i in sorted(by):
        if cell[:2] == "C1" and sm.random_sequence(rng, 63) != genome_of(by[i][0]["prompt"]):
            seeds_ok = False
        seeds_ok &= all(rng.getrandbits(31) == x["seed"] for x in by[i])
        if not by[i][-1]["valid"]:
            model.deterministic_mutate(genome_of(by[i][0]["prompt"]), 1.0 / 63 if cell[:2] == "C1" else 0.05, rng)
    return {"attempts": len(att), "calls": len(by), "fallbacks": len(fb), "errors": sum("error" in r for r in recs),
            "models": sorted({r["model"] for r in att}), "temperature": sorted({r["temperature"] for r in att}),
            "structure_ok": structure, "valid_flag_mismatches": valid_mismatch, "prompt_rebuild_mismatches": prompt_mismatch,
            "inputs_ok": inputs_ok, "seed_stream_ok": seeds_ok}


def score(cell: str, recs: list[dict], scope: str) -> dict:
    rows = [r for r in recs if "response" in r and (scope == "all" or r["attempt"] == 0)]
    c, bands = collections.Counter(), collections.defaultdict(lambda: [0, 0])
    invalid_kinds = collections.Counter()
    for r in rows:
        g = genome_of(r["prompt"])
        c["attempts"] += 1
        c["valid"] += r["valid"]
        pon = _PON.findall(r["response"]) if old_slot(cell) else []
        lines = [(int(p), o.upper(), n.upper()) for p, o, n in pon] if pon else [(int(p), None, n.upper()) for p, n in _PN.findall(r["response"])]
        lines = [x for x in lines if x[0] < len(g)]
        for p, o, n in lines:
            c["lines"] += 1
            c["repeats"] += n == g[p]
            if old_slot(cell) and o is not None:
                ok = o == g[p]
                c["old_lines"] += 1
                c["read_ok"] += ok
                c["cond_repeats"] += ok and n == g[p]
                c["misread_repeats"] += (not ok) and n == g[p]
                bands[band(p)][0] += ok
                bands[band(p)][1] += 1
        if old_slot(cell) and not r["valid"]:
            k = r["k"]
            wrong_old = any(o != g[p] for p, o, _ in lines if o is not None)
            same = any(n == g[p] for p, _, n in lines)
            invalid_kinds["no OLD-slot line" if not pon else "wrong number of lines" if len(pon) != k
                          else ("OLD wrong, and a NEW equals the current letter" if same else "OLD wrong, every NEW differs")
                          if wrong_old else "OLD right, other reason"] += 1
    d = dict(c)
    d["bands"] = {b: bands[b] for b in ("0-9", "10-19", "20+") if bands[b][1]}
    d["invalid_kinds"] = dict(invalid_kinds)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-md", default=str(ROOT / "results" / "SPARK_QWEN35_PROBE_TABLES.md"))
    a = ap.parse_args()
    model = genome_model.build_genome_model(HPGAConfig(genome_model="sequence"))
    recs = {c: [json.loads(l) for l in open(d / f"calls_{TAG}_{c}.jsonl")] for c, d in CELLS.items()}
    bases = json.load(open(ROOT / "results" / "raw" / "oldfield_gate" / "oldfield_replay_bases.json"))["bases"]
    spark_bases = json.load(open(GATE / "oldfield_replay_bases.json"))["bases"]
    grng = random.Random(2)
    c2_inputs = [(sm.random_sequence(grng, 63), 3) for _ in range(200)]
    c3_inputs = [(b["genome"], b["k"]) for b in bases]

    md = ["# Spark probe, nvidia/Qwen3.6-35B-A3B-NVFP4: generated tables (experiments/analyse_spark_qwen_probe.py)", "",
          "Not a controlled model-scale comparison: a different model from the one requested, served by vLLM on a DGX Spark, "
          "not by Ollama. Source: results/raw/spark_qwen/. Read with results/SPARK_QWEN35_PROBE.md.", "",
          "## 1. Log checks against the committed code", "",
          f"- Replay bases in the Spark gate directory identical to results/raw/oldfield_gate/oldfield_replay_bases.json: {spark_bases == bases}", "",
          "| cell | calls | attempts | fallbacks | transport errors | model field | temperature | attempt structure ok | `valid` flags not reproduced by the committed parser | prompts not rebuilt byte for byte by the committed code | inputs equal the fixed C2/C3 lists | request-seed stream replays |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cell in CELLS:
        k = checks(cell, recs[cell], c2_inputs, c3_inputs, model)
        md.append(f"| {cell} | {k['calls']} | {k['attempts']} | {k['fallbacks']} | {k['errors']} | {', '.join(k['models'])} | "
                  f"{k['temperature']} | {k['structure_ok']} | {k['valid_flag_mismatches']} | {k['prompt_rebuild_mismatches']} | "
                  f"{'n/a (C1)' if k['inputs_ok'] is None else k['inputs_ok']} | {k['seed_stream_ok']} |")

    a1 = [r for r in recs["C1_old"] if "prompt" in r]
    b1 = [r for r in recs["C1_drift"] if "prompt" in r]
    fa, fb = [r for r in a1 if r["attempt"] == 0], [r for r in b1 if r["attempt"] == 0]
    md += ["", "## 2. C1 drift check (spaced/base C1 of the Spark gate against the Spark numbered probe's C1_drift)", "",
           f"- First-attempt prompts: {sum(x['prompt'] == y['prompt'] for x, y in zip(fa, fb))} of {len(fa)} identical "
           f"(and {sum(x['seed'] == y['seed'] for x, y in zip(fa, fb))} of {len(fa)} request seeds identical).",
           f"- All-attempt prompt streams: {len(a1)} against {len(b1)} prompts; identical as lists: "
           f"{[x['prompt'] for x in a1] == [y['prompt'] for y in b1]} (the test `probe_mutate_numbered.py summarize` applies).",
           f"- Retries in C1_old at input_index {[r['input_index'] for r in a1 if r['attempt'] > 0]}; "
           f"in C1_drift at {[r['input_index'] for r in b1 if r['attempt'] > 0]}.",
           f"- Same prompt and same request seed, different response: {sum(x['response'] != y['response'] for x, y in zip(fa, fb))} "
           f"of {len(fa)} first-attempt pairs."]
    for om in ("gemma4_12b", "qwen2.5_7b", "mistral_7b", "llama3.2_3b"):
        o = [json.loads(l)["prompt"] for l in open(ROOT / "results" / "raw" / "oldfield_gate" / f"calls_{om}_C1_old.jsonl")
             if '"attempt": 0' in l]
        md.append(f"- Spark C1_old first-attempt prompts against the 2026-10-01 gate's {om} C1_old first attempts: "
                  f"{sum(x == y['prompt'] for x, y in zip(o, fa))} of {min(len(o), len(fa))} identical.")

    for scope in ("first", "all"):
        s = {c: score(c, recs[c], scope) for c in CELLS}
        md += ["", f"## {3 if scope == 'first' else 4}. {'First attempts (the pre-registered scope)' if scope == 'first' else 'All attempts (reported, not scored)'}", "",
               "| cell | display/response | attempts | valid | lines | NEW = current letter (rate) | reading (OLD correct) | reading by band 0-9 / 10-19 / 20+ | repeats on correctly read lines | repeats on misread lines |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for c, d in s.items():
            rd = f"{d['read_ok']}/{d['old_lines']} = {d['read_ok'] / d['old_lines']:.2f}" if d.get("old_lines") else ""
            bd = " / ".join(f"{v[0]}/{v[1]} = {v[0] / v[1]:.2f}" for v in d["bands"].values())
            md.append(f"| {c} | {LABEL[c]} | {d['attempts']} | {d['valid']} | {d['lines']} | {d['repeats']} ({d['repeats'] / d['lines']:.4f}) | "
                      f"{rd} | {bd} | {d.get('cond_repeats', '') if old_slot(c) else ''} | {d.get('misread_repeats', '') if old_slot(c) else ''} |")
        ok_lines = sum(d.get("read_ok", 0) for d in s.values())
        md += ["", f"Correctly read lines, all OLD-slot cells ({', '.join(c for c in CELLS if old_slot(c))}): {ok_lines}; "
               f"of those with NEW = the letter just stated: {sum(d.get('cond_repeats', 0) for d in s.values())}. "
               f"C2 and C3 only: {sum(s[c].get('read_ok', 0) for c in ('C2_new', 'C3_new', 'C2_numold', 'C3_numold'))}."]
        if scope == "first":
            md += ["", "Invalid first attempts in the OLD-slot cells, by reason (the strict parser rejects a line whose OLD is not the "
                   "letter at that position, hpga/sequence_model_oldfield.py:77):", ""]
            md += [f"- {c}: {s[c]['invalid_kinds']}" for c in CELLS if old_slot(c)]

    md += ["", "## 5. Execution (from the per-model JSON files; wall-clock time, nothing records GPU utilisation; no folding ran)", "",
           "| probe | cell | calls | HTTP requests | fallbacks | wall s |", "|---|---|---|---|---|---|"]
    for probe, path in (("gate", GATE / f"oldfield_{TAG}.json"), ("numbered", NUM / f"numbered_{TAG}.json")):
        for cell, v in json.load(open(path))["cells"].items():
            st = v["stats"]
            md.append(f"| {probe} | {cell} | {st['n_llm_calls']} | {st['n_llm_requests']} | {st['n_failures']} | {v['wall_s']:.1f} |")
    text = "\n".join(md) + "\n"
    Path(a.out_md).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
