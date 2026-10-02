"""Numbered-display probe (no GA, no ESMFold), pre-registered in results/PREREGISTERED_NUMBERED.md.

New cells per model (the spaced cells are the OLD-slot gate's, reused under rule 1 of the pre-registration):
  C1_drift   spaced display, base response, k = 1, the gate's C1 stream: its prompts must equal the gate's
             calls_<model>_C1_old.jsonl byte for byte, or the gate's spaced cells are not reused for that model
  C2_num     numbered display, base response       C2 inputs (200 random length-63 sequences from Random(2), k = 3)
  C2_numold  numbered display, OLD-slot response   same inputs
  C3_num     numbered display, base response       C3 inputs (the gate's 200 replayed GA bases, read from
  C3_numold  numbered display, OLD-slot response   results/raw/oldfield_gate/oldfield_replay_bases.json)
Request seeds: a fresh Random(0) per cell for C2/C3, exactly as the gate, so the inputs and seed streams match the
gate's spaced cells. All calls go through operators._run_llm_op (production retries, fallback, logging).

  python experiments/probe_mutate_numbered.py all --out-dir results/raw/numbered_probe [--models ...]
  python experiments/probe_mutate_numbered.py summarize --out-dir results/raw/numbered_probe
"""

import argparse
import collections
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import probe_mutate_oldfield as gate  # noqa: E402  (helpers only; the gate driver is not modified)

GATE_DIR = ROOT / "results" / "raw" / "oldfield_gate"
CELLS = ("C1_drift", "C2_num", "C2_numold", "C3_num", "C3_numold")
_SEQ_ANY = re.compile(r"positions 0-\d+\): ((?:\d+:)?[A-Z](?: (?:\d+:)?[A-Z])*)")
_PN = re.compile(r"POSITION\s*:\s*(\d+)\s*,\s*NEW\s*:\s*([A-Za-z])", re.I)
_PON = re.compile(r"POSITION\s*:\s*(\d+)\s*,\s*OLD\s*:\s*([A-Za-z])\s*,\s*NEW\s*:\s*([A-Za-z])", re.I)


def sequence_of(prompt: str) -> str:
    """The genome in a prompt, from either display."""
    return "".join(tok.split(":")[-1] for tok in _SEQ_ANY.search(prompt).group(1).split())


def run_model(out: Path) -> None:
    from hpga import genome_model, operators as ops, sequence_model as sm, sequence_model_numbered as nb
    from hpga.config import HPGAConfig

    os.environ["HPGA_OPERATOR_MODE"] = "llm"
    model = genome_model.build_genome_model(HPGAConfig(genome_model="sequence"))
    genome_model.set_active(model)
    bases = json.load(open(GATE_DIR / "oldfield_replay_bases.json"))["bases"]
    record = {"model": ops.LLM_MODEL, "cells": {}}

    def call(genome, k, rate, rng, cell, i):
        if cell == "C1_drift":
            plan, style = sm.plan_llm_mutate("position", genome, k), "position"
        else:
            old_slot = cell.endswith("numold")
            plan = nb.plan_llm_mutate_numbered(genome, k, old_slot=old_slot)
            style = "position_old_numbered" if old_slot else "position_numbered"
        return ops._run_llm_op(
            op="mutate", style=style, system=plan.system, build_prompt=plan.build_prompt, parse=plan.parse, rng=rng,
            num_predict=plan.num_predict, fallback=lambda: model.deterministic_mutate(genome, rate, rng),
            retry_hint_text=plan.retry_hint_text,
            extra_log_fields={"probe": "numbered", "cell": cell, "k": k, "input_index": i})

    for cell in CELLS:
        os.environ["HPGA_LLM_LOG_PATH"] = str(out / f"calls_{gate.tag(ops.LLM_MODEL)}_{cell}.jsonl")
        ops.reset_operator_stats()
        t0 = time.time()
        if cell == "C1_drift":
            rng = random.Random(0)
            for i in range(100):
                g = sm.random_sequence(rng, 63)
                call(g, 1, 1.0 / 63, rng, cell, i)
        else:
            if cell.startswith("C2"):
                grng = random.Random(2)
                inputs = [(sm.random_sequence(grng, 63), 3) for _ in range(200)]
            else:
                inputs = [(b["genome"], b["k"]) for b in bases]
            rng = random.Random(0)
            for i, (g, k) in enumerate(inputs):
                call(g, k, 0.05, rng, cell, i)
        st = ops.get_operator_stats()
        record["cells"][cell] = {"stats": st, "wall_s": time.time() - t0}
        print(f"{time.strftime('%H:%M:%S')} {ops.LLM_MODEL} {cell}: {st['n_llm_calls']} calls, {st['n_llm_requests']} requests, "
              f"{st['n_failures']} fallbacks, {time.time() - t0:.0f}s", flush=True)
        json.dump(record, open(out / f"numbered_{gate.tag(ops.LLM_MODEL)}.json", "w"), indent=1, default=str)
    os.environ.pop("HPGA_LLM_LOG_PATH", None)


def cmd_all(out: Path, models) -> None:
    out.mkdir(parents=True, exist_ok=True)
    meta_p = out / "numbered_meta.json"
    meta = json.load(open(meta_p)) if meta_p.exists() else {"launches": [], "models": {}}
    meta["launches"].append({"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "models": list(models),
                             "ollama_version": gate._http("/api/version")["version"],
                             "env": {k: os.environ.get(k) for k in ("HPGA_LLM_TIMEOUT_S", "PYTHONHASHSEED")}})
    tags = {m["name"]: m for m in gate._http("/api/tags")["models"]}
    for m in models:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
        if apps and "/scratch/pcanaste/ollama" not in apps:
            meta["launches"][-1]["stopped"] = f"GPU in use by another process before {m}: {apps}"
            json.dump(meta, open(meta_p, "w"), indent=1)
            raise SystemExit(f"GPU in use by another process: {apps}")
        for resident in gate._http("/api/ps").get("models", []):
            gate._http("/api/generate", {"model": resident["name"], "prompt": "", "keep_alive": 0})
        warm = gate._http("/api/generate", {"model": m, "prompt": "Reply with OK.", "stream": False, "think": False,
                                            "options": {"num_ctx": 4096, "num_predict": 4, "temperature": 0}, "keep_alive": "30m"})
        meta["models"][m] = {"digest": tags[m]["digest"], "warmup_load_s": round(warm.get("load_duration", 0) / 1e9, 2),
                             "start": time.strftime("%Y-%m-%dT%H:%M:%S")}
        json.dump(meta, open(meta_p, "w"), indent=1)
        rc = subprocess.run([sys.executable, __file__, "run-model", "--out-dir", str(out)],
                            env={**os.environ, "HPGA_LLM_MODEL": m}, cwd=ROOT).returncode
        meta["models"][m].update({"returncode": rc, "end": time.strftime("%Y-%m-%dT%H:%M:%S")})
        gate._http("/api/generate", {"model": m, "prompt": "", "keep_alive": 0})
        json.dump(meta, open(meta_p, "w"), indent=1)
        print(f"MODEL DONE {m} rc={rc}", flush=True)
    meta["launches"][-1]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(meta, open(meta_p, "w"), indent=1)


# --- scoring inputs (first attempts are the pre-registered scope; all-attempts reported alongside) ---------------

def score_file(path: Path, old_slot: bool) -> dict:
    recs = [json.loads(l) for l in open(path)]
    att = [r for r in recs if "prompt" in r and "response" in r]
    out = {"calls": len({r["input_index"] for r in att}), "fallbacks": sum(r.get("event") == "fallback_to_deterministic" for r in recs),
           "errors": sum("error" in r for r in recs)}
    for scope, rows in (("first", [r for r in att if r["attempt"] == 0]), ("all", att)):
        c = collections.Counter()
        bands = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            g = sequence_of(r["prompt"])
            c["attempts"] += 1
            c["invalid"] += not r["valid"]
            pon = _PON.findall(r["response"]) if old_slot else []
            lines = [(int(p), o.upper(), n.upper()) for p, o, n in pon] if pon else [(int(p), None, n.upper()) for p, n in _PN.findall(r["response"])]
            for p, o, n in lines:
                if p >= len(g):
                    continue
                c["lines"] += 1
                c["repeats"] += n == g[p]
                if old_slot and o is not None:
                    ok = o == g[p]
                    c["old_lines"] += 1
                    c["old_correct"] += ok
                    c["cond_repeats"] += ok and n == g[p]
                    b = "0-9" if p < 10 else "10-19" if p < 20 else "20+"
                    bands[b][0] += ok
                    bands[b][1] += 1
        d = dict(c)
        d["repeat_rate"] = c["repeats"] / c["lines"] if c["lines"] else None
        if old_slot:
            d["reading"] = c["old_correct"] / c["old_lines"] if c["old_lines"] else None
            d["cond_repeat_rate"] = c["cond_repeats"] / c["old_correct"] if c["old_correct"] else None
            d["reading_by_band"] = {b: v[0] / v[1] for b, v in bands.items() if v[1]}
        out[scope] = d
    return out


def cmd_summarize(out: Path) -> None:
    res = {}
    for m in gate.MODELS:
        t = gate.tag(m)
        cells = {}
        for cond in ("C2", "C3"):
            for name, path, old_slot in ((f"{cond}_spaced_base", GATE_DIR / f"calls_{t}_{cond}_old.jsonl", False),
                                         (f"{cond}_spaced_old", GATE_DIR / f"calls_{t}_{cond}_new.jsonl", True),
                                         (f"{cond}_num_base", out / f"calls_{t}_{cond}_num.jsonl", False),
                                         (f"{cond}_num_old", out / f"calls_{t}_{cond}_numold.jsonl", True)):
                if path.exists():
                    cells[name] = score_file(path, old_slot)
        drift = None
        mine, ref = out / f"calls_{t}_C1_drift.jsonl", GATE_DIR / f"calls_{t}_C1_old.jsonl"
        if mine.exists():
            a = [json.loads(l)["prompt"] for l in open(mine) if '"prompt"' in l]
            b = [json.loads(l)["prompt"] for l in open(ref) if '"prompt"' in l]
            drift = {"identical": sum(x == y for x, y in zip(a, b)), "lengths": [len(a), len(b)], "pass": a == b}
        res[m] = {"cells": cells, "drift": drift}
    json.dump(res, open(out / "numbered_summary.json", "w"), indent=1)
    md = ["# Numbered-display probe: scoring inputs (first attempts; generated by experiments/probe_mutate_numbered.py)", "",
          "| model | cell | calls | attempts | invalid | fallbacks | lines | repeats (rate) | reading | reading by band | repeats when read correctly |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, d in res.items():
        for name, c in d["cells"].items():
            f = c["first"]
            rd = "" if f.get("reading") is None else f"{f['reading']:.2f}"
            bands = " ".join(f"{b}:{v:.2f}" for b, v in sorted(f.get("reading_by_band", {}).items()))
            cond = "" if "cond_repeats" not in f else f"{f['cond_repeats']}/{f['old_correct']}"
            md.append(f"| {m} | {name} | {c['calls']} | {f['attempts']} | {f['invalid']} | {c['fallbacks']} | {f.get('lines', 0)} | "
                      f"{f.get('repeats', 0)} ({(f['repeat_rate'] or 0):.3f}) | {rd} | {bands} | {cond} |")
    md += ["", "Drift check (C1, spaced/base, against the gate's C1 old-format prompts):", ""]
    md += [f"- {m}: {d['drift']}" for m, d in res.items()]
    (out / "numbered_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("all", "run-model", "summarize"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", nargs="+", default=list(gate.MODELS))
    a = ap.parse_args()
    out = Path(a.out_dir)
    {"all": lambda: cmd_all(out, a.models), "run-model": lambda: run_model(out), "summarize": lambda: cmd_summarize(out)}[a.cmd]()


if __name__ == "__main__":
    main()
