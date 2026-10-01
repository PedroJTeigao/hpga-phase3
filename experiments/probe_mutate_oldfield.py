"""Isolated probe (no GA, no ESMFold): does an OLD slot in the mutate response format remove the repeated-letter failure?
Pre-registered in results/PREREGISTERED_OLDFIELD.md (committed bab55af before any call). Old format = the production
mutate/position plan (hpga/sequence_model.py, untouched); new format = hpga/sequence_model_oldfield.py. Both run through
operators._run_llm_op, so retries, fallback, logging and stats are the production code.

Conditions, for each model, each run under both formats:
  C1  replication: random length-63 sequences, k = 1, 100 calls; one Random(0) stream drives both the sequences and the
      per-request seeds, exactly as probe_sequence_operator_compliance.py's mutate loop does, so the old-format prompts
      must reproduce MODEL_HETEROGENEITY_STEP1's logged mutate prompts byte for byte (checked by `summarize`).
  C2  random length-63 sequences from Random(2), k = 3 (rate 0.05, the GA's), 200 calls.
  C3  replay: 200 real GA base sequences with their own k (2-4), 100 sampled from gemma4:12b arm C's first-attempt mutate
      prompts and 100 from qwen2.5:7b arm C's, Random(3); the same 200 for every model (oldfield_replay_bases.json).
  C2 and C3 inputs are fixed lists and request seeds come from a separate Random(0), so old and new see identical inputs.

  python experiments/probe_mutate_oldfield.py all --out-dir results/raw/oldfield_gate      # every model, one process each
  python experiments/probe_mutate_oldfield.py summarize --out-dir results/raw/oldfield_gate
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
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
sys.path.insert(0, str(ROOT))
MODELS = ("gemma4:12b", "qwen2.5:7b", "mistral:7b", "llama3.2:3b")
CONDITIONS = ("C1", "C2", "C3")
FORMATS = ("old", "new")
HOST = os.environ.get("HPGA_OLLAMA_HOST", "http://localhost:11434")
_SEQ = re.compile(r"positions 0-\d+\): ([A-Z ]+)")
_PN = re.compile(r"POSITION\s*:\s*(\d+)\s*,\s*NEW\s*:\s*([A-Za-z])", re.I)
_PON = re.compile(r"POSITION\s*:\s*(\d+)\s*,\s*OLD\s*:\s*([A-Za-z])\s*,\s*NEW\s*:\s*([A-Za-z])", re.I)


def tag(model: str) -> str:
    return model.replace(":", "_")


def _http(path, payload=None, timeout=900):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(HOST + path, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# --- inputs ---------------------------------------------------------------------------------------------------------

def replay_bases(out: Path) -> list[dict]:
    path = out / "oldfield_replay_bases.json"
    if path.exists():
        return json.load(open(path))["bases"]
    pools = {}
    for arm, files in (("gemma4:12b arm C", [json.load(open(RAW / f"sequence_ga_cmp_C_seed{s}.json"))["llm_call_log"] for s in range(5)]),
                       ("qwen2.5:7b arm C", [json.load(open(RAW / "armC_qwen2.5_7b" / f"sequence_ga_cmp_C_seed{s}.json"))["llm_call_log"] for s in range(5)])):
        pool = []
        for f in files:
            for line in open(ROOT / f):
                r = json.loads(line)
                if r.get("op") == "mutate" and "prompt" in r and r.get("attempt") == 0:
                    pool.append({"source": arm, "log": f, "genome": "".join(_SEQ.search(r["prompt"]).group(1).split()),
                                 "k": int(re.search(r"Change exactly (\d+)", r["prompt"]).group(1))})
        pools[arm] = pool
    rng = random.Random(3)
    bases = [b for arm in pools for b in rng.sample(pools[arm], 100)]
    json.dump({"note": "100 first-attempt mutate bases from each arm's call logs, Random(3); the same list for every model",
               "bases": bases}, open(path, "w"), indent=1)
    return bases


# --- one model, in its own process (hpga.operators reads HPGA_LLM_MODEL at import) ----------------------------------

def run_model(out: Path) -> None:
    from hpga import genome_model, operators as ops, sequence_model as sm, sequence_model_oldfield as of
    from hpga.config import HPGAConfig

    os.environ["HPGA_OPERATOR_MODE"] = "llm"
    model = genome_model.build_genome_model(HPGAConfig(genome_model="sequence"))
    genome_model.set_active(model)
    bases = replay_bases(out)
    record = {"model": ops.LLM_MODEL, "cells": {}}

    def call(genome, k, rate, rng, fmt, cond, i):
        plan = sm.plan_llm_mutate("position", genome, k) if fmt == "old" else of.plan_llm_mutate_oldfield(genome, k)
        return ops._run_llm_op(
            op="mutate", style="position" if fmt == "old" else of.STYLE, system=plan.system,
            build_prompt=plan.build_prompt, parse=plan.parse, rng=rng, num_predict=plan.num_predict,
            fallback=lambda: model.deterministic_mutate(genome, rate, rng), retry_hint_text=plan.retry_hint_text,
            extra_log_fields={"probe": "oldfield", "condition": cond, "format": fmt, "k": k, "input_index": i})

    for cond in CONDITIONS:
        for fmt in FORMATS:
            os.environ["HPGA_LLM_LOG_PATH"] = str(out / f"calls_{tag(ops.LLM_MODEL)}_{cond}_{fmt}.jsonl")
            ops.reset_operator_stats()
            t0 = time.time()
            if cond == "C1":  # the step-1 loop: one stream for sequences and request seeds
                rng = random.Random(0)
                for i in range(100):
                    g = sm.random_sequence(rng, 63)
                    call(g, 1, 1.0 / 63, rng, fmt, cond, i)
            else:
                if cond == "C2":
                    grng = random.Random(2)
                    inputs = [(sm.random_sequence(grng, 63), 3) for _ in range(200)]
                else:
                    inputs = [(b["genome"], b["k"]) for b in bases]
                rng = random.Random(0)
                for i, (g, k) in enumerate(inputs):
                    call(g, k, 0.05, rng, fmt, cond, i)
            record["cells"][f"{cond}_{fmt}"] = {"stats": ops.get_operator_stats(), "wall_s": time.time() - t0}
            print(f"{ops.LLM_MODEL} {cond} {fmt}: {ops.get_operator_stats()['n_llm_calls']} calls, "
                  f"{ops.get_operator_stats()['n_failures']} fallbacks, {time.time() - t0:.0f}s", flush=True)
    os.environ.pop("HPGA_LLM_LOG_PATH", None)
    json.dump(record, open(out / f"oldfield_{tag(ops.LLM_MODEL)}.json", "w"), indent=1, default=str)


def cmd_all(out: Path, models) -> None:
    out.mkdir(parents=True, exist_ok=True)
    replay_bases(out)
    meta = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ollama_version": _http("/api/version")["version"],
            "models": {}, "env": {k: os.environ.get(k) for k in ("HPGA_LLM_TIMEOUT_S", "HPGA_LLM_TEMPERATURE", "HPGA_LLM_NUM_CTX")}}
    tags = {m["name"]: m for m in _http("/api/tags")["models"]}
    for m in models:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
        if apps and "/scratch/pcanaste/ollama" not in apps:
            raise SystemExit(f"GPU in use by another process: {apps}")
        for resident in _http("/api/ps").get("models", []):  # unload only what is loaded (unloading by name would load it)
            _http("/api/generate", {"model": resident["name"], "prompt": "", "keep_alive": 0})
        warm = _http("/api/generate", {"model": m, "prompt": "Reply with OK.", "stream": False, "think": False,
                                       "options": {"num_ctx": 4096, "num_predict": 4, "temperature": 0}, "keep_alive": "30m"})
        meta["models"][m] = {"digest": tags[m]["digest"], "details": tags[m].get("details"),
                             "warmup_load_s": round(warm.get("load_duration", 0) / 1e9, 2), "start": time.strftime("%H:%M:%S")}
        env = {**os.environ, "HPGA_LLM_MODEL": m}
        rc = subprocess.run([sys.executable, __file__, "run-model", "--out-dir", str(out)], env=env, cwd=ROOT).returncode
        meta["models"][m].update({"returncode": rc, "end": time.strftime("%H:%M:%S")})
        _http("/api/generate", {"model": m, "prompt": "", "keep_alive": 0})
        json.dump(meta, open(out / "oldfield_meta.json", "w"), indent=1)
        if rc != 0:
            print(f"ALERT: {m} exited {rc}", flush=True)
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(meta, open(out / "oldfield_meta.json", "w"), indent=1)


# --- analysis -------------------------------------------------------------------------------------------------------

def classify(r: dict) -> dict:
    """Lenient per-line scoring of one attempt (NEW vs the actual letter, OLD ignored), plus the strict failure kind."""
    g = "".join(_SEQ.search(r["prompt"]).group(1).split())
    k = int(re.search(r"Change exactly (\d+)", r["prompt"]).group(1))
    resp = r.get("response", "")
    new_fmt = r.get("format") == "new"
    pon = _PON.findall(resp) if new_fmt else []
    lines = [(int(p), o.upper(), n.upper()) for p, o, n in pon] if pon else [(int(p), None, n.upper()) for p, n in _PN.findall(resp)]
    lines_in = [(p, o, n) for p, o, n in lines if p < len(g)]
    out = {"lines": len(lines_in), "same": sum(n == g[p] for p, o, n in lines_in),
           "old_present": sum(o is not None for _, o, _ in lines_in), "old_correct": sum(o == g[p] for p, o, _ in lines_in if o),
           "old_correct_and_new_eq_old": sum(o == g[p] and n == o for p, o, n in lines_in if o)}
    if r.get("valid"):
        kind = None
    elif not lines:
        kind = "no parseable line"
    elif new_fmt and not pon:
        kind = "OLD missing"
    elif len(lines) != k:
        kind = "wrong number of lines"
    elif any(p >= len(g) for p, _, _ in lines):
        kind = "position out of range"
    elif len({p for p, _, _ in lines}) < len(lines):
        kind = "repeated position"
    elif any(n == g[p] for p, _, n in lines):
        kind = "NEW equals the current letter"
    elif new_fmt and any(o != g[p] for p, o, _ in lines):
        kind = "OLD wrong (NEW differs)"
    else:
        kind = "other"
    out["kind"] = kind
    return out


def cmd_summarize(out: Path) -> None:
    res, md = {}, []
    for m in MODELS:
        for cond in CONDITIONS:
            for fmt in FORMATS:
                p = out / f"calls_{tag(m)}_{cond}_{fmt}.jsonl"
                if not p.exists():
                    continue
                recs = [json.loads(l) for l in open(p)]
                att = [r for r in recs if "prompt" in r and "response" in r]
                fb = sum(1 for r in recs if r.get("event") == "fallback_to_deterministic")
                calls = len({r["input_index"] for r in att})
                cell = {"calls": calls, "attempts": len(att), "invalid": sum(not r["valid"] for r in att), "fallbacks": fb,
                        "kinds": dict(collections.Counter(c["kind"] for c in map(classify, att) if c["kind"])),
                        "errors": sum(1 for r in recs if "error" in r)}
                for scope, rows in (("first", [r for r in att if r["attempt"] == 0]), ("all", att)):
                    cs = [classify(r) for r in rows]
                    L = sum(c["lines"] for c in cs)
                    agg = {key: sum(c[key] for c in cs) for key in ("same", "old_present", "old_correct", "old_correct_and_new_eq_old")}
                    cell[scope] = {"attempts": len(rows), "lines": L, **agg, "same_rate": agg["same"] / L if L else None,
                                   "old_accuracy": agg["old_correct"] / agg["old_present"] if agg["old_present"] else None,
                                   "invalid": sum(not r["valid"] for r in rows)}
                res[f"{m}|{cond}|{fmt}"] = cell
    # C1 old must reproduce MODEL_HETEROGENEITY_STEP1's mutate prompts
    repro = {}
    for m in MODELS:
        mine = out / f"calls_{tag(m)}_C1_old.jsonl"
        ref = RAW / f"llm_operator_calls_hetero_{tag(m)}.jsonl"
        if mine.exists() and ref.exists():
            a = [json.loads(l)["prompt"] for l in open(mine) if '"prompt"' in l]
            b = [r["prompt"] for r in map(json.loads, open(ref)) if r.get("op") == "mutate" and "prompt" in r]
            n = min(len(a), len(b))
            repro[m] = {"compared": n, "identical": sum(x == y for x, y in zip(a[:n], b[:n])), "first_mismatch":
                        next((i for i in range(n) if a[i] != b[i]), None), "lengths": [len(a), len(b)]}
    json.dump({"cells": res, "c1_old_reproduces_step1": repro}, open(out / "oldfield_summary.json", "w"), indent=1)

    md += ["# OLD-slot gate: generated tables (experiments/probe_mutate_oldfield.py summarize)", "",
           "Repeated-letter rate = parsed lines whose NEW equals the letter actually at that position (OLD ignored), the "
           "same definition under both formats. Invalid = strict validity of the format used.", ""]
    for scope in ("first", "all"):
        md += [f"## {'First attempts' if scope == 'first' else 'All attempts'}", "",
               "| model | condition | format | calls | attempts | invalid attempts | fallbacks | lines | repeated-letter lines (rate) | OLD accuracy | OLD correct and NEW = OLD |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for key, c in res.items():
            m, cond, fmt = key.split("|")
            s = c[scope]
            md.append(f"| {m} | {cond} | {fmt} | {c['calls']} | {s['attempts']} | {s['invalid']} | {c['fallbacks']} | {s['lines']} | "
                      f"{s['same']} ({s['same_rate']:.3f}) | {'' if s['old_accuracy'] is None else f'{s['old_accuracy']:.3f}'} | "
                      f"{s['old_correct_and_new_eq_old'] if fmt == 'new' else ''} |")
        md.append("")
    md += ["## Invalid attempts by kind (all attempts)", "", "| model | condition | format | kinds |", "|---|---|---|---|"]
    md += [f"| {k.split('|')[0]} | {k.split('|')[1]} | {k.split('|')[2]} | {c['kinds']} |" for k, c in res.items()]
    md += ["", "## C1 old format against MODEL_HETEROGENEITY_STEP1's logged mutate prompts", ""]
    md += [f"- {m}: {r['identical']} of {r['compared']} prompts identical (logged attempts {r['lengths'][0]} here, {r['lengths'][1]} in step 1); first mismatch at {r['first_mismatch']}" for m, r in repro.items()]
    (out / "oldfield_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("all", "run-model", "summarize"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    a = ap.parse_args()
    out = Path(a.out_dir)
    if a.cmd == "all":
        cmd_all(out, a.models)
    elif a.cmd == "run-model":
        run_model(out)
    else:
        cmd_summarize(out)


if __name__ == "__main__":
    main()
