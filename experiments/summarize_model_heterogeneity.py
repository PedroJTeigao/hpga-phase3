"""Tables for the model-heterogeneity step-1 probe (experiments/run_model_heterogeneity_probe.py).
Everything is computed from results/raw: model_heterogeneity_meta.json, the probe's own
sequence_operator_compliance_<tag>_len63_hetero.json, and the per-request call logs
llm_operator_calls_hetero_<tag>.jsonl.  Nothing in the output is typed by hand.

Writes results/raw/model_heterogeneity_step1_tables.md and prints it.

Cross-model comparison (the question this probe exists for): for each pair of models,
  * distribution level: total-variation distance between the two models' choice distributions
    (positions over 63; cut values over 99), with a label-permutation p-value (null: both models
    draw from one pooled distribution);
  * per-prompt level: the probe is seeded, so two models are given the SAME genomes / parent pairs
    only while neither has fallen back (a fallback consumes extra draws from the shared RNG and
    desynchronises the stream). In practice only pairs of models with no fallbacks share prompts;
    the table reports the number of shared prompts and is uninformative where it is ~0. Where
    prompts are shared: the fraction on which two models made the identical choice, against the
    same fraction after shuffling one model's answers across prompts (what agreement looks like
    when the models share a marginal distribution but ignore the prompt).
"""

import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
MODELS = ["gemma4:12b", "llama3.2:3b", "qwen2.5:7b", "mistral:7b"]
POS_RE = re.compile(r"POSITION:\s*(\d+)\s*,\s*NEW:\s*([A-Za-z])")
SEG_RE = re.compile(r"(\d+)\s*-\s*(\d+)\s*:\s*([12])")
N_PERM = 5000


def tag(m: str) -> str:
    return m.replace(":", "_")


def load(m: str):
    probe = json.load(open(RAW / f"sequence_operator_compliance_{tag(m)}_len63_hetero.json"))
    recs = [json.loads(l) for l in open(RAW / f"llm_operator_calls_hetero_{tag(m)}.jsonl")]
    return probe, recs


def calls_from_log(recs: list[dict], op: str, style: str) -> list[dict]:
    """One entry per operator call: key = the attempt-0 prompt (identifies the genome / parent pair),
    result = the first valid response, or None if every attempt failed (fell back)."""
    calls, cur = [], None
    for r in recs:
        if r.get("op") != op or r.get("prompt_style") != style:
            continue
        if r.get("attempt") == 0:
            cur = {"key": r["prompt"], "result": None, "latencies": [], "tokens_out": []}
            calls.append(cur)
        if cur is None:
            continue
        if "latency_s" in r:
            cur["latencies"].append(r["latency_s"])
            cur["tokens_out"].append(r.get("tokens_out", 0))
        if r.get("valid") is True and cur["result"] is None:
            cur["result"] = r["response"]
    return calls


def mutate_choice(resp: str):
    m = POS_RE.search(resp or "")
    return (int(m.group(1)), m.group(2).upper()) if m else None


def cut_set(resp: str):
    m = re.search(r"SEGMENTS\s*:\s*(.*)", resp or "")
    segs = sorted((int(s), int(e), int(p)) for s, e, p in SEG_RE.findall(m.group(1))) if m else []
    return tuple(s for s, _, _ in segs[1:])


def tv(a: Counter, b: Counter) -> float:
    na, nb = sum(a.values()), sum(b.values())
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in keys)


def tv_perm_p(xa: list, xb: list, rng) -> tuple[float, float]:
    obs = tv(Counter(xa), Counter(xb))
    pool = np.array(xa + xb, dtype=object)
    na = len(xa)
    ge = 0
    for _ in range(N_PERM):
        rng.shuffle(pool)
        if tv(Counter(pool[:na].tolist()), Counter(pool[na:].tolist())) >= obs - 1e-12:
            ge += 1
    return obs, (1 + ge) / (1 + N_PERM)


def agreement(da: dict, db: dict, rng) -> dict:
    """da, db: prompt -> choice.  Observed same-choice rate on shared prompts vs the mean rate after
    shuffling db's choices across the shared prompts."""
    keys = [k for k in da if k in db]
    if not keys:
        return {"n_shared": 0}
    a = [da[k] for k in keys]
    b = [db[k] for k in keys]
    obs = sum(x == y for x, y in zip(a, b)) / len(keys)
    sh = []
    for _ in range(2000):
        bb = b[:]
        rng.shuffle(bb)
        sh.append(sum(x == y for x, y in zip(a, bb)) / len(keys))
    return {"n_shared": len(keys), "observed": obs, "shuffled_mean": float(np.mean(sh)),
            "p_ge": (1 + sum(s >= obs - 1e-12 for s in sh)) / (1 + len(sh))}


def pct(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def main() -> None:
    meta = json.load(open(RAW / "model_heterogeneity_meta.json"))
    models = [m for m in MODELS if m in meta["models"] and (RAW / f"sequence_operator_compliance_{tag(m)}_len63_hetero.json").exists()]
    data = {m: load(m) for m in models}
    out = []
    P = out.append

    # ---- 1. models, load, memory ----------------------------------------------------------------
    P("### 1. Models, load time and GPU memory (num_ctx 4096, as the operators run)\n")
    P("| model | family | params | quant | disk (GB) | load time (s) | Ollama resident (GiB) | fully on GPU | nvidia-smi delta (MiB) |")
    P("|---|---|---|---|---|---|---|---|---|")
    for m in models:
        e = meta["models"][m]
        ps = e["ps"] or {}
        P(f"| {m} | {e['info']['family']} | {e['info']['parameter_size']} | {e['info']['quantization']} | "
          f"{e['info']['disk_bytes'] / 1e9:.2f} | {e['warmup']['load_duration_s']:.1f} | "
          f"{(ps.get('size_bytes') or 0) / 2**30:.2f} | {ps.get('fully_on_gpu')} | {e['gpu_used_mib_delta']} |")
    P("\nLoad time is Ollama's `load_duration` on a warm-up request after unloading everything (disk cache may be warm for some models).\n")

    P("### 2. Do two fit on the 15360 MiB T4 together? (each loaded by a tiny request, then the first re-touched)\n")
    P("| pair | both resident | both fully on GPU | combined nvidia-smi delta (MiB) | first model evicted or reloaded |")
    P("|---|---|---|---|---|")
    for p in meta.get("pairs", []):
        P(f"| {p['pair'][0]} + {p['pair'][1]} | {p['both_resident']} | {p['both_fully_on_gpu']} | "
          f"{p['gpu_used_mib_delta']} | {p['first_model_evicted']} (retouch load {p['retouch_first_load_s']:.2f} s) |")
    P("")

    # ---- 3. compliance --------------------------------------------------------------------------
    P("### 3. Format compliance (100 calls per operator, genome length 63, bounds [30, 80], temperature 0.7, seed 0)\n")
    P("| model | operator | fell back | fallback rate | requests / call | mean latency per request (s) | mean latency per call (s) | mean tokens out / request |")
    P("|---|---|---|---|---|---|---|---|")
    logcalls = {}
    for m in models:
        probe, recs = data[m]
        for c in probe["conditions"]:
            label = f"{c['op']}/{c['style']}"
            s = c["stats"]
            lat = s["mean_latency_s"]
            rc = calls_from_log(recs, c["op"], c["style"])
            logcalls[(m, label)] = rc
            tout = [t for x in rc for t in x["tokens_out"]]
            P(f"| {m} | {label} | {s['n_failures']}/{c['n_calls']} | {c['fallback_rate']:.3f} | {c['requests_per_call']:.2f} | "
              f"{lat:.2f} | {lat * c['requests_per_call']:.2f} | {np.mean(tout):.1f} |")
    P("\nFallback = the operator gave up after its retries and used the deterministic operator.\n")

    # ---- 4. mutate position coverage ------------------------------------------------------------
    P("### 4. mutate/position: which positions and letters? (LLM-successful calls only)\n")
    P("| model | successful calls | distinct positions (of 63; expected if uniform) | modal position (share) | top-3 share | position chi2 (df 62), MC p | distinct new letters (of 20) | modal letter (share) | zero-change outputs |")
    P("|---|---|---|---|---|---|---|---|---|")
    mut = {}
    for m in models:
        probe, _ = data[m]
        c = next(x for x in probe["conditions"] if x["op"] == "mutate")
        pu, lu = c["position_uniformity"], c["letter_uniformity"]
        pc = {int(k): v for k, v in c["position_counts"].items()}
        lc = c["letter_counts"]
        mut[m] = (pc, lc)
        if pu.get("n", 0) == 0:
            P(f"| {m} | 0 | - | - | - | - | - | - | {c['zero_diff_outputs']} |")
            continue
        mp, mn = max(pc.items(), key=lambda kv: kv[1])
        ml, mln = max(lc.items(), key=lambda kv: kv[1])
        exp_d = 63 * (1 - (62 / 63) ** pu["n"])
        P(f"| {m} | {pu['n']} | {pu['n_distinct']} ({exp_d:.1f}) | {mp} ({pct(mn / pu['n'])}) | {pct(pu['top3_share'])} | "
          f"{pu['chi2']:.0f}, p={pu['mc_p_value']:.4g} | {lu['n_distinct']} | {ml} ({pct(mln / lu['n'])}) | {c['zero_diff_outputs']} |")
    pooled_p = Counter()
    for m in models:
        pooled_p.update(mut[m][0])
    if pooled_p:
        tot = sum(pooled_p.values())
        mp, mn = pooled_p.most_common(1)[0]
        P(f"\nPooled over the {len(models)} models: {tot} successful mutations, {len(pooled_p)} of 63 positions used by at least one model, "
          f"modal position {mp} ({pct(mn / tot)}); positions that are the modal choice of each model: "
          + ", ".join(f"{m}: {max(mut[m][0].items(), key=lambda kv: kv[1])[0]}" for m in models if mut[m][0]) + ".")
    P("")

    # ---- 5. crossover cuts ----------------------------------------------------------------------
    P("### 5. crossover/segment: which cuts? (cuts are percentages; 99 possible values 1-99)\n")
    P("| model | valid declarations | segments per call | distinct cut values (of 99) | modal cut (share of cuts) | cuts within 45-55 | five most used cuts |")
    P("|---|---|---|---|---|---|---|")
    cutc = {}
    for m in models:
        probe, _ = data[m]
        c = next(x for x in probe["conditions"] if x["op"] == "crossover")
        st = c["segment_structure"]
        cc = {int(k): v for k, v in st["cut_percent_counts"].items()}
        cutc[m] = cc
        if not cc:
            P(f"| {m} | {st['n_valid_declarations']} | {st['segments_per_call']} | 0 | - | - | - |")
            continue
        tot = sum(cc.values())
        top5 = ", ".join(f"{k}x{v}" for k, v in sorted(cc.items(), key=lambda kv: -kv[1])[:5])
        P(f"| {m} | {st['n_valid_declarations']} | {st['segments_per_call']} | {st['n_distinct_cut_values']} | "
          f"{st['modal_cut']} ({pct(st['modal_cut_share'])}) | {pct(st['share_cuts_45_to_55'])} | {top5} |")
    pooled_c = Counter()
    for m in models:
        pooled_c.update(cutc[m])
    if pooled_c:
        tot = sum(pooled_c.values())
        P(f"\nPooled over the {len(models)} models: {tot} cuts, {len(pooled_c)} of 99 values used by at least one model "
          f"({', '.join(f'{k}x{v}' for k, v in sorted(pooled_c.items(), key=lambda kv: -kv[1]))}).")
    P("")

    # ---- 6. cross-model comparison --------------------------------------------------------------
    P("### 6. Do different models choose differently from each other?\n")
    rng = np.random.default_rng(0)
    P("**mutate/position** (choice = position; 'same (pos,letter)' also requires the same new letter)\n")
    P("| pair | positions used by both (of the union) | TV distance | permutation p | shared genomes | same position, observed | same position, shuffled baseline | same (pos,letter), observed |")
    P("|---|---|---|---|---|---|---|---|")
    mchoice = {}
    for m in models:
        d = {}
        for x in logcalls[(m, "mutate/position")]:
            ch = mutate_choice(x["result"]) if x["result"] else None
            if ch:
                d[x["key"]] = ch
        mchoice[m] = d
    for a, b in combinations(models, 2):
        pa, pb = mut[a][0], mut[b][0]
        if not pa or not pb:
            continue
        xa = [p for p, n in pa.items() for _ in range(n)]
        xb = [p for p, n in pb.items() for _ in range(n)]
        t, p = tv_perm_p(xa, xb, rng)
        both, union = len(set(pa) & set(pb)), len(set(pa) | set(pb))
        ag_pos = agreement({k: v[0] for k, v in mchoice[a].items()}, {k: v[0] for k, v in mchoice[b].items()}, rng)
        ag_pl = agreement(mchoice[a], mchoice[b], rng)
        P(f"| {a} vs {b} | {both} of {union} | {t:.3f} | {p:.4f} | {ag_pos.get('n_shared', 0)} | "
          f"{pct(ag_pos.get('observed'))} | {pct(ag_pos.get('shuffled_mean'))} | {pct(ag_pl.get('observed'))} |")
    P("\n**crossover/segment** (choice = the set of cut values in the declaration)\n")
    P("| pair | cut values used by both (of the union) | TV distance | permutation p | shared parent pairs | identical cut set, observed | identical cut set, shuffled baseline |")
    P("|---|---|---|---|---|---|---|")
    cchoice = {}
    for m in models:
        d = {}
        for x in logcalls[(m, "crossover/segment")]:
            cs = cut_set(x["result"]) if x["result"] else ()
            if cs:
                d[x["key"]] = cs
        cchoice[m] = d
    for a, b in combinations(models, 2):
        ca, cb = cutc[a], cutc[b]
        if not ca or not cb:
            continue
        xa = [c for c, n in ca.items() for _ in range(n)]
        xb = [c for c, n in cb.items() for _ in range(n)]
        t, p = tv_perm_p(xa, xb, rng)
        ag = agreement(cchoice[a], cchoice[b], rng)
        P(f"| {a} vs {b} | {len(set(ca) & set(cb))} of {len(set(ca) | set(cb))} | {t:.3f} | {p:.4f} | {ag.get('n_shared', 0)} | "
          f"{pct(ag.get('observed'))} | {pct(ag.get('shuffled_mean'))} |")
    P(f"\nPermutation p-values use {N_PERM} label shuffles (floor {1 / (1 + N_PERM):.1e}); TV = total-variation distance "
      "(0 = identical distributions, 1 = disjoint). The shuffled baseline shuffles the second model's answers across the shared "
      "prompts (2000 shuffles): it is the agreement expected if the models share a marginal distribution but ignore the prompt.\n")

    # ---- 7. consistency check of the log parse against the probe's own counts --------------------
    P("### 7. Consistency check (log parse vs the probe's own JSON)\n")
    for m in models:
        pc = mut[m][0]
        parsed = Counter(v[0] for v in mchoice[m].values())
        ncalls = len(logcalls[(m, "mutate/position")])
        # parsed counts collapse duplicate prompts; only flag if the number of calls differs from n_calls
        P(f"- {m}: {ncalls} mutate calls and {len(logcalls[(m, 'crossover/segment')])} crossover calls found in the log; "
          f"probe counted {sum(pc.values())} successful mutates; log-parse positions distinct {len(parsed)} vs probe {len(pc)}.")
    text = "\n".join(out) + "\n"
    (RAW / "model_heterogeneity_step1_tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
