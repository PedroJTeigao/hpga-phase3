"""Stage 2 gate for agent memory: does putting an agent's private record in its prompt change WHERE it proposes edits
at all? 100 proposal calls per condition, one agent, at the operator probes' settings (gemma4:12b, temperature 0.7,
genome length 63, num_ctx 4096, seed 0, k = 1 edit per call so each call is one independent (position, letter)
observation, fresh uniformly random sequence per call). Every call goes through hpga.agents_sequence.propose, the
function the GA uses, so the prompt is exactly the GA's prompt.

  A  no record        (memory off: the plain mutate/position prompt)
  B  record shown     (memory on: the agent's last 8 entries and a tally prepended)

Pairing: call i uses the same genome and the same first-attempt sampling seed in A and in B, so the two conditions
differ only in the record block (and, on a retry, in the shifted stream after it).

The record in B is SYNTHETIC, because a live agent's history needs the GA. It is built to look like what M2 will show:
  * positions and new letters are drawn from A's own proposals (an agent's past edits come from its own habits);
  * the fitness is a chain -- each entry's "after" is the next entry's "before" -- starting near 0.28 with steps
    N(0, 0.03), so about half the edits improve (the coin flip measured for circle proposals, SEQUENCE_GA_REPORT 9.11);
  * it is built BACKWARDS from the call's own genome, so the last edit's new letter is the letter now at that position:
    the record is consistent with the sequence shown. 11 entries (rounds 1-11, 'current round' 12), the last 8 shown,
    tally over all 11 -- the state of a real agent at generation 12.
Outcomes are independent of positions here (the operators do not condition on the genome, OPERATOR_BEHAVIOUR.md), which
is also the case in the GA; a record whose outcomes carried real signal is NOT tested by this gate.

Decision rule, fixed before any live call: the proposal distribution is called CHANGED if a paired permutation test
(20,000 label swaps within call pairs) of the total-variation distance between the A and B position distributions
gives p < 0.05; otherwise NO DETECTABLE CHANGE, and the 95th percentile of the null TV is reported so the size of a
shift this design could not have seen is visible. Also reported, not part of the rule: distinct positions, modal
position and share, JS divergence, fallback rate, requests per call, and, for B,
whether proposals copy / avoid / revert the positions in the record, split by that entry's outcome, against what A did
on the same genomes with the same records.

  python experiments/probe_agent_memory_gate.py            # live: 200 calls, writes results/raw/agent_memory_gate*.json
  python experiments/probe_agent_memory_gate.py --stub     # offline: stubbed model, checks the pipeline end to end
  python experiments/probe_agent_memory_gate.py --analyze results/raw/agent_memory_gate.json   # re-analyse saved calls
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RAW = ROOT / "results" / "raw"

GENOME_LEN, N_CALLS, SEED = 63, 100, 0
N_HISTORY, CURRENT_ROUND = 11, 12  # entries in the synthetic record; the round the agent is about to propose in
N_PERM = 20000
ALPHA = 0.05


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# --- the synthetic record ---------------------------------------------------------------------------

def build_record(genome: str, pos_dist: Counter, letter_dist: Counter, call_idx: int, alphabet: str) -> list[dict]:
    """N_HISTORY entries ending at `genome` (see module docstring). Entry dicts have agents_sequence's keys."""
    r = random.Random(_seed(SEED, call_idx, "record"))
    positions = r.choices(list(pos_dist), weights=list(pos_dist.values()), k=N_HISTORY)
    cur = list(genome)
    back = []  # latest first
    for p in reversed(positions):
        new = cur[p]
        # the old letter: drawn from the model's own new-letter habits, never equal to the letter it became
        cands = [(c, w) for c, w in letter_dist.items() if c != new] or [(c, 1) for c in alphabet if c != new]
        old = r.choices([c for c, _ in cands], weights=[w for _, w in cands], k=1)[0]
        back.append([p, old, new])
        cur[p] = old
    edits = list(reversed(back))  # oldest first
    f = min(0.6, max(0.12, r.gauss(0.28, 0.04)))
    entries = []
    for i, (p, old, new) in enumerate(edits):
        after = min(0.6, max(0.10, f + r.gauss(0.0, 0.03)))
        entries.append({"agent_id": 0, "generation": CURRENT_ROUND - N_HISTORY + i, "base_fitness": f, "changes": [[p, old, new]],
                        "fitness_after": after, "no_edit": False, "improved": after > f})
        f = after
    return entries


# --- collection -------------------------------------------------------------------------------------

def run_condition(cond: str, genomes: list[str], records: list[list[dict]] | None, use_stub: bool) -> list[dict]:
    from hpga import agents_sequence as aseq, operators as ops

    out = []
    for i, g in enumerate(genomes):
        rng = random.Random(_seed(SEED, i, "call"))
        before_fail, before_req = ops.get_operator_stats()["n_failures"], ops.get_operator_stats()["n_llm_requests"]
        entries = records[i] if records is not None else []
        new = aseq.propose(0, g, entries, CURRENT_ROUND, rng, show_record=(cond == "B"))
        st = ops.get_operator_stats()
        ch = aseq.changes_between(g, new)
        out.append({"i": i, "genome": g, "fallback": st["n_failures"] > before_fail, "requests": st["n_llm_requests"] - before_req,
                    "changes": ch, "proposal": new})
        if (i + 1) % 20 == 0:
            print(f"  {cond}: {i + 1}/{len(genomes)}  fallbacks so far {st['n_failures']}", flush=True)
    return out


def stub_call(prompt, system, num_predict, seed):
    """Offline stand-in: position drawn from a habit distribution that shifts when a record is present, so the analysis
    has something to find when run with --stub."""
    from hpga import sequence_model as sm

    r = random.Random(_seed(seed, prompt[-300:]))
    cur = "".join(re.search(r"Sequence \(0-indexed positions 0-\d+\): ([A-Z ]+)", prompt).group(1).split())
    with_rec = "Your own record" in prompt
    p = 10 if r.random() < (0.25 if with_rec else 0.40) else r.randrange(len(cur))
    letter = r.choice([c for c in sm.ALPHABET if c != cur[p]])
    bad = r.random() < 0.03
    return ("junk" if bad else f"POSITION: {p}, NEW: {letter}"), 0.05, len(prompt) // 4, 12


# --- analysis ---------------------------------------------------------------------------------------

def dist(counts: Counter, support: list) -> np.ndarray:
    v = np.array([counts.get(s, 0) for s in support], dtype=float)
    return v / v.sum() if v.sum() else v


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def paired_perm_tv(pa: list[int], pb: list[int], support: list, n_perm: int, seed: int = 0) -> dict:
    idx = {s: k for k, s in enumerate(support)}
    a = np.array([idx[x] for x in pa])
    b = np.array([idx[x] for x in pb])
    n, m = len(a), len(support)

    def tv_of(x, y):
        cx, cy = np.bincount(x, minlength=m) / len(x), np.bincount(y, minlength=m) / len(y)
        return 0.5 * np.abs(cx - cy).sum()

    obs = tv_of(a, b)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for t in range(n_perm):
        swap = rng.random(n) < 0.5
        null[t] = tv_of(np.where(swap, b, a), np.where(swap, a, b))
    return {"observed": float(obs), "p": float((1 + (null >= obs).sum()) / (1 + n_perm)), "null_mean": float(null.mean()),
            "null_p95": float(np.quantile(null, 0.95)), "null_p99": float(np.quantile(null, 0.99))}


def exact_sign_p(x: int, y: int) -> float:
    """Two-sided exact binomial (p = 0.5) on the discordant pairs x vs y."""
    n = x + y
    if n == 0:
        return 1.0
    from math import comb

    tail = sum(comb(n, i) for i in range(0, min(x, y) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarize(counts: Counter, n: int) -> dict:
    if not counts:
        return {"n": 0}
    (mode, mc), = counts.most_common(1)
    return {"n": n, "distinct": len(counts), "mode": mode, "mode_count": mc, "mode_share": mc / n,
            "top5": [[k, v] for k, v in counts.most_common(5)]}


def analyse(data: dict) -> dict:
    A, B = data["A"], data["B"]
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    both = [(a, b) for a, b in zip(A, B) if not a["fallback"] and not b["fallback"] and len(a["changes"]) == 1 and len(b["changes"]) == 1]
    posA, posB = Counter(a["changes"][0][0] for a in A if not a["fallback"] and len(a["changes"]) == 1), \
        Counter(b["changes"][0][0] for b in B if not b["fallback"] and len(b["changes"]) == 1)
    letA, letB = Counter(a["changes"][0][2] for a in A if not a["fallback"] and len(a["changes"]) == 1), \
        Counter(b["changes"][0][2] for b in B if not b["fallback"] and len(b["changes"]) == 1)
    support = list(range(GENOME_LEN))
    pA, pB = dist(posA, support), dist(posB, support)
    perm = paired_perm_tv([a["changes"][0][0] for a, _ in both], [b["changes"][0][0] for _, b in both], support, N_PERM)
    lsupport = list(alphabet)
    lperm = paired_perm_tv([a["changes"][0][2] for a, _ in both], [b["changes"][0][2] for _, b in both], lsupport, N_PERM)
    res = {
        "n_calls": {"A": len(A), "B": len(B)}, "n_valid": {"A": sum(posA.values()), "B": sum(posB.values())}, "n_pairs_both_valid": len(both),
        "fallback": {"A": sum(a["fallback"] for a in A), "B": sum(b["fallback"] for b in B)},
        "requests_per_call": {"A": sum(a["requests"] for a in A) / len(A), "B": sum(b["requests"] for b in B) / len(B)},
        "position": {"A": summarize(posA, sum(posA.values())), "B": summarize(posB, sum(posB.values())),
                     "tv": tv(pA, pB), "js_bits": js(pA, pB), "paired_perm_tv": perm,
                     "same_position_in_pair": sum(1 for a, b in both if a["changes"][0][0] == b["changes"][0][0]),
                     "same_position_expected_if_independent": float(len(both) * (pA * pB).sum()),
                     "positions_only_in_A": sorted(set(posA) - set(posB)), "positions_only_in_B": sorted(set(posB) - set(posA)),
                     "counts_A": dict(sorted(posA.items())), "counts_B": dict(sorted(posB.items()))},
        "new_letter": {"A": summarize(letA, sum(letA.values())), "B": summarize(letB, sum(letB.values())),
                       "tv": tv(dist(letA, lsupport), dist(letB, lsupport)), "paired_perm_tv": lperm},
    }
    # --- what B does with the record, against what A did on the same genomes with the same records
    rec = data["records"]
    rows = {"copy_any": [0, 0], "copy_better": [0, 0], "copy_worse": [0, 0], "revert_latest": [0, 0], "same_as_latest_position": [0, 0]}
    denom = 0
    disc = {k: [0, 0] for k in rows}  # discordant pairs: [A hit and B did not, B hit and A did not]
    for a, b in both:
        i = a["i"]
        shown = rec[i][-8:]
        latest_by_pos = {}
        for e in shown:
            latest_by_pos[e["changes"][0][0]] = "better" if e["improved"] else "worse"
        better = {p for p, o in latest_by_pos.items() if o == "better"}
        worse = {p for p, o in latest_by_pos.items() if o == "worse"}
        last = shown[-1]["changes"][0]
        denom += 1
        hit = []
        for k, x in enumerate((a, b)):
            p, _old, new = x["changes"][0]
            h = {"copy_any": p in latest_by_pos, "copy_better": p in better, "copy_worse": p in worse,
                 "revert_latest": (p == last[0] and new == last[1]), "same_as_latest_position": (p == last[0])}
            for key, v in h.items():
                rows[key][k] += v
            hit.append(h)
        for key in rows:
            disc[key][0] += int(hit[0][key] and not hit[1][key])
            disc[key][1] += int(hit[1][key] and not hit[0][key])
    res["record_use"] = {"n_pairs": denom, **{k: {"A": v[0], "B": v[1]} for k, v in rows.items()},
                         "posthoc_exact_sign_test_on_discordant_pairs": {k: {"A_only": d[0], "B_only": d[1], "p_two_sided": exact_sign_p(d[0], d[1])}
                                                                         for k, d in disc.items()},
                         "note": "A column = what the no-record proposal did on the same genome, scored against the same (unshown) record"}
    v = res["position"]["paired_perm_tv"]
    res["decision"] = {"rule": f"CHANGED if paired-permutation p(TV over positions) < {ALPHA}",
                       "verdict": "CHANGED" if v["p"] < ALPHA else "NO DETECTABLE CHANGE",
                       "tv": v["observed"], "p": v["p"], "null_p95_tv": v["null_p95"]}
    return res


def report(res: dict) -> None:
    p, d = res["position"], res["decision"]
    print(f"\ncalls: {res['n_calls']}  valid (one edit at one position): {res['n_valid']}  pairs both valid: {res['n_pairs_both_valid']}")
    print(f"fallback: {res['fallback']}  requests/call: {res['requests_per_call']}")
    for c in ("A", "B"):
        s = p[c]
        print(f"position {c}: distinct {s['distinct']}, mode {s['mode']} ({s['mode_share']:.0%}), top5 {s['top5']}")
    print(f"position TV {p['tv']:.3f}  JS {p['js_bits']:.3f} bits  paired-perm p={p['paired_perm_tv']['p']:.4f} "
          f"(null mean {p['paired_perm_tv']['null_mean']:.3f}, p95 {p['paired_perm_tv']['null_p95']:.3f})")
    print(f"same position in pair: {p['same_position_in_pair']} (independent-marginals expectation {p['same_position_expected_if_independent']:.1f})")
    L = res["new_letter"]
    print(f"new letter A: distinct {L['A']['distinct']} mode {L['A']['mode']} ({L['A']['mode_share']:.0%}); B: distinct {L['B']['distinct']} "
          f"mode {L['B']['mode']} ({L['B']['mode_share']:.0%}); TV {L['tv']:.3f} p={L['paired_perm_tv']['p']:.4f}")
    ru = res["record_use"]
    print(f"record use (pairs {ru['n_pairs']}): " + "; ".join(f"{k} A={v['A']} B={v['B']}" for k, v in ru.items() if isinstance(v, dict) and "A" in v))
    print("  post-hoc exact sign test on discordant pairs (A only / B only, p): " + "; ".join(
        f"{k} {v['A_only']}/{v['B_only']} p={v['p_two_sided']:.4f}" for k, v in ru["posthoc_exact_sign_test_on_discordant_pairs"].items()))
    print(f"\nDECISION: {d['verdict']}  ({d['rule']}; TV {d['tv']:.3f}, p {d['p']:.4f}, null 95th pct {d['null_p95_tv']:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--analyze", default=None)
    ap.add_argument("--n-calls", type=int, default=N_CALLS)
    ap.add_argument("--out", default=str(RAW / "agent_memory_gate.json"))
    a = ap.parse_args()
    if a.analyze:
        data = json.load(open(a.analyze))
        res = analyse(data)
        data["analysis"] = res  # raw calls are untouched; only the analysis block is regenerated
        Path(a.analyze).write_text(json.dumps(data, indent=1))
        report(res)
        return

    os.environ.update({"HPGA_OPERATOR_MODE": "llm", "HPGA_AGENTS_SEQ_EDIT_RATE": repr(1.0 / GENOME_LEN),
                       "HPGA_AGENTS_MEMORY_WINDOW": "8", "HPGA_LLM_NUM_CTX": "4096",
                       "HPGA_RUN_ID": "agent_memory_gate" + ("_stub" if a.stub else "")})
    os.environ.pop("HPGA_LLM_LOG_PATH", None)
    from hpga import agents_sequence as aseq, genome_model, operators as ops, sequence_model as sm
    from hpga.config import HPGAConfig

    assert aseq.n_edits(GENOME_LEN) == 1
    genome_model.set_active(genome_model.build_genome_model(HPGAConfig(genome_model="sequence")))
    if a.stub:
        os.environ["HPGA_LLM_LOG_PATH"] = os.environ.get("STUB_DIR", "/tmp") + "/_stub_gate_calls.jsonl"
        ops._call_ollama = stub_call
    print(f"model={ops.LLM_MODEL} temperature={ops.LLM_TEMPERATURE} num_ctx={ops.LLM_NUM_CTX} len={GENOME_LEN} n={a.n_calls} "
          f"k={aseq.n_edits(GENOME_LEN)} seed={SEED}  llm log: {ops._log_path()}", flush=True)
    if not a.stub:
        t = time.time()
        ops._get_client().generate(model=ops.LLM_MODEL, prompt="Reply with the single word OK.", stream=False, think=False,
                                   options={"num_predict": 4, "temperature": 0}, keep_alive=ops.LLM_KEEP_ALIVE)
        print(f"warm-up {time.time() - t:.1f}s", flush=True)

    genomes = [sm.random_sequence(random.Random(_seed(SEED, i, "genome")), GENOME_LEN) for i in range(a.n_calls)]
    t0 = time.time()
    ops.reset_operator_stats()
    print("=== A: no record ===", flush=True)
    A = run_condition("A", genomes, None, a.stub)
    validA = [x for x in A if not x["fallback"] and len(x["changes"]) == 1]
    pos_dist = Counter(x["changes"][0][0] for x in validA)
    let_dist = Counter(x["changes"][0][2] for x in validA)
    records = [build_record(g, pos_dist, let_dist, i, sm.ALPHABET) for i, g in enumerate(genomes)]
    ops.reset_operator_stats()
    print("=== B: record shown ===", flush=True)
    B = run_condition("B", genomes, records, a.stub)
    data = {"settings": {"model": ops.LLM_MODEL, "temperature": ops.LLM_TEMPERATURE, "num_ctx": ops.LLM_NUM_CTX, "genome_len": GENOME_LEN,
                         "n_calls": a.n_calls, "k": 1, "seed": SEED, "n_history": N_HISTORY, "shown": 8, "current_round": CURRENT_ROUND,
                         "stub": a.stub, "wall_s": time.time() - t0, "llm_log": str(ops._log_path().relative_to(ROOT))
                         if str(ops._log_path()).startswith(str(ROOT)) else str(ops._log_path())},
            "A": A, "B": B, "records": records}
    res = analyse(data)
    data["analysis"] = res
    out = Path(a.out if not a.stub else os.environ.get("STUB_DIR", "/tmp") + "/_stub_gate.json")
    out.write_text(json.dumps(data, indent=1))
    print(f"wrote {out}  (wall {time.time() - t0:.0f}s)")
    report(res)


if __name__ == "__main__":
    main()
