"""Format probe for agent memory. Stage 3 (the GA) found the model coming back to positions its own record just
called "worse" more than ones it called "better", and often answering with the letter already at that position --
suggestive that the model reads the record but can't use its sign. This probe asks the Phase 2 question of that
finding: is the record failing on FORMAT (the way "full" mutate/crossover prompts failed until asked for edits
instead -- PHASE2_RESULTS.md Sec 4) or on PRINCIPLE (the model cannot use this kind of signal at all)?

One isolated probe, no GA, no fitness. 100 mutate/position calls per condition, gemma4:12b, the operator-probe
settings (temperature 0.7, num_ctx 4096, genome length 63, k = 1 edit per call, seed 0, fresh uniformly random
genome per call), one agent (the same call hpga.agents_sequence.propose makes: hpga.sequence_model.plan_llm_mutate).

  a  no record                the plain mutate/position prompt
  b  the current prose record hpga.agents_sequence.render_record's own format, PRODUCTION CODE, unchanged
  c  explicit avoid list      "Do not change these positions: x, y, z."
  d  explicit prefer list     "These positions improved before: x, y, z."

Pairing, so the right answer is known. Per call i, ONE genome and TWO disjoint 3-position sets are drawn once --
avoid_i (positions with a "worse" entry) and prefer_i (positions with a "better" entry) -- and reused in EVERY
condition for that call, so a, b, c and d differ only in what is shown, never in the genome or the sets themselves.
Condition b's record is built by construction: one entry per position in avoid_i (improved=False, fitness after <
before) and one per position in prefer_i (improved=True), each entry's NEW letter set to the genome's own letter at
that position now (consistent with what a real agent's history would show for the genome it currently holds), fed
into agents_sequence.render_record unmodified. Conditions c and d show only one of the two lists, in the wording
above. Condition a never sees avoid_i/prefer_i, so a's own shares (below) are the NULL: what a call would land on
those positions anyway, with nothing shown.

Reported per condition: fallback rate; NO-OP RATE (share of calls with at least one attempt whose parsed
POSITION/NEW pair proposed the letter already at that position -- read from the raw call log after the fact, the
mechanism identified in the Stage 3 GA runs, not a new parser); and, of the calls that produced a valid edit, the
share whose chosen position is in THAT CALL'S avoid_i and the share in THAT CALL'S prefer_i. Also, paired against
condition a (exact sign test on the paired indicator, calls valid in both conditions): if c's avoid share falls far
below a's null and b's does not, memory failed on format, not on principle; the same logic for d's prefer share.

  python experiments/probe_agent_memory_format.py            # live: 400 calls, writes results/raw/agent_memory_format.json
  python experiments/probe_agent_memory_format.py --stub     # offline: stubbed model, checks the pipeline end to end
  python experiments/probe_agent_memory_format.py --analyze results/raw/agent_memory_format.json
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
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RAW = ROOT / "results" / "raw"

GENOME_LEN, N_CALLS, SEED, K = 63, 100, 0, 1
N_LIST = 3  # positions per avoid / prefer list
CONDITIONS = ("a", "b", "c", "d")


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# --- per-call fixtures (shared across all 4 conditions) ---------------------------------------------------------

def build_lists(call_idx: int) -> tuple[list[int], list[int]]:
    r = random.Random(_seed(SEED, call_idx, "lists"))
    chosen = r.sample(range(GENOME_LEN), 2 * N_LIST)
    return sorted(chosen[:N_LIST]), sorted(chosen[N_LIST:])


def build_record_entries(genome: str, avoid: list[int], prefer: list[int], call_idx: int, alphabet: str) -> list[dict]:
    r = random.Random(_seed(SEED, call_idx, "record"))
    gens = list(range(1, 1 + N_LIST + N_LIST))
    r.shuffle(gens)
    entries, gi = [], 0
    for p in avoid:
        new = genome[p]
        old = r.choice([c for c in alphabet if c != new])
        f0 = round(r.uniform(0.25, 0.35), 3)
        entries.append({"agent_id": 0, "generation": gens[gi], "base_fitness": f0, "changes": [[p, old, new]],
                        "fitness_after": round(f0 - r.uniform(0.03, 0.08), 3), "no_edit": False, "improved": False})
        gi += 1
    for p in prefer:
        new = genome[p]
        old = r.choice([c for c in alphabet if c != new])
        f0 = round(r.uniform(0.25, 0.35), 3)
        entries.append({"agent_id": 0, "generation": gens[gi], "base_fitness": f0, "changes": [[p, old, new]],
                        "fitness_after": round(f0 + r.uniform(0.03, 0.08), 3), "no_edit": False, "improved": True})
        gi += 1
    entries.sort(key=lambda e: e["generation"])  # render_record expects oldest first
    return entries


def build_context(cond: str, genome: str, avoid: list[int], prefer: list[int], call_idx: int) -> str:
    from hpga import agents_sequence as aseq

    if cond == "a":
        return ""
    if cond == "c":
        return f"Do not change these positions: {', '.join(map(str, avoid))}.\n\n"
    if cond == "d":
        return f"These positions improved before: {', '.join(map(str, prefer))}.\n\n"
    assert cond == "b"
    from hpga import sequence_model as sm

    entries = build_record_entries(genome, avoid, prefer, call_idx, sm.ALPHABET)
    return aseq.render_record(entries, window=2 * N_LIST)


# --- collection ---------------------------------------------------------------------------------------------------

def run_condition(cond: str, genomes: list[str], avoids: list[list[int]], prefers: list[list[int]]) -> None:
    from hpga import operators as ops
    from hpga import sequence_model as sm

    for i, g in enumerate(genomes):
        rng = random.Random(_seed(SEED, i, "call"))
        plan = sm.plan_llm_mutate("position", g, K)
        context = build_context(cond, g, avoids[i], prefers[i], i)

        def build_prompt(retry_hint: str, context=context) -> str:
            return context + plan.build_prompt(retry_hint)

        ops._run_llm_op(
            op="format_probe", style="position", system=plan.system, build_prompt=build_prompt, parse=plan.parse,
            rng=rng, num_predict=plan.num_predict, fallback=lambda: g, retry_hint_text=plan.retry_hint_text,
            extra_log_fields={"condition": cond, "call_idx": i},
        )
        if (i + 1) % 20 == 0:
            print(f"  {cond}: {i + 1}/{len(genomes)}", flush=True)


def stub_call(prompt, system, num_predict, seed):
    """Offline stand-in: obeys an explicit avoid list strongly, an explicit prefer list strongly, the prose record
    weakly, and is uniform (over the genome, biased slightly to 10) with no list."""
    from hpga import sequence_model as sm

    r = random.Random(_seed(seed, prompt[-350:]))
    cur = "".join(re.search(r"Sequence \(0-indexed positions 0-\d+\): ([A-Z ]+)", prompt).group(1).split())
    n = len(cur)
    avoid_m = re.search(r"Do not change these positions: ([\d, ]+)\.", prompt)
    prefer_m = re.search(r"These positions improved before: ([\d, ]+)\.", prompt)
    avoid = {int(x) for x in avoid_m.group(1).split(",")} if avoid_m else set()
    prefer = {int(x) for x in prefer_m.group(1).split(",")} if prefer_m else set()
    record_prefer = set()
    if "Your own record" in prompt:
        for m in re.finditer(r"position (\d+) \S+->\S+; fitness [\d.]+ -> [\d.]+ \((better|worse)\)", prompt):
            if m.group(2) == "better":
                record_prefer.add(int(m.group(1)))
    pool = [p for p in range(n) if p not in avoid] or list(range(n))
    if prefer and r.random() < 0.75:
        p = r.choice(sorted(prefer))
    elif record_prefer and r.random() < 0.3:
        p = r.choice(sorted(record_prefer))
    else:
        p = 10 if r.random() < 0.3 and 10 in pool else r.choice(pool)
    same_letter_junk = r.random() < 0.05
    letter = cur[p] if same_letter_junk else r.choice([c for c in sm.ALPHABET if c != cur[p]])
    bad = r.random() < 0.03
    return ("junk" if bad else f"POSITION: {p}, NEW: {letter}"), 0.03, len(prompt) // 4, 10


# --- analysis -------------------------------------------------------------------------------------------------

def load_calls(log_path: Path) -> dict:
    """{(condition, call_idx): [records in attempt order] + fallback record if any}"""
    groups: dict = {}
    for line in open(log_path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("op") != "format_probe":
            continue
        key = (r["condition"], r["call_idx"])
        groups.setdefault(key, []).append(r)
    return groups


def analyse(genomes: list[str], avoids: list[list[int]], prefers: list[list[int]], groups: dict) -> dict:
    from hpga import sequence_model as sm

    per_cond = {c: [] for c in CONDITIONS}
    for (cond, i), recs in groups.items():
        g, av, pf = genomes[i], avoids[i], prefers[i]
        fell_back = any(r.get("event") == "fallback_to_deterministic" for r in recs)
        attempts = [r for r in recs if "attempt" in r]
        same_letter = False
        for r in attempts:
            for pos_s, letter in sm._POSITION_PAIR.findall(r.get("response") or ""):
                if int(pos_s) < len(g) and letter.upper() == g[int(pos_s)]:
                    same_letter = True
        chosen = None
        if not fell_back:
            valid_attempts = [r for r in attempts if r.get("valid")]
            if valid_attempts:
                pairs = sm._POSITION_PAIR.findall(valid_attempts[-1]["response"])
                if len(pairs) == 1:
                    chosen = int(pairs[0][0])
        per_cond[cond].append({"i": i, "fallback": fell_back, "same_letter_any_attempt": same_letter, "chosen": chosen,
                               "in_avoid": chosen in av if chosen is not None else None,
                               "in_prefer": chosen in pf if chosen is not None else None})

    def share(rows, key):
        v = [r[key] for r in rows if r[key] is not None]
        return {"n": len(v), "hit": sum(v), "share": sum(v) / len(v) if v else None}

    def exact_sign_p(x: int, y: int) -> float:
        n = x + y
        if n == 0:
            return 1.0
        tail = sum(comb(n, k) for k in range(0, min(x, y) + 1)) / 2 ** n
        return min(1.0, 2 * tail)

    def paired_vs_a(cond: str, key: str) -> dict:
        by_i_a = {r["i"]: r[key] for r in per_cond["a"] if r[key] is not None}
        by_i_c = {r["i"]: r[key] for r in per_cond[cond] if r[key] is not None}
        common = sorted(set(by_i_a) & set(by_i_c))
        a_only = sum(1 for i in common if by_i_a[i] and not by_i_c[i])
        c_only = sum(1 for i in common if by_i_c[i] and not by_i_a[i])
        return {"n_pairs": len(common), "hit_a": sum(by_i_a[i] for i in common), "hit_" + cond: sum(by_i_c[i] for i in common),
               "a_only": a_only, cond + "_only": c_only, "exact_sign_p": exact_sign_p(a_only, c_only)}

    out = {}
    for cond, rows in per_cond.items():
        distinct = Counter(r["chosen"] for r in rows if r["chosen"] is not None)
        out[cond] = {
            "n_calls": len(rows), "fallback": share(rows, "fallback"), "no_op_rate": share(rows, "same_letter_any_attempt"),
            "avoid_share": share(rows, "in_avoid"), "prefer_share": share(rows, "in_prefer"),
            "distinct_positions": len(distinct), "modal_position": distinct.most_common(1)[0] if distinct else None,
        }
    out["paired_vs_a"] = {
        "c_avoid_vs_a": paired_vs_a("c", "in_avoid"), "b_avoid_vs_a": paired_vs_a("b", "in_avoid"),
        "d_prefer_vs_a": paired_vs_a("d", "in_prefer"), "b_prefer_vs_a": paired_vs_a("b", "in_prefer"),
    }
    return out


def fmt_share(d: dict) -> str:
    return f"{d['hit']}/{d['n']} = {d['share']:.0%}" if d["share"] is not None else "n/a"


def report(res: dict) -> None:
    print("\n| condition | calls | fallback | no-op rate (any attempt) | avoid share | prefer share | distinct pos | mode |")
    print("|---|---|---|---|---|---|---|---|")
    label = {"a": "a: no record", "b": "b: prose record", "c": "c: explicit avoid list", "d": "d: explicit prefer list"}
    for c in CONDITIONS:
        r = res[c]
        mode = f"{r['modal_position'][0]} ({r['modal_position'][1]}/{r['avoid_share']['n'] + (r['n_calls'] - r['avoid_share']['n'])})" if r["modal_position"] else "n/a"
        print(f"| {label[c]} | {r['n_calls']} | {fmt_share(r['fallback'])} | {fmt_share(r['no_op_rate'])} | {fmt_share(r['avoid_share'])} | {fmt_share(r['prefer_share'])} | {r['distinct_positions']} | {mode} |")
    print("\npaired against condition a (exact sign test on calls valid in both):")
    for key, p in res["paired_vs_a"].items():
        other = key.split("_")[0]  # "c", "b" or "d"
        print(f"  {key}: a={p['hit_a']}/{p['n_pairs']}  {other}={p['hit_' + other]}/{p['n_pairs']}  "
              f"discordant a-only={p['a_only']} {other}-only={p[other + '_only']}  exact sign p={p['exact_sign_p']:.4f}")
    a, c, d, b = res["a"]["avoid_share"]["share"], res["c"]["avoid_share"]["share"], res["d"]["prefer_share"]["share"], res["b"]["avoid_share"]["share"]
    print(f"\nDECISION-RELEVANT: avoid share -- a (null) {fmt_share(res['a']['avoid_share'])}, b (prose) {fmt_share(res['b']['avoid_share'])}, "
          f"c (explicit) {fmt_share(res['c']['avoid_share'])}. prefer share -- a (null) {fmt_share(res['a']['prefer_share'])}, "
          f"b (prose) {fmt_share(res['b']['prefer_share'])}, d (explicit) {fmt_share(res['d']['prefer_share'])}.")
    if a is not None and c is not None:
        verdict = "FORMAT: c drops far below a's null" if c < a * 0.5 else ("NOT FORMAT: c does not drop below a's null" if c >= a else "unclear")
        print(f"  avoid: {verdict} (c {c:.0%} vs a {a:.0%})" + (f"; b {'stayed near a' if b is not None and abs(b - a) < abs(c - a) else 'also moved'} ({fmt_share(res['b']['avoid_share'])})" if b is not None else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--analyze", default=None)
    ap.add_argument("--n-calls", type=int, default=N_CALLS)
    ap.add_argument("--out", default=str(RAW / "agent_memory_format.json"))
    a = ap.parse_args()

    if a.analyze:
        data = json.load(open(a.analyze))
        # analysis is always regenerated from the raw call log (never from a copy of it)
        groups = load_calls(ROOT / data["settings"]["llm_log"])
        res = analyse(data["genomes"], data["avoids"], data["prefers"], groups)
        data["analysis"] = res
        Path(a.analyze).write_text(json.dumps(data, indent=1))
        report(res)
        return

    os.environ.update({"HPGA_OPERATOR_MODE": "llm", "HPGA_LLM_NUM_CTX": "4096",
                       "HPGA_RUN_ID": "agent_memory_format" + ("_stub" if a.stub else "")})
    os.environ.pop("HPGA_LLM_LOG_PATH", None)
    from hpga import genome_model, operators as ops, sequence_model as sm
    from hpga.config import HPGAConfig

    genome_model.set_active(genome_model.build_genome_model(HPGAConfig(genome_model="sequence")))
    if a.stub:
        os.environ["HPGA_LLM_LOG_PATH"] = str(Path(os.environ.get("STUB_DIR", "/tmp")) / "_stub_format_calls.jsonl")
        ops._call_ollama = stub_call
    print(f"model={ops.LLM_MODEL} temperature={ops.LLM_TEMPERATURE} num_ctx={ops.LLM_NUM_CTX} len={GENOME_LEN} n={a.n_calls} "
          f"k={K} seed={SEED}  llm log: {ops._log_path()}", flush=True)
    if not a.stub:
        t = time.time()
        ops._get_client().generate(model=ops.LLM_MODEL, prompt="Reply with the single word OK.", stream=False, think=False,
                                   options={"num_predict": 4, "temperature": 0}, keep_alive=ops.LLM_KEEP_ALIVE)
        print(f"warm-up {time.time() - t:.1f}s", flush=True)

    genomes = [sm.random_sequence(random.Random(_seed(SEED, i, "genome")), GENOME_LEN) for i in range(a.n_calls)]
    fixtures = [build_lists(i) for i in range(a.n_calls)]
    avoids, prefers = [f[0] for f in fixtures], [f[1] for f in fixtures]

    t0 = time.time()
    for cond in CONDITIONS:
        print(f"=== condition {cond} ===", flush=True)
        ops.reset_operator_stats()
        run_condition(cond, genomes, avoids, prefers)

    groups = load_calls(ops._log_path())
    res = analyse(genomes, avoids, prefers, groups)
    llm_log_rel = str(ops._log_path().relative_to(ROOT)) if str(ops._log_path()).startswith(str(ROOT)) else str(ops._log_path())
    data = {"settings": {"model": ops.LLM_MODEL, "temperature": ops.LLM_TEMPERATURE, "num_ctx": ops.LLM_NUM_CTX,
                         "genome_len": GENOME_LEN, "n_calls": a.n_calls, "k": K, "seed": SEED, "n_list": N_LIST,
                         "stub": a.stub, "wall_s": time.time() - t0, "llm_log": llm_log_rel},
            "genomes": genomes, "avoids": avoids, "prefers": prefers, "analysis": res}
    out = Path(a.out if not a.stub else str(Path(os.environ.get("STUB_DIR", "/tmp")) / "_stub_format.json"))
    out.write_text(json.dumps(data, indent=1))
    print(f"wrote {out}  (wall {time.time() - t0:.0f}s)")
    report(res)


if __name__ == "__main__":
    main()
