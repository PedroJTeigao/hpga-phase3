"""Tables for the prose-number sweep (experiments/run_cut_number_sweep.py).  Each model and prose number is reported
separately; nothing is pooled.  Computed from results/raw only; writes results/raw/model_cutnum_tables.md.

Definitions (100 crossover/segment calls per cell; a call is one operator invocation, which may retry):
  cuts               every interior boundary declared in the VALID reply of each LLM-successful call (a 2-segment reply has
                     one cut, a 3-segment reply two); the probe's own boundary_summary counts the same thing.
  share of cuts = N  cuts equal to the prose number / all cuts.
  calls with N       LLM-successful calls whose declaration contains N as a cut / 100 calls.
  calls, only cut N  LLM-successful calls whose declaration is a single cut, at N / 100 calls (the strictest reading of
                     "the cut equals the prose number").
  within +-5 of N    cuts with |cut - N| <= 5 / all cuts.   midpoint 45-55: cuts in 45..55 / all cuts.
  first attempt      the same cut counts taken from the FIRST attempt of every call (valid or not) in the call log, so a
                     retry that moved the answer cannot hide a first reply.
"""

import json
import re
from collections import Counter
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "results" / "raw"
MODELS = ["qwen2.5:7b", "gemma4:12b"]
NUMBERS = [10, 25, 37, 60, 90]
SEG = re.compile(r"(\d+)\s*-\s*(\d+)\s*:\s*([12])")


def tag(m):
    return m.replace(":", "_")


def pct(x):
    return f"{100 * x:.1f}%"


def cuts_of(response):
    m = re.search(r"SEGMENTS\s*:\s*(.*)", response or "")
    segs = sorted((int(a), int(b), int(c)) for a, b, c in SEG.findall(m.group(1))) if m else []
    return [a for a, _, _ in segs[1:]]


def load(m, n):
    s = f"prose{n}"
    res = json.load(open(RAW / f"sequence_operator_compliance_{tag(m)}_len63_cutnum_segex_{s}.json"))
    assert res.get("complete") is True, f"{m} {s}: run not complete"
    cond = res["conditions"][0]
    recs = [json.loads(l) for l in open(RAW / f"llm_operator_calls_cutnum_{s}_{tag(m)}.jsonl", encoding="utf-8")]
    recs = [r for r in recs if r.get("op") == "crossover" and r.get("prompt_style") == "segment"]
    return cond, recs


def dist_str(cnt):
    return ", ".join(f"{k}x{v}" for k, v in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])))


def stats(m, n):
    cond, recs = load(m, n)
    valid = [r for r in recs if r.get("valid") is True]
    decl = [cuts_of(r.get("response")) for r in valid]
    cnt = Counter(c for d in decl for c in d)
    tot = sum(cnt.values())
    first = Counter(c for r in recs if r.get("attempt") == 0 for c in cuts_of(r.get("response")))
    bs = cond["boundary_summary"]
    assert {int(k): v for k, v in bs["cut_percent_counts"].items()} == dict(cnt), "log and probe summaries disagree"
    top = max(cnt.values()) if cnt else 0
    modal = sorted(v for v, c in cnt.items() if c == top)
    # prompt check over every logged attempt
    nb = re.compile(rf"(?<!\d){n}(?!\d)")
    p_prose = p_eg = p_hdr = p_extra = 0
    for r in recs:
        p = r["prompt"]
        p_prose += f"boundary at {n} falls {n}%" in p
        p_eg += "e.g." in p
        lines = p.splitlines()
        hdr = [l for l in lines if l.startswith(("Parent 1 (length", "Parent 2 (length"))]
        p_hdr += any(f"length {n})" in l for l in hdr)
        body = "\n".join(l for l in lines if l not in hdr)
        p_extra += len(nb.findall(body)) != 3  # expected exactly the three N's of the prose sentence (at N, N%, and N%)
    return {
        "n_calls": cond["n_calls"], "fell_back": cond["stats"]["n_failures"], "rpc": cond["requests_per_call"],
        "n_valid": len(valid), "segs": cond["segment_structure"]["segments_per_call"], "cnt": cnt, "tot": tot,
        "distinct": len(cnt), "modal": modal, "top": top, "at_n": cnt.get(n, 0),
        "calls_with_n": sum(1 for d in decl if n in d), "calls_only_n": sum(1 for d in decl if d == [n]),
        "near": sum(c for v, c in cnt.items() if abs(v - n) <= 5), "mid": sum(c for v, c in cnt.items() if 45 <= v <= 55),
        "first": first, "n_attempts": len(recs), "p_prose": p_prose, "p_eg": p_eg, "p_hdr": p_hdr, "p_extra": p_extra,
    }


def main():
    out = []
    P = out.append
    P("100 crossover/segment calls per cell, genome length 63, bounds [30, 80], temperature 0.7, seed 0, worked example "
      "removed in every setting, prose sentence 'a boundary at N falls N% of the way ...' with N as shown. 99 possible cuts.\n")
    S = {(m, n): stats(m, n) for m in MODELS for n in NUMBERS}
    for m in MODELS:
        P(f"## {m}\n")
        P("| prose N | fell back | requests/call | valid declarations | segments per call | distinct cuts (of 99) | modal cut (share of cuts) | "
          "cuts = N (share of cuts) | calls containing N (of 100) | calls whose only cut is N (of 100) | cuts within +-5 of N | cuts in 45-55 |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for n in NUMBERS:
            s = S[m, n]
            P(f"| {n} | {s['fell_back']}/{s['n_calls']} | {s['rpc']:.2f} | {s['n_valid']} | {s['segs']} | {s['distinct']} | "
              f"{'/'.join(map(str, s['modal']))} ({pct(s['top'] / s['tot'])}) | {s['at_n']} of {s['tot']} ({pct(s['at_n'] / s['tot'])}) | "
              f"{s['calls_with_n']} | {s['calls_only_n']} | {pct(s['near'] / s['tot'])} | {pct(s['mid'] / s['tot'])} |")
        P("\nFull cut distribution (value x count), most used first:\n")
        for n in NUMBERS:
            P(f"- **N={n}**: {dist_str(S[m, n]['cnt'])}")
        P("\nFirst-attempt cuts (every call's first reply, valid or not), most used first:\n")
        for n in NUMBERS:
            P(f"- **N={n}**: {dist_str(S[m, n]['first'])}")
        P("")
    P("## Prompt check (from the logged prompts, every attempt, both models)\n")
    P("| model | prose N | logged attempts | prose sentence with N | 'e.g.' present | attempts with N anywhere but the 3 prose slots | parent length N printed in a header (data) |")
    P("|---|---|---|---|---|---|---|")
    for m in MODELS:
        for n in NUMBERS:
            s = S[m, n]
            P(f"| {m} | {n} | {s['n_attempts']} | {s['p_prose']} | {s['p_eg']} | {s['p_extra']} | {s['p_hdr']} |")
    P("\n## Replication against earlier runs (same model, seed, prompt)\n")
    for m, ref in (("qwen2.5:7b", "sequence_operator_compliance_qwen2.5_7b_len63_hetero_segex_absent_prose25.json"),
                   ("gemma4:12b", None)):
        new = dict(S[m, 25]["cnt"])
        if ref and (RAW / ref).exists():
            old = {int(k): v for k, v in json.load(open(RAW / ref))["conditions"][0]["boundary_summary"]["cut_percent_counts"].items()}
            P(f"- {m} N=25: step 2 `absent_prose25` {old} vs this run {new} -> {'identical' if old == new else 'DIFFERENT'}")
        else:
            P(f"- {m} N=25: this run {new} (no earlier file with this exact setting on disk; PHASE3_RESULTS.md 9.3 reports 50x100)")
    text = "\n".join(out) + "\n"
    (RAW / "model_cutnum_tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
