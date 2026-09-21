"""Tables for the qwen2.5:7b 2x2 (prose number x worked example) crossover-cut factorial
(experiments/run_qwen_segex_factorial.py).  Computed from results/raw only.  Writes
results/raw/model_factorial_tables.md and prints it.
"""

import json
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "results" / "raw"
CELLS = [("current", "40", "0-40:1, 40-100:2 present"),
         ("absent", "40", "removed"),
         ("prose25_example40", "25", "0-40:1, 40-100:2 present"),
         ("absent_prose25", "25", "removed")]


def load(s):
    return json.load(open(RAW / f"sequence_operator_compliance_qwen2.5_7b_len63_factorial_segex_{s}.json"))["conditions"][0]


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    out = []
    P = out.append
    P("qwen2.5:7b, 100 crossover/segment calls per setting, genome length 63, bounds [30, 80], temperature 0.7, seed 0. "
      "Cuts are percentages; 99 possible values.\n")
    P("| setting | prose sentence says | worked example | fell back | requests/call | valid declarations | segments per call | distinct cuts (of 99) | modal cut (share of cuts) | cuts at 40 | cuts at 25 | declarations containing 40 / 25 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    cells = {}
    for s, prose, ex in CELLS:
        c = load(s)
        cells[s] = c
        bs = c["boundary_summary"]
        cuts = {int(k): v for k, v in bs["cut_percent_counts"].items()}
        tot = sum(cuts.values())
        # declaration-level containment from the call log
        import re
        seg = re.compile(r"(\d+)\s*-\s*(\d+)\s*:\s*([12])")
        decl = []
        for line in open(RAW / f"llm_operator_calls_factorial_{s}_qwen2.5_7b.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r.get("op") == "crossover" and r.get("prompt_style") == "segment" and r.get("valid") is True:
                m = re.search(r"SEGMENTS\s*:\s*(.*)", r.get("response", ""))
                ss = sorted((int(a), int(b), int(c2)) for a, b, c2 in seg.findall(m.group(1))) if m else []
                decl.append({a for a, _, _ in ss[1:]})
        modal = "/".join(str(v) for v in bs["modal_values"])
        P(f"| {s} | {prose} | {ex} | {c['stats']['n_failures']}/{c['n_calls']} | {c['requests_per_call']:.2f} | "
          f"{bs['n_valid_declarations']} | {c['segment_structure']['segments_per_call']} | {bs['n_distinct_values']} | "
          f"{modal} ({pct(bs['modal_share_of_cuts'])}) | {cuts.get(40, 0)} of {tot} | {cuts.get(25, 0)} of {tot} | "
          f"{sum(1 for d in decl if 40 in d)} / {sum(1 for d in decl if 25 in d)} of {len(decl)} |")
    P("\nFull cut distribution (value x count), most used first:\n")
    for s, prose, ex in CELLS:
        cuts = {int(k): v for k, v in cells[s]["boundary_summary"]["cut_percent_counts"].items()}
        P(f"- **{s}**: " + ", ".join(f"{k}x{v}" for k, v in sorted(cuts.items(), key=lambda kv: (-kv[1], kv[0]))))
    P("\nPrompt check (from the logged prompts, every attempt):\n")
    P("| setting | logged attempts | prose says 40 | prose says 25 | example 0-40 present | '40' outside Parent header lines | parent length 40 printed in a header (data) |")
    P("|---|---|---|---|---|---|---|")
    for s, _, _ in CELLS:
        k = cells[s]["segment_example_check"]
        P(f"| {s} | {k['n_logged_attempts']} | {k['with_prose_40']} | {k['with_prose_25']} | {k['with_0-40_example']} | "
          f"{k['with_40_outside_parent_headers']} | {k['with_parent_length_40_in_header']} |")
    # replication of step 2 (same settings, same model, same seed)
    P("\nReplication check against step 2 (same model, seed and prompt settings, run earlier):\n")
    for s, s2 in (("current", "current"), ("absent_prose25", "absent_prose25")):
        f = RAW / f"sequence_operator_compliance_qwen2.5_7b_len63_hetero_segex_{s2}.json"
        if f.exists():
            old = json.load(open(f))["conditions"][0]["boundary_summary"]["cut_percent_counts"]
            new = cells[s]["boundary_summary"]["cut_percent_counts"]
            P(f"- {s}: step 2 {dict(old)} vs this run {dict(new)} -> {'identical' if old == new else 'DIFFERENT'}")
    text = "\n".join(out) + "\n"
    (RAW / "model_factorial_tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
