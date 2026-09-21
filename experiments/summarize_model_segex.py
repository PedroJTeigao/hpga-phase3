"""Tables for the per-model segment-example / prose-number ablation
(experiments/run_model_segex_ablation.py).  Each model is reported separately; nothing is pooled.
Computed from results/raw only: sequence_operator_compliance_<tag>_len63_hetero_segex_<setting>.json (the probe's own
output) and the per-request logs llm_operator_calls_hetero_segex_<setting>_<tag>.jsonl.

Writes results/raw/model_segex_tables.md and prints it.
"""

import json
import re
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "results" / "raw"
MODELS = ["llama3.2:3b", "qwen2.5:7b", "mistral:7b"]
SETTINGS = [("current", "a. current (prose number 40, example 0-40:1, 40-100:2)"),
            ("absent_prose25", "b. absent_prose25 (no example; prose number 25; no 40 in prompt or retry hint)")]
SEG_RE = re.compile(r"(\d+)\s*-\s*(\d+)\s*:\s*([12])")


def tag(m):
    return m.replace(":", "_")


def declarations(m, s):
    """cut sets of the valid declarations in the call log (one per successful call)."""
    p = RAW / f"llm_operator_calls_hetero_segex_{s}_{tag(m)}.jsonl"
    out = []
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r.get("op") == "crossover" and r.get("prompt_style") == "segment" and r.get("valid") is True:
            mm = re.search(r"SEGMENTS\s*:\s*(.*)", r.get("response", ""))
            segs = sorted((int(a), int(b), int(c)) for a, b, c in SEG_RE.findall(mm.group(1))) if mm else []
            out.append(tuple(a for a, _, _ in segs[1:]))
    return out


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    out = []
    P = out.append
    P("Per model, per setting: 100 crossover/segment calls, genome length 63, bounds [30, 80], temperature 0.7, seed 0. "
      "Cuts are percentages; 99 possible values (1-99). Each model is reported on its own.\n")
    for m in MODELS:
        P(f"## {m}\n")
        rows = {}
        for s, label in SETTINGS:
            f = RAW / f"sequence_operator_compliance_{tag(m)}_len63_hetero_segex_{s}.json"
            if not f.exists():
                P(f"(missing: {f.name})\n")
                continue
            j = json.load(open(f))
            c = j["conditions"][0]
            rows[s] = (j, c, declarations(m, s))
        if not rows:
            continue
        P("| setting | fell back | requests/call | valid declarations | segments per call | distinct cut values (of 99) | modal cut (share of cuts) | cuts equal to 40 | declarations containing a 40 cut |")
        P("|---|---|---|---|---|---|---|---|---|")
        for s, label in SETTINGS:
            if s not in rows:
                continue
            j, c, decl = rows[s]
            bs = c["boundary_summary"]
            cuts = {int(k): v for k, v in bs["cut_percent_counts"].items()}
            tot = sum(cuts.values())
            modal = "/".join(str(v) for v in bs["modal_values"])
            n40 = cuts.get(40, 0)
            d40 = sum(1 for d in decl if 40 in d)
            P(f"| {label} | {c['stats']['n_failures']}/{c['n_calls']} ({c['fallback_rate']:.3f}) | {c['requests_per_call']:.2f} | "
              f"{bs['n_valid_declarations']} | {c['segment_structure']['segments_per_call']} | {bs['n_distinct_values']} | "
              f"{modal} ({pct(bs['modal_share_of_cuts'])}) | {n40} of {tot} ({pct(n40 / tot) if tot else 'n/a'}) | "
              f"{d40} of {len(decl)} |")
        P("\nFull cut distribution (value x count), most used first:\n")
        for s, label in SETTINGS:
            if s not in rows:
                continue
            j, c, decl = rows[s]
            cuts = {int(k): v for k, v in c["boundary_summary"]["cut_percent_counts"].items()}
            P(f"- **{s}**: " + ", ".join(f"{k}x{v}" for k, v in sorted(cuts.items(), key=lambda kv: (-kv[1], kv[0]))))
        P("\nPrompt check (what the logged prompts actually contained, over every logged attempt):\n")
        P("| setting | logged attempts | prose says 40 | prose says 25 | example 0-40 present | any e.g. | '40' anywhere outside the Parent header lines | parent length 40 printed in a header (data) |")
        P("|---|---|---|---|---|---|---|---|")
        for s, label in SETTINGS:
            if s not in rows:
                continue
            k = rows[s][1]["segment_example_check"]
            P(f"| {s} | {k['n_logged_attempts']} | {k['with_prose_40']} | {k['with_prose_25']} | {k['with_0-40_example']} | "
              f"{k['with_any_e.g.']} | {k['with_40_outside_parent_headers']} | {k['with_parent_length_40_in_header']} |")
        step1 = RAW / f"sequence_operator_compliance_{tag(m)}_len63_hetero.json"
        if step1.exists():
            c1 = next(x for x in json.load(open(step1))["conditions"] if x["op"] == "crossover")
            st = c1["segment_structure"]
            P(f"\nFor reference, step 1 (same prompt as setting a, different RNG stream because mutate ran first): "
              f"fallback {c1['stats']['n_failures']}/100, {st['n_distinct_cut_values']} distinct cuts, modal {st['modal_cut']} "
              f"({pct(st['modal_cut_share'])}).\n")
    text = "\n".join(out) + "\n"
    (RAW / "model_segex_tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
