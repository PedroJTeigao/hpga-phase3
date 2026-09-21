"""Tables for the mutate/position relabelling probe (experiments/run_mutate_relabel_probe.py).  Each model and setting is
reported separately; nothing is pooled.  Computed from results/raw only; writes results/raw/model_relabel_tables.md.

Two views of the labels the model used:
  accepted    labels of the LLM-successful calls (the probe's own position counts, converted to labels = index + offset);
  first reply the label written in the FIRST attempt of every call, valid or not, parsed from the call log -- this is the
              only place an out-of-range answer (e.g. an unshifted 10 under the 100-162 labelling) is visible, because
              such a reply is rejected and retried.
"""

import json
import re
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "results" / "raw"
MODELS = ["gemma4:12b", "qwen2.5:7b"]
SETTINGS = [("normal", 0, "0-62 (as shipped)"), ("offset100", 100, "100-162"), ("offset107", 107, "107-169 (extra setting)")]
L = 63
POS = re.compile(r"POSITION\s*:\s*(\d+)", re.IGNORECASE)


def tag(m):
    return m.replace(":", "_")


def pct(x):
    return f"{100 * x:.1f}%"


def first_replies(m, s):
    """label in the first attempt of each call (None if no parseable label)."""
    out = []
    for line in open(RAW / f"llm_operator_calls_relabel_{s}_{tag(m)}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("op") == "mutate" and r.get("prompt_style") == "position" and r.get("attempt") == 0:
            mm = POS.search(r.get("response", "") or "")
            out.append(int(mm.group(1)) if mm else None)
    return out


def dist(d):
    return ", ".join(f"{k}x{v}" for k, v in sorted(d.items(), key=lambda kv: (-kv[1], kv[0])))


def main():
    out = []
    P = out.append
    P("100 mutate/position calls per cell, genome length 63, temperature 0.7, seed 0, one change per call. Labels are the "
      "numbers the model is shown and answers with; the genome is identical in every setting.\n")
    for m in MODELS:
        P(f"## {m}\n")
        P("| setting (labels shown) | fell back | requests/call | successful calls | distinct accepted labels (of 63) | modal accepted label (share) | modal label as index (label - offset) | modal label is a multiple of 10 | first replies outside the stated range | modal first-reply label (share of 100) |")
        P("|---|---|---|---|---|---|---|---|---|---|")
        det = []
        for s, off, label in SETTINGS:
            f = RAW / f"sequence_operator_compliance_{tag(m)}_len63_relabel_{s}.json"
            if not f.exists():
                P(f"| {label} | (missing) |")
                continue
            c = json.load(open(f))["conditions"][0]
            acc = {int(k) + off: v for k, v in c["position_counts"].items()}
            tot = sum(acc.values())
            fr = first_replies(m, s)
            raw = {}
            for x in fr:
                if x is not None:
                    raw[x] = raw.get(x, 0) + 1
            oor = {k: v for k, v in raw.items() if not (off <= k <= off + L - 1)}
            n_none = sum(1 for x in fr if x is None)
            if acc:
                ml, mn = max(acc.items(), key=lambda kv: kv[1])
                mod = f"{ml} ({pct(mn / tot)})"
                idx, r10 = ml - off, "yes" if ml % 10 == 0 else "no"
            else:
                mod, idx, r10 = "-", "-", "-"
            rl, rn = max(raw.items(), key=lambda kv: kv[1]) if raw else ("-", 0)
            P(f"| {label} | {c['stats']['n_failures']}/{c['n_calls']} | {c['requests_per_call']:.2f} | {tot} | {len(acc)} | {mod} | {idx} | {r10} | "
              f"{sum(oor.values())} of {len(fr)} | {rl} ({rn}%) |")
            det.append((s, label, off, acc, raw, oor, n_none))
        P("")
        for s, label, off, acc, raw, oor, n_none in det:
            P(f"**{label}**")
            P(f"- accepted labels (label x count): {dist(acc)}")
            P(f"- first-reply labels (label x count): {dist(raw)}" + (f"; {n_none} first replies had no parseable label" if n_none else ""))
            P("- first replies outside the stated range: " + (dist(oor) if oor else "none"))
            P("")
    text = "\n".join(out) + "\n"
    (RAW / "model_relabel_tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
