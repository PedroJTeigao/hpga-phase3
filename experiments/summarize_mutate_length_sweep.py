"""Tables for the mutate/position genome-length sweep (experiments/run_mutate_length_sweep.py).  Each model reported
separately.  Computed from results/raw only; writes results/raw/model_lensweep_tables.md and prints it.

'Absolute' hypothesis: the favoured position keeps the same index at every length (a salient round-number index).
'Relative' hypothesis: it moves with length, i.e. position/(length-1) stays put.  For each model the table gives the
modal position at each length, its relative position, and what each hypothesis would predict from the length-63 mode.
"""

import json
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "results" / "raw"
MODELS = ["gemma4:12b", "qwen2.5:7b"]
LENGTHS = [40, 63, 80]


def tag(m):
    return m.replace(":", "_")


def load(m, n):
    j = json.load(open(RAW / f"sequence_operator_compliance_{tag(m)}_len{n}_lensweep.json"))
    return j["conditions"][0]


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    out = []
    P = out.append
    P("100 mutate/position calls per cell, temperature 0.7, seed 0, one change per call (k = 1). Each model reported on its own.\n")
    summary = {}
    for m in MODELS:
        P(f"## {m}\n")
        P("| genome length N | fell back | requests/call | successful calls | distinct positions (of N; expected if uniform) | modal position (share) | modal position / (N-1) | top-3 share | chi2 vs uniform (df N-1), MC p | choices at multiples of 10 (share; uniform would give) |")
        P("|---|---|---|---|---|---|---|---|---|---|")
        cells = {}
        for n in LENGTHS:
            c = load(m, n)
            cells[n] = c
            pu = c["position_uniformity"]
            pc = {int(k): v for k, v in c["position_counts"].items()}
            if not pc:
                P(f"| {n} | {c['stats']['n_failures']}/100 | {c['requests_per_call']:.2f} | 0 | - | - | - | - | - | - |")
                continue
            tot = sum(pc.values())
            mp, mn = max(pc.items(), key=lambda kv: kv[1])
            exp_d = n * (1 - ((n - 1) / n) ** tot)
            m10 = sum(v for k, v in pc.items() if k % 10 == 0)
            u10 = sum(1 for k in range(n) if k % 10 == 0) / n
            summary[(m, n)] = (mp, mn / tot)
            P(f"| {n} | {c['stats']['n_failures']}/100 | {c['requests_per_call']:.2f} | {tot} | {pu['n_distinct']} ({exp_d:.1f}) | "
              f"{mp} ({pct(mn / tot)}) | {mp / (n - 1):.3f} | {pct(pu['top3_share'])} | {pu['chi2']:.0f}, p={pu['mc_p_value']:.4g} | "
              f"{pct(m10 / tot)} ({pct(u10)}) |")
        P("\nFull distribution (position x count), most used first:\n")
        for n in LENGTHS:
            pc = {int(k): v for k, v in cells[n]["position_counts"].items()}
            P(f"- **N={n}**: " + ", ".join(f"{k}x{v}" for k, v in sorted(pc.items(), key=lambda kv: (-kv[1], kv[0]))))
        if (m, 63) in summary:
            m63 = summary[(m, 63)][0]
            P(f"\nPredictions from the N=63 mode ({m63}): absolute hypothesis -> the same index at N=40 and N=80 ({m63}, {m63}); "
              f"relative hypothesis -> {m63 / 62 * 39:.1f} at N=40 and {m63 / 62 * 79:.1f} at N=80. Observed modal positions: "
              + ", ".join(f"N={n}: {summary[(m, n)][0]}" for n in LENGTHS if (m, n) in summary) + ".\n")
        # replication of step 1 at N=63
        f1 = RAW / f"sequence_operator_compliance_{tag(m)}_len63_hetero.json"
        if f1.exists():
            c1 = next(x for x in json.load(open(f1))["conditions"] if x["op"] == "mutate")
            new = cells[63]["position_counts"]
            P(f"Replication at N=63: step 1 position counts {'identical to' if c1['position_counts'] == new else 'DIFFERENT from'} this run.\n")
    text = "\n".join(out) + "\n"
    (RAW / "model_lensweep_tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
