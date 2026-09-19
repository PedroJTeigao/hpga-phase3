"""Tables and chart for the sequence-mode comparison (run_sequence_ga_comparison.py), built only from
the raw result files -- nothing typed by hand.

  python experiments/summarize_sequence_ga_comparison.py [--raw DIR] [--out-md FILE] [--out-png FILE]

Sections written to --out-md (per-seed tables are never pooled; the one cross-seed table is the
mean / range summary, and it is labelled as such):
  1 runs (what finished)          2 best-so-far vs DISTINCT evaluations, per seed (+ chart)
  3 final best per seed, then mean and range over seeds
  4 mean pairwise edit distance per generation (B and C)
  5 final best genomes and their edit distance to the reference
  6 wall time split      7 arm C fallback rate and requests per call
  8 comparisons: B vs A, C vs A, C vs B at EQUAL distinct-evaluation budget

Comparison rule (fixed here, before any result was read): for a pair X, Y and a seed, compare
best-so-far at n = min(distinct evals of X, of Y) -- arm A's curve is read at n, since A runs longer.
X "beats" Y only if X is strictly higher in EVERY seed compared and the mean difference is positive.
Anything else is reported as no demonstrated difference (or the reverse, if Y wins every seed).
With k seeds the smallest two-sided sign-test p-value possible is 2 * 0.5**k, printed alongside.
"""

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARMS = ("A", "B", "C")
ARM_NAME = {"A": "A random search", "B": "B GA (deterministic ops)", "C": "C GA (LLM ops)"}
CHECKPOINTS = (16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320)
SLOT = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}  # validated categorical slots 1-3 (all-pairs)


def load(raw: Path) -> dict:
    runs = {}
    for p in sorted(raw.glob("sequence_ga_cmp_[ABC]_seed*.json")):
        d = json.load(open(p))
        if d.get("complete"):
            runs[(d["arm"], d["seed"])] = d
    return runs


def best_at(run: dict, n: int):
    curve = run["best_so_far_by_distinct_evaluation"]
    return curve[min(n, len(curve)) - 1] if n >= 1 and curve else None


def n_distinct(run: dict) -> int:
    return run["summary"]["n_distinct_evaluations"]


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f4(x) -> str:
    return "—" if x is None else f"{x:.4f}"


def llm_behaviour(run: dict | None):
    """[mutate calls, positions changed, distinct indices, top index, distinct letters, top letter,
    crossover calls, distinct cuts, cut counts] from the call log of one arm-C run."""
    import re
    from collections import Counter
    if not run or not run.get("llm_call_log"):
        return None
    path = Path(run["llm_call_log"])
    path = path if path.is_absolute() else ROOT / path
    if not path.exists():
        return None
    pos, let, cuts, n_mut, n_x = Counter(), Counter(), Counter(), 0, 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("valid") is not True:
            continue
        if r.get("op") == "mutate":
            n_mut += 1
            for a, b in re.findall(r"POSITION\s*:\s*(\d+)\s*,\s*NEW\s*:\s*([A-Za-z])", r["response"], re.I):
                pos[int(a)] += 1
                let[b.upper()] += 1
        elif r.get("op") == "crossover":
            n_x += 1
            m = re.search(r"SEGMENTS\s*:\s*(.*)", r["response"])
            trip = sorted((int(a), int(b), int(c)) for a, b, c in re.findall(r"(\d+)\s*-\s*(\d+)\s*:\s*([12])", m.group(1))) if m else []
            for t in trip[1:]:
                cuts[t[0]] += 1
    tp, tl = pos.most_common(1)[0] if pos else (None, 0), let.most_common(1)[0] if let else (None, 0)
    return [n_mut, sum(pos.values()), len(pos), f"{tp[0]} ({tp[1] / max(1, sum(pos.values())):.0%})",
            len(let), f"{tl[0]} ({tl[1] / max(1, sum(let.values())):.0%})", n_x, len(cuts),
            ", ".join(f"{c}: {n}" for c, n in sorted(cuts.items()))]


def sections(runs: dict, status: dict | None, check: dict | None) -> tuple[str, dict]:
    seeds = sorted({s for (_, s) in runs})
    md, facts = [], {}

    # 1 ----------------------------------------------------------------- runs
    rows = []
    for s in seeds:
        for a in ARMS:
            r = runs.get((a, s))
            if r is None:
                rows.append([s, ARM_NAME[a], "not completed", "", "", "", "", ""])
                continue
            m = r["summary"]
            rows.append([s, ARM_NAME[a], m["n_distinct_evaluations"], m["n_cache_hits"], f4(m["final_best"]),
                         m["final_best_length"], m["final_best_edit_distance_to_reference"], r.get("attempt", 1)])
    md.append("### 1. Runs\n\n" + md_table(
        ["seed", "arm", "distinct evaluations", "cache hits", "final best TM", "final best length",
         "edit distance to reference", "attempt"], rows))
    if status and status.get("failed"):
        md.append("\nFailed after one retry: " + "; ".join(f"{f['arm']} seed {f['seed']} ({f['error']})" for f in status["failed"]))

    # 2 ------------------------------------------------------------ best-so-far
    md.append("\n### 2. Best-so-far TM-score vs distinct evaluations (per seed, never pooled)\n")
    for s in seeds:
        have = {a: runs[(a, s)] for a in ARMS if (a, s) in runs}
        top = max((n_distinct(r) for r in have.values()), default=0)
        pts = [n for n in CHECKPOINTS if n <= top]
        rows = []
        for n in pts:
            rows.append([n] + [f4(best_at(have[a], n)) if a in have and n <= len(have[a]["best_so_far_by_distinct_evaluation"]) else "—" for a in ARMS])
        rows.append(["own final (n)"] + [f"{f4(have[a]['summary']['final_best'])} ({n_distinct(have[a])})" if a in have else "—" for a in ARMS])
        md.append(f"**Seed {s}**\n\n" + md_table(["distinct evaluations"] + [ARM_NAME[a] for a in ARMS], rows) + "\n")

    # 3 ------------------------------------------------------------ final best
    md.append("### 3. Final best per seed; then mean and range over seeds\n")
    rows = []
    for a in ARMS:
        rows.append([ARM_NAME[a] + " (own budget)"] + [f4(runs[(a, s)]["summary"]["final_best"]) if (a, s) in runs else "—" for s in seeds])
    for tag, arm in (("B", "B"), ("C", "C")):
        rows.append([f"A read at {tag}'s distinct-evaluation count"] + [
            f4(best_at(runs[("A", s)], n_distinct(runs[(arm, s)]))) if ("A", s) in runs and (arm, s) in runs else "—" for s in seeds])
    md.append(md_table(["arm"] + [f"seed {s}" for s in seeds], rows) + "\n")
    agg = []
    for label, get in (
        ("A (own budget)", lambda s: runs[("A", s)]["summary"]["final_best"] if ("A", s) in runs else None),
        ("A at B's budget", lambda s: best_at(runs[("A", s)], n_distinct(runs[("B", s)])) if ("A", s) in runs and ("B", s) in runs else None),
        ("A at C's budget", lambda s: best_at(runs[("A", s)], n_distinct(runs[("C", s)])) if ("A", s) in runs and ("C", s) in runs else None),
        ("B", lambda s: runs[("B", s)]["summary"]["final_best"] if ("B", s) in runs else None),
        ("C", lambda s: runs[("C", s)]["summary"]["final_best"] if ("C", s) in runs else None),
    ):
        v = [x for x in (get(s) for s in seeds) if x is not None]
        agg.append([label, len(v), f4(statistics.mean(v)) if v else "—", f"{f4(min(v))} – {f4(max(v))}" if v else "—"])
    md.append("Cross-seed summary (the only pooled table; n = number of seeds that finished):\n\n" +
              md_table(["arm", "n seeds", "mean final best", "range (min – max)"], agg) + "\n")

    # 4 ------------------------------------------------------------ diversity
    md.append("### 4. Mean pairwise edit distance within the population, per generation\n")
    cols = [(a, s) for a in ("B", "C") for s in seeds if (a, s) in runs]
    if cols:
        ngen = max(len(runs[c]["generations"]) for c in cols)
        rows = [[g] + [f"{runs[c]['generations'][g]['mean_pairwise_edit_distance']:.1f}" if g < len(runs[c]["generations"]) else "—" for c in cols]
                for g in range(ngen)]
        md.append(md_table(["gen"] + [f"{a} s{s}" for a, s in cols], rows) + "\n")
        rows = [[g] + [str(runs[c]["generations"][g]["n_distinct_in_population"]) if g < len(runs[c]["generations"]) else "—" for c in cols]
                for g in range(ngen)]
        md.append("Distinct genomes in the population (of 16), same columns:\n\n" + md_table(["gen"] + [f"{a} s{s}" for a, s in cols], rows) + "\n")

    # 5 ------------------------------------------------------------ genomes
    md.append("### 5. Final best genomes and their edit distance to the reference sequence\n")
    rows = [[s, ARM_NAME[a], f4(runs[(a, s)]["summary"]["final_best"]), runs[(a, s)]["summary"]["final_best_length"],
             runs[(a, s)]["summary"]["final_best_edit_distance_to_reference"]] for s in seeds for a in ARMS if (a, s) in runs]
    md.append(md_table(["seed", "arm", "TM", "length", "edit distance to reference (63 aa)"], rows) + "\n")
    md.append("```\n" + "\n".join(f"seed {s} {a}: {runs[(a, s)]['summary']['final_best_genome']}" for s in seeds for a in ARMS if (a, s) in runs) + "\n```\n")

    # 6 ------------------------------------------------------------ wall time
    md.append("### 6. Wall time split (seconds; percentages of that run's wall time)\n")
    rows = []
    for s in seeds:
        for a in ARMS:
            if (a, s) not in runs:
                continue
            m = runs[(a, s)]["summary"]
            w = m["wall_s"]
            rows.append([s, ARM_NAME[a], f"{w:.0f}", f"{m['fitness_s']:.0f} ({100 * m['fitness_s'] / w:.1f}%)",
                         f"{m['llm_s']:.0f} ({100 * m['llm_s'] / w:.1f}%)", f"{m['swap_s']:.0f} ({100 * m['swap_s'] / w:.1f}%)",
                         f"{m['other_s']:.1f} ({100 * m['other_s'] / w:.2f}%)", f"{m['mean_fitness_s_per_distinct_evaluation']:.2f}"])
    md.append(md_table(["seed", "arm", "wall", "fitness (folds)", "LLM calls", "GPU swap (C only)", "everything else", "s per distinct fold"], rows) +
              "\n\nLLM time is time inside the two operator wrappers: retries and the Ollama model reload that follows each swap are included. "
              "\"Everything else\" is selection/crossover/mutation bookkeeping, diversity metrics and logging.\n")

    # 7 ------------------------------------------------------------ fallback
    md.append("### 7. Arm C: fallback rate and requests per call\n")
    rows = []
    for s in seeds:
        if ("C", s) not in runs:
            continue
        m = runs[("C", s)]["summary"]
        for op in ("mutate", "crossover"):
            o = m["operators"][op]
            rows.append([s, f"{op} ({o['style']})", o["wrapper_calls"], o["n_skipped_no_op"], o["n_llm_calls"], o["n_failures"],
                         f"{o['fallback_rate']:.3f}" if o["fallback_rate"] is not None else "—",
                         f"{o['requests_per_call']:.3f}" if o["requests_per_call"] is not None else "—", o["n_retries"]])
        rows.append([s, "both", "", "", sum(m["operators"][o]["n_llm_calls"] for o in m["operators"]),
                     sum(m["operators"][o]["n_failures"] for o in m["operators"]),
                     f"{m['fallback_rate_all_llm_calls']:.3f}" if m["fallback_rate_all_llm_calls"] is not None else "—",
                     f"{m['requests_per_call_all_llm_calls']:.3f}" if m["requests_per_call_all_llm_calls"] is not None else "—", ""])
    if rows:
        md.append(md_table(["seed", "operator", "operator calls", "gated off (crossover rate)", "LLM calls", "fell back",
                            "fallback rate", "requests / LLM call", "retries"], rows) +
                  "\n\nFallback rate = fell back / LLM calls (a crossover the 0.9 rate gate skipped never reaches the LLM and is excluded); "
                  "requests / LLM call is 1.0 when every call succeeded first time.\n")
    if check:
        md.append(f"Stage-1 gate (one generation, pop 16, seed {check.get('seed', 0)}): fallback rate by operator "
                  f"{check['fallback_rate_by_operator']}; LLM calls {{'mutate': {check['operators']['mutate']['n_llm_calls']}, "
                  f"'crossover': {check['operators']['crossover']['n_llm_calls']}}}.\n")

    # 8 ------------------------------------------------------------ comparisons
    md.append("### 8. Comparisons at equal distinct-evaluation budget\n")
    verdicts = {}
    for x, y in (("B", "A"), ("C", "A"), ("C", "B")):
        rows, diffs = [], []
        for s in seeds:
            if (x, s) in runs and (y, s) in runs:
                n = min(n_distinct(runs[(x, s)]), n_distinct(runs[(y, s)]))
                bx, by = best_at(runs[(x, s)], n), best_at(runs[(y, s)], n)
                diffs.append(bx - by)
                rows.append([s, n, f4(bx), f4(by), f"{bx - by:+.4f}", x if bx > by else (y if by > bx else "tie")])
        if not diffs:
            continue
        k = len(diffs)
        wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
        mean_d = statistics.mean(diffs)
        if wins == k and mean_d > 0:
            verdict = f"{x} BEATS {y} under the pre-set rule (higher in all {k} of {k} seeds)"
        elif losses == k and mean_d < 0:
            verdict = f"{y} BEATS {x} under the pre-set rule ({x} lower in all {k} of {k} seeds)"
        else:
            verdict = f"NO DEMONSTRATED DIFFERENCE between {x} and {y}: {x} higher in {wins}, lower in {losses}, tied in {k - wins - losses} of {k} seeds"
        if k < 3:
            verdict = f"ONLY {k} SEED(S), NOT INTERPRETABLE ({verdict})"
        verdicts[f"{x}_vs_{y}"] = {"verdict": verdict, "wins": wins, "losses": losses, "k": k, "mean_diff": mean_d,
                                   "min_diff": min(diffs), "max_diff": max(diffs), "min_p": 2 * 0.5 ** k}
        md.append(f"**{x} vs {y}**\n\n" + md_table(["seed", "budget n", f"{x} best@n", f"{y} best@n", "difference", "higher"], rows) +
                  f"\n\nMean difference {mean_d:+.4f} (range {min(diffs):+.4f} to {max(diffs):+.4f}). **{verdict}.** "
                  f"Smallest two-sided sign-test p possible with {k} seeds: {2 * 0.5 ** k:.4f}.\n")
    # 9 ------------------------------------------------------------ LLM behaviour
    md.append("### 9. What the LLM operators did inside arm C (from the per-call logs; accepted, i.e. valid, responses only)\n")
    rows = []
    for s in seeds:
        b = llm_behaviour(runs.get(("C", s)))
        if b:
            rows.append([s] + b)
    if rows:
        md.append(md_table(["seed", "mutate calls", "positions changed", "distinct position indices", "most-used index (share)",
                            "distinct new letters (of 20)", "most-used letter (share)", "crossover calls", "distinct cut values",
                            "cut values (count)"], rows) + "\n")
    facts["verdicts"] = verdicts
    return "\n".join(md), facts


def chart(runs: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = sorted({s for (_, s) in runs})
    if not seeds:
        return
    ncols = min(3, len(seeds))
    nrows = -(-len(seeds) // ncols)
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(8.0, 4.6 * ncols), 3.7 * nrows + 0.6), sharey=True, squeeze=False, facecolor=surface)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, s in enumerate(seeds):
        ax = axes.flat[i]
        ax.set_visible(True)
        ax.set_facecolor(surface)
        ends = []
        for a in ARMS:
            if (a, s) not in runs:
                continue
            y = runs[(a, s)]["best_so_far_by_distinct_evaluation"]
            ax.plot(range(1, len(y) + 1), y, drawstyle="steps-post", color=SLOT[a], lw=2.0, label=ARM_NAME[a], solid_capstyle="round")
            ax.plot([len(y)], [y[-1]], "o", color=SLOT[a], ms=6, mec=surface, mew=1.5)
            ends.append([a, len(y), y[-1], y[-1]])  # arm, x, true y, label y
        ends.sort(key=lambda e: e[2])
        gap = 0.017  # data units: keeps 8pt end labels from overprinting when final values are close
        for i in range(1, len(ends)):
            ends[i][3] = max(ends[i][3], ends[i - 1][3] + gap)
        for a, x, yv, ly in ends:
            ax.annotate(f"{a} {yv:.3f}", (x, ly), xytext=(6, 0), textcoords="offset points", color=ink2, fontsize=8, va="center")
        ax.set_title(f"seed {s}", loc="left", color=ink, fontsize=10)
        ax.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=8)
        ax.margins(x=0.14)
        ax.set_xlabel("distinct evaluations (folds)", color=ink2, fontsize=9)
        if i % ncols == 0:
            ax.set_ylabel("best-so-far TM-score vs 7UR7", color=ink2, fontsize=9)
    h, l = [], []
    for ax in axes.flat:
        if ax.get_visible():
            hh, ll = ax.get_legend_handles_labels()
            for a, b in zip(hh, ll):
                if b not in l:
                    h.append(a), l.append(b)
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, labelcolor=ink, fontsize=9, bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=150, facecolor=surface)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw"))
    ap.add_argument("--out-md", default=str(ROOT / "results" / "raw" / "sequence_ga_report_tables.md"))
    ap.add_argument("--out-png", default=str(ROOT / "results" / "sequence_ga_best_so_far.png"))
    ap.add_argument("--out-facts", default=None)
    args = ap.parse_args()
    raw = Path(args.raw)
    runs = load(raw)
    status = json.load(open(raw / "sequence_ga_comparison_status.json")) if (raw / "sequence_ga_comparison_status.json").exists() else None
    check = json.load(open(raw / "sequence_ga_llm_check.json")) if (raw / "sequence_ga_llm_check.json").exists() else None
    text, facts = sections(runs, status, check)
    Path(args.out_md).write_text(text + "\n")
    chart(runs, Path(args.out_png))
    if args.out_facts:
        Path(args.out_facts).write_text(json.dumps(facts, indent=1))
    print(f"{len(runs)} completed runs; wrote {args.out_md} and {args.out_png}")
    for k, v in facts["verdicts"].items():
        print(f"  {k}: {v['verdict']}")


if __name__ == "__main__":
    main()
