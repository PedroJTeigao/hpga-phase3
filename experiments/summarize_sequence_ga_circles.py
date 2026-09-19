"""Section 9 of results/SEQUENCE_GA_REPORT.md: arms D (GA + LLM + circles) and E (GA + LLM + random
immigrants) against A random search, B GA, C GA + LLM -- generated only from the raw result files.

  python experiments/summarize_sequence_ga_circles.py [--raw DIR] [--out-md FILE] [--out-png FILE] [--out-facts FILE]

BUDGET AXIS. Circles and immigrants add folds, so every comparison here is made at the SAME distinct-fold
count per seed: n_cut(seed) = the smallest distinct-evaluation count reached by any of A-E for that seed, each
run's best-so-far read at that count (its curve), with full curves shown too.
RULE (unchanged from the first comparison): X beats Y only if X is strictly higher in EVERY seed compared, at
n_cut. With k seeds the smallest two-sided sign-test p possible is 2 * 0.5**k.
Per-seed tables are never pooled; the one cross-seed table is labelled.
"""

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import summarize_sequence_ga_comparison as s1  # same directory; md_table, f4, best_at, n_distinct, ROOT

ROOT = s1.ROOT
ARMS = ("A", "B", "C", "D", "E")
NAME = {"A": "A random search", "B": "B GA (deterministic ops)", "C": "C GA (LLM ops)",
        "D": "D GA + LLM + circles", "E": "E GA + LLM + random immigrants"}
COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100", "E": "#e87ba4"}  # validated categorical slots 1-5
N_SLOTS = 4


def load(raw: Path) -> dict:
    runs = {}
    for p in sorted(raw.glob("sequence_ga_cmp_[A-E]_seed*.json")):
        d = json.load(open(p))
        if d.get("complete"):
            runs[(d["arm"], d["seed"])] = d
    return runs


def n_cut(runs, seed):
    return min(s1.n_distinct(runs[(a, seed)]) for a in ARMS)


def best_genome_at(run, n):
    """The genome whose fitness equals the best-so-far value after n distinct evaluations (fitnesses are
    deterministic and effectively unique per genome, so this identifies it)."""
    target = s1.best_at(run, n)
    if run["arm"] == "A":  # random search stores no populations; its draws are deterministic (Random(seed), one genome per
        import random, sys  # distinct evaluation, no duplicates), so regenerate them and take the first index that reaches the value
        sys.path.insert(0, str(ROOT))
        from hpga import sequence_model as sm
        idx = next(i for i, v in enumerate(run["best_so_far_by_distinct_evaluation"]) if v == target)
        rng = random.Random(run["seed"])
        draws = [sm.random_sequence(rng, None) for _ in range(idx + 1)]
        assert len(set(draws)) == len(draws), "duplicate draw: the index would not be the distinct-evaluation index"
        return draws[idx]
    for g in run["generations"]:
        for genome, f in zip(g["population"], g["fitnesses"]):
            if f == target:
                return genome
    return None


def estimate_reload_s(run) -> float | None:
    """For arm C only (no separate model-load accounting): the excess latency of the first request after each
    fold phase over that operator's median latency, from the call log. An ESTIMATE."""
    path = Path(run.get("llm_call_log", ""))
    path = path if path.is_absolute() else ROOT / path
    if not path.exists():
        return None
    from datetime import datetime
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    recs = [r for r in recs if "latency_s" in r and "timestamp" in r]
    med = {}
    for op in {r["op"] for r in recs}:
        med[op] = statistics.median(r["latency_s"] for r in recs if r["op"] == op)
    total, prev_end = 0.0, None
    for r in recs:
        start = datetime.fromisoformat(r["timestamp"]).timestamp() - r["latency_s"]
        if prev_end is None or start - prev_end > 15:
            total += max(0.0, r["latency_s"] - med[r["op"]])
        prev_end = datetime.fromisoformat(r["timestamp"]).timestamp()
    return total


def sections(runs, status):
    seeds = sorted({s for (_, s) in runs if all((a, s) in runs for a in ARMS)})
    partial = sorted({s for (_, s) in runs} - set(seeds))
    md, facts = [], {"seeds": seeds}
    cut = {s: n_cut(runs, s) for s in seeds}
    facts["n_cut"] = cut

    # 9.1 runs
    rows = []
    for s in sorted({s for (_, s) in runs}):
        for a in ("D", "E"):
            r = runs.get((a, s))
            if r:
                m = r["summary"]
                rows.append([s, NAME[a], m["n_distinct_evaluations"], m["n_cache_hits"], m["n_population_evaluations"], f"{s1.f4(m['final_best'])}", r.get("attempt", 1)])
            else:
                rows.append([s, NAME[a], "not completed", "", "", "", ""])
    md.append("### 9.1 Runs and the common budget\n\n" + s1.md_table(
        ["seed", "arm", "distinct folds", "cache hits", "genomes evaluated (incl. repeats)", "final best TM (own budget)", "attempt"], rows))
    if status and status.get("failed"):
        md.append("\nFailed after one retry: " + "; ".join(f"{f['arm']} seed {f['seed']} ({f['error']})" for f in status["failed"]))
    if partial:
        md.append(f"\nSeeds with an incomplete set of arms are excluded from the comparisons below: {partial}.")
    md.append("\nDistinct folds reached by each arm, and the common cut n_cut (the smallest of the five):\n\n" + s1.md_table(
        ["seed"] + [NAME[a].split(" ")[0] for a in ARMS] + ["n_cut"],
        [[s] + [s1.n_distinct(runs[(a, s)]) for a in ARMS] + [cut[s]] for s in seeds]) + "\n")

    # 9.2 final best at the common budget
    md.append("### 9.2 Best TM-score at the common distinct-fold count (per seed), and over seeds\n")
    rows = [[NAME[a]] + [s1.f4(s1.best_at(runs[(a, s)], cut[s])) for s in seeds] for a in ARMS]
    md.append(s1.md_table(["arm (read at n_cut)"] + [f"seed {s} (n={cut[s]})" for s in seeds], rows) + "\n")
    rows = [[NAME[a]] + [s1.f4(runs[(a, s)]["summary"]["final_best"]) for s in seeds] for a in ARMS]
    md.append("Each run's own final best over its full budget (D and E ran further than the cut; A, B, C as before):\n\n" +
              s1.md_table(["arm (full run)"] + [f"seed {s}" for s in seeds], rows) + "\n")
    agg = []
    for a in ARMS:
        v = [s1.best_at(runs[(a, s)], cut[s]) for s in seeds]
        agg.append([NAME[a], len(v), s1.f4(statistics.mean(v)), f"{s1.f4(min(v))} - {s1.f4(max(v))}"])
    md.append("Cross-seed summary at the common cut (the only pooled table; n = complete seeds):\n\n" +
              s1.md_table(["arm", "n seeds", "mean best-so-far at n_cut", "range (min - max)"], agg) + "\n")
    facts["mean_at_cut"] = {a: statistics.mean(s1.best_at(runs[(a, s)], cut[s]) for s in seeds) for a in ARMS}

    # 9.3 diversity
    md.append("### 9.3 Mean pairwise edit distance within the evaluated population, per generation (per seed)\n")
    md.append("The evaluated population is 16 genomes at generation 0 and 20 afterwards for D and E (their 4 injected genomes included), 16 for B and C.\n")
    for s in seeds:
        ngen = max(len(runs[(a, s)]["generations"]) for a in "BCDE")
        rows = [[g] + [f"{runs[(a, s)]['generations'][g]['mean_pairwise_edit_distance']:.1f}" if g < len(runs[(a, s)]["generations"]) else "-" for a in "BCDE"] for g in range(ngen)]
        md.append(f"**Seed {s}**\n\n" + s1.md_table(["gen", "B", "C", "D", "E"], rows) + "\n")

    # 9.4 comparisons
    md.append("### 9.4 Comparisons at the common budget\n")
    verdicts = {}
    for x, y in (("D", "C"), ("D", "E"), ("E", "C")):
        rows, diffs = [], []
        for s in seeds:
            bx, by = s1.best_at(runs[(x, s)], cut[s]), s1.best_at(runs[(y, s)], cut[s])
            diffs.append(bx - by)
            rows.append([s, cut[s], s1.f4(bx), s1.f4(by), f"{bx - by:+.4f}", x if bx > by else (y if by > bx else "tie")])
        k = len(diffs)
        wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
        mean_d = statistics.mean(diffs)
        if k < 3:
            verdict = f"ONLY {k} SEED(S), NOT INTERPRETABLE"
        elif wins == k and mean_d > 0:
            verdict = f"{x} BEATS {y} under the pre-set rule (higher in all {k} of {k} seeds)"
        elif losses == k and mean_d < 0:
            verdict = f"{y} BEATS {x} under the pre-set rule ({x} lower in all {k} of {k} seeds)"
        else:
            verdict = f"NO DEMONSTRATED DIFFERENCE between {x} and {y}: {x} higher in {wins}, lower in {losses}, tied in {k - wins - losses} of {k} seeds"
        verdicts[f"{x}_vs_{y}"] = {"verdict": verdict, "wins": wins, "losses": losses, "k": k, "mean_diff": mean_d,
                                   "min_diff": min(diffs), "max_diff": max(diffs), "min_p": 2 * 0.5 ** k}
        md.append(f"**{x} vs {y}**\n\n" + s1.md_table(["seed", "n_cut", f"{x} best@n_cut", f"{y} best@n_cut", "difference", "higher"], rows) +
                  f"\n\nMean difference {mean_d:+.4f} (range {min(diffs):+.4f} to {max(diffs):+.4f}). **{verdict}.** "
                  f"Smallest two-sided sign-test p possible with {k} seeds: {2 * 0.5 ** k:.4f}.\n")
    facts["verdicts"] = verdicts

    # 9.5 does extra diversity convert into fitness
    md.append("### 9.5 Does D's extra diversity, if any, convert into fitness?\n")
    rows, div = [], {}
    for s in seeds:
        def dmean(a, lo=1):
            g = runs[(a, s)]["generations"]
            return statistics.mean(x["mean_pairwise_edit_distance"] for x in g[lo:])
        def dlast(a):
            return runs[(a, s)]["generations"][-1]["mean_pairwise_edit_distance"]
        row = {"D-C mean": dmean("D") - dmean("C"), "D-E mean": dmean("D") - dmean("E"), "D-C last": dlast("D") - dlast("C"), "D-E last": dlast("D") - dlast("E"),
               "fit D-C": s1.best_at(runs[("D", s)], cut[s]) - s1.best_at(runs[("C", s)], cut[s]),
               "fit D-E": s1.best_at(runs[("D", s)], cut[s]) - s1.best_at(runs[("E", s)], cut[s])}
        div[s] = row
        rows.append([s] + [f"{row[k]:+.1f}" for k in ("D-C mean", "D-C last", "D-E mean", "D-E last")] + [f"{row['fit D-C']:+.4f}", f"{row['fit D-E']:+.4f}"])
    md.append(s1.md_table(["seed", "diversity D-C (mean gens 1-19)", "D-C (gen 19)", "diversity D-E (mean gens 1-19)", "D-E (gen 19)",
                           "best TM D-C @n_cut", "best TM D-E @n_cut"], rows) +
              "\n\nDiversity differences are in edit-distance units (positive = D more diverse); fitness differences in TM-score at n_cut (positive = D higher). "
              "Population diversity is over the evaluated population (16-20 genomes).\n")
    facts["diversity"] = div

    # 9.6 what the injected genomes were worth
    md.append("### 9.6 The injected genomes themselves (D: circle proposals; E: random immigrants)\n")
    rows = []
    for s in seeds:
        line = [s]
        for a in ("D", "E"):
            gens = runs[(a, s)]["generations"][1:]
            inj = [f for g in gens for f in g["fitnesses"][-N_SLOTS:]]
            rest = [f for g in gens for f in g["fitnesses"][:-N_SLOTS]]
            line += [f"{statistics.mean(inj):.4f}", f"{max(inj):.4f}", f"{statistics.mean(rest):.4f}"]
        rows.append(line)
    md.append(s1.md_table(["seed", "D injected: mean TM", "D injected: best", "D rest of population: mean", "E injected: mean TM", "E injected: best",
                           "E rest of population: mean"], rows) +
              "\n\nMeans over generations 1-19 of the last 4 members of each evaluated population (the injected ones) and of the other members.\n")

    # 9.7 LLM use and wall time
    md.append("### 9.7 LLM calls, tokens and wall time (seconds; percentages of that run's wall time)\n")
    rows = []
    for s in seeds:
        for a in ("C", "D", "E"):
            m = runs[(a, s)]["summary"]
            w = m["wall_s"]
            tin = sum(t["total_tokens_in"] for t in m["operators"].values())
            tout = sum(t["total_tokens_out"] for t in m["operators"].values())
            ncall = sum(t["n_llm_calls"] for t in m["operators"].values())
            nreq = sum(t["n_llm_requests"] for t in m["operators"].values())
            model_load = f"{m['ollama_load_s']:.0f}" if "ollama_load_s" in m else "in LLM time"
            rows.append([s, NAME[a], ncall, nreq, f"{tin:,}", f"{tout:,}", f"{w:.0f}", f"{m['fitness_s']:.0f} ({100 * m['fitness_s'] / w:.1f}%)",
                         f"{m['llm_s']:.0f} ({100 * m['llm_s'] / w:.1f}%)", f"{m['swap_s']:.0f} ({100 * m['swap_s'] / w:.1f}%)", model_load,
                         f"{m['other_s']:.1f} ({100 * m['other_s'] / w:.2f}%)"])
    md.append(s1.md_table(["seed", "arm", "LLM calls", "requests", "tokens in", "tokens out", "wall", "fitness (folds)", "LLM calls time",
                           "swap between models", "  of which Ollama load", "everything else"], rows) +
              "\n\nSwap = unloading Ollama, moving ESMFold between CPU and GPU, and (D, E) loading the Ollama model, timed separately from the LLM calls. "
              "**Arm C was run with different accounting:** its swap time excludes the Ollama reload, which fell inside the first LLM call of each breeding step and is "
              "therefore in its LLM time. Estimate of that reload for C (excess latency of the first request after each fold phase over the operator median, from its call log): "
              + ", ".join(f"seed {s}: {est:.0f} s" for s in seeds if (est := estimate_reload_s(runs[('C', s)])) is not None) + ". D and E carry the measured load in swap time.\n")
    dseeds = sorted(s for (a, s) in runs if a == "D")  # every completed D run, including a seed whose control (E) did not finish
    tab = []
    for s in dseeds:
        m = runs[("D", s)]["summary"]["operators"]
        for op in ("mutate", "crossover", "propose", "observe", "consult", "central_directive", "curate"):
            if op in m:
                t = m[op]
                tab.append([s, op, t["calls"], t["n_llm_calls"], t["n_failures"], f"{t['fallback_rate']:.3f}" if t["fallback_rate"] is not None else "-",
                            f"{t['requests_per_call']:.3f}" if t["requests_per_call"] is not None else "-", f"{t['wall_s']:.0f}"])
    md.append("### 9.8 Arm D, per operator: calls, fallbacks, requests per call\n\n" + s1.md_table(
        ["seed", "operator", "operator calls", "LLM calls", "fell back", "fallback rate", "requests / call", "wall s"], tab) +
        "\n\n**Circle proposal fallback rate** (`propose`, position edits to a population member) is the `propose` rows; the stage-1 probe gave 1 of 50. "
        "`observe`, `consult`, `central_directive` and `curate` fall back to nothing (no observation written, empty note, deterministic directive, board unchanged).\n")
    tot = {op: [0, 0] for op in ("propose", "observe", "consult", "central_directive", "curate", "mutate", "crossover")}
    for s in dseeds:
        for op, t in runs[("D", s)]["summary"]["operators"].items():
            if op in tot:
                tot[op][0] += t["n_llm_calls"]; tot[op][1] += t["n_failures"]
    facts["D_fallbacks"] = {op: {"llm_calls": v[0], "fell_back": v[1], "rate": (v[1] / v[0]) if v[0] else None} for op, v in tot.items()}
    md.append(f"Over all {len(dseeds)} completed D runs (seeds {dseeds}): " + "; ".join(f"{op} {v[1]}/{v[0]}" for op, v in tot.items() if v[0]) + " (fell back / LLM calls).\n")
    bbrows = [[s, *(runs[("D", s)]["summary"]["blackboard"][k] for k in ("entries_written", "live_at_end")),
               *(runs[("D", s)]["summary"]["blackboard"]["by_type"][t] for t in ("observation", "directive", "summary"))] for s in dseeds]
    md.append("Blackboard at the end of each D run:\n\n" + s1.md_table(["seed", "entries written", "live at end", "observations", "directives", "summaries (curation)"], bbrows) + "\n")

    # 9.9 edit distance to reference
    md.append("### 9.9 Edit distance of the best genomes to the 63-residue reference sequence\n")
    rows = []
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from hpga import sequence_model as sm
    from esmfold import tm_fitness as tf
    ref = tf.reference_sequence()
    for s in seeds:
        for a in ARMS:
            r = runs[(a, s)]
            g_cut = best_genome_at(r, cut[s])
            rows.append([s, NAME[a], s1.f4(s1.best_at(r, cut[s])), sm.edit_distance(g_cut, ref) if g_cut else "-", len(g_cut) if g_cut else "-",
                         s1.f4(r["summary"]["final_best"]), r["summary"]["final_best_edit_distance_to_reference"], r["summary"]["final_best_length"]])
    md.append(s1.md_table(["seed", "arm", "best TM @n_cut", "edit dist to reference", "length", "final best TM (own budget)", "edit dist", "length"], rows) + "\n")
    facts["edit_to_ref_final"] = {a: [runs[(a, s)]["summary"]["final_best_edit_distance_to_reference"] for s in seeds] for a in ARMS}

    # 9.10 what circles did
    md.append("### 9.10 What the circle proposals did (from the D call logs; accepted responses only)\n")
    rows = []
    for s in seeds:
        run = runs[("D", s)]
        path = Path(run["llm_call_log"]); path = path if path.is_absolute() else ROOT / path
        pos, let, n = Counter(), Counter(), 0
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("op") == "propose" and r.get("valid") is True:
                n += 1
                for a_, b_ in re.findall(r"POSITION\s*:\s*(\d+)\s*,\s*NEW\s*:\s*([A-Za-z])", r["response"], re.I):
                    pos[int(a_)] += 1; let[b_.upper()] += 1
        tp = pos.most_common(1)[0] if pos else (None, 0)
        tl = let.most_common(1)[0] if let else (None, 0)
        improved = tot_n = 0
        gens = run["generations"]
        for g in range(2, len(gens)):
            for slot in range(N_SLOTS):
                tot_n += 1
                improved += gens[g]["fitnesses"][-N_SLOTS + slot] > gens[g - 1]["fitnesses"][-N_SLOTS + slot]
        rows.append([s, n, sum(pos.values()), len(pos), f"{tp[0]} ({tp[1] / max(1, sum(pos.values())):.0%})", len(let), f"{tl[0]} ({tl[1] / max(1, sum(let.values())):.0%})",
                     f"{improved}/{tot_n}"])
    md.append(s1.md_table(["seed", "accepted proposals", "positions changed", "distinct position indices", "most-used index (share)", "distinct new letters",
                           "most-used letter (share)", "slot proposals that beat their base (gens 2-19)"], rows) + "\n")
    return "\n".join(md), facts


def chart(runs, seeds, cut, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncols = min(3, len(seeds))
    nrows = -(-len(seeds) // ncols)
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(9.0, 4.9 * ncols), 3.9 * nrows + 0.7), sharey=True, squeeze=False, facecolor=surface)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, s in enumerate(seeds):
        ax = axes.flat[i]
        ax.set_visible(True); ax.set_facecolor(surface)
        ends = []
        for a in ARMS:
            y = runs[(a, s)]["best_so_far_by_distinct_evaluation"]
            ax.plot(range(1, len(y) + 1), y, drawstyle="steps-post", color=COLOR[a], lw=2.0, ls="--" if a == "E" else "-", label=NAME[a], solid_capstyle="round")
            ends.append([a, len(y), y[-1], y[-1]])
        ends.sort(key=lambda e: e[2])
        for j in range(1, len(ends)):
            ends[j][3] = max(ends[j][3], ends[j - 1][3] + 0.017)
        for a, x, yv, ly in ends:
            ax.plot([x], [yv], "o", color=COLOR[a], ms=5, mec=surface, mew=1.2)
            ax.annotate(f"{a} {yv:.3f}", (x, ly), xytext=(6, 0), textcoords="offset points", color=ink2, fontsize=8, va="center")
        ax.axvline(cut[s], color=ink2, lw=0.9, ls=":")
        ax.set_title(f"seed {s}   (dotted: common cut n={cut[s]})", loc="left", color=ink, fontsize=9.5)
        ax.grid(True, color=grid, lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=8); ax.margins(x=0.16)
        ax.set_xlabel("distinct evaluations (folds)", color=ink2, fontsize=9)
        if i % ncols == 0:
            ax.set_ylabel("best-so-far TM-score vs 7UR7", color=ink2, fontsize=9)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, labelcolor=ink, fontsize=8.5, bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=150, facecolor=surface)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw"))
    ap.add_argument("--out-md", default=str(ROOT / "results" / "raw" / "sequence_ga_circles_report_tables.md"))
    ap.add_argument("--out-png", default=str(ROOT / "results" / "sequence_ga_circles_best_so_far.png"))
    ap.add_argument("--out-facts", default=None)
    args = ap.parse_args()
    raw = Path(args.raw)
    runs = load(raw)
    sp = raw / "sequence_ga_circles_status.json"
    status = json.load(open(sp)) if sp.exists() else None
    text, facts = sections(runs, status)
    Path(args.out_md).write_text(text + "\n")
    chart(runs, facts["seeds"], facts["n_cut"], Path(args.out_png))
    if args.out_facts:
        Path(args.out_facts).write_text(json.dumps(facts, indent=1))
    print(f"{len(runs)} completed runs, {len(facts['seeds'])} complete seeds; wrote {args.out_md} and {args.out_png}")
    for k, v in facts["verdicts"].items():
        print(f"  {k}: {v['verdict']}")


if __name__ == "__main__":
    main()
