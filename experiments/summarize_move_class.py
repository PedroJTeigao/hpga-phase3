"""Tables for results/MOVE_CLASS.md from results/raw/move_class_{arm}_seed{n}.json and move_class_bases_{arm}_seed{n}.json.

Budget: per seed, n_cut = the smallest distinct-evaluation count any arm (B included) reached; every arm's best is read
off its best-so-far curve at n_cut. Move payoff: a move improves if child fitness > base fitness (base = the
post-crossover child the move was applied to); gain | improved = mean of (child - base) over improving moves. Thirds are
by breeding step: early 0-5, middle 6-11, late 12-18 (children evaluated in generations 1-6, 7-12, 13-19).

  summarize_move_class.py [--seeds 0 1 2]    writes results/MOVE_CLASS_TABLES.md and results/move_class_summary.json
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
ARMS = ("S1", "S2", "S3", "S4", "B")
THIRDS = {"early (steps 0-5)": range(0, 6), "middle (steps 6-11)": range(6, 12), "late (steps 12-18)": range(12, 19)}


def load(arm, seed):
    return (json.load(open(RAW / f"move_class_{arm}_seed{seed}.json")),
            json.load(open(RAW / f"move_class_bases_{arm}_seed{seed}.json")))


def payoff(moves):
    n = len(moves)
    imp = [m["gain"] for m in moves if m["improved"]]
    return {"n": n, "share_improved": len(imp) / n if n else None, "mean_gain_if_improved": st.mean(imp) if imp else None,
            "share_equal": sum(m["gain"] == 0 for m in moves) / n if n else None,
            "share_worse": sum(m["gain"] < 0 for m in moves) / n if n else None,
            "mean_gain_all": st.mean(m["gain"] for m in moves) if n else None,
            "expected_gain_per_move_positive_part": sum(imp) / n if n else None,
            "n_no_op": sum(m["no_op"] for m in moves)}


def f(x, d=4):
    return "–" if x is None else f"{x:.{d}f}"


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


COLORS = {"S1": "#2a78d6", "S2": "#eb6834", "S3": "#1baf7a", "S4": "#eda100", "B": "#8a8984"}
LABELS = {"S1": "S1 single sub", "S2": "S2 2-5 subs", "S3": "S3 segment", "S4": "S4 indel", "B": "B per-site p=0.05 (ref.)"}


def plot(traj: dict, n_cut: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    n = len(next(iter(traj.values())))
    xs = range(1, n + 1)
    ends = sorted(((traj[a][-1], a) for a in traj), reverse=True)
    label_y, last = {}, None
    for y, a in ends:  # direct labels, nudged apart so they never collide
        y = y if last is None or last - y >= 0.008 else last - 0.008
        label_y[a], last = y, y
    for a in ("B", "S1", "S2", "S3", "S4"):
        ax.step(xs, traj[a], where="post", color=COLORS[a], lw=2, ls="--" if a == "B" else "-", label=LABELS[a])
        ax.text(n + 3, label_y[a], f"{LABELS[a]}  {traj[a][-1]:.3f}", color="#52514e", fontsize=8, va="center")
    ax.set_xlim(0, n + 75)
    ax.set_xlabel("distinct evaluations (ESMFold folds)", color="#52514e")
    ax.set_ylabel("best-so-far TM-score, mean over seeds", color="#52514e")
    ax.set_title(f"Best-so-far vs evaluations spent, mean of seeds {', '.join(map(str, n_cut))} "
                 f"(cut at the smallest common count, {n})", fontsize=9, color="#0b0b0b", loc="left")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ad")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    seeds = args.seeds
    runs = {(a, s): load(a, s) for a in ARMS for s in seeds}
    out, L = {}, []

    # --- 1. best at common budget
    n_cut = {s: min(runs[a, s][0]["summary"]["n_distinct_evaluations"] for a in ARMS) for s in seeds}
    best = {(a, s): runs[a, s][0]["best_so_far_by_distinct_evaluation"][n_cut[s] - 1] for a in ARMS for s in seeds}
    out["n_cut"], out["best_at_n_cut"] = n_cut, {f"{a}_seed{s}": v for (a, s), v in best.items()}
    L += ["## 1. Best fitness at the common evaluation count", "",
          "n_cut per seed = the smallest distinct-fold count any of the five arms reached: "
          + ", ".join(f"seed {s}: {n_cut[s]}" for s in seeds) + ".", "",
          "| arm | " + " | ".join(f"seed {s}" for s in seeds) + " | mean | distinct folds at end of run |",
          "|---|" + "---|" * (len(seeds) + 2)]
    for a in ARMS:
        ends = ", ".join(str(runs[a, s][0]["summary"]["n_distinct_evaluations"]) for s in seeds)
        L.append(f"| {a} | " + " | ".join(f(best[a, s]) for s in seeds) + f" | {f(st.mean(best[a, s] for s in seeds))} | {ends} |")
    L += ["", "Pairwise, higher in every seed (row beats column = ✓):", "",
          "| | " + " | ".join(ARMS) + " |", "|---|" + "---|" * len(ARMS)]
    beats = {}
    for a in ARMS:
        row = []
        for b in ARMS:
            w = a != b and all(best[a, s] > best[b, s] for s in seeds)
            beats[f"{a}>{b}"] = w
            row.append("—" if a == b else ("✓" if w else ""))
        L.append(f"| {a} | " + " | ".join(row) + " |")
    out["beats_every_seed"] = beats

    # --- 2. trajectory
    n_min = min(n_cut.values())
    marks = sorted(set(list(range(16, n_min + 1, 16)) + [n_min]))
    traj = {a: [st.mean(runs[a, s][0]["best_so_far_by_distinct_evaluation"][n - 1] for s in seeds) for n in marks] for a in ARMS}
    out["trajectory"] = {"evaluations": marks, **traj}
    out["trajectory_full"] = {a: [st.mean(runs[a, s][0]["best_so_far_by_distinct_evaluation"][n] for s in seeds)
                                  for n in range(n_min)] for a in ARMS}
    L += ["", "## 2. Best-so-far vs distinct evaluations, mean over seeds", "",
          "| evaluations | " + " | ".join(ARMS) + " |", "|---|" + "---|" * len(ARMS)]
    for i, n in enumerate(marks):
        L.append(f"| {n} | " + " | ".join(f(traj[a][i]) for a in ARMS) + " |")

    # --- 3. payoff per arm
    moves = {(a, s): runs[a, s][1]["moves"] for a in ARMS for s in seeds}
    L += ["", "## 3. Move payoff per arm (all 19 breeding steps; every child the fill produced is one move)", "",
          "| arm | seed | moves | improved | mean gain if improved | equal | worse | mean gain, all moves | no-op moves |",
          "|---|---|---|---|---|---|---|---|---|"]
    out["payoff"] = {}
    for a in ARMS:
        for s in list(seeds) + ["pooled"]:
            ms = sum((moves[a, t] for t in seeds), []) if s == "pooled" else moves[a, s]
            p = payoff(ms)
            out["payoff"][f"{a}_{s}"] = p
            L.append(f"| {a} | {s} | {p['n']} | {pct(p['share_improved'])} | {f(p['mean_gain_if_improved'])} | "
                     f"{pct(p['share_equal'])} | {pct(p['share_worse'])} | {f(p['mean_gain_all'])} | {p['n_no_op']} |")
    L += ["", "Per-seed direction (higher in every seed):", ""]
    for key in ("share_improved", "mean_gain_if_improved"):
        wins = [f"{a}>{b}" for a in ARMS for b in ARMS if a != b
                and all((out["payoff"][f"{a}_{s}"][key] or 0) > (out["payoff"][f"{b}_{s}"][key] or 0) for s in seeds)]
        L.append(f"- {key}: " + (", ".join(wins) if wins else "no pair ordered in every seed"))
        out[f"payoff_beats_every_seed_{key}"] = wins

    # --- 4. thirds
    L += ["", "## 4. Move payoff by generation thirds (pooled over seeds; per-seed in move_class_summary.json)", "",
          "| arm | " + " | ".join(f"{t}: improved / gain if improved" for t in THIRDS) + " |", "|---|" + "---|" * len(THIRDS)]
    out["payoff_thirds"] = {}
    for a in ARMS:
        cells = []
        for t, rng in THIRDS.items():
            for s in list(seeds) + ["pooled"]:
                src = sum((moves[a, u] for u in seeds), []) if s == "pooled" else moves[a, s]
                out["payoff_thirds"][f"{a}_{s}_{t.split()[0]}"] = payoff([m for m in src if m["step"] in rng])
            p = out["payoff_thirds"][f"{a}_pooled_{t.split()[0]}"]
            cells.append(f"{pct(p['share_improved'])} / {f(p['mean_gain_if_improved'])}")
        L.append(f"| {a} | " + " | ".join(cells) + " |")

    # --- 5. S4 lengths
    L += ["", "## 5. S4 genome length per generation (all evaluated genomes, pooled over seeds)", "",
          "| generation | min | q1 | median | q3 | max | mean | per-seed mean |", "|---|---|---|---|---|---|---|---|"]
    out["s4_lengths"] = []
    for g in range(len(runs["S4", seeds[0]][0]["generations"])):
        per = {s: runs["S4", s][0]["generations"][g]["lengths"] for s in seeds}
        allv = sorted(sum(per.values(), []))
        q = st.quantiles(allv, n=4)
        row = {"generation": g, "min": allv[0], "q1": q[0], "median": st.median(allv), "q3": q[2], "max": allv[-1],
               "mean": st.mean(allv), "per_seed_mean": {s: st.mean(v) for s, v in per.items()}}
        out["s4_lengths"].append(row)
        L.append(f"| {g} | {row['min']} | {q[0]:.1f} | {row['median']:.1f} | {q[2]:.1f} | {row['max']} | {row['mean']:.1f} | "
                 + ", ".join(f"{v:.1f}" for v in row["per_seed_mean"].values()) + " |")
    ops = [m for s in seeds for m in moves["S4", s]]
    L.append("")
    for kind in ("del", "ins"):
        p = payoff([m for m in ops if m["kind"] == kind])
        out[f"s4_payoff_{kind}"] = p
        L.append(f"- S4 {kind}: {p['n']} moves, improved {pct(p['share_improved'])}, mean gain if improved {f(p['mean_gain_if_improved'])}")
    L.append(f"- S4 draws redrawn at a length bound: {sum(m['n_redraws'] > 0 for m in ops)} of {len(ops)} moves")

    # --- measurement cost and verification
    L += ["", "## Measurement cost and verification", "",
          "| arm | seed | measurement folds | GA folds (distinct) | measurement fold s | GA fold s | measurement share of fold time | inert (digest + replay) | reproduces sequence_ga_cmp_B |",
          "|---|---|---|---|---|---|---|---|---|"]
    out["measurement"] = {}
    for a in ARMS:
        for s in seeds:
            r, b = runs[a, s]
            inert = r["summary"]["inertness"]
            rm = r["summary"]["reference_match"]
            out["measurement"][f"{a}_{s}"] = {k: b[k] for k in ("n_measurement_folds", "measurement_fold_s", "ga_fold_s",
                                                                 "measurement_share_of_fold_time")}
            L.append(f"| {a} | {s} | {b['n_measurement_folds']} | {r['summary']['n_distinct_evaluations']} | "
                     f"{b['measurement_fold_s']:.0f} | {b['ga_fold_s']:.0f} | {pct(b['measurement_share_of_fold_time'])} | "
                     f"{inert['digest_after_measurement_identical'] and inert['replay_without_measurement_identical']} | "
                     f"{'–' if rm is None else rm['identical']} |")

    plot(out["trajectory_full"], n_cut, ROOT / "results" / "move_class.png")
    (ROOT / "results" / "MOVE_CLASS_TABLES.md").write_text(
        "# Move class: generated tables\n\nGenerated by experiments/summarize_move_class.py from results/raw/move_class_*.json. "
        "Read with results/MOVE_CLASS.md.\n\n" + "\n".join(L) + "\n")
    json.dump(out, open(ROOT / "results" / "move_class_summary.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
