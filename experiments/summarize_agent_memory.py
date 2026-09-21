"""Tables and chart for the agent-memory GA runs (run_agent_memory_ga.py). Everything in the output is computed from
results/raw/agent_memory_M{1,2,3}_seed{s}.json; nothing is typed by hand.

  python experiments/summarize_agent_memory.py [--raw DIR] [--out-md results/AGENT_MEMORY_STAGE3_TABLES.md]
                                               [--out-png results/agent_memory_stage3.png] [--seeds 0 1 2]

Definitions, fixed before any result was read:
  budget        each seed is read at n_cut = the smallest distinct-fold count reached by any of its arms; "best at the
                common evaluation count" is best-so-far after n_cut distinct folds. X beats Y only if higher in EVERY seed
                compared (with s seeds the smallest two-sided sign-test p is 2 / 2^s).
  proposal      an agent's edit at breeding step g (g = 0 .. G-2). Its base is the agent's tail slot at g (unevaluated at
                g = 0, so g = 0 proposals have no base fitness and are left out of the improvement rate). Its fitness is
                the fitness of population[pop_size + i] at generation g + 1, read from the generation records (so the
                last breeding step is included, which the module's own record log cannot resolve).
  improved      fitness after > fitness of the base (strict); a fallback (no valid edit) is not a proposal and is excluded.
  improvement   share of an arm's valid proposals that improved, per generation and per third of the run (gens 1-6, 7-12,
                13-18 for G = 20), pooled over the agents and seeds present; "trend" is the Spearman correlation between
                generation and improved over the pooled proposals. Descriptions, not tests.
  divergence    distance between the two agents' proposal distributions: total variation (TV) between the histograms of
                the positions each agent has edited (cumulative from generation 0; and over a trailing window of 6
                generations). Two agents drawing from ONE distribution still show a TV > 0 at these sample sizes (3 positions
                per proposal), so every TV is reported next to its NULL: the mean TV of two multinomial samples of the same
                sizes drawn from the pooled histogram (2000 draws). EXCESS = observed - null. Flat, near zero excess means
                the agents stayed identical; a rising excess means they came apart.
  record use    per edited position of a proposal from generation 2 on: was that position in the agent's own last-8
                record, and was its latest entry better or worse? Computed for M1 as well, whose record exists but is
                never shown, so the M1 rate is what the same agent does without seeing it.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARMS = ("M1", "M2", "M3")
NAME = {"M1": "M1 agents, no memory", "M2": "M2 agents, private record", "M3": "M3 random immigrants"}
COLOR = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a"}  # slots 1-3 of the reference palette, validated all-pairs
WINDOW, SUPPORT, N_NULL, SHOWN = 6, 80, 2000, 8


def load(raw: Path, seeds) -> dict:
    runs = {}
    for arm in ARMS:
        for s in seeds:
            p = raw / f"agent_memory_{arm}_seed{s}.json"
            if p.exists():
                r = json.load(open(p))
                if r.get("complete"):
                    runs[(arm, s)] = r
    return runs


def pop_size(run) -> int:
    return run["config"]["pop_size"]


def n_cut(runs, seed):
    ls = [len(runs[(a, seed)]["best_so_far_by_distinct_evaluation"]) for a in ARMS if (a, seed) in runs]
    return min(ls)


def best_at(run, n):
    return run["best_so_far_by_distinct_evaluation"][n - 1]


def proposals(run) -> list[dict]:
    """Every agent proposal with its base fitness and its measured fitness (None where unknown)."""
    ev = run["summary"].get("agent_events") or []
    gens, pop = run["generations"], pop_size(run)
    out = []
    for e in ev:
        if e["event"] != "proposal":
            continue
        g, i = e["generation"], e["agent_id"]
        after = None
        if g + 1 < len(gens) and len(gens[g + 1]["population"]) > pop + i and gens[g + 1]["population"][pop + i] == e["proposal"]:
            after = gens[g + 1]["fitnesses"][pop + i]
        valid = (e["base_fitness"] is not None) and (not e["fallback"]) and after is not None
        out.append({"agent": i, "gen": g, "base_fitness": e["base_fitness"], "after": after, "fallback": e["fallback"], "valid": valid,
                    "improved": bool(valid and after > e["base_fitness"]), "changes": e["changes"],
                    "positions": [c[0] for c in e["changes"]], "letters": [c[2] for c in e["changes"]]})
    return out


def resolved(run) -> list[dict]:
    return [e for e in (run["summary"].get("agent_events") or []) if e["event"] == "resolved"]


def thirds(g_last: int) -> list[tuple[int, int]]:
    n = g_last  # proposals with a base fitness are gens 1..g_last
    k = -(-n // 3)
    return [(1 + j * k, min(n, (j + 1) * k)) for j in range(3)]


def spearman(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None

    def avg_rank(v):
        order = np.argsort(v, kind="stable")
        ranks = np.empty(len(v))
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2 + 1
            i = j + 1
        return ranks
    rx, ry = avg_rank(x), avg_rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def tv_hist(a: Counter, b: Counter) -> float:
    va, vb = np.array([a.get(p, 0) for p in range(SUPPORT)], float), np.array([b.get(p, 0) for p in range(SUPPORT)], float)
    if va.sum() == 0 or vb.sum() == 0:
        return float("nan")
    return 0.5 * float(np.abs(va / va.sum() - vb / vb.sum()).sum())


def null_tv(a: Counter, b: Counter, rng) -> float:
    va, vb = np.array([a.get(p, 0) for p in range(SUPPORT)], float), np.array([b.get(p, 0) for p in range(SUPPORT)], float)
    na, nb = int(va.sum()), int(vb.sum())
    if na == 0 or nb == 0:
        return float("nan")
    pool = (va + vb) / (va + vb).sum()
    xa, xb = rng.multinomial(na, pool, size=N_NULL), rng.multinomial(nb, pool, size=N_NULL)
    return float((0.5 * np.abs(xa / na - xb / nb).sum(axis=1)).mean())


def divergence(run, seed) -> list[dict]:
    """Per generation: cumulative and trailing-window TV between agent 0 and agent 1, each with its null."""
    props = proposals(run)
    if not props:
        return []
    G = max(p["gen"] for p in props)
    rng = np.random.default_rng(1000 + seed)
    out = []
    for g in range(G + 1):
        row = {"generation": g}
        for kind, lo in (("cum", 0), ("win", max(0, g - WINDOW + 1))):
            h = {0: Counter(), 1: Counter()}
            for p in props:
                if lo <= p["gen"] <= g and p["agent"] in h:
                    h[p["agent"]].update(p["positions"])
            obs, nul = tv_hist(h[0], h[1]), null_tv(h[0], h[1], rng)
            row[kind] = {"tv": obs, "null": nul, "excess": obs - nul, "n0": sum(h[0].values()), "n1": sum(h[1].values())}
        out.append(row)
    return out


def record_use(run) -> dict:
    props, res = proposals(run), resolved(run)
    c = Counter()
    for p in props:
        if p["fallback"] or p["gen"] < 2:
            continue
        known = [e for e in res if e["agent_id"] == p["agent"] and e["resolved_at_generation"] <= p["gen"] and not e["no_edit"]][-SHOWN:]
        latest = {}
        for e in known:
            f0, f1 = e["base_fitness"], e["fitness_after"]
            for ch in e["changes"]:
                latest[ch[0]] = "better" if f1 > f0 else ("worse" if f1 < f0 else "same")
        for pos in p["positions"]:
            c["edits"] += 1
            if pos in latest:
                c["in_record"] += 1
                c[latest[pos]] += 1
    return dict(c)


def fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def sections(runs, seeds) -> tuple[str, dict]:
    md, data = [], {}
    have = [s for s in seeds if all((a, s) in runs for a in ARMS)]
    part = [s for s in seeds if any((a, s) in runs for a in ARMS)]
    cut = {s: n_cut(runs, s) for s in part}

    md.append("### 1. Runs\n")
    md.append("| seed | arm | distinct folds | cache hits | final best TM (own budget) | wall s | LLM calls | agent fallback | attempt |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for s in part:
        for a in ARMS:
            if (a, s) not in runs:
                continue
            r, sm_ = runs[(a, s)], runs[(a, s)]["summary"]
            ap = sm_["operators"].get("agent_propose")
            fb = f"{ap['n_failures']}/{ap['n_llm_calls']}" if ap else "-"
            md.append(f"| {s} | {NAME[a]} | {sm_['n_distinct_evaluations']} | {sm_['n_cache_hits']} | {fmt(sm_['final_best'])} | {sm_['wall_s']:.0f} | {sm_['n_llm_calls_total']} | {fb} | {r.get('attempt')} |")
    md.append("")

    md.append("### 2. Best TM-score at the common distinct-fold count, per seed\n")
    md.append("n_cut = smallest distinct-fold count among the seed's arms: " + ", ".join(f"seed {s}: {cut[s]}" for s in part) + "\n")
    md.append("| seed | n_cut | " + " | ".join(NAME[a] for a in ARMS) + " |")
    md.append("|---|---|" + "---|" * len(ARMS))
    at = {}
    for s in part:
        row = []
        for a in ARMS:
            if (a, s) in runs:
                at[(a, s)] = best_at(runs[(a, s)], cut[s])
                row.append(fmt(at[(a, s)]))
            else:
                row.append("-")
        md.append(f"| {s} | {cut[s]} | " + " | ".join(row) + " |")
    md.append("")
    if have:
        md.append("| arm | mean over seeds " + str(have) + " | range |")
        md.append("|---|---|---|")
        for a in ARMS:
            v = [at[(a, s)] for s in have]
            md.append(f"| {NAME[a]} | {np.mean(v):.4f} | {min(v):.4f} - {max(v):.4f} |")
        md.append("")
        md.append(f"Rule: X beats Y only if higher in every seed; with {len(have)} seeds the smallest two-sided sign-test p is {2 / 2 ** len(have):.3g}, so nothing here can be called significant at 0.05.\n")
        for x, y in (("M2", "M1"), ("M2", "M3"), ("M1", "M3")):
            d = [at[(x, s)] - at[(y, s)] for s in have]
            hi, lo = sum(v > 0 for v in d), sum(v < 0 for v in d)
            verdict = f"{x} BEATS {y} under the pre-set rule" if hi == len(d) else f"NO DEMONSTRATED DIFFERENCE between {x} and {y}"
            md.append(f"- **{x} vs {y}:** {verdict}. {x} higher in {hi}, lower in {lo}, tied in {len(d) - hi - lo} of {len(d)} seeds; mean difference {np.mean(d):+.4f}; per seed ({x} minus {y}): "
                      + ", ".join(f"seed {s} {v:+.4f}" for s, v in zip(have, d)) + ".")
        md.append("")
        data["best_at_cut"] = {f"{a}_{s}": at[(a, s)] for (a, s) in at}

    # --- improvement rate
    md.append("### 3. Improvement rate: how often an agent's proposal beats its own base genome\n")
    P = {(a, s): proposals(runs[(a, s)]) for a in ("M1", "M2") for s in part if (a, s) in runs}
    if P:
        G = max(p["gen"] for v in P.values() for p in v)
        bins = thirds(G)
        md.append("Per third of the run (gens " + ", ".join(f"{lo}-{hi}" for lo, hi in bins) + "), improved / valid proposals (2 agents per seed):\n")
        md.append("| arm | seed | " + " | ".join(f"gens {lo}-{hi}" for lo, hi in bins) + " | all | fallbacks | trend (Spearman, gen vs improved) |")
        md.append("|---|---|" + "---|" * len(bins) + "---|---|---|")
        for a in ("M1", "M2"):
            for s in part:
                if (a, s) not in P:
                    continue
                v = [p for p in P[(a, s)] if p["valid"]]
                cells = []
                for lo, hi in bins:
                    w = [p for p in v if lo <= p["gen"] <= hi]
                    cells.append(f"{sum(p['improved'] for p in w)}/{len(w)}")
                sp = spearman([p["gen"] for p in v], [p["improved"] for p in v])
                md.append(f"| {NAME[a]} | {s} | " + " | ".join(cells) + f" | {sum(p['improved'] for p in v)}/{len(v)} | {sum(p['fallback'] for p in P[(a, s)])} | {fmt(sp, 2)} |")
        md.append("")
        md.append("Pooled over the seeds present:\n")
        md.append("| arm | " + " | ".join(f"gens {lo}-{hi}" for lo, hi in bins) + " | all | trend (Spearman) |")
        md.append("|---|" + "---|" * len(bins) + "---|---|")
        pooled = {}
        for a in ("M1", "M2"):
            v = [p for s in part if (a, s) in P for p in P[(a, s)] if p["valid"]]
            pooled[a] = v
            cells = []
            for lo, hi in bins:
                w = [p for p in v if lo <= p["gen"] <= hi]
                cells.append(f"{sum(p['improved'] for p in w)}/{len(w)} = {sum(p['improved'] for p in w) / max(1, len(w)):.0%}")
            md.append(f"| {NAME[a]} | " + " | ".join(cells) + f" | {sum(p['improved'] for p in v)}/{len(v)} = {sum(p['improved'] for p in v) / max(1, len(v)):.0%} | {fmt(spearman([p['gen'] for p in v], [p['improved'] for p in v]), 2)} |")
        md.append("")
        md.append("Per generation, pooled (improved/valid):\n")
        md.append("| gen | " + " | ".join(NAME[a] for a in ("M1", "M2")) + " |")
        md.append("|---|---|---|")
        for g in range(1, G + 1):
            cells = []
            for a in ("M1", "M2"):
                w = [p for p in pooled[a] if p["gen"] == g]
                cells.append(f"{sum(p['improved'] for p in w)}/{len(w)}")
            md.append(f"| {g} | " + " | ".join(cells) + " |")
        md.append("")
        data["improvement"] = {a: {"n_valid": len(pooled[a]), "n_improved": sum(p["improved"] for p in pooled[a])} for a in pooled}

    # --- divergence
    md.append("### 4. Divergence: distance between the two agents' proposal distributions\n")
    D = {(a, s): divergence(runs[(a, s)], s) for a in ("M1", "M2") for s in part if (a, s) in runs}
    if D:
        Gd = max(len(v) for v in D.values()) - 1
        marks = [g for g in (3, 6, 9, 12, 15, 18) if g <= Gd] or [Gd]
        md.append("Total variation over edited positions (obs / null / excess), cumulative from generation 0 and over a trailing window of "
                  f"{WINDOW} generations, at selected generations. Null = two samples of the same sizes from the pooled histogram; excess near 0 means the agents are indistinguishable from one distribution.\n")
        md.append("| arm | seed | kind | " + " | ".join(f"gen {g}" for g in marks) + " | mean excess over gens 3-" + str(Gd) + " |")
        md.append("|---|---|---|" + "---|" * len(marks) + "---|")
        for a in ("M1", "M2"):
            for s in part:
                if (a, s) not in D:
                    continue
                for kind in ("cum", "win"):
                    rows = D[(a, s)]
                    cells = [f"{rows[g][kind]['tv']:.2f} / {rows[g][kind]['null']:.2f} / {rows[g][kind]['excess']:+.2f}" for g in marks if g < len(rows)]
                    ex = [r[kind]["excess"] for r in rows if r["generation"] >= 3 and not np.isnan(r[kind]["excess"])]
                    md.append(f"| {NAME[a]} | {s} | {'cumulative' if kind == 'cum' else 'window'} | " + " | ".join(cells) + f" | {np.mean(ex):+.3f} |")
        md.append("")
        md.append("Mean over the seeds present of the excess TV, by third of the run:\n")
        md.append("| arm | kind | " + " | ".join(f"gens {lo}-{hi}" for lo, hi in thirds(Gd)) + " |")
        md.append("|---|---|" + "---|" * 3)
        for a in ("M1", "M2"):
            for kind in ("cum", "win"):
                cells = []
                for lo, hi in thirds(Gd):
                    v = [r[kind]["excess"] for s in part if (a, s) in D for r in D[(a, s)] if lo <= r["generation"] <= hi and not np.isnan(r[kind]["excess"])]
                    cells.append(f"{np.mean(v):+.3f}" if v else "n/a")
                md.append(f"| {NAME[a]} | {'cumulative' if kind == 'cum' else 'window'} | " + " | ".join(cells) + " |")
        md.append("")
        data["divergence"] = {f"{a}_{s}": D[(a, s)] for (a, s) in D}

    # --- what the proposals did
    md.append("### 5. What the agents' proposals were (M1, M2) and what the injected genomes were worth (M1, M2, M3)\n")
    md.append("| arm | seed | proposals (valid edit) | edited positions | distinct positions | most-edited position (share) | distinct new letters | most-used letter (share) | injected: mean TM | injected: best | rest of population: mean TM |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        for s in part:
            if (a, s) not in runs:
                continue
            r = runs[(a, s)]
            pop = pop_size(r)
            inj = [f for g in r["generations"][1:] for f in g["fitnesses"][pop:]]
            rest = [f for g in r["generations"][1:] for f in g["fitnesses"][:pop]]
            if a == "M3":
                md.append(f"| {NAME[a]} | {s} | - | - | - | - | - | - | {np.mean(inj):.4f} | {max(inj):.4f} | {np.mean(rest):.4f} |")
                continue
            pr = [p for p in proposals(r) if not p["fallback"]]
            pos, let = Counter(x for p in pr for x in p["positions"]), Counter(x for p in pr for x in p["letters"])
            n_pos = sum(pos.values())
            (pm, pc), (lm, lc) = pos.most_common(1)[0], let.most_common(1)[0]
            md.append(f"| {NAME[a]} | {s} | {len(pr)} | {n_pos} | {len(pos)} | {pm} ({pc / n_pos:.0%}) | {len(let)} | {lm} ({lc / n_pos:.0%}) | {np.mean(inj):.4f} | {max(inj):.4f} | {np.mean(rest):.4f} |")
    md.append("")
    md.append("Injected = the last 2 members of each evaluated population from generation 1 (agent proposals in M1/M2, random immigrants in M3).\n")

    # --- record use
    md.append("### 6. Do the agents act on their record? (per edited position, from generation 2)\n")
    md.append("| arm | seed | edited positions | in the agent's last-8 record | latest entry better | latest entry worse |")
    md.append("|---|---|---|---|---|---|")
    tot = {a: Counter() for a in ("M1", "M2")}
    for a in ("M1", "M2"):
        for s in part:
            if (a, s) not in runs:
                continue
            c = Counter(record_use(runs[(a, s)]))
            tot[a].update(c)
            md.append(f"| {NAME[a]} | {s} | {c['edits']} | {c['in_record']} ({c['in_record'] / max(1, c['edits']):.0%}) | {c['better']} ({c['better'] / max(1, c['edits']):.0%}) | {c['worse']} ({c['worse'] / max(1, c['edits']):.0%}) |")
    for a in ("M1", "M2"):
        c = tot[a]
        md.append(f"| **{NAME[a]}** | all | {c['edits']} | {c['in_record']} ({c['in_record'] / max(1, c['edits']):.0%}) | {c['better']} ({c['better'] / max(1, c['edits']):.0%}) | {c['worse']} ({c['worse'] / max(1, c['edits']):.0%}) |")
    md.append("")
    md.append("M1's record is kept but never shown, so its row is the baseline: where the same agent puts edits when it cannot see its record.\n")

    # --- cost
    md.append("### 7. LLM calls, tokens and wall time (per run)\n")
    md.append("| seed | arm | LLM calls | tokens in | tokens out | wall s | LLM s | fitness s | swap s |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for s in part:
        for a in ARMS:
            if (a, s) not in runs:
                continue
            x = runs[(a, s)]["summary"]
            st = x["operator_stats_total"]
            md.append(f"| {s} | {NAME[a]} | {x['n_llm_calls_total']} | {st['total_tokens_in']} | {st['total_tokens_out']} | {x['wall_s']:.0f} | {x['llm_s']:.0f} | {x['fitness_s']:.0f} | {x['swap_s']:.0f} |")
    md.append("")
    return "\n".join(md), {"data": data, "D": D, "P": P, "cut": cut, "part": part}


def chart(runs, ctx, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
    part, cut, D, P = ctx["part"], ctx["cut"], ctx["D"], ctx["P"]
    ns = max(1, len(part))
    fig = plt.figure(figsize=(max(9.5, 4.3 * ns), 8.4), facecolor=surface)
    gs = fig.add_gridspec(2, max(ns, 3), hspace=0.42, wspace=0.28)

    def style(ax, xl, yl=None, title=None):
        ax.set_facecolor(surface)
        ax.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=8)
        ax.set_xlabel(xl, color=ink2, fontsize=9)
        if yl:
            ax.set_ylabel(yl, color=ink2, fontsize=9)
        if title:
            ax.set_title(title, loc="left", color=ink, fontsize=9.5)

    for j, s in enumerate(part):
        ax = fig.add_subplot(gs[0, j])
        ends = []
        for a in ARMS:
            if (a, s) not in runs:
                continue
            y = runs[(a, s)]["best_so_far_by_distinct_evaluation"][: cut[s]]
            ax.plot(range(1, len(y) + 1), y, drawstyle="steps-post", color=COLOR[a], lw=2.0, ls="--" if a == "M3" else "-", label=NAME[a])
            ends.append([a, len(y), y[-1], y[-1]])
        ends.sort(key=lambda e: e[2])
        for k in range(1, len(ends)):
            ends[k][3] = max(ends[k][3], ends[k - 1][3] + 0.016)
        for a, x, yv, ly in ends:
            ax.plot([x], [yv], "o", color=COLOR[a], ms=5, mec=surface, mew=1.2)
            ax.annotate(f"{a} {yv:.3f}", (x, ly), xytext=(6, 0), textcoords="offset points", color=ink2, fontsize=8, va="center")
        ax.margins(x=0.2)
        style(ax, "distinct evaluations (folds), cut at the common count", "best-so-far TM-score" if j == 0 else None, f"seed {s}  (n_cut {cut[s]})")

    G = max((len(v) for v in D.values()), default=1) - 1
    ax = fig.add_subplot(gs[1, 0])
    for a in ("M1", "M2"):
        rates = []
        for g in range(1, G + 1):
            w = [p for s in part if (a, s) in P for p in P[(a, s)] if p["valid"] and p["gen"] == g]
            rates.append(sum(p["improved"] for p in w) / len(w) if w else np.nan)
        r = np.array(rates)
        sm_ = np.array([np.nanmean(r[max(0, i - 2): i + 1]) for i in range(len(r))])
        ax.plot(range(1, G + 1), r, color=COLOR[a], lw=0.8, alpha=0.35, marker="o", ms=3)
        ax.plot(range(1, G + 1), sm_, color=COLOR[a], lw=2.0, label=NAME[a])
        if len(sm_):
            ax.annotate(a, (G, sm_[-1]), xytext=(6, 0), textcoords="offset points", color=ink2, fontsize=8, va="center")
    ax.axhline(0.5, color=ink2, lw=0.8, ls=":")
    ax.set_ylim(-0.02, 1.02)
    ax.margins(x=0.12)
    style(ax, "generation of the proposal", "share of proposals that beat their base", "Improvement rate (dots: per generation; line: 3-gen mean)")

    for col, (kind, title) in enumerate((("cum", "Divergence, cumulative"), ("win", f"Divergence, trailing {WINDOW}-gen window")), start=1):
        ax = fig.add_subplot(gs[1, col])
        for a in ("M1", "M2"):
            vals = []
            for g in range(0, G + 1):
                v = [D[(a, s)][g][kind]["excess"] for s in part if (a, s) in D and g < len(D[(a, s)]) and not np.isnan(D[(a, s)][g][kind]["excess"])]
                vals.append(np.mean(v) if v else np.nan)
            ax.plot(range(0, G + 1), vals, color=COLOR[a], lw=2.0, label=NAME[a])
            ok = [i for i, x in enumerate(vals) if not np.isnan(x)]
            if ok:
                ax.annotate(a, (ok[-1], vals[ok[-1]]), xytext=(6, 0), textcoords="offset points", color=ink2, fontsize=8, va="center")
        ax.axhline(0, color=ink2, lw=0.8, ls=":")
        ax.margins(x=0.12)
        style(ax, "generation", "excess TV over the null (0 = identical agents)" if col == 1 else None, title)

    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, labelcolor=ink, fontsize=9, bbox_to_anchor=(0.5, 0.995))
    fig.savefig(out_png, dpi=150, facecolor=surface, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out-md", default=str(ROOT / "results" / "AGENT_MEMORY_STAGE3_TABLES.md"))
    ap.add_argument("--out-png", default=str(ROOT / "results" / "agent_memory_stage3.png"))
    ap.add_argument("--out-json", default=str(ROOT / "results" / "raw" / "agent_memory_stage3_summary.json"))
    a = ap.parse_args()
    runs = load(Path(a.raw), a.seeds)
    if not runs:
        raise SystemExit("no complete result files found")
    text, ctx = sections(runs, a.seeds)
    head = ("<!-- generated by experiments/summarize_agent_memory.py from results/raw/agent_memory_M*_seed*.json; do not edit by hand -->\n"
            f"Runs found: {sorted(f'{arm}/seed{s}' for arm, s in runs)}\n\n")
    Path(a.out_md).write_text(head + text + "\n")
    chart(runs, ctx, Path(a.out_png))
    Path(a.out_json).write_text(json.dumps(ctx["data"], indent=1, default=str))
    print(head + text)


if __name__ == "__main__":
    main()
