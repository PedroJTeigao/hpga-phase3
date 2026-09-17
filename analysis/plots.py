"""Phase 1 figures. Reads only from results/raw/ (never re-runs the sweep),
writes PNGs to results/figures/. Three figures:

  a) speedup_vs_workers.png   - DIS-phase / overall speedup vs. worker count
  b) worker_efficiency.png    - speedup(N)/N vs. worker count
  c) dis_ga_ratio.png         - DIS/GA phase time ratio vs. worker count

Source data: results/raw/sweep_repeats.json (the 3-repeat batch), not the
single original sweep_runs.json run. PHASE1_RESULTS.md presents the repeat
batch as the primary numbers, and these figures need to match it rather than
show a different run's data under the same axis labels. Each repeat's
speedup/Amdahl-prediction/C/1+C is computed against *that repeat's own* N=1
baseline (paired), then the three repeats are aggregated into a mean and a
+/-1 stdev error bar per N -- that pairing-then-aggregating is what makes the
error bars meaningful (a shared baseline would leak that baseline's own noise
into every point identically instead of cancelling out).

Figure (a) plots the DIS-phase (fitness-calculation) speedup as the primary
curve, since that is the quantity Eq. 7 actually bounds (see model.py and
PHASE1_SUMMARY.md) - plotting overall wall-clock speedup against C there
would be comparing the wrong things. It also overlays observed overall
speedup and the Amdahl-law prediction built from the observed DIS speedup,
to show that those two nearly coincide (the overall-speedup ceiling is
explained by the fixed serial GA fraction, not by the injection-channel
capacity C).

C and 1+C are each recomputed per repeat from that repeat's own N=1 data
(analysis/master_overhead.py's subtraction method), then shown as a
min-to-max band across the 3 repeats rather than a single line -- per
PHASE1_SUMMARY.md's "Review round 1/2 corrections", these numbers are
sensitive to which run supplies the N=1 baseline, so a single line would
overstate precision the data doesn't have. The naive C (~190, from the
original enqueue-call-gap T_interval) is shown the same way in the small
rescaled inset, since it's still ~60x off the top of the readable main-panel
axis.
"""

import json
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.model import amdahl_speedup, max_speedup_finite, network_capacity, worker_efficiency

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "results" / "raw"
FIG_DIR = ROOT / "results" / "figures"
REPEATS_JSON = RAW_DIR / "sweep_repeats.json"

# dataviz reference palette (references/palette.md), light-mode categorical slots.
BLUE = "#2a78d6"     # slot 1 - primary series
ORANGE = "#eb6834"   # slot 2 - secondary series
AQUA = "#1baf7a"     # slot 3 - tertiary / validation series
MUTED = "#898781"    # reference / ideal lines, non-data
GRIDLINE = "#e1e0d9"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"

ERRORBAR_KW = dict(capsize=3, capthick=1.2, elinewidth=1.2)


def _style_axes(ax) -> None:
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors=SECONDARY_INK)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(SECONDARY_INK)
    ax.yaxis.label.set_color(SECONDARY_INK)


PHYSICAL_CORES = 6  # psutil.cpu_count(logical=False); review round 1 caught
                     # os.cpu_count()'s 12 as logical, not physical.


def _physical_core_idx(Ns: list[int], physical: int = PHYSICAL_CORES) -> float:
    """Interpolated x position (on the categorical idx axis) of the physical
    core count, between whichever two swept N values it falls between."""
    for i in range(len(Ns) - 1):
        if Ns[i] <= physical <= Ns[i + 1]:
            frac = (physical - Ns[i]) / (Ns[i + 1] - Ns[i])
            return i + frac
    return -1.0  # physical count below the swept range; caller should skip


def _add_physical_core_line(ax, Ns: list[int]) -> None:
    x = _physical_core_idx(Ns)
    if x < 0:
        return
    ymin, ymax = ax.get_ylim()
    ax.axvline(x, color=SECONDARY_INK, linewidth=1.2, linestyle=(0, (2, 2)), zorder=1)
    ax.text(x, ymin + (ymax - ymin) * 0.03, f" {PHYSICAL_CORES} physical cores", color=SECONDARY_INK,
            fontsize=8, rotation=90, va="bottom", ha="left")


def _mean_std(xs: list[float]) -> tuple[float, float]:
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def load_repeat_batch() -> dict:
    """Loads results/raw/sweep_repeats.json and computes, per repeat and per
    N, the quantities each figure needs -- paired to that repeat's own N=1
    run -- then aggregates (mean, stdev) across repeats per N.

    Also computes, per repeat, C and 1+C from that repeat's own N=1 data
    (same subtraction method as analysis/master_overhead.py, applied here
    directly to the run-level summary rather than the per-generation CSV,
    since sweep_repeats.json only has run-level aggregates), plus the naive
    (enqueue-call) C for the small inset.
    """
    with open(REPEATS_JSON) as f:
        runs = json.load(f)

    by_repeat: dict[int, dict[int, dict]] = {}
    for r in runs:
        by_repeat.setdefault(r["repeat"], {})[r["n_workers"]] = r

    Ns = sorted(next(iter(by_repeat.values())).keys())

    dis_speedup = {n: [] for n in Ns}
    overall_speedup = {n: [] for n in Ns}
    amdahl_pred = {n: [] for n in Ns}
    dis_ga_ratio = {n: [] for n in Ns}

    C_list = []
    ceiling_list = []
    C_naive_list = []

    for repeat, by_n in by_repeat.items():
        base = by_n[1]
        s_frac = base["ga_total_s"] / (base["dis_total_s"] + base["ga_total_s"])
        for n in Ns:
            r = by_n[n]
            g_dis = base["dis_total_s"] / r["dis_total_s"]
            dis_speedup[n].append(g_dis)
            overall_speedup[n].append(base["wall_time_s"] / r["wall_time_s"])
            amdahl_pred[n].append(amdahl_speedup(s_frac, g_dis))
            dis_ga_ratio[n].append(r["dis_ga_ratio"])

        t_calc_mean = base["t_calc"]["mean"]
        t_calc_n = base["t_calc"]["n"]
        t_interval_orig = base["t_interval"]["mean"]
        master_overhead = (base["dis_total_s"] - t_calc_mean * t_calc_n) / t_calc_n

        C_list.append(network_capacity(t_calc_mean, master_overhead))
        ceiling_list.append(max_speedup_finite(t_calc_mean, master_overhead))
        C_naive_list.append(network_capacity(t_calc_mean, t_interval_orig))

    def agg(d: dict[int, list[float]]) -> dict[int, tuple[float, float]]:
        return {n: _mean_std(vals) for n, vals in d.items()}

    return {
        "Ns": Ns,
        "dis_speedup": agg(dis_speedup),
        "overall_speedup": agg(overall_speedup),
        "amdahl_pred": agg(amdahl_pred),
        "dis_ga_ratio": agg(dis_ga_ratio),
        "C_band": (min(C_list), max(C_list)),
        "ceiling_band": (min(ceiling_list), max(ceiling_list)),
        "C_naive_band": (min(C_naive_list), max(C_naive_list)),
    }


def plot_speedup(data: dict) -> None:
    Ns = data["Ns"]
    idx = list(range(len(Ns)))  # even spacing: N is a sparse, non-uniform set (1,2,4,8,12,16,32)

    dis_mean = [data["dis_speedup"][n][0] for n in Ns]
    dis_std = [data["dis_speedup"][n][1] for n in Ns]
    overall_mean = [data["overall_speedup"][n][0] for n in Ns]
    overall_std = [data["overall_speedup"][n][1] for n in Ns]
    amdahl_mean = [data["amdahl_pred"][n][0] for n in Ns]
    amdahl_std = [data["amdahl_pred"][n][1] for n in Ns]

    C_lo, C_hi = data["C_band"]
    ceil_lo, ceil_hi = data["ceiling_band"]
    C_naive_lo, C_naive_hi = data["C_naive_band"]

    fig, (ax, ax_zoom) = plt.subplots(
        1, 2, figsize=(11.5, 5.5), dpi=150, gridspec_kw={"width_ratios": [2.4, 1]}
    )
    fig.patch.set_facecolor("#fcfcfb")
    _style_axes(ax)
    _style_axes(ax_zoom)

    ax.plot(idx, Ns, linestyle=(0, (4, 3)), color=MUTED, linewidth=1.5, label="Linear (ideal) speedup", zorder=2)
    ax.errorbar(idx, dis_mean, yerr=dis_std, marker="o", markersize=7, color=BLUE, linewidth=2,
                label="Observed DIS-phase speedup (what Eq. 7 bounds)", zorder=3, **ERRORBAR_KW)
    ax.errorbar(idx, overall_mean, yerr=overall_std, marker="s", markersize=7, color=ORANGE, linewidth=2,
                linestyle="--", label="Observed overall (wall-clock) speedup", zorder=3, **ERRORBAR_KW)
    ax.errorbar(idx, amdahl_mean, yerr=amdahl_std, marker="^", markersize=7, color=AQUA, linewidth=2,
                linestyle=":", label="Amdahl-predicted overall (from observed DIS speedup)", zorder=3, **ERRORBAR_KW)

    ax.axhspan(ceil_lo, ceil_hi, color=ORANGE, alpha=0.15, zorder=1,
               label=f"Predicted DIS ceiling, finite-N form 1+C ≈{ceil_lo:.2f}–{ceil_hi:.2f}")
    ax.axhspan(C_lo, C_hi, color=ORANGE, alpha=0.25, zorder=1,
               label=f"Eq.7 asymptotic C (N→∞) ≈{C_lo:.2f}–{C_hi:.2f}")

    ax.set_xlim(-0.3, len(idx) - 0.7)
    ax.set_ylim(0, 5)
    ax.set_xticks(idx)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("Worker count N")
    ax.set_ylabel("Speedup relative to N=1")
    ax.set_title("Speedup vs. Worker Count")
    _add_physical_core_line(ax, Ns)
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    # Companion panel (not an overlapping inset - a separate axes) showing
    # where Eq.7's predicted C sits relative to what's achievable in this
    # swept range: same DIS-phase curve, y-axis zoomed out to include C.
    ax_zoom.errorbar(idx, dis_mean, yerr=dis_std, color=BLUE, linewidth=2, marker="o", markersize=5,
                      label="Observed DIS-phase speedup", **ERRORBAR_KW)
    ax_zoom.axhspan(C_naive_lo, C_naive_hi, color=ORANGE, alpha=0.3,
                     label=f"Eq.3/7 predicted C (naive T_interval)≈{C_naive_lo:.0f}–{C_naive_hi:.0f}")
    ax_zoom.set_ylim(0, C_naive_hi * 1.08)
    ax_zoom.set_xlim(-0.3, len(idx) - 0.7)
    ax_zoom.set_xticks(idx)
    ax_zoom.set_xticklabels([str(n) for n in Ns])
    ax_zoom.set_xlabel("Worker count N")
    ax_zoom.set_title("Same DIS-phase curve,\ny-axis to scale with naive C", fontsize=9.5, color=SECONDARY_INK)
    ax_zoom.legend(loc="center left", frameon=False, fontsize=8)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "speedup_vs_workers.png")
    plt.close(fig)


def plot_efficiency(data: dict) -> None:
    Ns = data["Ns"]
    idx = list(range(len(Ns)))

    dis_eff_mean = [worker_efficiency(data["dis_speedup"][n][0], n) for n in Ns]
    dis_eff_std = [data["dis_speedup"][n][1] / n for n in Ns]
    overall_eff_mean = [worker_efficiency(data["overall_speedup"][n][0], n) for n in Ns]
    overall_eff_std = [data["overall_speedup"][n][1] / n for n in Ns]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    _style_axes(ax)

    ax.axhline(1.0, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), label="Ideal (efficiency=1)")
    ax.errorbar(idx, dis_eff_mean, yerr=dis_eff_std, marker="o", markersize=7, color=BLUE, linewidth=2,
                label="DIS-phase efficiency", **ERRORBAR_KW)
    ax.errorbar(idx, overall_eff_mean, yerr=overall_eff_std, marker="s", markersize=7, color=ORANGE,
                linewidth=2, linestyle="--", label="Overall efficiency", **ERRORBAR_KW)

    ax.set_xlim(-0.3, len(idx) - 0.7)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(idx)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("Worker count N")
    ax.set_ylabel("Efficiency = Speedup(N) / N")
    ax.set_title("Worker Efficiency vs. Worker Count")
    _add_physical_core_line(ax, Ns)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "worker_efficiency.png")
    plt.close(fig)


def plot_dis_ga_ratio(data: dict) -> None:
    Ns = data["Ns"]
    idx = list(range(len(Ns)))
    ratio_mean = [data["dis_ga_ratio"][n][0] for n in Ns]
    ratio_std = [data["dis_ga_ratio"][n][1] for n in Ns]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    _style_axes(ax)

    ax.axhline(1.0, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(len(idx) - 1, 1.05, "Ratio=1", color=SECONDARY_INK, fontsize=9, ha="right")
    ax.errorbar(idx, ratio_mean, yerr=ratio_std, marker="o", markersize=7, color=BLUE, linewidth=2,
                **ERRORBAR_KW)

    ax.set_xlim(-0.3, len(idx) - 0.7)
    ax.set_ylim(0, max(m + s for m, s in zip(ratio_mean, ratio_std)) * 1.15)
    ax.set_xticks(idx)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("Worker count N")
    ax.set_ylabel("DIS time / GA time")
    ax.set_title("DIS/GA Phase Time Ratio vs. Worker Count")
    _add_physical_core_line(ax, Ns)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "dis_ga_ratio.png")
    plt.close(fig)


def main() -> None:
    if not REPEATS_JSON.exists():
        raise SystemExit(f"no repeat-batch data at {REPEATS_JSON} - run experiments/run_sweep_repeats.py first")
    data = load_repeat_batch()
    plot_speedup(data)
    plot_efficiency(data)
    plot_dis_ga_ratio(data)
    print(f"wrote 3 figures to {FIG_DIR} (source: sweep_repeats.json, 3-repeat batch)")


if __name__ == "__main__":
    main()
