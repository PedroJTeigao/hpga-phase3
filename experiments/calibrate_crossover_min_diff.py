"""Calibrates CROSSOVER_MIN_DIFF against a measured distribution instead of
a guess. hpga/operators.py's own comment already flagged the existing
default (2) as uncalibrated for the 5-symbol 3D alphabet; Phase 3's circles
smoke runs additionally found it uncalibrated against genome length (0%
parse failures under "segment" style, but still 51-79% fallback from
_crossover_sufficiently_mixed at both length 8 and length 18) -- an
unvalidated free parameter, not a bug to patch around.

No LLM calls, no GPU: this measures the DETERMINISTIC crossover() (single-
point, Phase 1's original operator) against random parent pairs at
genome_length=18, the length every real Phase 3 run so far actually uses.
_crossover_sufficiently_mixed itself never gates the deterministic path --
it only validates LLM output -- so the deterministic operator is exactly
the right reference for "what does genuine recombination look like,"
independent of whether an LLM produced it.

rate=1.0 forces crossover on every trial (no rate-gate skip). A single-
point crossover at a uniformly random point is itself a 2-segment split,
structurally identical in shape to what "segment" style asks the model to
produce -- so this is a direct, not analogous, reference for that style,
and incidentally relevant to "full"/"diff" too since all three styles are
gated by the same _crossover_sufficiently_mixed check.

The binding constraint in _crossover_sufficiently_mixed is the MINIMUM of
the four child-vs-parent diff counts (child1 vs parent1, child1 vs parent2,
child2 vs parent1, child2 vs parent2) -- all four must clear the threshold,
so that minimum's own distribution across trials is what determines how
often a genuine deterministic-quality crossover would itself pass the gate
at a given threshold value.
"""

import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpga import operators as ops

GENOME_LEN = 18
N_TRIALS = 5000
SEED = 0


def main() -> None:
    rng = random.Random(SEED)
    mins = []
    all_four = []
    for _ in range(N_TRIALS):
        p1 = ops.random_genome(GENOME_LEN, rng)
        p2 = ops.random_genome(GENOME_LEN, rng)
        c1, c2 = ops.crossover(p1, p2, 1.0, rng)
        d = (
            ops._diff_count(c1, p1), ops._diff_count(c1, p2),
            ops._diff_count(c2, p1), ops._diff_count(c2, p2),
        )
        all_four.extend(d)
        mins.append(min(d))

    mins.sort()

    def pct(p: float) -> int:
        idx = min(len(mins) - 1, max(0, round(p / 100 * (len(mins) - 1))))
        return mins[idx]

    percentiles = {p: pct(p) for p in (0, 1, 5, 10, 25, 50, 75, 90, 99, 100)}

    # Fraction of genuine deterministic crossovers that would PASS the gate
    # at each candidate threshold -- the number that actually matters for
    # picking one, more directly than the percentile table alone.
    pass_rate_by_threshold = {
        t: sum(1 for m in mins if m >= t) / len(mins)
        for t in range(0, 10)
    }

    chosen = percentiles[5]

    print(f"genome_length={GENOME_LEN}  n_trials={N_TRIALS}  seed={SEED}\n")
    print("Distribution of min(diff(c1,p1), diff(c1,p2), diff(c2,p1), diff(c2,p2)) "
          "across trials (this is the quantity _crossover_sufficiently_mixed thresholds):")
    for p, v in percentiles.items():
        print(f"  p{p:>3}: {v}")
    print(f"  mean(all four, per trial): {statistics.mean(all_four):.2f}")
    print()
    print("Pass rate of a genuine deterministic crossover against candidate CROSSOVER_MIN_DIFF values:")
    for t, rate in pass_rate_by_threshold.items():
        print(f"  threshold={t}: {rate:.3f} of trials would pass")
    print()
    print(f"Chosen: CROSSOVER_MIN_DIFF = {chosen} (5th percentile of the measured minimum -- "
          f"~95% of genuine deterministic crossovers on random parent pairs at this length "
          f"would themselves clear this bar). Length-18-specific: this does not generalize "
          f"to other genome lengths without re-running this script at that length.")

    out_path = Path(__file__).resolve().parent.parent / "results" / "raw" / "crossover_min_diff_calibration.json"
    import json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "genome_length": GENOME_LEN, "n_trials": N_TRIALS, "seed": SEED,
            "percentiles": percentiles,
            "mean_all_four": statistics.mean(all_four),
            "pass_rate_by_threshold": pass_rate_by_threshold,
            "chosen_crossover_min_diff": chosen,
            "chosen_rationale": "5th percentile of the measured min-of-four-diffs distribution",
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
