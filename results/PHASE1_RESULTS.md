# Phase 1 Results — Classical HPGA Baseline

*Standalone results summary. For full derivations, raw data pointers, and the
two-round correction history behind every number here, see
`PHASE1_SUMMARY.md` — not restated in this document.*

## 1. What was built and measured

Phase 1 implements a classical, single-island master/slave hierarchical
parallel genetic algorithm (HPGA) for 2D HP-lattice protein folding, in
Python (`multiprocessing`) — a software re-derivation of the NoC hardware
bottleneck analysis in Xue et al. (NoCS 2014, Section III.A). A master
process holds the population and dispatches individuals to a persistent pool
of N worker processes for fitness evaluation; N is swept to characterize how
dispatch/compute timing and search speedup scale with worker count.

**Run configuration:** population size 256, 40 generations per run, worker
counts N ∈ {1, 2, 4, 8, 12, 16, 32}, on a 6-physical / 12-logical
(hyperthreaded) core machine. Two workloads are used for different purposes:
a synthetic 300-residue random H/P sequence for the timing sweep (chosen only
to make per-individual compute time non-trivial relative to dispatch
overhead), and the real Unger & Moult (1993) n=20 benchmark sequence (known
optimal energy E*=-9) for GA correctness validation.

## 2. GA validation

The deterministic GA operators (tournament selection, single-point
crossover, point mutation, elitism) were validated against the real n=20
Unger & Moult benchmark, independent of the timing harness. Across 5 seeds ×
500 generations: **3 of 5 seeds reached the exact global optimum** (9
non-consecutive H-H contacts, E*=-9); the other 2 landed one contact short
(8). **All five final folds were valid** — positive fitness is only
achievable without lattice self-collisions. Convergence was fast (by
generation ~25–50 of 500) and stable once reached.

The synthetic 300-residue sequence used for the timing sweep below always
ends at fitness -73 (an invalid, self-colliding fold) — this is a
generation-budget artifact of a sequence that exists only to keep
per-individual compute time non-trivial for timing purposes, not a GA
defect; it was never intended to converge in 40 generations.

## 3. Timing results

Primary numbers below are from a 3-repeat batch (same configuration and GA
seed, repeated back-to-back) rather than the original single sweep run, so
error bars reflect real run-to-run timing noise rather than a single sample.

| N | T_calc (ms) | DIS-phase speedup | Overall speedup | DIS/GA ratio | Worker utilisation |
|---|---|---|---|---|---|
| 1 | 0.201 ± 0.001 | 1.000 ± 0.000 | 1.000 ± 0.000 | 10.56 ± 0.26 | 0.619 ± 0.007 |
| 2 | 0.214 ± 0.007 | 1.777 ± 0.048 | 1.651 ± 0.044 | 5.62 ± 0.11 | 0.544 ± 0.002 |
| 4 | 0.254 ± 0.017 | 2.760 ± 0.151 | 2.348 ± 0.115 | 3.49 ± 0.10 | 0.459 ± 0.005 |
| 8 | 0.371 ± 0.015 | 3.319 ± 0.122 | 2.695 ± 0.093 | 2.87 ± 0.03 | 0.385 ± 0.000 |
| 12 | 0.461 ± 0.013 | 3.120 ± 0.086 | 2.575 ± 0.074 | 3.07 ± 0.13 | 0.305 ± 0.001 |
| 16 | 0.466 ± 0.015 | 2.913 ± 0.111 | 2.420 ± 0.102 | 3.15 ± 0.14 | 0.217 ± 0.004 |
| 32 | 0.486 ± 0.019 | 2.591 ± 0.234 | 2.146 ± 0.144 | 3.15 ± 0.34 | 0.100 ± 0.004 |

(mean ± standard deviation across 3 repeats; speedups computed against each
repeat's own N=1 run)

Both DIS-phase and overall speedup peak in the N=8–12 range (point estimate
N=8), above the 6 physical cores. Worker utilisation declines monotonically
from 0.62 at N=1. Two mechanisms contribute in different regimes: below the
physical core count, workers idle because the master cannot dispatch fast
enough and because they wait through the serial GA phase — the dispatch-side
bottleneck of Eq. 3, visible directly in utilisation. Beyond ~6 workers, core
contention compounds this, visible as growth in T_calc itself, which grows
0.20ms → 0.49ms from N=1 to N=32 and nearly flattens between N=16 and N=32,
consistent with a shared-resource bottleneck rather than simple time-slicing.

The original single sweep run's numbers are consistent in shape but differ
by several percent in magnitude: T_calc(N=1) was 0.2144ms (vs. 0.201±0.001ms
here), and overall speedup peaked at 2.50x at N=8 (vs. 2.695±0.093x here).
See Limitations below.

## 4. Model comparison

Xue et al.'s Eq. 3 channel-capacity model **transfers to this software
substrate in order of magnitude, not in precise numeric agreement.** The
paper's `T_interval` (their per-individual dispatch cost) cannot be read as
just the `queue.put()` enqueue call on this substrate — that call returns in
~1µs regardless of load, since `mp.Queue` hands the actual send off to a
background thread — so a literal transcription puts the predicted DIS-phase
speedup ceiling `C = T_calc/T_interval` at ≈190, about 60x above what's
observed. Redefining `T_interval` as the master's **total serial cost per
individual** (pickling, unpickling, result bookkeeping — everything the
enqueue-call reading misses) moves `C` to **≈2.2**, the same order of
magnitude as the observed DIS-phase speedup peaks (3.09–3.32x, across the
original run and the repeat batch).

The finite-N form of Eq. 7, `1+C ≈ 3.2`, is presented **as an approximate
scale for this regime, not a strict bound**: recomputed from the repeat
batch's own baseline data it comes out lower (3.06–3.21), and is exceeded by
the repeat batch's own observed peak by 3.5–8.6%. The paper's broader
two-bottleneck framing — a dispatch/channel bottleneck limiting the DIS
phase, and a separate fixed serial phase (here, the GA step, ~8.9% of N=1
wall time) capping overall speedup via Amdahl's law — does transfer cleanly;
core contention plays the role the paper's channel bandwidth plays.

## 5. Limitations

- **Session-level timing variance.** Absolute timing numbers differ by
  several percent between runs taken at different times (e.g., T_calc(N=1):
  0.2144ms vs. 0.201ms), beyond what within-run noise explains — likely
  background load, CPU turbo/thermal state, or other activity on a shared,
  non-dedicated machine. Single-run comparisons, including against later
  phases, should allow for this rather than treating small percentage
  differences as meaningful.
- **N=8 vs. N=12 not statistically separable at n=3 repeats.** Their speedup
  ranges overlap and a two-sample comparison does not reach conventional
  significance at this sample size. "Saturates around N=8–12" is the
  supported claim; "at N=8" is not.
- **Untested hypothesis on the ceiling derivation.** The 1+C derivation
  assumes no worker/master overlap at N=1. `mp.Queue`'s background feeder
  threads make some overlap plausible even at N=1 (a worker can start its
  next computation before the master finishes processing the prior result),
  which would undermine that assumption and is a plausible, unverified
  explanation for why the ceiling is exceeded rather than respected.

## 6. What Phase 1 establishes, and what's next

Phase 1 establishes a working, independently-validated classical HPGA
baseline with an instrumented timing harness: the GA operators are confirmed
correct on a known benchmark, and the paper's bottleneck framing (a
dispatch-side bottleneck plus a separate serial-phase ceiling) transfers in
kind to a software substrate, with core contention substituting for the
original hardware's channel bandwidth as the mechanism that limits the first
bottleneck. The paper's specific numeric constants do not transfer literally
and should not be quoted as such.

For Phase 2 (LLM-based operators replacing the deterministic ones), timing
comparisons against this baseline should use repeated runs rather than
single runs, given the session-level variance noted above, and should not
assume Eq. 3/7's constants will hold numerically on whatever substrate LLM
inference introduces.
