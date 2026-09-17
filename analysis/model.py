"""Pure-Python implementation of the Xue et al. (NoCS 2014) bottleneck
equations, mapped onto this project's software (multiprocessing) substrate.

Paper: "An Efficient Network-on-Chip (NoC) based Multicore Platform for
Hierarchical Parallel Genetic Algorithms" (Xue, Qian, Wei, Bogdan, Tsui,
Marculescu; NoCS 2014), Section III.A.

Vocabulary mapping (paper's NoC hardware -> this codebase's software harness):

  T_calc      Paper's Tcalc: average fitness-evaluation time on one slave.
              Measured directly (worker.py, per-individual t_calc).

  T_interval  Paper's Tinterval (Eq. 2): the time for the master to
              dispatch one individual. The paper decomposes this as
              Tturnaround + Lchm because their master injects one flit of
              a chromosome packet per cycle over a fixed-width channel, so
              a longer chromosome takes proportionally longer to inject.
              This software harness has no flit-width channel — one
              dispatch is a single queue.put() call — so there is no
              separate "packet length in flits" term to add; the measured
              T_interval (island.py, gaps between consecutive
              dispatch_timestamps) already *is* the master's whole
              per-individual dispatch cost, so it is used directly as the
              paper's Tinterval.

  T_sdelay    Paper's Tsdelay: round-trip time seen by the master from
              dispatching an individual to receiving its fitness back,
              Tsdelay = Tcalc + 2*Tcom (unlabelled equation just before
              Eq. 7). The paper itself assumes Tcom << Tcalc and
              simplifies Tsdelay ≈ Tcalc. That assumption holds at least
              as strongly here: this harness's measured T_interval (its
              closest proxy for per-message IPC/communication cost) runs
              ~1-2 microseconds against a T_calc of ~200-500
              microseconds, i.e. two-plus orders of magnitude smaller — so
              we keep the paper's own simplification (t_com=0 default)
              rather than inventing a separate T_com measurement this
              harness has no channel to take.

No fitting: every function below is a direct transcription of the paper's
formula, taking the harness's measured constants as inputs. Nothing here is
tuned against the sweep results — that comparison is the point (see
results/PHASE1_SUMMARY.md).
"""


def t_sdelay(t_calc: float, t_com: float = 0.0) -> float:
    """Tsdelay = Tcalc + 2*Tcom (paper, unlabelled eqn before Eq. 7).

    t_com defaults to 0 to reproduce the paper's own Tcom << Tcalc
    simplification (Tsdelay ≈ Tcalc); see module docstring.
    """
    return t_calc + 2 * t_com


def network_capacity(t_calc: float, t_interval: float, t_com: float = 0.0) -> float:
    """Eq. 3: C ≈ Tsdelay / Tinterval.

    The number of slave processors the master's injection channel can
    keep saturated with work before adding more stops helping.
    """
    return t_sdelay(t_calc, t_com) / t_interval


def max_speedup(t_calc: float, t_interval: float) -> float:
    """Eq. 7: G_N ≈ Tcalc / Tinterval ≈ C.

    The predicted speedup ceiling of the direct (non-multiplexed)
    island-based design, i.e. the horizontal asymptote the measured
    speedup-vs-N curve should approach as N -> infinity. This is the
    *asymptotic* form; see max_speedup_finite() for the +1 correction that
    matters when comparing against a finite swept N range rather than the
    N -> infinity limit.
    """
    return t_calc / t_interval


def max_speedup_finite(t_calc: float, t_interval: float) -> float:
    """1 + C: the DIS-phase speedup ceiling including the N=1 baseline term
    that Eq. 7's asymptotic form (max_speedup, above) omits.

    Derivation (review round 1, from this harness's own DIS-phase timing
    structure -- not verified against the paper's original derivation, which
    isn't accessible on this machine; see PHASE1_SUMMARY.md for the caveat):
    at N=1 there is no worker/master overlap, so the per-individual DIS-phase
    time is the *sum* of T_calc and T_interval (both happen serially). As
    N -> infinity, worker compute is fully hidden behind parallelism and the
    per-individual time is bounded below by T_interval alone (the master's
    serial, unparallelisable per-individual cost). The ratio of these two
    per-individual times gives the speedup ceiling:

        (T_calc + T_interval) / T_interval = 1 + T_calc/T_interval = 1 + C

    This is the quantity that should be compared against a DIS-phase speedup
    curve measured over a finite N range (rather than max_speedup's pure C,
    which is only exact in the N -> infinity limit).
    """
    return 1.0 + max_speedup(t_calc, t_interval)


def observed_saturation_point(worker_counts: list[int], speedups: list[float]) -> int:
    """Eq. 1: C = argmax{N : delta_G(N) > 0}, delta_G(N) = G(N+1) - G(N).

    The paper's own operational definition of where a speedup-vs-N curve
    saturates: the largest N in the swept range for which the *next*
    worker count still measurably increased speedup. Applied here to the
    observed sweep (as opposed to the model's predicted C above) so the
    two can be compared directly.
    """
    if len(worker_counts) != len(speedups):
        raise ValueError("worker_counts and speedups must be the same length")
    pairs = sorted(zip(worker_counts, speedups))
    best_n = pairs[0][0]
    for (n_a, g_a), (n_b, g_b) in zip(pairs, pairs[1:]):
        if g_b > g_a:
            best_n = n_b
    return best_n


def worker_efficiency(speedup: float, n_workers: int) -> float:
    """G(N) / N — fraction of linear (ideal) speedup actually achieved.
    Not a numbered equation in the paper, but it is exactly what Fig. 9
    plots ("Core Efficiency"), so it's included here alongside the
    numbered equations rather than duplicated ad hoc in plots.py.
    """
    return speedup / n_workers


def amdahl_speedup(serial_fraction: float, parallel_speedup: float) -> float:
    """Standard Amdahl's law — not one of Xue et al.'s numbered equations.

    The paper's Eq. 7 (G_N ≈ C) bounds speedup of the DIS phase only: it
    is derived from the ratio of Tdis values (fitness *distribution*
    time), which is exactly what their Fig. 7 plots ("Fitness Calculation
    Speedup"), as distinct from their Fig. 11 ("Overall Speedup"). Overall
    wall-clock speedup also has to account for the GA phase, which runs
    serially on the master and does not shrink with N. Given the fraction
    of total (N=1) time spent in that serial phase and the speedup
    actually achieved on the parallel (DIS) portion, this is the standard
    Amdahl bound on the resulting overall speedup:

        Speedup = 1 / ((1 - serial_fraction) / parallel_speedup + serial_fraction)

    Used to check whether Eq. 7's DIS-phase prediction, combined with the
    measured DIS/GA time split, is *sufficient* to explain the observed
    overall speedup — i.e. whether the two effects compound cleanly
    rather than leaving an unexplained residual.
    """
    return 1.0 / ((1 - serial_fraction) / parallel_speedup + serial_fraction)
