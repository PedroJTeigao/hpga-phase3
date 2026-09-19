"""Checks for making operators.next_generation work with the sequence GenomeModel
(deterministic operators; no GPU, no Ollama, no fitness model -- a stub fitness).

  A. LATTICE PARITY. hpga/operators.py at a git ref (default main) vs the working
     tree, on the deterministic path only: random_population, tournament_select and
     next_generation (selection, elitism copy, crossover, mutation), chained over
     generations, with no active model and with a lattice model active. Compared per
     step: the population's repr, the RNG state afterwards, and whether any returned
     genome aliases an input genome. Unlike verify_llm_operator_parity.py this needs no
     stubbed Ollama and is valid for any baseline: none of these functions dispatches
     back into hpga.operators through the GenomeModel. Self-check: two deliberately
     broken copies of the baseline MUST diverge.
  B. SEQUENCE MODE. With a sequence GenomeModel active: random_population,
     tournament_select, next_generation return str genomes of the right count, inside
     the length bounds, with elitism honoured and runs reproducible by seed.
  C. LOUD FAILURES. Sequence mode with agents / circles / diversity logging enabled,
     a non-str population under a sequence model, and a str population with no
     sequence model each raise -- before any RNG draw or generation tick.

Usage: python experiments/verify_sequence_next_generation.py [--baseline REF]
Exit status 0 only if every check passes.
"""

import argparse
import importlib.util
import itertools
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from hpga import agents, genome_model as gm  # noqa: E402
from hpga import operators as after  # noqa: E402
from hpga import sequence_model as sm  # noqa: E402
from hpga.config import HPGAConfig  # noqa: E402

FAILS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail and not ok else ''}")
    if not ok:
        FAILS.append(label)


def load_baseline(ref: str, tag: str, mutate=None):
    src = subprocess.run(["git", "-C", str(PROJECT), "show", f"{ref}:hpga/operators.py"],
                         capture_output=True, text=True, check=True).stdout
    if mutate:
        old, new = mutate
        assert old in src, f"mutant pattern absent from baseline: {old!r}"
        src = src.replace(old, new, 1)
    path = Path(tempfile.mkdtemp(prefix="seq_ng_")) / f"{tag}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(tag, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[tag] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ A. parity

def lattice_trace(mod, case) -> list:
    """Everything observable about a short chained lattice run through `mod`."""
    seed, length, pop, elit, k, crate, mrate = case
    os.environ["HPGA_OPERATOR_MODE"] = "deterministic"
    rng = random.Random(seed)
    trace = []
    population = mod.random_population(pop, length, rng)
    trace.append(("init", repr(population), rng.getstate()))
    for _ in range(4):
        # fitness with ties, a pure function of the genome
        fit = [float(sum(g) % 5) for g in population]
        sel = mod.tournament_select(population, fit, min(k, len(population)), rng)
        trace.append(("select", repr(sel), any(sel is g for g in population), rng.getstate()))
        new = mod.next_generation(population, fit, pop, k, crate, mrate, elit, rng)
        alias = any(c is g for c in new for g in population)
        trace.append(("next", repr(new), alias, rng.getstate()))
        population = new
    return trace


def parity_cases():
    for case in itertools.product(
        (0, 1, 7), (2, 5, 18, 25), (2, 3, 8, 16), (0, 1, 2), (1, 3), (0.0, 0.9, 1.0), (0.0, 0.05, 1.0)
    ):
        if case[4] <= case[2]:  # tournament_k must not exceed the population size (invalid otherwise, in the baseline too)
            yield case


def run_parity(before, label: str) -> int:
    bad = 0
    n = 0
    for model_kind in ("none", "lattice"):
        gm.set_active(None if model_kind == "none" else gm.build_genome_model(HPGAConfig(sequence="HPHPHPHPHP")))
        for case in parity_cases():
            n += 1
            if lattice_trace(before, case) != lattice_trace(after, case):
                bad += 1
    gm.set_active(None)
    print(f"  [{label}] {n} lattice cases (x4 chained generations), diverged: {bad}")
    return bad


# ------------------------------------------------------------ B. sequence mode

def stub_fitness(g: str) -> float:
    return (g.count("A") + 0.01 * g.count("L")) / len(g)  # deterministic, has structure, ties are rare


def run_sequence_checks() -> None:
    os.environ["HPGA_OPERATOR_MODE"] = "deterministic"
    for var in ("HPGA_AGENTS_ENABLED", "HPGA_CIRCLES_ENABLED", "HPGA_LOG_DIVERSITY"):
        os.environ.pop(var, None)
    gm.set_active(gm.build_genome_model(HPGAConfig(genome_model="sequence")))
    lo, hi = sm.MIN_LENGTH, sm.MAX_LENGTH
    letters = set(sm.ALPHABET)

    rng = random.Random(0)
    pop = after.random_population(8, None, rng)
    check("random_population(8, None): 8 str genomes", len(pop) == 8 and all(isinstance(g, str) for g in pop))
    check("random_population(None): lengths inside bounds and varied",
          all(lo <= len(g) <= hi for g in pop) and len({len(g) for g in pop}) > 1)
    check("random_population(63): every genome length 63", {len(g) for g in after.random_population(6, 63, rng)} == {63})
    try:
        after.random_population(2, 5, rng)
        check("random_population(length below bounds) raises", False)
    except ValueError:
        check("random_population(length below bounds) raises", True)

    fit = [stub_fitness(g) for g in pop]
    sel = after.tournament_select(pop, fit, 3, rng)
    check("tournament_select returns the str genome itself, not a list of characters",
          isinstance(sel, str) and sel in pop)

    ok_count = ok_bounds = ok_alpha = ok_elite = ok_str = True
    for seed in range(12):
        r = random.Random(seed)
        p = after.random_population(8, None, r)
        for _ in range(30):
            f = [stub_fitness(g) for g in p]
            top2 = sorted(range(len(p)), key=lambda i: f[i], reverse=True)[:2]
            nxt = after.next_generation(p, f, 8, 3, 0.9, 0.05, 2, r)
            ok_count &= len(nxt) == 8
            ok_str &= all(isinstance(g, str) for g in nxt)
            ok_bounds &= all(lo <= len(g) <= hi for g in nxt)
            ok_alpha &= all(set(g) <= letters for g in nxt)
            ok_elite &= nxt[:2] == [p[i] for i in top2]
            p = nxt
    check("next_generation x360 steps: always pop_size genomes", ok_count)
    check("next_generation: every genome is a str", ok_str)
    check(f"next_generation: every length inside [{lo}, {hi}]", ok_bounds)
    check("next_generation: only the 20 canonical letters", ok_alpha)
    check("next_generation: the top-2 genomes carried over as the first two", ok_elite)

    def run(seed):
        r = random.Random(seed)
        p = after.random_population(8, None, r)
        for _ in range(10):
            p = after.next_generation(p, [stub_fitness(g) for g in p], 8, 3, 0.9, 0.05, 2, r)
        return p, r.getstate()

    check("same seed -> identical population and RNG state", run(3) == run(3))
    check("different seeds -> different populations", run(3)[0] != run(4)[0])

    p0 = after.random_population(8, None, random.Random(5))
    nxt = after.next_generation(p0, [stub_fitness(g) for g in p0], 8, 3, 0.0, 0.0, 2, random.Random(5))
    check("crossover=0, mutation=0: every child is an unchanged parent", all(g in p0 for g in nxt))
    gm.set_active(None)


# ------------------------------------------------------------- C. loud failures

def raises(fn, label, contains=""):
    try:
        fn()
    except RuntimeError as e:
        check(label, contains in str(e), f"message was: {e}")
        return
    check(label, False, "did not raise")


def run_failure_checks() -> None:
    os.environ["HPGA_OPERATOR_MODE"] = "deterministic"
    for var in ("HPGA_AGENTS_ENABLED", "HPGA_CIRCLES_ENABLED", "HPGA_LOG_DIVERSITY"):
        os.environ.pop(var, None)
    seq_pop = ["ACDEFGHIKLMNPQRSTVWY" * 2] * 4
    fit = [0.1, 0.2, 0.3, 0.4]
    gm.set_active(gm.build_genome_model(HPGAConfig(genome_model="sequence")))
    for var, name in (("HPGA_AGENTS_ENABLED", "agents"), ("HPGA_CIRCLES_ENABLED", "circles"),
                      ("HPGA_LOG_DIVERSITY", "diversity logging")):
        r = random.Random(1)
        before_state, before_gen = r.getstate(), agents._generation
        os.environ[var] = "1"
        raises(lambda: after.next_generation(seq_pop, fit, 4, 3, 0.9, 0.05, 2, r),
               f"sequence mode + {var}=1 raises (operator mode deterministic)", var)
        os.environ.pop(var)
        check(f"  ...and touched nothing first ({name}): RNG state and generation counter unchanged",
              r.getstate() == before_state and agents._generation == before_gen)
    raises(lambda: after.next_generation([[0, 1, 2]] * 4, fit, 4, 3, 0.9, 0.05, 2, random.Random(1)),
           "sequence model active + list genomes raises", "non-str")
    gm.set_active(None)
    raises(lambda: after.next_generation(seq_pop, fit, 4, 3, 0.9, 0.05, 2, random.Random(1)),
           "str genomes with no sequence model active raises", "no sequence GenomeModel active")
    gm.set_active(gm.build_genome_model(HPGAConfig(sequence="HPHPHPHPHP")))
    raises(lambda: after.next_generation(seq_pop, fit, 4, 3, 0.9, 0.05, 2, random.Random(1)),
           "str genomes with a LATTICE model active raises", "no sequence GenomeModel active")
    gm.set_active(None)
    # lattice, with the flags set: unchanged behaviour (the guard never runs for lists)
    os.environ["HPGA_AGENTS_ENABLED"] = "1"
    try:
        after.next_generation([[0, 1, 2, 3]] * 4, fit, 4, 3, 0.9, 0.05, 2, random.Random(1))
        check("lattice + HPGA_AGENTS_ENABLED=1 in deterministic mode still runs (unchanged)", True)
    except Exception as e:  # noqa: BLE001
        check("lattice + HPGA_AGENTS_ENABLED=1 in deterministic mode still runs (unchanged)", False, repr(e))
    os.environ.pop("HPGA_AGENTS_ENABLED")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="main")
    args = ap.parse_args()
    print(f"baseline = {args.baseline}")

    print("A. lattice parity (deterministic path)")
    before = load_baseline(args.baseline, "ops_before")
    check("baseline vs working tree: identical on every case", run_parity(before, "before-vs-after") == 0)
    mutants = {
        "tournament_select returns an alias": ("return list(population[best])", "return population[best]"),
        "tournament_select samples in another order": ("idxs = rng.sample(range(len(population)), k)",
                                                        "idxs = rng.sample(range(len(population)), k)[::-1]"),
    }
    for name, m in mutants.items():
        mut = load_baseline(args.baseline, "ops_mutant", mutate=m)
        check(f"self-check: mutant '{name}' is detected", run_parity(mut, f"mutant {name}") > 0)

    print("B. sequence mode")
    run_sequence_checks()
    print("C. loud failures")
    run_failure_checks()

    print(f"\n{'ALL CHECKS PASSED' if not FAILS else str(len(FAILS)) + ' CHECK(S) FAILED: ' + '; '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
