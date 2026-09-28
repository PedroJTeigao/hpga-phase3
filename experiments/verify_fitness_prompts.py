"""Verifier for hpga/sequence_model_fitness.py: the fitness line is the ONLY
difference from the prompts arm C was measured with, and fitness=None is arm C
exactly.

The lattice parity harness (verify_llm_operator_parity.py) covers
operators.py's shared edits; it speaks S/L/R/U/D and does not reach the
sequence prompts, which is what this file checks.

  print   both prompts verbatim, all three cases (crossover, mutate
          inherited, mutate own), plus one retry-hint rendering
  check   over every generated case:
            1. strip_fitness_lines(fitness prompt) == base prompt, byte for byte
            2. the fitness prompt is exactly N lines longer, and the inserted
               lines sit immediately after the sequence they describe
            3. fitness=None returns a plan whose every field -- system, parse,
               num_predict, retry_hint_text AND build_prompt output -- equals
               sequence_model.py's plan
            4. only build_prompt differs when a fitness context IS passed
            5. log_fields() records basis and parent count for every case
            6. a base prompt whose anchor line moved raises rather than
               inserting the line in the wrong place
            7. the two out-of-scope styles raise NotImplementedError

Usage:  python experiments/verify_fitness_prompts.py {print,check}
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hpga import sequence_model as sm  # noqa: E402
from hpga import sequence_model_fitness as smf  # noqa: E402

RETRY_HINTS = ("", "\nIMPORTANT: your previous response did not match the required format.")

# plan_llm_* builds a fresh `parse` closure on every call, so two plans never
# share one object and identity proves nothing. Parity is therefore checked
# behaviourally: both plans' parsers must agree on every probe response,
# valid and malformed, including the near-miss rejections that decide the
# invalid-output rate this arm reports.


def _mutate_probes(genome: str, k: int) -> list[str]:
    def other(c):
        return "W" if c != "W" else "Y"

    valid = "\n".join(f"POSITION: {i}, NEW: {other(genome[i])}" for i in range(k))
    return [
        valid,                                                        # valid
        valid + f"\nPOSITION: {k}, NEW: {other(genome[k])}",          # one line too many
        "\n".join(f"POSITION: 0, NEW: {other(genome[0])}" for _ in range(k)),  # repeated position
        f"POSITION: 0, NEW: {genome[0]}",                             # NEW == current letter
        f"POSITION: {len(genome)}, NEW: W",                           # out of range
        "MUTATED: " + " ".join(genome),                               # wrong style's format
        "I changed a few residues in the middle.",                    # prose
        "",
    ]


def _crossover_probes(p1: str, p2: str) -> list[str]:
    return [
        "SEGMENTS: 0-40:1, 40-100:2",          # valid
        "SEGMENTS: 0-100:1",                   # one segment, one parent
        "SEGMENTS: 0-30:1, 40-100:2",          # gap
        "SEGMENTS: 0-50:1, 40-100:2",          # overlap
        "SEGMENTS: 0-40:1, 40-90:2",           # does not reach 100
        "CHILD1: " + " ".join(p1) + "\nCHILD2: " + " ".join(p2),  # wrong style's format
        "I recombined the parents in the middle.",
        "",
    ]


def cases():
    """(label, kind, args) over lengths, k, fitness pairs and retry hints."""
    rng = random.Random(12345)
    out = []
    for n1 in (30, 42, 63, 80):
        for n2 in (30, 57, 80):
            g1, g2 = sm.random_sequence(rng, n1), sm.random_sequence(rng, n2)
            for f1, f2 in ((0.0, 1.0), (0.24329, 0.46325), (0.5, 0.5), (0.41234, 0.38912)):
                for hint in RETRY_HINTS:
                    out.append(("crossover", g1, g2, f1, f2, hint))
                    for k in (1, 3, 8):
                        # inherited: a child that differs from both parents
                        child = ("W" if g1[0] != "W" else "Y") + g1[1:]
                        out.append(("mutate_inherited", child, g1, g2, f1, f2, k, hint))
                        # own: the child IS parent 1
                        out.append(("mutate_own", g1, g1, g2, f1, f2, k, hint))
    return out


def check_one(case) -> list[str]:
    fails = []
    model = smf.FitnessAwareSequenceGenomeModel.__new__(smf.FitnessAwareSequenceGenomeModel)

    if case[0] == "crossover":
        _, g1, g2, f1, f2, hint = case
        fbg = {g1: f1, g2: f2}
        ctx = model.crossover_fitness_context(g1, g2, fbg)
        base = sm.plan_llm_crossover("segment", g1, g2)
        new = smf.plan_llm_crossover("segment", g1, g2, ctx)
        n_inserted, anchors = 2, [(0, smf._CROSSOVER_ANCHOR_1), (2, smf._CROSSOVER_ANCHOR_2)]
        expect_log = {"fitness_shown": True, "fitness_basis": "own",
                      "fitness_values_shown": [round(f1, 5), round(f2, 5)],
                      "n_parents_shown": 2, "fitness_decimals": 5}
        none_plan = smf.plan_llm_crossover("segment", g1, g2)
        probes = _crossover_probes(g1, g2)
    else:
        _, genome, g1, g2, f1, f2, k, hint = case
        fbg = {g1: f1, g2: f2}
        ctx = model.mutate_fitness_context(genome, g1, g2, fbg)
        base = sm.plan_llm_mutate("position", genome, k)
        new = smf.plan_llm_mutate("position", genome, k, ctx)
        n_inserted, anchors = 1, [(0, smf._MUTATE_ANCHOR)]
        own = case[0] == "mutate_own"
        expect_log = {"fitness_shown": True, "fitness_basis": "own" if own else "inherited",
                      "fitness_values_shown": [round(f1, 5)] if own else [round(f1, 5), round(f2, 5)],
                      "n_parents_shown": 0 if own else 2, "fitness_decimals": 5}
        none_plan = smf.plan_llm_mutate("position", genome, k)
        probes = _mutate_probes(genome, k)

    base_text, new_text = base.build_prompt(hint), new.build_prompt(hint)

    # 1. the fitness line is the only difference
    if smf.strip_fitness_lines(new_text) != base_text:
        fails.append("stripped fitness prompt != base prompt")

    # 2. exactly n_inserted lines longer, each immediately after its sequence
    bl, nl = base_text.split("\n"), new_text.split("\n")
    if len(nl) != len(bl) + n_inserted:
        fails.append(f"line count {len(nl)} != {len(bl)} + {n_inserted}")
    for anchor_idx, anchor_prefix in anchors:
        if not nl[anchor_idx].startswith(anchor_prefix):
            fails.append(f"line {anchor_idx} is not the expected sequence line")
        elif not any(nl[anchor_idx + 1].startswith(f"{lab}:") for lab in smf.FITNESS_LABELS):
            fails.append(f"line {anchor_idx + 1} is not a fitness line: {nl[anchor_idx + 1]!r}")

    # 3. fitness=None is sequence_model.py's plan, in every field
    for field in ("system", "num_predict", "retry_hint_text"):
        if getattr(none_plan, field) != getattr(base, field):
            fails.append(f"fitness=None changed {field}")
    if none_plan.build_prompt(hint) != base_text:
        fails.append("fitness=None changed the prompt")
    for i, probe in enumerate(probes):
        if repr(none_plan.parse(probe)) != repr(base.parse(probe)):
            fails.append(f"fitness=None changed parse on probe {i}")

    # 4. with a context passed, still only build_prompt differs
    for field in ("system", "num_predict", "retry_hint_text"):
        if getattr(new, field) != getattr(base, field):
            fails.append(f"fitness context changed {field}")
    for i, probe in enumerate(probes):
        if repr(new.parse(probe)) != repr(base.parse(probe)):
            fails.append(f"fitness context changed parse on probe {i}")

    # 5. log fields
    if ctx.log_fields() != expect_log:
        fails.append(f"log_fields {ctx.log_fields()} != {expect_log}")

    return fails


def check_anchor_guard() -> list[str]:
    """A base prompt whose anchor moved must raise, not misplace the line."""
    fails = []
    try:
        smf._insert_lines("Some other opening line\n\nrest", [(0, "Fitness: 1.0", smf._MUTATE_ANCHOR)])
        fails.append("moved anchor did not raise")
    except RuntimeError:
        pass
    try:
        smf._insert_lines("", [(5, "Fitness: 1.0", smf._MUTATE_ANCHOR)])
        fails.append("out-of-range anchor did not raise")
    except RuntimeError:
        pass
    return fails


def check_style_scope() -> list[str]:
    """Only arm C's two styles are defined; the rest must raise."""
    fails = []
    ctx_m = smf.MutateFitness((0.4,), "own")
    ctx_x = smf.CrossoverFitness(0.4, 0.3)
    g = "A" * 40
    for style in ("full", ""):
        try:
            smf.plan_llm_mutate(style, g, 2, ctx_m)
            fails.append(f"mutate style {style!r} did not raise")
        except NotImplementedError:
            pass
        try:
            smf.plan_llm_crossover(style, g, g, ctx_x)
            fails.append(f"crossover style {style!r} did not raise")
        except NotImplementedError:
            pass
    return fails


def cmd_print() -> None:
    rng = random.Random(0)
    g1, g2 = sm.random_sequence(rng, 63), sm.random_sequence(rng, 58)
    f1, f2 = 0.41234, 0.38912
    model = smf.FitnessAwareSequenceGenomeModel.__new__(smf.FitnessAwareSequenceGenomeModel)
    fbg = {g1: f1, g2: f2}
    child = ("W" if g1[0] != "W" else "Y") + g1[1:]

    blocks = [
        ("CROSSOVER, style 'segment' (both numbers exact)",
         smf.plan_llm_crossover("segment", g1, g2, model.crossover_fitness_context(g1, g2, fbg)).build_prompt("")),
        ("MUTATE, style 'position', basis 'inherited' (child differs from both parents)",
         smf.plan_llm_mutate("position", child, 3,
                             model.mutate_fitness_context(child, g1, g2, fbg)).build_prompt("")),
        ("MUTATE, style 'position', basis 'own' (child is byte-identical to parent 1)",
         smf.plan_llm_mutate("position", g1, 3,
                             model.mutate_fitness_context(g1, g1, g2, fbg)).build_prompt("")),
    ]
    for title, text in blocks:
        print(f"{'=' * 78}\n{title}\n{'=' * 78}")
        print(text)
        print()
    print(f"{'=' * 78}\nMUTATE 'inherited' WITH THE RETRY HINT APPENDED\n{'=' * 78}")
    print(smf.plan_llm_mutate("position", child, 3,
                              model.mutate_fitness_context(child, g1, g2, fbg)).build_prompt(
        sm.plan_llm_mutate("position", child, 3).retry_hint_text))


def cmd_check() -> int:
    all_cases = cases()
    failed = 0
    for case in all_cases:
        fails = check_one(case)
        if fails:
            failed += 1
            if failed <= 5:
                print(f"FAIL {case[0]} len={len(case[1])}: {fails}")
    guard = check_anchor_guard() + check_style_scope()
    print(f"cases checked: {len(all_cases)}  failed: {failed}")
    print(f"anchor-guard and style-scope checks: {'PASS' if not guard else guard}")
    ok = failed == 0 and not guard
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("print", "check"):
        raise SystemExit(__doc__)
    sys.exit(cmd_check() if sys.argv[1] == "check" else (cmd_print() or 0))


if __name__ == "__main__":
    main()
