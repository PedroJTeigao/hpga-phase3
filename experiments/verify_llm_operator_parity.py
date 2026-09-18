"""Behavioural-parity harness for hpga/operators.py's LLM operators: a
baseline (a git ref's operators.py) vs. the working tree, driven identically
through a stubbed _call_ollama. Built for the GenomeModel-dispatch refactor,
where the constraint was that the experiment scripts' behaviour -- and hence
the published results behind them -- must not change; kept because the next
refactor of this file will want the same guarantee.

The stub's reply is a pure function of (seed, case inputs), never of call
count, so a refactor that changes RNG draw order, prompt text, num_predict,
retry behaviour or log fields diverges on the very next prompt or record.
Compared per case: return value (repr, so list-vs-tuple counts), raised
exception, stats summary, RNG state after the call, every (prompt, system,
num_predict, seed) the stub saw, and every JSONL log record minus timestamp.
4,410 direct-call cases (7 style values x 7 lengths x 5 rates, mixed
valid/malformed/near-echo/transport-error replies) + 13 next_generation runs
(P=1 and P=3), plus a check that every name the baseline module defined still
exists with an equal value.

Usage (from anywhere):
  python experiments/verify_llm_operator_parity.py [--baseline REF] MODE
    --baseline REF   git ref whose hpga/operators.py is "before" (default HEAD,
                     i.e. "did my uncommitted change alter behaviour?"). The
                     GenomeModel-dispatch refactor's own baseline is 50de6f8.
    MODE:
      self     baseline vs. a second load of itself -- proves determinism
      check    baseline vs. working tree, no GenomeModel active (what every
               existing experiment script sees)
      active   same, with a lattice GenomeModel set_active'd
      mutants  baseline vs. deliberately broken copies of it; every mutant
               MUST diverge, or the harness is too weak to trust

What is reusable and what is not:
  * Reusable as-is for any change to operators.py's LLM path on the 5-symbol
    lattice genome: case generation, reply factory, comparison, namespace check.
  * Tied to the lattice formats: the reply factory speaks CHILD1/CHILD2,
    MUTATED, CHANGES, POSITION and SEGMENTS over S/L/R/U/D. If those formats
    change, update it -- a parity harness that stops exercising the parsers
    silently proves nothing.
  * The MUTANTS table is tied to the source text of the baseline it was
    written against (50de6f8). Against a different baseline, patterns that no
    longer appear are reported SKIPPED; the mode still fails if any applicable
    mutant escapes, or if none were applicable (validating nothing). Refresh
    the table when you change the baseline.
"""

import importlib
import importlib.util
import itertools
import json
import os
import random
import re
import sys
import subprocess
import tempfile
import hashlib
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

BASELINE_REF = "HEAD"
_WORKDIR = Path(tempfile.mkdtemp(prefix="llm_op_parity_"))

CH = "SLRUD"


# --------------------------------------------------------------------- loading

def load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # @dataclass needs this
    spec.loader.exec_module(mod)
    return mod


def baseline_source() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT), "show", f"{BASELINE_REF}:hpga/operators.py"],
        capture_output=True, text=True, check=True,
    ).stdout


def load_before(name="ops_before"):
    path = _WORKDIR / f"{name}.py"
    path.write_text(baseline_source())
    return load_file(name, path)


def load_after():
    return importlib.import_module("hpga.operators")


# ------------------------------------------------------------ response factory

def L(g):
    return " ".join(CH[x] for x in g)


def rand_other(r, cur):
    return r.choice([c for c in range(5) if c != cur])


def crossover_reply(kind, r, p1, p2):
    n = len(p1)
    if kind.startswith("full") or kind == "chatter":
        cut = r.randint(1, max(1, n - 1))
        c1, c2 = p1[:cut] + p2[cut:], p2[:cut] + p1[cut:]
        for _ in range(r.randint(0, 2)):  # jitter
            c1 = list(c1)
            if n:
                c1[r.randrange(n)] = r.randrange(5)
        if kind == "full_echo":
            c1, c2 = list(p1), list(p2)
        if kind == "full_wronglen" and c1:
            c1 = c1[:-1]
        if kind == "full_onlyone":
            return f"CHILD1: {L(c1)}"
        if kind == "full_lower":
            return f"CHILD1: {', '.join(CH[x].lower() for x in c1)}\nCHILD2: {L(c2).lower()}"
        txt = f"CHILD1: {L(c1)}\nCHILD2: {L(c2)}"
        return ("Sure! Here you go:\n" + txt + "\nHope that helps.") if kind == "chatter" else txt
    if kind.startswith("diff"):
        def side(label, base):
            if kind == "diff_none":
                return f"{label}: none"
            m = r.randint(0, 3)
            pos = r.sample(range(n), min(m, n)) if n else []
            if kind == "diff_oor":
                pos = pos[:1] + [n]
            if not pos:
                return f"{label}: none."
            return f"{label}: " + ", ".join(f"{p}:{CH[r.randrange(5)]}" for p in pos)
        return side("CHILD1 diff from Parent1", p1) + "\n" + side("CHILD2 diff from Parent2", p2)
    if kind.startswith("seg"):
        if kind == "seg_single":
            return f"SEGMENTS: 0-{n - 1}:1"
        cuts = sorted(r.sample(range(1, n), min(r.randint(1, 3), max(0, n - 1)))) if n > 1 else []
        bounds = [0] + cuts + [n]
        segs = [(bounds[i], bounds[i + 1] - 1) for i in range(len(bounds) - 1)]
        owners = [r.choice([1, 2]) for _ in segs]
        if kind == "seg_oneparent":
            owners = [1] * len(segs)
        if kind == "seg_valid" and len(segs) >= 2 and set(owners) != {1, 2}:
            owners[0], owners[-1] = 1, 2
        parts = [f"{a}-{b}:{o}" for (a, b), o in zip(segs, owners)]
        if kind == "seg_gap" and len(parts) > 1:
            a, b = segs[0]
            parts[0] = f"{a}-{max(a, b - 1)}:{owners[0]}" if b > a else parts[0]
            if b == a:
                parts[0] = f"{a + 1}-{b + 1}:{owners[0]}"
        if kind == "seg_overlap" and len(parts) > 1:
            a, b = segs[1]
            parts[1] = f"{a - 1}-{b}:{owners[1]}"
        if kind == "seg_unsorted":
            r.shuffle(parts)
        return "SEGMENTS: " + ", ".join(parts)
    if kind == "empty":
        return ""
    return "I'm sorry, I can't help with that request."


def mutate_reply(kind, r, g, k):
    n = len(g)
    if kind.startswith("mfull"):
        kk = k + r.choice([1, -1]) if kind == "mfull_wrongk" else k
        kk = max(0, min(n, kk))
        out = list(g)
        for p in r.sample(range(n), kk):
            out[p] = rand_other(r, g[p])
        if kind == "mfull_lower":
            return "MUTATED: " + ",".join(CH[x].lower() for x in out)
        return "MUTATED: " + L(out)
    if kind.startswith("mdiff"):
        if kind == "mdiff_none":
            return "CHANGES: none"
        kk = k + 1 if kind == "mdiff_wrongk" else k
        kk = max(0, min(n, kk))
        pos = r.sample(range(n), kk)
        if kind == "mdiff_samelet":
            return "CHANGES: " + ", ".join(f"{p}:{CH[g[p]]}" for p in pos)
        return "CHANGES: " + ", ".join(f"{p}:{CH[rand_other(r, g[p])]}" for p in pos)
    if kind.startswith("mpos"):
        kk = k + 1 if kind == "mpos_extra" else k
        kk = max(0, min(n, kk))
        pos = r.sample(range(n), kk)
        lines = []
        for p in pos:
            letter = CH[g[p]] if kind == "mpos_same" else CH[rand_other(r, g[p])]
            if kind == "mpos_oor":
                p = n
            if kind == "mpos_lower":
                lines.append(f"position : {p} , new : {letter.lower()}")
            else:
                lines.append(f"POSITION: {p}, NEW: {letter}")
        if kind == "mpos_dup" and lines:
            lines = lines + [lines[0]] if len(lines) < k + 1 else lines
            lines = lines[:k] if k > 1 else lines
            if k > 1:
                lines[-1] = lines[0]
        return "\n".join(lines)
    if kind == "empty":
        return ""
    return "I'm sorry, I can't help with that request."


CROSS_PREF = {
    "full": ["full_valid", "full_valid", "full_lower", "full_echo", "full_wronglen", "full_onlyone", "chatter"],
    "diff": ["diff_valid", "diff_valid", "diff_none", "diff_oor"],
    "segment": ["seg_valid", "seg_valid", "seg_unsorted", "seg_gap", "seg_single", "seg_oneparent", "seg_overlap"],
}
CROSS_PREF["best"] = CROSS_PREF["segment"]
CROSS_ALL = sorted({k for v in CROSS_PREF.values() for k in v}) + ["garbage", "empty"]
MUT_PREF = {
    "full": ["mfull_valid", "mfull_valid", "mfull_wrongk", "mfull_lower"],
    "diff": ["mdiff_valid", "mdiff_valid", "mdiff_none", "mdiff_wrongk", "mdiff_samelet"],
    "position": ["mpos_valid", "mpos_valid", "mpos_extra", "mpos_same", "mpos_oor", "mpos_dup", "mpos_lower"],
}
MUT_PREF["best"] = MUT_PREF["position"]
MUT_ALL = sorted({k for v in MUT_PREF.values() for k in v}) + ["garbage", "empty"]


def style_bucket(style, prefs):
    return prefs.get(style, prefs["full"])  # unknown style falls through to 'full'


def make_stub(ctx, calls):
    """ctx: dict with op, style, p1, p2, genome, k, raise_on (attempt idx or None)."""
    def stub(prompt, system, num_predict, seed):
        calls.append((prompt, system, num_predict, seed))
        r = random.Random(seed)
        if ctx.get("raise_on") is not None and len(calls) - 1 == ctx["raise_on"]:
            raise ConnectionError("simulated")
        if ctx["op"] == "crossover":
            pool = style_bucket(ctx["style"], CROSS_PREF) if r.random() < 0.6 else CROSS_ALL
            text = crossover_reply(r.choice(pool), r, ctx["p1"], ctx["p2"])
        else:
            pool = style_bucket(ctx["style"], MUT_PREF) if r.random() < 0.6 else MUT_ALL
            text = mutate_reply(r.choice(pool), r, ctx["genome"], ctx["k"])
        return text, 0.001, 10, 5
    return stub


# ---------------------------------------------------------------------- driving

def read_log(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        rec.pop("timestamp", None)
        out.append(rec)
    return out


def rng_fingerprint(r):
    return hashlib.sha256(repr(r.getstate()).encode()).hexdigest()[:16]


def run_case(mod, case):
    """One direct _llm_crossover/_llm_mutate call. Returns a comparable dict."""
    mod.reset_operator_stats()
    mod.CROSSOVER_MIN_DIFF = case["min_diff"]
    mod.LLM_MAX_RETRIES = case["max_retries"]
    os.environ["HPGA_LLM_PROMPT_STYLE"] = case["style"]
    fd, logpath = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.unlink(logpath)
    os.environ["HPGA_LLM_LOG_PATH"] = logpath
    calls = []
    ctx = dict(case)
    mod._call_ollama = make_stub(ctx, calls)
    rng = random.Random(case["seed"])
    result, exc = None, None
    try:
        if case["op"] == "crossover":
            kwargs = {"generation": case["generation"]} if case["generation"] is not None else {}
            result = mod._llm_crossover(list(case["p1"]), list(case["p2"]), case["rate"], rng, **kwargs)
        else:
            result = mod._llm_mutate(list(case["genome"]), case["rate"], rng)
    except Exception as e:  # noqa: BLE001
        exc = f"{type(e).__name__}: {e}"
    rec = {
        "result": repr(result), "exc": exc, "stats": mod.get_operator_stats(),
        "rng": rng_fingerprint(rng), "calls": calls, "log": read_log(logpath),
    }
    if os.path.exists(logpath):
        os.unlink(logpath)
    return rec


def build_cases():
    cases = []
    styles = ["full", "diff", "position", "segment", "best", "bogus", ""]
    lengths = [1, 2, 3, 5, 8, 18, 30]
    rates = [0.0, 0.05, 0.3, 0.7, 1.0]
    cid = 0
    for style, n, rate in itertools.product(styles, lengths, rates):
        for inst in range(9):
            cid += 1
            r = random.Random(1000 + cid)
            p1 = [r.randrange(5) for _ in range(n)]
            if inst % 3 == 0:
                p2 = list(p1)                                   # identical parents
            elif inst % 3 == 1:
                p2 = list(p1)
                if n:
                    p2[r.randrange(n)] = r.randrange(5)          # near-identical
            else:
                p2 = [r.randrange(5) for _ in range(n)]
            base = dict(style=style, rate=rate, seed=50000 + cid, p1=p1, p2=p2, genome=p1,
                        k=max(1, round(rate * n)), min_diff=[2, 0, 5][cid % 3],
                        max_retries=[2, 2, 0, 1][cid % 4], raise_on=None, generation=None)
            cases.append(dict(base, op="crossover", generation=[None, 0, 7][cid % 3],
                              raise_on=[None, None, None, 0, 1][cid % 5]))
            cases.append(dict(base, op="mutate", raise_on=[None, None, None, 0, 1][(cid + 2) % 5]))
    return cases


def run_next_generation(mod, spec):
    from hpga import agents
    agents.reset_agent_state()
    mod.reset_operator_stats()
    mod.CROSSOVER_MIN_DIFF = 2
    mod.LLM_MAX_RETRIES = 2
    os.environ["HPGA_OPERATOR_MODE"] = spec["mode"]
    os.environ["HPGA_GA_DISPATCH_P"] = str(spec["P"])
    os.environ["HPGA_LLM_PROMPT_STYLE"] = spec["style"]
    for v in ("HPGA_AGENTS_ENABLED", "HPGA_CIRCLES_ENABLED", "HPGA_LOG_DIVERSITY"):
        os.environ.pop(v, None)
    fd, logpath = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.unlink(logpath)
    os.environ["HPGA_LLM_LOG_PATH"] = logpath
    calls = []

    # Stub keyed only on (seed, prompt) so thread scheduling under P>1 can't matter.
    def stub(prompt, system, num_predict, seed):
        calls.append((prompt, system, num_predict, seed))
        r = random.Random(seed)
        genomes = [[CH.index(c) for c in line.replace(" ", "")]
                   for line in re.findall(r"(?:Parent \d|Genome)[^:\n]*: ([SLRUD ]+)", prompt)]
        if "Parent 1" in prompt:
            p1, p2 = genomes[0], genomes[1]
            pool = style_bucket(spec["style"], CROSS_PREF) if r.random() < 0.7 else CROSS_ALL
            return crossover_reply(r.choice(pool), r, p1, p2), 0.001, 10, 5
        g = genomes[0]
        m = re.search(r"EXACTLY (\d+)", prompt) or re.search(r"exactly (\d+)", prompt)
        k = int(m.group(1)) if m else max(1, round(spec["mrate"] * len(g)))
        pool = style_bucket(spec["style"], MUT_PREF) if r.random() < 0.7 else MUT_ALL
        return mutate_reply(r.choice(pool), r, g, k), 0.001, 10, 5

    mod._call_ollama = stub
    r0 = random.Random(spec["seed"])
    n = spec["length"]
    pop = [[r0.randrange(5) for _ in range(n)] for _ in range(spec["pop"])]
    fit = [r0.random() for _ in pop]
    rng = random.Random(spec["seed"] + 1)
    gens = []
    exc = None
    try:
        for _ in range(spec["gens"]):
            pop = mod.next_generation(pop, fit, spec["pop"], 3, spec["crate"], spec["mrate"], 2, rng)
            fit = [r0.random() for _ in pop]
            gens.append(repr(pop))
    except Exception as e:  # noqa: BLE001
        exc = f"{type(e).__name__}: {e}"
    log = read_log(logpath)
    if spec["P"] > 1:  # interleaving is nondeterministic; content must still match
        log = sorted(log, key=lambda x: json.dumps(x, sort_keys=True, default=str))
        cs = sorted(calls)
    else:
        cs = calls
    if os.path.exists(logpath):
        os.unlink(logpath)
    return {"gens": gens, "exc": exc, "stats": mod.get_operator_stats(), "rng": rng_fingerprint(rng),
            "calls": cs, "log": log}


def build_ng_specs():
    specs = []
    i = 0
    for style, P, mode in itertools.product(["full", "diff", "position", "segment", "best", "bogus"],
                                            [1, 3], ["llm"]):
        i += 1
        specs.append(dict(style=style, P=P, mode=mode, seed=900 + i, length=10, pop=8, gens=3,
                          crate=[0.9, 0.5][i % 2], mrate=[0.2, 0.05][i % 2]))
    specs.append(dict(style="full", P=1, mode="deterministic", seed=7, length=10, pop=8, gens=3, crate=0.9, mrate=0.1))
    return specs


def namespace_check(before, after):
    """Every name `before` defines must still exist in `after`; plain-data
    constants must be equal (regexes compared by pattern+flags)."""
    problems = []
    for name, val in vars(before).items():
        if name.startswith("__"):
            continue
        if not hasattr(after, name):
            problems.append(f"missing name: {name}")
            continue
        av = getattr(after, name)
        if isinstance(val, (str, int, float, tuple, dict, list, bool, type(None))):
            if name in ("_DEFAULT_RUN_ID", "_stats", "_client", "CROSSOVER_MIN_DIFF", "LLM_MAX_RETRIES"):
                continue  # process-varying / harness-mutated
            if val != av:
                problems.append(f"value differs: {name}")
        elif isinstance(val, re.Pattern):
            if (val.pattern, val.flags) != (av.pattern, av.flags):
                problems.append(f"regex differs: {name}")
    return problems


def compare(before, after, label, activate=None):
    n_bad = 0
    diffs = []
    cases = build_cases()
    stats_cover = {"valid_first": 0, "fallback": 0, "retry_success": 0, "raised": 0}
    for i, case in enumerate(cases):
        if activate:
            activate(True)
        b = run_case(before, case)
        a = run_case(after, case)
        if b != a:
            n_bad += 1
            if len(diffs) < 5:
                keys = [k for k in b if b[k] != a[k]]
                diffs.append((i, {k: case[k] for k in ("op", "style", "rate", "seed", "generation", "min_diff", "max_retries", "raise_on")}, keys))
        s = b["stats"]
        if b["exc"]:
            stats_cover["raised"] += 1
        elif s["n_failures"]:
            stats_cover["fallback"] += 1
        elif s["n_retries"]:
            stats_cover["retry_success"] += 1
        elif s["n_llm_calls"]:
            stats_cover["valid_first"] += 1
    ng_bad = 0
    ng_specs = build_ng_specs()
    for spec in ng_specs:
        b = run_next_generation(before, spec)
        a = run_next_generation(after, spec)
        if b != a:
            ng_bad += 1
            diffs.append(("next_generation", spec, [k for k in b if b[k] != a[k]]))
    ns = namespace_check(before, after)
    print(f"[{label}] direct-call cases: {len(cases)}  diverged: {n_bad}   coverage(before side): {stats_cover}")
    print(f"[{label}] next_generation specs: {len(ng_specs)}  diverged: {ng_bad}")
    print(f"[{label}] namespace problems: {len(ns)} {ns[:5]}")
    for d in diffs[:6]:
        print("   DIFF", d)
    return n_bad + ng_bad + len(ns)


# ----------------------------------------------------------------------- mutants

MUTANTS = {
    "prompt_word_in_full_mutate": ("Change EXACTLY {k} of the {length} positions to a different letter from\n{{S,L,R,U,D}}; leave the other",
                                   "Change EXACTLY {k} of the {length} positions to a different letter from\n{{S,L,R,U,D}}; keep the other"),
    "retry_hint_diff_text": ('"restate the full sequence."', '"re-state the full sequence."'),
    "segment_num_predict": ("num_predict = min(2048, max(32, 6 * length))\n\n        def build_prompt(retry_hint: str) -> str:\n            return _crossover_segment_prompt",
                            "num_predict = min(2048, max(32, 7 * length))\n\n        def build_prompt(retry_hint: str) -> str:\n            return _crossover_segment_prompt"),
    "mixed_gate_strictness": ("_diff_count(c1, p1) >= CROSSOVER_MIN_DIFF", "_diff_count(c1, p1) > CROSSOVER_MIN_DIFF"),
    "extra_rng_draw_in_mutate": ("    length = len(genome)\n    genome_str = _genome_to_str(genome)\n    style = _prompt_style()\n    k = max(1, round(rate * length))",
                                 "    length = len(genome)\n    genome_str = _genome_to_str(genome)\n    style = _prompt_style()\n    rng.random()\n    k = max(1, round(rate * length))"),
    "crossover_fallback_rate": ("        return crossover(parent1, parent2, rate, rng)", "        return crossover(parent1, parent2, 1.0, rng)"),
    "log_field_rename": ('"op": op, "prompt_style": style, "attempt": attempt, "seed": seed,\n            "temperature": LLM_TEMPERATURE,\n            "model": LLM_MODEL, "prompt": prompt, "response": text,',
                         '"op": op, "prompt_style": style, "attempt": attempt, "seed": seed,\n            "temperature": LLM_TEMPERATURE,\n            "model": LLM_MODEL, "prompt": prompt, "raw_response": text,'),
    "identical_parents_check": ("identical_parents = list(parent1) == list(parent2)", "identical_parents = parent1 == parent2 and False"),
    "position_parse_same_letter_allowed": ("        if letter_code == base[pos]:\n            return None\n", ""),
    "unknown_style_falls_to_diff": ('    elif style == "diff":\n        num_predict = min(2048, max(48, 3 * length))', '    elif style in ("diff", "bogus"):\n        num_predict = min(2048, max(48, 3 * length))'),
}


def run_mutants():
    src = baseline_source()
    before = load_before()
    escaped = applicable = 0
    for name, (old, new) in MUTANTS.items():
        if old not in src:
            print(f"  mutant {name:38s} SKIPPED (pattern absent from baseline {BASELINE_REF})")
            continue
        applicable += 1
        path = _WORKDIR / "ops_mutant.py"
        path.write_text(src.replace(old, new, 1))
        mut = load_file("ops_mutant", path)
        n = compare_quiet(before, mut)
        verdict = "DETECTED" if n else "ESCAPED"
        if not n:
            escaped += 1
        print(f"  mutant {name:38s} {verdict} ({n} divergences)")
    if applicable == 0:
        print("  NO applicable mutants for this baseline -- harness is unvalidated; refresh MUTANTS")
        return 1
    return escaped


def compare_quiet(before, after):
    bad = 0
    for case in build_cases():
        if run_case(before, case) != run_case(after, case):
            bad += 1
    for spec in build_ng_specs():
        if run_next_generation(before, spec) != run_next_generation(after, spec):
            bad += 1
    return bad


def main():
    global BASELINE_REF
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--baseline", default="HEAD", help="git ref for the 'before' operators.py (default HEAD)")
    ap.add_argument("mode", choices=["self", "check", "active", "mutants"])
    args = ap.parse_args()
    BASELINE_REF = args.baseline
    mode = args.mode
    print(f"baseline={BASELINE_REF}  mode={mode}")
    before = load_before()
    if mode == "self":
        sys.exit(1 if compare(before, load_before("ops_before2"), "self") else 0)
    if mode == "mutants":
        esc = run_mutants()
        print("ALL APPLICABLE MUTANTS DETECTED" if not esc else f"{esc} PROBLEM(S) -- harness not trustworthy for this baseline")
        sys.exit(1 if esc else 0)
    after = load_after()
    if mode == "check":
        try:
            from hpga import genome_model as gm
            gm.set_active(None)
        except (ImportError, AttributeError):
            pass  # baseline predates genome_model
        sys.exit(1 if compare(before, after, "check (no active model)") else 0)
    if mode == "active":
        from hpga import genome_model as gm
        from hpga.config import HPGAConfig
        gm.set_active(gm.build_genome_model(HPGAConfig(sequence="HPHPHPHPHP")))
        sys.exit(1 if compare(before, after, "active lattice model") else 0)


if __name__ == "__main__":
    try:
        main()
    finally:
        import shutil
        shutil.rmtree(_WORKDIR, ignore_errors=True)
