"""Tests for the sequence-model LLM prompt styles (hpga/sequence_model.py's
plan_llm_crossover / plan_llm_mutate) and for the dispatch guards in
hpga/operators.py's _model_for -- no live Ollama server needed.

Drives the real operators._llm_crossover / _llm_mutate (retry, fallback,
logging and stats included) through an active SequenceGenomeModel with
_call_ollama replaced by a stub, so what's tested is the same code path a live
run takes, minus the network. Covers: each implemented style's success path
and every rejection rule in its parser; the 20-letter label-swallowing hazard
in the full-style parser; the percentage-boundary segment convention; that
diff/best raise NotImplementedError and unknown names raise ValueError while
cross-operator names fall through to 'full'; that gate/fallback paths return
str (not a list of characters); and the model-resolution guards (str genome
with no active model, genome_type mismatch, lattice legacy fallback).

Not covered here, by design: lattice behaviour (that is
verify_llm_operator_parity.py's job) and live-model compliance (that is
probe_sequence_operator_compliance.py's).

Exits non-zero on any failure. Never touches results/raw: the operator log is
pointed at a temp directory before the first call.
"""
import os, random, shutil, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hpga import operators as ops, genome_model as gm, sequence_model as sm
from hpga.config import HPGAConfig

_TMP = tempfile.mkdtemp(prefix="test_sequence_llm_")
os.environ["HPGA_LLM_LOG_PATH"] = os.path.join(_TMP, "calls.jsonl")
_SAVED = {"call": ops._call_ollama, "retries": ops.LLM_MAX_RETRIES,
          "style": os.environ.get("HPGA_LLM_PROMPT_STYLE")}
A = sm.ALPHABET
fails = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -- {detail}"))
    if not cond: fails.append(name)

def seq(r, n): return "".join(r.choice(A) for _ in range(n))
def sp(s): return " ".join(s)

def run(op, style, reply, *args, retries=2, seed=0):
    """Run one op with a stub replying `reply` (str, list of str, or callable(prompt))."""
    os.environ["HPGA_LLM_PROMPT_STYLE"] = style
    ops.LLM_MAX_RETRIES = retries
    ops.reset_operator_stats()
    seen = []
    def stub(prompt, system, num_predict, seed_):
        seen.append((prompt, system, num_predict))
        r = reply(prompt) if callable(reply) else (reply[min(len(seen)-1, len(reply)-1)] if isinstance(reply, list) else reply)
        return r, 0.001, 1, 1
    ops._call_ollama = stub
    rng = random.Random(seed)
    fn = ops._llm_crossover if op == "crossover" else ops._llm_mutate
    return fn(*args, rng) if op == "mutate" else fn(args[0], args[1], args[2], rng), ops.get_operator_stats(), seen

gm.set_active(gm.build_genome_model(HPGAConfig(genome_model="sequence")))
r = random.Random(42)
p1, p2 = seq(r, 60), seq(r, 45)

# ---- mutate full
g = seq(r, 50); k = max(1, round(0.1 * 50))
mut = list(g)
for i in r.sample(range(50), k): mut[i] = r.choice([c for c in A if c != g[i]])
mut = "".join(mut)
out, st, seen = run("mutate", "full", f"MUTATED: {sp(mut)}", g, 0.1)
check("mutate/full valid -> parsed str", out == mut and isinstance(out, str) and st["n_retries"] == 0, (out, st))
prompt = seen[0][0]
check("mutate/full prompt is space-separated 20-letter", sp(g) in prompt and "{A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y}" in prompt, prompt)
out, st, _ = run("mutate", "full", f"MUTATED: {sp(g)}", g, 0.1)  # zero changes
check("mutate/full no-op reply rejected -> fallback str", st["n_failures"] == 1 and isinstance(out, str) and len(out) == 50, st)
out, st, _ = run("mutate", "full", f"MUTATED: {sp(mut[:-1])}", g, 0.1)
check("mutate/full wrong length rejected", st["n_failures"] == 1, st)
out, st, _ = run("mutate", "full", f"MUTATED: {','.join(mut).lower()}", g, 0.1)
check("mutate/full lowercase+commas accepted", out == mut and st["n_retries"] == 0, (out, st))

# ---- mutate position
pos = [i for i in range(50) if mut[i] != g[i]]
reply = "\n".join(f"POSITION: {i}, NEW: {mut[i]}" for i in pos)
out, st, seen = run("mutate", "position", reply, g, 0.1)
check("mutate/position valid", out == mut and st["n_retries"] == 0, (out, st))
check("mutate/position prompt says k and range", f"exactly {k} position" in seen[0][0] and "0-49" in seen[0][0], seen[0][0])
bad = reply.replace(f"NEW: {mut[pos[0]]}", f"NEW: {g[pos[0]]}", 1)
out, st, _ = run("mutate", "position", bad, g, 0.1)
check("mutate/position same-letter rejected", st["n_failures"] == 1, st)
out, st, _ = run("mutate", "position", reply + f"\nPOSITION: 3, NEW: A", g, 0.1)
check("mutate/position wrong count rejected", st["n_failures"] == 1, st)
out, st, _ = run("mutate", "position", f"POSITION: 50, NEW: A\n" * k, g, 0.1)
check("mutate/position out-of-range rejected", st["n_failures"] == 1, st)

# ---- mutate: cross-op style falls through to full; diff/best/bogus raise
out, st, seen = run("mutate", "segment", f"MUTATED: {sp(mut)}", g, 0.1)
check("mutate with style 'segment' runs full", out == mut and "MUTATED:" in seen[0][0], (out, seen[0][0][:80]))
for style, exc in (("diff", NotImplementedError), ("best", NotImplementedError), ("bogus", ValueError)):
    try: run("mutate", style, "x", g, 0.1); check(f"mutate/{style} raises", False, "no raise")
    except exc: check(f"mutate/{style} raises {exc.__name__}", True)
out, st, seen = run("mutate", "", f"MUTATED: {sp(mut)}", g, 0.1)
check("mutate with empty style == full", out == mut, out)

# ---- crossover full
cut = 25
c1, c2 = p1[:cut] + p2[cut - 5:], p2[:cut - 5] + p1[cut:]
assert 30 <= len(c1) <= 80 and 30 <= len(c2) <= 80
out, st, seen = run("crossover", "full", f"CHILD1: {sp(c1)}\nCHILD2: {sp(c2)}", p1, p2, 1.0)
check("crossover/full valid -> (str,str), no retry", out == (c1, c2) and st["n_retries"] == 0, (out, st))
check("crossover/full prompt has both parents + bounds", sp(p1) in seen[0][0] and sp(p2) in seen[0][0] and "between 30 and 80" in seen[0][0])
out, st, _ = run("crossover", "full", f"CHILD1: {sp(p1)}\nCHILD2: {sp(p2)}", p1, p2, 1.0)
check("crossover/full parent echo rejected", st["n_failures"] == 1, st)
out, st, _ = run("crossover", "full", f"CHILD1: {sp(c1[:10])}\nCHILD2: {sp(c2)}", p1, p2, 1.0)
check("crossover/full child below MIN_LENGTH rejected", st["n_failures"] == 1, st)
out, st, _ = run("crossover", "full", f"CHILD1: {sp(c1)}\nCHILD2: {sp(c2 + c2)}", p1, p2, 1.0)
check("crossover/full child above MAX_LENGTH rejected", st["n_failures"] == 1, st)
# 20-letter hazard: CHILD2's label is made of valid amino letters; must not be swallowed into CHILD1's run
out, st, _ = run("crossover", "full", f"CHILD1: {sp(c1)}\nCHILD2: {sp(c2)}", p1, p2, 1.0)
check("crossover/full label-swallow hazard: CHILD1 has exactly its own letters", out[0] == c1)
check("_extract_labelled stops at newline", sm._extract_labelled("CHILD1: A C D\nCHILD2: E F", "CHILD1") == "ACD")
check("_extract_labelled trailing-period tolerated", sm._extract_labelled("CHILD1: A C D.", "CHILD1") == "ACD")
check("_extract_labelled missing label -> None", sm._extract_labelled("nothing here", "CHILD1") is None)

# ---- crossover segment
def seg_case(spec):
    return run("crossover", "segment", f"SEGMENTS: {spec}", p1, p2, 1.0)
out, st, seen = seg_case("0-50:1, 50-100:2")
exp1 = p1[:round(50*60/100)] + p2[round(50*45/100):]; exp2 = p2[:round(50*45/100)] + p1[round(50*60/100):]
check("crossover/segment 2-seg valid, built from declaration", out == (exp1, exp2) and st["n_retries"] == 0, (out, st))
check("crossover/segment prompt shows both lengths", "(length 60)" in seen[0][1 - 1] and "(length 45)" in seen[0][0])
out, st, _ = seg_case("0-30:1, 30-70:2, 70-100:1")
c1s = p1[:round(30*60/100)] + p2[round(30*45/100):round(70*45/100)] + p1[round(70*60/100):]  # exact int math, as the code does
check("crossover/segment 3-seg valid", out is not None and out[0] == c1s and st["n_retries"] == 0, (out, st))
out, st, _ = seg_case("70-100:1, 0-30:2, 30-70:1")
check("crossover/segment unsorted declaration accepted", st["n_retries"] == 0, st)
for name, spec in (("lattice-style inclusive (gap)", "0-49:1, 50-100:2"), ("single segment", "0-100:1"),
                   ("one parent", "0-50:1, 50-100:1"), ("doesn't start at 0", "10-50:1, 50-100:2"),
                   ("doesn't end at 100", "0-50:1, 50-90:2"), ("overlap", "0-60:1, 50-100:2"),
                   ("zero-width", "0-50:1, 50-50:2, 50-100:2")):
    out, st, _ = seg_case(spec)
    check(f"crossover/segment rejects {name}", st["n_failures"] == 1, st)
# child length out of bounds: tiny parents can't be cut into in-bound children
short1, short2 = seq(r, 30), seq(r, 30)
out, st, _ = run("crossover", "segment", "SEGMENTS: 0-50:1, 50-100:2", short1, short2, 1.0)
check("crossover/segment in-bounds at MIN_LENGTH parents", st["n_failures"] == 0, st)
# In-bounds parents can't yield an out-of-bounds child from one cut (convex combination), so
# tighten MAX_LENGTH for this case: 80/30 parents cut at 50% -> child1 = 40+15 = 55 > 50.
_saved = sm.MAX_LENGTH; sm.MAX_LENGTH = 50
try:
    out, st, _ = run("crossover", "segment", "SEGMENTS: 0-50:1, 50-100:2", seq(r, 80), seq(r, 30), 1.0)
finally:
    sm.MAX_LENGTH = _saved
check("crossover/segment child out of bounds rejected", st["n_failures"] == 1, (out, st))

# ---- crossover: cross-op fallthrough / unimplemented
out, st, seen = run("crossover", "position", f"CHILD1: {sp(c1)}\nCHILD2: {sp(c2)}", p1, p2, 1.0)
check("crossover with style 'position' runs full", out == (c1, c2) and "CHILD1:" in seen[0][0])
for style, exc in (("diff", NotImplementedError), ("best", NotImplementedError), ("bogus", ValueError)):
    try: run("crossover", style, "x", p1, p2, 1.0); check(f"crossover/{style} raises", False, "no raise")
    except exc: check(f"crossover/{style} raises {exc.__name__}", True)

# ---- gate + fallback return str, not list-of-chars
out, st, seen = run("crossover", "full", "unused", p1, p2, 0.0)  # rate 0 -> gate no-op
check("crossover rate-gate no-op returns the strs", out == (p1, p2) and all(isinstance(x, str) for x in out) and not seen, out)
out, st, _ = run("crossover", "full", "garbage", p1, p2, 1.0)
check("crossover fallback goes through sequence_model.crossover (strs, in bounds)",
      st["n_failures"] == 1 and all(isinstance(x, str) and 30 <= len(x) <= 80 for x in out), (out, st))
out, st, _ = run("mutate", "full", "garbage", g, 0.1)
check("mutate fallback is a str of same length over ALPHABET", isinstance(out, str) and len(out) == len(g) and set(out) <= set(A), out)

# ---- guards
gm.set_active(None)
try: run("mutate", "full", "x", g, 0.1); check("str genome + no active model raises", False, "no raise")
except RuntimeError as e: check("str genome + no active model raises", "no active GenomeModel" in str(e), e)
try: run("crossover", "full", "x", p1, p2, 1.0); check("str parents + no active model raises", False, "no raise")
except RuntimeError: check("str parents + no active model raises", True)
out, st, _ = run("mutate", "full", "garbage", [0, 1, 2, 3, 4, 0], 0.5)
check("list genome + no active model -> legacy lattice still works", isinstance(out, list), out)
gm.set_active(gm.build_genome_model(HPGAConfig(genome_model="sequence")))
try: run("mutate", "full", "x", [0, 1, 2], 0.5); check("list genome + sequence active raises", False, "no raise")
except RuntimeError as e: check("list genome + sequence active raises", "genome_type" in str(e), e)
gm.set_active(gm.build_genome_model(HPGAConfig(sequence="HPHPHPHP")))
try: run("mutate", "full", "x", g, 0.1); check("str genome + lattice active raises", False, "no raise")
except RuntimeError as e: check("str genome + lattice active raises", "genome_type" in str(e), e)
gm.set_active(None)
try: gm.current(); check("current() still raises when unset", False, "no raise")
except RuntimeError: check("current() still raises when unset", True)
lm = gm.LatticeGenomeModel()
try: lm.evaluate_fitness([0, 1]); check("config-less LatticeGenomeModel.evaluate_fitness raises clearly", False, "no raise")
except RuntimeError as e: check("config-less LatticeGenomeModel.evaluate_fitness raises clearly", "HPGAConfig" in str(e), e)

# ---- fuzz: valid-format replies always parse into in-bound str genomes
ok = True
fr = random.Random(7)
for _ in range(300):
    a, b = seq(fr, fr.randint(30, 80)), seq(fr, fr.randint(30, 80))
    f = fr.random(); c = fr.randint(1, 99)
    spec = f"0-{c}:1, {c}-100:2"
    pl = sm.plan_llm_crossover("segment", a, b)
    res = pl.parse(f"SEGMENTS: {spec}")
    if res is not None:
        ok &= all(isinstance(x, str) and 30 <= len(x) <= 80 and set(x) <= set(A) for x in res)
        ok &= len(res[0]) + len(res[1]) == len(a) + len(b)   # complement conserves total residues
check("fuzz: segment parse -> in-bound strs, residues conserved (300 draws)", ok)

# ---- restore process state
ops._call_ollama = _SAVED["call"]
ops.LLM_MAX_RETRIES = _SAVED["retries"]
if _SAVED["style"] is None:
    os.environ.pop("HPGA_LLM_PROMPT_STYLE", None)
else:
    os.environ["HPGA_LLM_PROMPT_STYLE"] = _SAVED["style"]
gm.set_active(None)
shutil.rmtree(_TMP, ignore_errors=True)

print("\n%d FAILED: %s" % (len(fails), fails) if fails else "\nALL SEQUENCE TESTS PASSED")
sys.exit(1 if fails else 0)
