"""Sanity check, ~30 calls total, before building the "agents with different
prompts" architecture (explore/refine/recombine agents proposing folds for
the same HP problem).

Question: does prompting the SAME model with genuinely different framings
(explore a new region / refine the current best / recombine two folds)
produce measurably different output distributions on this S/L/R move-string
representation? Phase 2's own data already found the model collapsing to
low-variance behaviour under the existing GA-operator prompts (mutate:
150/150 no-ops; free crossover: 39/40 exact-parent copies; segment
crossover: 36/40 exact-midpoint splits) -- so there is prior evidence
against the premise before this probe even runs. This script tests it
directly rather than assuming it either way.

Not part of hpga/operators.py -- this bypasses the GA-operator machinery
entirely (no retry-until-valid, no fallback-on-failure) because the point
here is to observe what the model actually does under each framing,
including its failures, not to make a working operator.

Usage:
    Phase 2/.venv/Scripts/python.exe experiments/probe_prompt_diversity.py

Requires a running local Ollama server (see hpga/operators.py env vars;
this script reuses the same HPGA_LLM_* defaults for comparability with
existing Phase 2 findings).
"""

import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpga.config import BENCHMARK_SEQUENCE
from hpga.hp_model import evaluate_fitness, MOVES

N_CALLS_PER_TYPE = 10
SEQUENCE = BENCHMARK_SEQUENCE  # HPHPPHHPHPPHPHHPPHPH, n=20, genome_length=18
LENGTH = len(SEQUENCE) - 2

LLM_MODEL = os.environ.get("HPGA_LLM_MODEL", "gemma4:12b")
LLM_HOST = os.environ.get("HPGA_OLLAMA_HOST", "http://localhost:11434")
LLM_TEMPERATURE = float(os.environ.get("HPGA_LLM_TEMPERATURE", "0.7"))
LLM_NUM_CTX = int(os.environ.get("HPGA_LLM_NUM_CTX", "4096"))
LLM_KEEP_ALIVE = os.environ.get("HPGA_LLM_KEEP_ALIVE", "30m")
LLM_REQUEST_TIMEOUT_S = float(os.environ.get("HPGA_LLM_TIMEOUT_S", "120"))
MAX_RETRIES = 2  # extra attempts beyond the first, matching operators.py

_MOVE_TO_CHAR = {0: "S", 1: "L", 2: "R"}
_CHAR_TO_MOVE = {"S": 0, "L": 1, "R": 2}

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "raw"
_LOG_PATH = _RESULTS_DIR / f"prompt_diversity_probe_{int(time.time())}.jsonl"

import re

_FOLD_LINE = re.compile(r"FOLD\s*:\s*([SLRslr][SLRslr\s,]*)", re.IGNORECASE)
_VALID_CHARS = re.compile(r"[SLR]", re.IGNORECASE)


def genome_to_str(genome):
    return " ".join(_MOVE_TO_CHAR[g] for g in genome)


def parse_fold(text, expected_len):
    m = _FOLD_LINE.search(text)
    if not m:
        return None
    letters = _VALID_CHARS.findall(m.group(1))
    if len(letters) != expected_len:
        return None
    return [_CHAR_TO_MOVE[c.upper()] for c in letters]


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def hill_climb(rng, iters=4000):
    """Cheap local search standing in for 'a current best fold the GA
    found' -- not the full multiprocessing island/worker harness, just
    enough to get a real, non-random evolved fold to hand the model."""
    genome = [rng.choice(MOVES) for _ in range(LENGTH)]
    best_fit = evaluate_fitness(genome, SEQUENCE)
    for _ in range(iters):
        cand = list(genome)
        pos = rng.randrange(LENGTH)
        cand[pos] = rng.choice(MOVES)
        fit = evaluate_fitness(cand, SEQUENCE)
        if fit >= best_fit:
            genome, best_fit = cand, fit
    return genome, best_fit


SYSTEM = (
    "You are solving a 2D HP protein lattice folding problem. A fold is a "
    "sequence of moves, one per residue after the first two, each move "
    "relative to the current heading: S (straight), L (turn left), or R "
    "(turn right). Higher fitness (more non-consecutive H-H lattice "
    "contacts) is better. Follow the requested output format exactly and "
    "output nothing else."
)

_RETRY_HINT = (
    "\nIMPORTANT: your previous response did not match the required "
    "format. Respond with ONLY the line 'FOLD: <{n} letters from S, L, R "
    "separated by single spaces>' and nothing else."
)


def build_prompts(seq, best_str, best_fit, second_str, second_fit, n):
    explore = f"""The HP sequence being folded (H=hydrophobic, P=polar) is: {seq}

The best fold found so far for this sequence is:
FOLD: {best_str}
(fitness = {best_fit})

Propose a SUBSTANTIALLY DIFFERENT fold for the same sequence -- explore a
different region of the space of possible folds rather than making small
tweaks to this one. It does not need to score better; the goal is
diversity from the fold above.

Respond with EXACTLY one line and nothing else:
FOLD: <{n} letters from {{S,L,R}} separated by single spaces>"""

    refine = f"""The HP sequence being folded (H=hydrophobic, P=polar) is: {seq}

The best fold found so far for this sequence is:
FOLD: {best_str}
(fitness = {best_fit})

Propose an IMPROVED fold for the same sequence: make targeted changes to
this fold that could increase the number of H-H contacts, keeping most of
its structure intact rather than replacing it wholesale.

Respond with EXACTLY one line and nothing else:
FOLD: <{n} letters from {{S,L,R}} separated by single spaces>"""

    recombine = f"""The HP sequence being folded (H=hydrophobic, P=polar) is: {seq}

Two candidate folds for this sequence are:
FOLD A: {best_str}
(fitness = {best_fit})
FOLD B: {second_str}
(fitness = {second_fit})

Propose a new fold that COMBINES the strengths of FOLD A and FOLD B.

Respond with EXACTLY one line and nothing else:
FOLD: <{n} letters from {{S,L,R}} separated by single spaces>"""

    return {"explore": explore, "refine": refine, "recombine": recombine}


def call_ollama(client, prompt, seed, num_predict):
    t0 = time.perf_counter()
    response = client.generate(
        model=LLM_MODEL,
        prompt=prompt,
        system=SYSTEM,
        stream=False,
        think=False,
        options={
            "num_ctx": LLM_NUM_CTX,
            "num_predict": num_predict,
            "temperature": LLM_TEMPERATURE,
            "seed": seed,
        },
        keep_alive=LLM_KEEP_ALIVE,
    )
    latency = time.perf_counter() - t0
    text = getattr(response, "response", "") or ""
    return text, latency


def log(record):
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def run_one(client, rng, prompt_type, base_prompt, n):
    num_predict = min(2048, max(64, 8 * n))
    retry_hint = ""
    for attempt in range(MAX_RETRIES + 1):
        prompt = base_prompt + (retry_hint if attempt else "")
        seed = rng.getrandbits(31)
        text, latency = call_ollama(client, prompt, seed, num_predict)
        parsed = parse_fold(text, n)
        log({
            "prompt_type": prompt_type, "attempt": attempt, "seed": seed,
            "prompt": prompt, "response": text, "latency_s": latency,
            "valid": parsed is not None,
        })
        if parsed is not None:
            return parsed, latency, attempt
        retry_hint = _RETRY_HINT.format(n=n)
    return None, latency, MAX_RETRIES


def mann_whitney_u(a, b):
    """Two-sided Mann-Whitney U, normal approximation. No scipy in this
    venv; small enough (n=10ish) to hand-roll. Returns (U, z, p)."""
    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = {}
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    rank_sum_a = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 0)
    n1, n2 = len(a), len(b)
    u1 = rank_sum_a - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    mu = n1 * n2 / 2
    sigma = (n1 * n2 * (n1 + n2 + 1) / 12) ** 0.5
    if sigma == 0:
        return u, 0.0, 1.0
    z = (u - mu) / sigma
    # two-sided normal approx p-value
    p = 2 * (1 - 0.5 * (1 + _erf(abs(z) / 2 ** 0.5)))
    return u, z, p


def _erf(x):
    # Abramowitz-Stegun approximation, good to ~1e-7
    sign = 1 if x >= 0 else -1
    x = abs(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (2.718281828459045 ** (-x * x))
    return sign * y


def describe(vals):
    if not vals:
        return "n=0"
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"n={len(vals)} mean={mean:.2f} sd={sd:.2f} min={min(vals)} max={max(vals)}"


def main():
    from ollama import Client
    client = Client(host=LLM_HOST, timeout=LLM_REQUEST_TIMEOUT_S)

    rng = random.Random(0)
    best_genome, best_fit = hill_climb(random.Random(1))
    second_genome, second_fit = hill_climb(random.Random(2))
    best_str = genome_to_str(best_genome)
    second_str = genome_to_str(second_genome)

    print(f"sequence = {SEQUENCE} (length={LENGTH})")
    print(f"best fold   (A): {best_str}  fitness={best_fit}")
    print(f"second fold (B): {second_str}  fitness={second_fit}")
    print(f"hamming(A,B) = {hamming(best_genome, second_genome)} / {LENGTH}")
    print()

    prompts = build_prompts(SEQUENCE, best_str, best_fit, second_str, second_fit, LENGTH)

    results = {}  # prompt_type -> list of dicts
    for prompt_type, base_prompt in prompts.items():
        print(f"--- {prompt_type} ---")
        calls = []
        for i in range(N_CALLS_PER_TYPE):
            genome, latency, attempts = run_one(client, rng, prompt_type, base_prompt, LENGTH)
            if genome is None:
                print(f"  call {i}: FAILED to parse after {MAX_RETRIES + 1} attempts")
                calls.append({"valid": False})
                continue
            fit = evaluate_fitness(genome, SEQUENCE)
            d_a = hamming(genome, best_genome)
            d_b = hamming(genome, second_genome)
            print(f"  call {i}: valid, fitness={fit:+.0f}, hamming(A)={d_a}, hamming(B)={d_b}, "
                  f"attempts={attempts + 1}, latency={latency:.1f}s")
            calls.append({
                "valid": True, "genome": genome, "fitness": fit,
                "hamming_a": d_a, "hamming_b": d_b,
            })
        results[prompt_type] = calls
        print()

    # --- Summary ---
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary = {}
    for prompt_type, calls in results.items():
        valid = [c for c in calls if c["valid"]]
        n_valid = len(valid)
        n_failed = len(calls) - n_valid
        ref_dist = [c["hamming_a"] for c in valid]  # distance from the fold each prompt was given as primary reference
        fits = [c["fitness"] for c in valid]
        exact_copy_a = sum(1 for c in valid if c["hamming_a"] == 0)
        exact_copy_b = sum(1 for c in valid if c["hamming_b"] == 0)
        summary[prompt_type] = {
            "n_valid": n_valid, "n_failed": n_failed,
            "hamming_from_A": ref_dist, "fitness": fits,
            "exact_copy_of_A": exact_copy_a, "exact_copy_of_B": exact_copy_b,
        }
        print(f"\n{prompt_type}:")
        print(f"  valid={n_valid}/{N_CALLS_PER_TYPE}  failed_to_parse={n_failed}")
        print(f"  hamming distance from fold A (input best): {describe(ref_dist)}")
        print(f"  returned fold fitness:                     {describe(fits)}")
        print(f"  exact byte-copy of fold A: {exact_copy_a}/{n_valid}   exact byte-copy of fold B: {exact_copy_b}/{n_valid}")

    print("\n" + "-" * 70)
    print("Cross-type comparison (Mann-Whitney U on hamming-from-A, two-sided, normal approx):")
    types = list(summary.keys())
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            t1, t2 = types[i], types[j]
            a, b = summary[t1]["hamming_from_A"], summary[t2]["hamming_from_A"]
            if len(a) < 2 or len(b) < 2:
                print(f"  {t1} vs {t2}: insufficient valid samples")
                continue
            u, z, p = mann_whitney_u(a, b)
            print(f"  {t1} vs {t2}: U={u:.1f} z={z:.2f} p={p:.3f}  "
                  f"({t1} mean={statistics.mean(a):.2f}, {t2} mean={statistics.mean(b):.2f})")

    print("\nCross-type comparison (Mann-Whitney U on returned-fold fitness):")
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            t1, t2 = types[i], types[j]
            a, b = summary[t1]["fitness"], summary[t2]["fitness"]
            if len(a) < 2 or len(b) < 2:
                print(f"  {t1} vs {t2}: insufficient valid samples")
                continue
            u, z, p = mann_whitney_u(a, b)
            print(f"  {t1} vs {t2}: U={u:.1f} z={z:.2f} p={p:.3f}  "
                  f"({t1} mean={statistics.mean(a):.2f}, {t2} mean={statistics.mean(b):.2f})")

    print(f"\nFull per-call log written to: {_LOG_PATH}")


if __name__ == "__main__":
    main()
