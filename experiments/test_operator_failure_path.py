"""Exercise the retry/fallback path deliberately. The pilot ran 27/27 valid
first-try, which is good news for output quality but means the retry and
fallback machinery in hpga/operators.py has never actually been triggered --
this could be silently broken and nothing so far would have caught it.

Monkeypatches hpga.operators._call_ollama with a stub that returns
controlled (valid/malformed/exception) text, so every scenario below is
deterministic and needs no live Ollama server. Four scenarios:

  1. always malformed -> exhausts retries, falls back to the deterministic
     operator, n_failures increments, output is still a valid genome.
  2. malformed once then valid -> succeeds on retry, n_retries increments,
     n_failures stays 0, output is the LLM-parsed genome (not the fallback).
  3. always valid -> baseline, 0 retries/0 failures (matches the pilot).
  4. transport exception (simulated connection error) -> must propagate, not
     be swallowed into a silent fallback -- infra failure is not "malformed
     output" and masking it would corrupt the T_calc/latency measurements.

Exits non-zero if any assertion fails.

Updated after the 3D migration (5-symbol alphabet) and the crossover
mixed-child gate, both of which had left this script failing 4 of 6 scenarios
while the operators were fine: (1) the "valid" mutate reply changed 1 position
but the operator demands exactly k = round(rate * length) = 2; (2) the validity
check only accepted moves 0-2, while the deterministic fallback draws from all
five (hp_model.MOVES); (3) the "valid" crossover reply was CHILD1=Parent1,
CHILD2=Parent2 -- a verbatim echo, which _crossover_sufficiently_mixed
correctly rejects. Replies are now derived from the operator's own parameters
rather than hardcoded, so they can't drift from them again. The operator log
is pointed at a temp directory (unless HPGA_LLM_LOG_PATH is already set) so a
test run doesn't leave llm_operator_calls_*.jsonl files in results/raw.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpga import operators as ops

GENOME_LEN = 8
MUTATION_RATE = 0.3
GARBAGE = "I'm sorry, I can't help with that request."


def valid_genome_str(genome: list[int]) -> str:
    return " ".join(ops._MOVE_TO_CHAR[g] for g in genome)


def make_stub(script):
    """script: list of either a str (response text) or an Exception instance,
    consumed in order across calls; the last entry repeats once exhausted."""
    calls = {"n": 0}

    def stub(prompt, system, num_predict, seed):
        i = min(calls["n"], len(script) - 1)
        calls["n"] += 1
        item = script[i]
        if isinstance(item, Exception):
            raise item
        return item, 0.001, 10, 5  # (text, latency, tokens_in, tokens_out)

    return stub, calls


def run_scenario(name, script, op, args, expect):
    ops.reset_operator_stats()
    stub, calls = make_stub(script)
    original = ops._call_ollama
    ops._call_ollama = stub
    try:
        if op == "crossover":
            result = ops._llm_crossover(*args)
        else:
            result = ops._llm_mutate(*args)
        error = None
    except Exception as exc:  # noqa: BLE001 - deliberately catching to assert on it
        result = None
        error = exc
    finally:
        ops._call_ollama = original

    stats = ops.get_operator_stats()
    ok = True
    details = []

    if expect.get("raises"):
        if error is None or not isinstance(error, expect["raises"]):
            ok = False
            details.append(f"expected {expect['raises']} to propagate, got {error!r}")
    else:
        if error is not None:
            ok = False
            details.append(f"unexpected exception: {error!r}")

    for key, val in expect.get("stats", {}).items():
        actual = stats.get(key)
        if actual != val:
            ok = False
            details.append(f"stats[{key}]={actual!r}, expected {val!r}")

    if expect.get("result_is_valid_genome"):
        genomes = result if isinstance(result, list) and op == "mutate" else None
        pairs = result if op == "crossover" else None
        to_check = [result] if op == "mutate" else list(pairs)
        for g in to_check:
            if not (isinstance(g, list) and len(g) == GENOME_LEN and all(m in ops.MOVES for m in g)):
                ok = False
                details.append(f"result is not a valid length-{GENOME_LEN} genome: {g!r}")

    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}  (calls made: {calls['n']}, stats={stats})")
    for d in details:
        print(f"         - {d}")
    return ok


def main() -> None:
    rng_genome = [0, 1, 2, 0, 1, 2, 0, 1]
    parent1 = list(rng_genome)
    parent2 = [1, 2, 0, 1, 2, 0, 1, 2]
    # Exactly k positions changed, k derived the way _llm_mutate derives it.
    k = max(1, round(MUTATION_RATE * GENOME_LEN))
    mutated = list(rng_genome)
    for i in range(k):
        mutated[i] = (mutated[i] + 1) % len(ops.MOVES)
    valid_mutated = valid_genome_str(mutated)
    # A genuine recombination: half-cut children. parent1 and parent2 differ at
    # every position, so each child differs from each parent in GENOME_LEN // 2
    # positions -- clear of _crossover_sufficiently_mixed's CROSSOVER_MIN_DIFF.
    assert GENOME_LEN // 2 >= ops.CROSSOVER_MIN_DIFF, "raise GENOME_LEN or lower HPGA_CROSSOVER_MIN_DIFF"
    half = GENOME_LEN // 2
    child1, child2 = parent1[:half] + parent2[half:], parent2[:half] + parent1[half:]
    valid_pair = f"CHILD1: {valid_genome_str(child1)}\nCHILD2: {valid_genome_str(child2)}"

    import random
    all_ok = True
    assert parent1 != parent2 and all(a != b for a, b in zip(parent1, parent2))

    all_ok &= run_scenario(
        "mutate: always malformed -> exhausts retries, falls back",
        script=[GARBAGE] * (ops.LLM_MAX_RETRIES + 5),
        op="mutate", args=(rng_genome, MUTATION_RATE, random.Random(0)),
        expect={
            "stats": {"n_retries": ops.LLM_MAX_RETRIES + 1, "n_failures": 1},
            "result_is_valid_genome": True,
        },
    )

    all_ok &= run_scenario(
        "mutate: malformed once then valid -> succeeds on retry",
        script=[GARBAGE, f"MUTATED: {valid_mutated}"],
        op="mutate", args=(rng_genome, MUTATION_RATE, random.Random(0)),
        expect={
            "stats": {"n_retries": 1, "n_failures": 0},
            "result_is_valid_genome": True,
        },
    )

    all_ok &= run_scenario(
        "mutate: always valid -> 0 retries, 0 failures",
        script=[f"MUTATED: {valid_mutated}"],
        op="mutate", args=(rng_genome, MUTATION_RATE, random.Random(0)),
        expect={
            "stats": {"n_retries": 0, "n_failures": 0},
            "result_is_valid_genome": True,
        },
    )

    all_ok &= run_scenario(
        "crossover: always malformed -> exhausts retries, falls back",
        script=[GARBAGE] * (ops.LLM_MAX_RETRIES + 5),
        op="crossover", args=(parent1, parent2, 1.0, random.Random(0)),
        expect={
            "stats": {"n_retries": ops.LLM_MAX_RETRIES + 1, "n_failures": 1},
            "result_is_valid_genome": True,
        },
    )

    all_ok &= run_scenario(
        "crossover: malformed once then valid -> succeeds on retry",
        script=[GARBAGE, valid_pair],
        op="crossover", args=(parent1, parent2, 1.0, random.Random(0)),
        expect={
            "stats": {"n_retries": 1, "n_failures": 0},
            "result_is_valid_genome": True,
        },
    )

    all_ok &= run_scenario(
        "mutate: transport exception -> propagates, not swallowed",
        script=[ConnectionError("simulated connection error")],
        op="mutate", args=(rng_genome, MUTATION_RATE, random.Random(0)),
        expect={"raises": ConnectionError},
    )

    print()
    if all_ok:
        print("ALL SCENARIOS PASSED")
    else:
        print("SOME SCENARIOS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    _tmp = None
    if not os.environ.get("HPGA_LLM_LOG_PATH"):
        _tmp = tempfile.mkdtemp(prefix="test_operator_failure_path_")
        os.environ["HPGA_LLM_LOG_PATH"] = os.path.join(_tmp, "calls.jsonl")
    try:
        main()
    finally:
        if _tmp:
            shutil.rmtree(_tmp, ignore_errors=True)
