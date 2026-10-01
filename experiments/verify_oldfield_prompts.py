"""Checks for hpga/sequence_model_oldfield.py, no model calls (results/PREREGISTERED_OLDFIELD.md).

  1. Byte-identity: for every (length, k, retry state, sequence) case, the oldfield prompt with its format line put back
     equals sequence_model.plan_llm_mutate("position", ...)'s prompt byte for byte; the new format line occurs exactly
     once; the system prompt and retry hint are the base plan's; num_predict is max(48, 16k) capped at 2048.
  2. Base plan untouched: sequence_model.plan_llm_mutate's prompt is the same before and after importing the module.
  3. Parser: valid answers accepted and applied; every invalid kind rejected (wrong OLD, NEW equal to the current
     letter, wrong line count, repeated position, out of range, OLD missing as in a base-format answer).

  python experiments/verify_oldfield_prompts.py     -> prints the case count and RESULT: PASS, or raises
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpga import sequence_model as sm  # noqa: E402

BEFORE = {(n, k): sm.plan_llm_mutate("position", "A" * n, k).build_prompt("") for n in (30, 63, 80) for k in (1, 3)}

from hpga import sequence_model_oldfield as of  # noqa: E402


def main() -> None:
    rng = random.Random(0)
    cases = 0
    for n in (30, 31, 40, 49, 50, 51, 63, 69, 70, 79, 80):
        for k in (1, 2, 3, 4):
            for _ in range(5):
                g = sm.random_sequence(rng, n)
                base, new = sm.plan_llm_mutate("position", g, k), of.plan_llm_mutate_oldfield(g, k)
                for hint in ("", base.retry_hint_text):
                    bp, np_ = base.build_prompt(hint), new.build_prompt(hint)
                    assert np_.count(of._new_format_line(n)) == 1, (n, k)
                    assert bp.count(of._base_format_line(n)) == 1, (n, k)
                    assert of.restore_base_format(np_, n) == bp, (n, k, hint)
                    assert np_ != bp
                    cases += 1
                assert new.system == base.system and new.retry_hint_text == base.retry_hint_text
                assert new.num_predict == min(2048, max(48, 16 * k)) and base.num_predict == min(2048, max(24, 10 * k))
    for (n, k), p in BEFORE.items():
        assert sm.plan_llm_mutate("position", "A" * n, k).build_prompt("") == p, "base prompt changed by import"

    g = "ACDEFGHIKL"
    ok = "POSITION: 0, OLD: A, NEW: W\nPOSITION: 3, OLD: E, NEW: Y"
    assert of.extract(ok, g, 2) == "WCDYFGHIKL"
    assert of.extract(ok.lower(), g, 2) == "WCDYFGHIKL"  # case-insensitive like the base parser
    rejects = {
        "wrong OLD": "POSITION: 0, OLD: C, NEW: W\nPOSITION: 3, OLD: E, NEW: Y",
        "NEW equal to current letter": "POSITION: 0, OLD: A, NEW: A\nPOSITION: 3, OLD: E, NEW: Y",
        "NEW equal to actual letter, OLD wrong": "POSITION: 0, OLD: C, NEW: A\nPOSITION: 3, OLD: E, NEW: Y",
        "too few lines": "POSITION: 0, OLD: A, NEW: W",
        "too many lines": ok + "\nPOSITION: 5, OLD: G, NEW: W",
        "repeated position": "POSITION: 0, OLD: A, NEW: W\nPOSITION: 0, OLD: A, NEW: Y",
        "out of range": "POSITION: 0, OLD: A, NEW: W\nPOSITION: 10, OLD: A, NEW: Y",
        "OLD missing (base format)": "POSITION: 0, NEW: W\nPOSITION: 3, NEW: Y",
    }
    for name, text in rejects.items():
        assert of.extract(text, g, 2) is None, name
    print(f"{cases} prompt cases, {len(BEFORE)} base-prompt checks, 2 accepts, {len(rejects)} rejects")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
