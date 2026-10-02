"""Checks for hpga/sequence_model_numbered.py, no model calls (results/PREREGISTERED_NUMBERED.md).

For every (length, k, retry state, sequence, response format) case:
  - restore_spaced(numbered prompt) equals the base prompt byte for byte, where the base is
    sequence_model.plan_llm_mutate("position") (old_slot=False) or sequence_model_oldfield's plan (old_slot=True);
  - only the first line differs, and it is exactly "Sequence (0-indexed positions 0-N): 0:A 1:C ...";
  - system prompt, retry hint and num_predict are the base plan's, and parse gives the base parse on probe answers.
Also: the production plan's prompt is unchanged by importing the module.

  python experiments/verify_numbered_prompts.py   -> prints the case count and RESULT: PASS, or raises
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpga import sequence_model as sm  # noqa: E402

BEFORE = {(n, k): sm.plan_llm_mutate("position", "A" * n, k).build_prompt("") for n in (30, 63, 80) for k in (1, 3)}

from hpga import sequence_model_numbered as nb  # noqa: E402
from hpga import sequence_model_oldfield as of  # noqa: E402


def main() -> None:
    rng = random.Random(0)
    cases = 0
    for n in (30, 31, 40, 49, 50, 51, 63, 69, 70, 79, 80):
        for k in (1, 2, 3, 4):
            for _ in range(5):
                g = sm.random_sequence(rng, n)
                for old_slot in (False, True):
                    base = of.plan_llm_mutate_oldfield(g, k) if old_slot else sm.plan_llm_mutate("position", g, k)
                    num = nb.plan_llm_mutate_numbered(g, k, old_slot=old_slot)
                    for hint in ("", base.retry_hint_text):
                        bp, np_ = base.build_prompt(hint), num.build_prompt(hint)
                        assert nb.restore_spaced(np_, g) == bp, (n, k, old_slot)
                        bl, nl = bp.split("\n"), np_.split("\n")
                        assert len(bl) == len(nl) and bl[1:] == nl[1:], "a line other than the first differs"
                        assert nl[0] == f"Sequence (0-indexed positions 0-{n - 1}): " + " ".join(f"{i}:{c}" for i, c in enumerate(g))
                        cases += 1
                    assert num.system == base.system and num.retry_hint_text == base.retry_hint_text
                    assert num.num_predict == base.num_predict
                    pos = rng.sample(range(n), k)
                    ans = "\n".join(f"POSITION: {p}, " + (f"OLD: {g[p]}, " if old_slot else "") +
                                    f"NEW: {'W' if g[p] != 'W' else 'Y'}" for p in pos)
                    assert num.parse(ans) == base.parse(ans) is not None
                    assert num.parse("junk") is None and base.parse("junk") is None
    for (n, k), p in BEFORE.items():
        assert sm.plan_llm_mutate("position", "A" * n, k).build_prompt("") == p, "production prompt changed by import"
    print(f"{cases} prompt cases (both response formats), {len(BEFORE)} production-prompt checks")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
