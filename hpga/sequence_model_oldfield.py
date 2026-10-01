"""The `position_old` mutate format: `mutate/position` with an OLD slot in the response format, and nothing else changed.

Pre-registered in results/PREREGISTERED_OLDFIELD.md. The rule "the new letter MUST differ from whatever letter is
currently at that position" is stated in prose in hpga/sequence_model.py's _mutate_position_prompt. Every invalid mutate
attempt in the project's GA runs broke exactly that rule. This format adds one slot to the response format, so the model
must write the current letter before the new one:

    POSITION: <0-62>, NEW: <one letter from {...}>                                               (sequence_model.py)
    POSITION: <0-62>, OLD: <the letter currently at that position>, NEW: <one letter from {...}>  (this module)

Built the way hpga/sequence_model_fitness.py builds arm F: from the plan sequence_model.plan_llm_mutate("position", ...)
returns, never from a copy of its text. The format line is replaced only after asserting it occurs exactly once, and
restore_base_format() inverts the replacement, which is how experiments/verify_oldfield_prompts.py proves every other
byte is the base prompt's. The system prompt, the instruction prose (including the MUST-differ sentence) and the retry
hint are the base plan's own objects.

One difference besides the prompt: num_predict is max(48, 16 * k) instead of max(24, 10 * k). Each line is about four
tokens longer, and at k = 3 the base limit would cut an answer off and score it invalid for a reason that has nothing to
do with the rule under test.

Validity (strict): exactly k lines, positions in range and distinct, OLD equal to the letter actually at that position,
NEW different from it. The analysis also scores every line leniently (OLD ignored), which is the base format's own rule,
so the repeated-letter rate is defined identically under both formats.

Not reachable from any existing arm: nothing in hpga/ or experiments/ imports this module except the oldfield probe and
its verifier. sequence_model.py, operators.py and genome_model.py are untouched.
"""

import re
from dataclasses import replace

from hpga import sequence_model as sm
from hpga.genome_model import LLMOpPlan

STYLE = "position_old"
_OLD_SLOT = "OLD: <the letter currently at that position>, "
_POS_OLD_NEW = re.compile(
    rf"POSITION\s*:\s*(\d+)\s*,\s*OLD\s*:\s*({sm._LETTER})\s*,\s*NEW\s*:\s*({sm._LETTER})", re.IGNORECASE)


def _base_format_line(length: int) -> str:
    return f"POSITION: <0-{length - 1}>, NEW: <one letter from {sm._ALPHA_SET}>"


def _new_format_line(length: int) -> str:
    return f"POSITION: <0-{length - 1}>, {_OLD_SLOT}NEW: <one letter from {sm._ALPHA_SET}>"


def _swap_format_line(text: str, length: int) -> str:
    old, new = _base_format_line(length), _new_format_line(length)
    n = text.count(old)
    if n != 1:
        raise RuntimeError(
            f"OLD slot cannot be placed: expected the base format line exactly once, found {n} times. "
            "hpga/sequence_model.py's mutate prompt changed; update hpga/sequence_model_oldfield.py deliberately.")
    return text.replace(old, new)


def restore_base_format(text: str, length: int) -> str:
    """Inverse of the swap: the oldfield prompt back to the base prompt, byte for byte."""
    return text.replace(_new_format_line(length), _base_format_line(length))


def parse_lines(text: str) -> list[tuple[int, str, str]]:
    """(position, OLD, NEW) for every well-formed line, upper-cased."""
    return [(int(p), o.upper(), n.upper()) for p, o, n in _POS_OLD_NEW.findall(text)]


def extract(text: str, base: str, k: int) -> str | None:
    lines = parse_lines(text)
    if len(lines) != k:
        return None
    result, seen = list(base), set()
    for pos, old, new in lines:
        if pos < 0 or pos >= len(base) or pos in seen:
            return None
        if old != base[pos] or new == base[pos]:
            return None
        seen.add(pos)
        result[pos] = new
    return "".join(result)


def plan_llm_mutate_oldfield(genome: str, k: int) -> LLMOpPlan:
    base = sm.plan_llm_mutate("position", genome, k)
    length = len(genome)
    _swap_format_line(base.build_prompt(""), length)  # fail at plan time, not mid-call, if the anchor is missing

    def build_prompt(retry_hint: str) -> str:
        return _swap_format_line(base.build_prompt(retry_hint), length)

    return replace(base, build_prompt=build_prompt, parse=lambda text: extract(text, genome, k),
                   num_predict=min(2048, max(48, 16 * k)))
