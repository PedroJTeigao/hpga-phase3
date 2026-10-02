"""The numbered sequence display for mutate/position: every letter labelled by its 0-based index, nothing else changed.

Pre-registered in results/PREREGISTERED_NUMBERED.md. results/INDEXING_NOT_INSTRUCTION.md found that the repeated-letter
failure goes with misreading the letter at a stated position in the space-separated display. This display makes the
position of every letter explicit:

    Sequence (0-indexed positions 0-62): A C D E ...            (hpga/sequence_model.py, production)
    Sequence (0-indexed positions 0-62): 0:A 1:C 2:D 3:E ...    (this module)

Built like hpga/sequence_model_oldfield.py: from the plan the base module returns, by replacing exactly one string
(the space-separated sequence, asserted to occur exactly once in the prompt), with restore_spaced() as the inverse.
Parser, system prompt, retry hint and num_predict are the base plan's own (the display changes the input, not the
answer). With old_slot=True the base plan is the OLD-slot plan, so the four cells of the design differ from each other
in exactly one thing each. Nothing in hpga/ or experiments/ imports this module except the numbered probe and its
verifier.
"""

from dataclasses import replace

from hpga import sequence_model as sm
from hpga import sequence_model_oldfield as of
from hpga.genome_model import LLMOpPlan


def spaced(genome: str) -> str:
    return sm._spaced(genome)


def numbered(genome: str) -> str:
    return " ".join(f"{i}:{c}" for i, c in enumerate(genome))


def _swap_display(text: str, genome: str) -> str:
    old = f"): {spaced(genome)}\n"
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"numbered display cannot be placed: the spaced sequence occurs {n} times in the prompt; "
                           "hpga/sequence_model.py's mutate prompt changed")
    return text.replace(old, f"): {numbered(genome)}\n")


def restore_spaced(text: str, genome: str) -> str:
    return text.replace(f"): {numbered(genome)}\n", f"): {spaced(genome)}\n")


def plan_llm_mutate_numbered(genome: str, k: int, old_slot: bool = False) -> LLMOpPlan:
    base = of.plan_llm_mutate_oldfield(genome, k) if old_slot else sm.plan_llm_mutate("position", genome, k)
    _swap_display(base.build_prompt(""), genome)  # fail at plan time if the anchor is missing

    def build_prompt(retry_hint: str) -> str:
        return _swap_display(base.build_prompt(retry_hint), genome)

    return replace(base, build_prompt=build_prompt)
