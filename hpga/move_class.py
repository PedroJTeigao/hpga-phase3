"""Move classes for the sequence GA's mutation step (experiments/run_move_class.py).

Each function takes the post-crossover child and the GA's own `rng` and returns (new genome, move description).
Every move class except "B" applies EXACTLY ONE move per call, whatever the mutation rate -- the point is to compare
move classes one move against one move. "B" is the existing deterministic operator (sequence_model.mutate: per-site
substitution with p=rate, the new letter uniform over all 20 so it can equal the old one), called unchanged so the
arm-B runs of run_sequence_ga_comparison.py are reproduced draw for draw.

  B   per-site substitution, p = rate (the existing operator, unchanged)
  S1  single-position substitution: one position uniform, new letter uniform over the other 19
  S2  k-position substitution: k uniform in 2-5, positions without replacement, each new letter uniform over the other 19
  S3  segment replacement: span length uniform in 5-15, start uniform over the starts where the whole span fits,
      span replaced by fresh letters uniform over all 20
  S4  indel: deletion or insertion with equal probability, span length uniform in 1-5; a deletion removes a span whose
      start is uniform over the starts where it fits, an insertion puts fresh letters (uniform over all 20) at a gap
      uniform over 0..len. A draw that would leave [MIN_LENGTH, MAX_LENGTH] is redrawn (direction, span and position
      all), so at a length bound only the direction that stays inside is possible.

Nothing in hpga/ calls this module; it is reached only through the driver's wrapper around the active model's
deterministic_mutate.
"""

import random

from hpga import sequence_model as sm

MOVE_CLASSES = ("B", "S1", "S2", "S3", "S4")
S2_K = (2, 5)
S3_SPAN = (5, 15)
S4_SPAN = (1, 5)


def _other_letter(c: str, rng: random.Random) -> str:
    return rng.choice([a for a in sm.ALPHABET if a != c])


def _fresh(n: int, rng: random.Random) -> str:
    return "".join(rng.choice(sm.ALPHABET) for _ in range(n))


def move_b(g: str, rate: float, rng: random.Random) -> tuple[str, dict]:
    child = sm.mutate(g, rate, rng)
    changed = [i for i, (a, b) in enumerate(zip(g, child)) if a != b]
    return child, {"kind": "per_site", "n_changed": len(changed), "positions": changed}


def move_s1(g: str, rate: float, rng: random.Random) -> tuple[str, dict]:
    p = rng.randrange(len(g))
    new = _other_letter(g[p], rng)
    return g[:p] + new + g[p + 1:], {"kind": "sub", "n_changed": 1, "positions": [p]}


def move_s2(g: str, rate: float, rng: random.Random) -> tuple[str, dict]:
    k = min(rng.randint(*S2_K), len(g))
    pos = rng.sample(range(len(g)), k)
    s = list(g)
    for p in pos:
        s[p] = _other_letter(g[p], rng)
    return "".join(s), {"kind": "sub", "k": k, "n_changed": k, "positions": sorted(pos)}


def move_s3(g: str, rate: float, rng: random.Random) -> tuple[str, dict]:
    n = min(rng.randint(*S3_SPAN), len(g))
    start = rng.randint(0, len(g) - n)
    child = g[:start] + _fresh(n, rng) + g[start + n:]
    return child, {"kind": "segment", "start": start, "span": n,
                   "n_changed": sum(a != b for a, b in zip(g, child))}


def move_s4(g: str, rate: float, rng: random.Random) -> tuple[str, dict]:
    n_redraws = 0
    while True:
        op = "del" if rng.random() < 0.5 else "ins"
        n = rng.randint(*S4_SPAN)
        new_len = len(g) - n if op == "del" else len(g) + n
        if sm.MIN_LENGTH <= new_len <= sm.MAX_LENGTH:
            break
        n_redraws += 1
    if op == "del":
        start = rng.randint(0, len(g) - n)
        child = g[:start] + g[start + n:]
    else:
        start = rng.randint(0, len(g))
        child = g[:start] + _fresh(n, rng) + g[start:]
    return child, {"kind": op, "start": start, "span": n, "n_redraws": n_redraws,
                   "len_before": len(g), "len_after": len(child)}


MOVES = {"B": move_b, "S1": move_s1, "S2": move_s2, "S3": move_s3, "S4": move_s4}
