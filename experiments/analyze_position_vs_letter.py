"""Offline analysis (NO model calls): is the position an LLM mutate/position operator chooses predicted by the
letter that sits there?

Data: the existing mutate/position call logs -- the four step-1 logs (llm_operator_calls_hetero_<model>.jsonl, 100
calls each, genome length 63) and the earlier gemma4:12b length-63 run behind PHASE3_RESULTS.md sec. 9.2
(llm_operator_calls_1789749658_409662.jsonl, 400 mutate/position calls).  Every call used a fresh uniform-random
genome, so the letters 'available' are ~uniform; the comparison uses the actual genomes.

Per call: genome = the letters printed in the attempt-0 prompt; chosen position = the POSITION in the first valid
response; letter_at_choice = genome[position] (the letter the model chose to replace).  Calls that fell back to the
deterministic operator have no valid response and are excluded.

Two nulls, both by Monte Carlo (20000 draws), statistic = chi-square of the 20 letter counts at chosen positions
against the null's expected counts:
  A. 'uniform position': the position is drawn uniformly over the genome, independent of content.  This is the
     comparison of chosen-position letters against the letters available in the genomes.  It CAN reject because of the
     model's positional bias alone (e.g. if position 10 always wins and letter frequencies at position 10 in these
     particular 100 genomes differ from the mean), so it is not by itself evidence of content sensitivity.
  B. 'positions kept, content shuffled': keep the model's chosen positions, pair each with a genome from a different
     call (random permutation of the genomes).  This keeps the positional bias and breaks any link between the chosen
     position and the letter found there; rejecting B means the choice depends on the content.
The same two tests are repeated on 4 residue classes (hydrophobic AVILMFW, charged DEKRH, polar STNQCY, special GP).

Writes results/raw/model_position_vs_letter_tables.md and prints it.
"""

import json
import re
from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parent.parent / "results" / "raw"
SOURCES = [
    ("gemma4:12b (step 1, n=100)", "llm_operator_calls_hetero_gemma4_12b.jsonl"),
    ("gemma4:12b (P3 sec. 9 run, n=400)", "llm_operator_calls_1789749658_409662.jsonl"),
    ("llama3.2:3b (step 1)", "llm_operator_calls_hetero_llama3.2_3b.jsonl"),
    ("qwen2.5:7b (step 1)", "llm_operator_calls_hetero_qwen2.5_7b.jsonl"),
    ("mistral:7b (step 1)", "llm_operator_calls_hetero_mistral_7b.jsonl"),
]
ALPHA = "ACDEFGHIKLMNPQRSTVWY"
CLASSES = {"hydrophobic": "AVILMFW", "charged": "DEKRH", "polar": "STNQCY", "special": "GP"}
GENOME_RE = re.compile(r"Sequence \(0-indexed positions 0-(\d+)\):\s*([A-Z ]+)")
POS_RE = re.compile(r"POSITION:\s*(\d+)\s*,\s*NEW:\s*([A-Za-z])")
N_SIM = 20000


def read_calls(fname: str):
    """(genome string, chosen position) per mutate/position call that produced a valid response."""
    calls, cur = [], None
    for line in open(RAW / fname, encoding="utf-8"):
        r = json.loads(line)
        if r.get("op") != "mutate" or r.get("prompt_style") != "position":
            continue
        if r.get("attempt") == 0:
            m = GENOME_RE.search(r["prompt"])
            cur = {"genome": m.group(2).replace(" ", ""), "pos": None} if m else None
            if cur:
                calls.append(cur)
        if cur is not None and cur["pos"] is None and r.get("valid") is True:
            pm = POS_RE.search(r.get("response", ""))
            if pm:
                cur["pos"] = int(pm.group(1))
    return [(c["genome"], c["pos"]) for c in calls if c["pos"] is not None and c["pos"] < len(c["genome"])]


def counts(letters, alphabet):
    idx = {a: i for i, a in enumerate(alphabet)}
    v = np.zeros(len(alphabet))
    for ch in letters:
        v[idx[ch]] += 1
    return v


def chi2(obs, exp):
    m = exp > 0
    return float(((obs[m] - exp[m]) ** 2 / exp[m]).sum())


def analyse(calls, groups, rng):
    """groups: dict name -> letters (partition of the alphabet)."""
    names = list(groups)
    lut = {ch: i for i, (n, ls) in enumerate(groups.items()) for ch in ls}
    K = len(names)
    G = np.array([[lut[c] for c in g] for g, _ in calls])          # n x L, class index per position
    pos = np.array([p for _, p in calls])
    n, L = G.shape
    obs = np.bincount(G[np.arange(n), pos], minlength=K).astype(float)
    bg = np.bincount(G.ravel(), minlength=K).astype(float) / (n * L)
    # null A: uniform position
    expA = bg * n
    statA = chi2(obs, expA)
    simA = np.empty(N_SIM)
    for s in range(N_SIM):
        p = rng.integers(0, L, size=n)
        simA[s] = chi2(np.bincount(G[np.arange(n), p], minlength=K).astype(float), expA)
    # null B: keep positions, shuffle genomes among calls
    simB_counts = np.empty((N_SIM, K))
    for s in range(N_SIM):
        perm = rng.permutation(n)
        simB_counts[s] = np.bincount(G[perm, pos], minlength=K)
    expB = simB_counts.mean(axis=0)
    statB = chi2(obs, expB)
    simB = np.array([chi2(c, expB) for c in simB_counts])
    return {"n": n, "names": names, "obs": obs, "bg_share": bg, "expA": expA, "expB": expB,
            "chi2_A": statA, "p_A": float((1 + (simA >= statA - 1e-9).sum()) / (1 + N_SIM)),
            "chi2_B": statB, "p_B": float((1 + (simB >= statB - 1e-9).sum()) / (1 + N_SIM)), "df": K - 1}


def main() -> None:
    rng = np.random.default_rng(0)
    out = []
    P = out.append
    P("Offline: is the chosen position predicted by the letter that sits there? (no model calls; calls that fell back are excluded)\n")
    P("| source | calls used | letters: chi2 (df 19) vs A (uniform position), MC p | letters: chi2 vs B (positions kept, content shuffled), MC p | classes: chi2 (df 3) vs A, p | classes: chi2 vs B, p |")
    P("|---|---|---|---|---|---|")
    detail = []
    for label, fname in SOURCES:
        calls = read_calls(fname)
        r_l = analyse(calls, {a: a for a in ALPHA}, rng)
        r_c = analyse(calls, CLASSES, rng)
        P(f"| {label} | {r_l['n']} | {r_l['chi2_A']:.1f}, p={r_l['p_A']:.3f} | {r_l['chi2_B']:.1f}, p={r_l['p_B']:.3f} | "
          f"{r_c['chi2_A']:.2f}, p={r_c['p_A']:.3f} | {r_c['chi2_B']:.2f}, p={r_c['p_B']:.3f} |")
        detail.append((label, r_l, r_c))
    P(f"\nMonte Carlo p-values, {N_SIM} draws each (floor {1 / (1 + N_SIM):.1e}). Null A = position uniform over the genome; "
      "null B = the model's own positions kept but paired with genomes from other calls. Expected count per letter is ~5 at n=100 "
      "and ~20 at n=400, so a modest content preference would not be detectable in the n=100 rows.\n")
    P("### Residue classes at the chosen position\n")
    P("| source | class | share of all genome letters | share at chosen positions | expected share under B |")
    P("|---|---|---|---|---|")
    for label, _, rc in detail:
        for i, nm in enumerate(rc["names"]):
            P(f"| {label} | {nm} | {100 * rc['bg_share'][i]:.1f}% | {100 * rc['obs'][i] / rc['n']:.1f}% | {100 * rc['expB'][i] / rc['n']:.1f}% |")
    P("\n### Letter-level detail (share of all genome letters / share at chosen positions / expected under B)\n")
    for label, rl, _ in detail:
        P(f"**{label}**, {rl['n']} calls:\n")
        P("| letter | " + " | ".join(rl["names"]) + " |")
        P("|---|" + "---|" * len(rl["names"]))
        P("| available (%) | " + " | ".join(f"{100 * b:.1f}" for b in rl["bg_share"]) + " |")
        P("| at chosen position (count) | " + " | ".join(f"{int(o)}" for o in rl["obs"]) + " |")
        P("| expected under B (count) | " + " | ".join(f"{e:.1f}" for e in rl["expB"]) + " |")
        P("")
    text = "\n".join(out) + "\n"
    (RAW / "model_position_vs_letter_tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
