"""Does the mutate operator read the letter at the position it changes? Offline, from existing call logs; no new calls.

For every mutate/position attempt, each response line names (position, NEW letter). A line "repeats" when NEW equals the
letter already at that position, which the prompt forbids. If the model never consulted the current letter, its NEW
letters would collide with it at a chance rate. Two chance baselines, each computed within one data set:

  marginal  for each line, the probability that a letter drawn from this data set's own NEW-letter distribution equals the
            actual letter at that position; summed over lines. Keeps the model's letter preferences, ignores positions.
  shuffle   each attempt's chosen (position, NEW) lines are re-paired with the sequence of another attempt in the same
            data set (positions beyond that sequence's length dropped), 200 random pairings. Keeps letter preferences,
            positional preferences and the joint (position, letter) choices; breaks only the link to the actual sequence.

ratio = observed repeats / expected repeats. Near 1: the choice behaves as if the current letter were not read. Below 1:
the model avoids the current letter. Above 1: it is drawn toward it. Primary: first attempts only (a retry follows a
hint that says the previous answer repeated a letter, so retries are not independent). All attempts reported as
secondary. Per-seed values for the GA arms, so the project's every-seed rule can be applied.

  python experiments/analyse_blind_choice.py   ->  results/raw/blind_choice.json, results/raw/blind_choice_tables.md
"""

import collections
import glob
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
POS = re.compile(r"POSITION\s*:\s*(\d+)\s*,\s*NEW\s*:\s*([A-Za-z])", re.I)
SEQ = re.compile(r"positions 0-\d+\): ([A-Z ]+)")
N_SHUFFLE = 200


def ga_logs(arm: str, seed: int) -> list[Path]:
    if arm == "gemma4:12b arm C":
        return [ROOT / json.load(open(RAW / f"sequence_ga_cmp_C_seed{seed}.json"))["llm_call_log"]]
    if arm == "gemma4:12b arm F":
        return [Path(p) for p in glob.glob(str(RAW / f"llm_operator_calls_fitprompt_F_seed{seed}_*.jsonl"))]
    if arm == "qwen2.5:7b arm C":
        return [ROOT / json.load(open(RAW / "armC_qwen2.5_7b" / f"sequence_ga_cmp_C_seed{seed}.json"))["llm_call_log"]]
    raise ValueError(arm)


GA = {"gemma4:12b arm C": range(5), "gemma4:12b arm F": range(3), "qwen2.5:7b arm C": range(5)}
PROBES = {  # isolated probes: random sequences, k = 1 (MODEL_HETEROGENEITY_STEP1 / STEP4)
    "gemma4:12b probe": ["llm_operator_calls_hetero_gemma4_12b.jsonl", "llm_operator_calls_lensweep_len40_gemma4_12b.jsonl",
                         "llm_operator_calls_lensweep_len80_gemma4_12b.jsonl"],
    "qwen2.5:7b probe": ["llm_operator_calls_hetero_qwen2.5_7b.jsonl", "llm_operator_calls_lensweep_len40_qwen2.5_7b.jsonl",
                         "llm_operator_calls_lensweep_len80_qwen2.5_7b.jsonl"],
    "mistral:7b probe": ["llm_operator_calls_hetero_mistral_7b.jsonl"],
    "llama3.2:3b probe": ["llm_operator_calls_hetero_llama3.2_3b.jsonl"],
}


def attempts(paths, first_only: bool) -> list[tuple[str, list[tuple[int, str]]]]:
    out = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("op") != "mutate" or "prompt" not in r or "response" not in r:
                continue
            if first_only and r.get("attempt", 0) != 0:
                continue
            m = SEQ.search(r["prompt"])
            if not m:
                continue
            g = "".join(m.group(1).split())
            lines = [(int(a), b.upper()) for a, b in POS.findall(r["response"]) if int(a) < len(g)]
            if lines:
                out.append((g, lines))
    return out


def measure(att, rng) -> dict:
    n_lines = sum(len(ls) for _, ls in att)
    observed = sum(g[a] == b for g, ls in att for a, b in ls)
    newc = collections.Counter(b for _, ls in att for _, b in ls)
    tot = sum(newc.values())
    marginal = sum(newc[g[a]] / tot for g, ls in att for a, _ in ls)
    rates = []
    seqs = [g for g, _ in att]
    for _ in range(N_SHUFFLE):
        rng.shuffle(seqs)
        hit = n = 0
        for g2, (_, ls) in zip(seqs, att):
            for a, b in ls:
                if a < len(g2):
                    n += 1
                    hit += g2[a] == b
        rates.append(hit / n if n else 0.0)
    rates.sort()
    shuffle_rate = sum(rates) / len(rates)
    return {"attempts": len(att), "lines": n_lines, "observed": observed, "observed_rate": observed / n_lines,
            "marginal_expected": marginal, "marginal_rate": marginal / n_lines, "ratio_marginal": observed / marginal,
            "shuffle_rate": shuffle_rate, "shuffle_rate_95": [rates[int(0.025 * N_SHUFFLE)], rates[int(0.975 * N_SHUFFLE) - 1]],
            "ratio_shuffle": (observed / n_lines) / shuffle_rate if shuffle_rate else None}


def main() -> None:
    rng = random.Random(0)
    res = {}
    for first_only in (True, False):
        key = "first_attempts" if first_only else "all_attempts"
        res[key] = {}
        for arm, seeds in GA.items():
            per_seed = {s: measure(attempts(ga_logs(arm, s), first_only), rng) for s in seeds}
            pooled = measure(attempts([p for s in seeds for p in ga_logs(arm, s)], first_only), rng)
            res[key][arm] = {"pooled": pooled, "per_seed": per_seed}
        for name, files in PROBES.items():
            res[key][name] = {"pooled": measure(attempts([RAW / f for f in files], first_only), rng)}
    json.dump(res, open(RAW / "blind_choice.json", "w"), indent=1)

    def row(name, m, seed="pooled"):
        lo, hi = m["shuffle_rate_95"]
        return (f"| {name} | {seed} | {m['attempts']} | {m['lines']} | {m['observed']} ({m['observed_rate']:.3f}) | "
                f"{m['marginal_rate']:.3f} | {m['shuffle_rate']:.3f} [{lo:.3f}, {hi:.3f}] | **{m['ratio_marginal']:.2f}** | "
                f"**{m['ratio_shuffle']:.2f}** |")

    hdr = ("| data set | seed | attempts | lines | repeats observed (rate) | chance, marginal | chance, shuffle [95% range] | "
           "ratio vs marginal | ratio vs shuffle |\n|---|---|---|---|---|---|---|---|---|")
    md = ["# Blind-choice tables (generated by experiments/analyse_blind_choice.py)", ""]
    for key, title in (("first_attempts", "First attempts only (primary)"), ("all_attempts", "All attempts, retries included")):
        md += [f"## {title}", "", hdr]
        for name, d in res[key].items():
            md.append(row(name, d["pooled"]))
            for s, m in d.get("per_seed", {}).items():
                md.append(row(name, m, str(s)))
        md.append("")
    (RAW / "blind_choice_tables.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
