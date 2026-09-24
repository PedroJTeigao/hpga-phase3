"""Score every move-class move with ESM-2 150M's masked likelihood.

Read-only on the move-class results: this reads results/raw/move_class_bases_{arm}_seed{n}.json
and writes results/raw/esm2_likelihood_{arm}_seed{n}.json. It runs no GA and touches no run file.

The question it feeds (results/ESM2_LIKELIHOOD_SCREEN.md): does a protein language model's own
preference predict realised TM-score gain on moves that were already made? The moves were
produced by the deterministic move classes, for reasons unrelated to this model, so they are an
unbiased sample with respect to it.

Two quantities per move:

  dlogit  sum over CHANGED positions of [logit(new) - logit(old)], read from the BASE genome's
          masked context. This is what an operator would consult when proposing a substitution.
          Defined only where base and child align position-for-position, i.e. the substitution
          arms (S1, S2, S3, B). The softmax normaliser cancels in the difference, so restricting
          to the 20 canonical letters does not affect it.

  dPLL    PLL(child) - PLL(base), where PLL(s) = sum_i log p(s[i] | s masked at position i),
          renormalised over the 20 canonical letters. Needs no positional correspondence, so it
          covers every arm including S4's indels, all of which change length. This is the usual
          ESM variant-effect score.

Cost: 6,446 unique genomes (3,144 bases + 3,978 children, deduplicated), 6,086 s on an idle T4.
Run it on an idle GPU:

    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader   # must be empty
    /scratch/pcanaste/venv/bin/python experiments/score_esm2_likelihood.py

--reuse <dump.json> rebuilds the per-arm/seed outputs from a previous run's flat dump, skipping
the GPU scoring. That is how the committed files were produced from the first run's output
without spending the 101 minutes again. The one-genome native calibration is still computed on
the GPU either way (0.4 s).

NOT saved: the per-position (L, 20) log-probability matrices. They are ~7.5M floats and would
dominate the repo. A new per-position analysis therefore needs a re-run.
"""

import argparse
import glob
import json
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
MODEL = "facebook/esm2_t30_150M_UR50D"
CANON = "ACDEFGHIKLMNPQRSTVWY"
IDX = {c: i for i, c in enumerate(CANON)}
ROW_BUDGET = 1024  # masked rows per forward call; ~14.9 GB peak on a T4, so run on an idle card
CHUNK = 400  # genomes per progress line
# esmfold/fitness.py TARGET_SEQ: 7UR7 chain A, the fitness reference, for calibration only.
NATIVE = "SEVKELLEEFLKRNKPVRIHHKNGEEIKVRITHIGEDTVEFELNGRTHRINIKDILDVKEWLE"


def load_model():
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForMaskedLM.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
    cid = torch.tensor([tok.convert_tokens_to_ids(c) for c in CANON], device="cuda")
    return torch, tok, model, cid


def score_many(seqs, torch, tok, model, cid):
    """seq -> (L, 20) log-probs over the canonical letters, position i masked in row i."""
    out, pend, rows = {}, [], 0

    def flush(group):
        if not group:
            return
        maxlen = max(len(tok(s)["input_ids"]) for s in group)
        ids, ams, spans = [], [], []
        for s in group:
            enc = tok(s)["input_ids"]
            row0 = enc + [tok.pad_token_id] * (maxlen - len(enc))
            am = [1] * len(enc) + [0] * (maxlen - len(enc))
            spans.append((s, len(ids), len(s)))
            for i in range(len(s)):
                r = list(row0)
                r[1 + i] = tok.mask_token_id  # +1 for BOS
                ids.append(r)
                ams.append(am)
        with torch.no_grad():
            lg = model(input_ids=torch.tensor(ids, device="cuda"),
                       attention_mask=torch.tensor(ams, device="cuda")).logits
        for s, start, L in spans:
            pos = torch.arange(L, device="cuda")
            sel = lg[start:start + L][pos, 1 + pos][:, cid]
            out[s] = torch.log_softmax(sel, dim=-1).float().cpu()

    for s in seqs:
        pend.append(s)
        rows += len(s)
        if rows >= ROW_BUDGET:
            flush(pend)
            pend, rows = [], 0
    flush(pend)
    return out


def read_moves():
    moves = []
    for f in sorted(glob.glob(str(RAW / "move_class_bases_*.json"))):
        d = json.load(open(f))
        for m in d["moves"]:
            moves.append(dict(arm=d["arm"], seed=d["seed"], step=m["step"],
                              base=m["base"], child=m["child"],
                              base_fitness=m["base_fitness"], child_fitness=m["child_fitness"],
                              gain=m["child_fitness"] - m["base_fitness"],
                              improved=m["child_fitness"] > m["base_fitness"]))
    return moves


def derive(moves, scores):
    """Attach the likelihood quantities to each move, in place."""
    def pll(lp, s):
        return sum(lp[i, IDX[s[i]]].item() for i in range(len(s)))

    for m in moves:
        lb, lc = scores[m["base"]], scores[m["child"]]
        m["pll_base"], m["pll_child"] = pll(lb, m["base"]), pll(lc, m["child"])
        m["dPLL"] = m["pll_child"] - m["pll_base"]
        m["mean_ll_base"] = m["pll_base"] / len(m["base"])
        m["mean_ll_child"] = m["pll_child"] / len(m["child"])
        if len(m["base"]) == len(m["child"]):
            ch = [i for i in range(len(m["base"])) if m["base"][i] != m["child"][i]]
            m["n_changed"] = len(ch)
            m["dlogit"] = sum(lb[i, IDX[m["child"][i]]].item() - lb[i, IDX[m["base"][i]]].item()
                              for i in ch)
        else:  # indel: no positional correspondence, so dlogit is undefined by construction
            m["n_changed"], m["dlogit"] = None, None


def write_per_arm_seed(moves, native_ll, elapsed_s, reused_from):
    keep = ("step", "base_fitness", "child_fitness", "gain", "improved", "len_base", "len_child",
            "pll_base", "pll_child", "dPLL", "mean_ll_base", "mean_ll_child", "n_changed", "dlogit")
    groups = defaultdict(list)
    for m in moves:
        m["len_base"], m["len_child"] = len(m["base"]), len(m["child"])
        groups[(m["arm"], m["seed"])].append({k: m[k] for k in keep})
    written = []
    for (arm, seed), rows in sorted(groups.items()):
        p = RAW / f"esm2_likelihood_{arm}_seed{seed}.json"
        json.dump({
            "arm": arm, "seed": seed, "model": MODEL,
            "source": f"results/raw/move_class_bases_{arm}_seed{seed}.json",
            "note": ("masked-likelihood screen, measurement only: no GA was run and no move here "
                     "was proposed by the model. dlogit is null for length-changing (indel) moves."),
            "canonical_renormalised": True,
            "native_reference_mean_ll": native_ll,
            "scoring_seconds_total": elapsed_s,
            "rebuilt_from_dump": reused_from,
            "moves": rows,
        }, open(p, "w"), indent=1)
        written.append((p, len(rows)))
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", help="flat scored dump from a previous run; skips GPU scoring")
    ap.add_argument("--dump", help="also write the flat scored dump here")
    args = ap.parse_args()

    t0 = time.perf_counter()
    moves = read_moves()
    print(f"moves: {len(moves)}")
    torch, tok, model, cid = load_model()

    if args.reuse:
        # Matched by position, not by (arm, seed, base, child): a run can produce the same
        # base->child pair twice, so that tuple is not unique (3 collisions across the 3,990).
        # The dump was written from this same read_moves() traversal, so order is identical;
        # verified element-wise below rather than assumed.
        cached = json.load(open(args.reuse))
        if len(cached) != len(moves):
            raise SystemExit(f"dump has {len(cached)} moves, expected {len(moves)}")
        for m, c in zip(moves, cached):
            if (m["arm"], m["seed"], m["base"], m["child"]) != (c["arm"], c["seed"], c["base"], c["child"]):
                raise SystemExit("dump is not in the same order as the move-class files")
            for k in ("pll_base", "pll_child", "dPLL", "mean_ll_base", "n_changed", "dlogit"):
                m[k] = c[k]
            m["mean_ll_child"] = m["pll_child"] / len(m["child"])
        elapsed = None
        print(f"reused {len(cached)} scored moves from {args.reuse} (GPU scoring skipped)")
    else:
        need = sorted({m["base"] for m in moves} | {m["child"] for m in moves})
        print(f"unique genomes to score: {len(need)}")
        scores, done = {}, 0
        for i in range(0, len(need), CHUNK):
            scores.update(score_many(need[i:i + CHUNK], torch, tok, model, cid))
            done += len(need[i:i + CHUNK])
            print(f"  scored {done}/{len(need)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
        derive(moves, scores)
        elapsed = round(time.perf_counter() - t0, 1)
        if args.dump:
            json.dump(moves, open(args.dump, "w"))

    nat = score_many([NATIVE], torch, tok, model, cid)[NATIVE]
    native_ll = sum(nat[i, IDX[NATIVE[i]]].item() for i in range(len(NATIVE))) / len(NATIVE)
    print(f"native 7UR7 chain A mean per-position LL: {native_ll:.4f}")

    for p, n in write_per_arm_seed(moves, native_ll, elapsed, args.reuse):
        print(f"wrote {p.relative_to(ROOT)}  ({n} moves)")


if __name__ == "__main__":
    main()
