"""One-run sanity check of esmfold/tm_fitness.py -> results/raw/tm_fitness_sanity.json.

  a. the reference's own sequence, scored against the reference: expect high
  b. 6 uniform-random genomes of length 63: expect low
  c. the same genome scored twice: must give the same number (done for the
     reference sequence AND for the first random genome)
  d. per-call wall time, split into predictor time and TM-align time

Calls run in that order, so call 1 is the cold call (first CUDA work after the
model load); every call's timing is recorded individually and the summary
gives the mean/median both with and without call 1. The model-load time is
recorded separately and is not part of any per-call time.

Decision rule, fixed before the run: the fitness can tell good from bad only
if (a) is above every (b). The verdict fields record that comparison; nothing
here is thresholded beyond it.

Refuses to run if any process is using the GPU (the timings and the ~13.7GB
model would both be compromised).
"""

import argparse
import hashlib
import json
import logging
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import transformers

from esmfold import fitness as esm_fitness
from esmfold import tm_fitness as tf

ROOT = Path(__file__).resolve().parent.parent
GENOME_LEN = 63
N_RANDOM = 6


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as exc:
        return f"<call failed: {exc!r}>"


def gpu_snapshot() -> dict:
    return {
        "gpu": _run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                     "--format=csv,noheader"]),
        "compute_apps": _run(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
                              "--format=csv,noheader"]) or "(none)",
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=str(ROOT / "results" / "raw" / "tm_fitness_sanity.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    gpu_before = gpu_snapshot()
    print(f"gpu before: {gpu_before}")
    if gpu_before["compute_apps"] != "(none)":
        raise SystemExit("GPU is in use by another process; not running")

    ref_seq = tf.reference_sequence()
    env = {
        "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
        "tmalign_bin": str(tf.TMALIGN_BIN), "tmalign_sha256": sha256(tf.TMALIGN_BIN),
        "tmalign_source_sha256": sha256(ROOT / "esmfold" / "tmalign" / "TMalign.cpp"),
        "reference_pdb": str(tf.REFERENCE_PDB.relative_to(ROOT)), "reference_pdb_sha256": sha256(tf.REFERENCE_PDB),
        "reference_length": len(ref_seq), "reference_sequence": ref_seq,
        "reference_sequence_equals_fitness_py_TARGET_SEQ": ref_seq == esm_fitness.TARGET_SEQ,
        "git_head": _run(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
    }

    t0 = time.perf_counter()
    fit = tf.TMFitness()  # loads ESMFold once
    model_load_s = time.perf_counter() - t0
    print(f"model load: {model_load_s:.1f}s  reference length {fit.reference_length}")

    rng = random.Random(args.seed)
    randoms = ["".join(rng.choice(sorted(tf.CANONICAL)) for _ in range(GENOME_LEN)) for _ in range(N_RANDOM)]
    plan = [("a_reference", ref_seq)] + [(f"b_random_{i}", g) for i, g in enumerate(randoms)]
    plan += [("c_repeat_of_a_reference", ref_seq), ("c_repeat_of_b_random_0", randoms[0])]

    rows = []
    for label, genome in plan:
        r = fit.score_detailed(genome)
        rows.append({"call": len(rows) + 1, "label": label, "genome": genome, "tm_score": r.tm_score,
                     "aligned_length": r.aligned_length, "rmsd": r.rmsd, "genome_length": r.genome_length,
                     "predictor_s": r.predictor_s, "tmalign_s": r.tmalign_s})
        print(f"call {rows[-1]['call']:>2} {label:<26} TM={r.tm_score:.5f} aligned={r.aligned_length:>2} rmsd={r.rmsd:.2f} "
              f"predictor={r.predictor_s:.3f}s tmalign={r.tmalign_s:.3f}s", flush=True)

    by = {r["label"]: r for r in rows}
    b_scores = [by[f"b_random_{i}"]["tm_score"] for i in range(N_RANDOM)]
    a_score = by["a_reference"]["tm_score"]

    def timing(rs):
        p, t = [r["predictor_s"] for r in rs], [r["tmalign_s"] for r in rs]
        return {"n_calls": len(rs), "predictor_s_mean": statistics.mean(p), "predictor_s_median": statistics.median(p),
                "tmalign_s_mean": statistics.mean(t), "tmalign_s_median": statistics.median(t),
                "total_s_mean": statistics.mean(x + y for x, y in zip(p, t))}

    summary = {
        "a_reference_self_score": a_score,
        "b_random_scores": b_scores, "b_min": min(b_scores), "b_max": max(b_scores), "b_mean": statistics.mean(b_scores),
        "a_minus_b_max": a_score - max(b_scores), "a_above_every_b": a_score > max(b_scores),
        "c_repeat_reference": {"first": a_score, "second": by["c_repeat_of_a_reference"]["tm_score"],
                               "identical": a_score == by["c_repeat_of_a_reference"]["tm_score"]},
        "c_repeat_random_0": {"first": b_scores[0], "second": by["c_repeat_of_b_random_0"]["tm_score"],
                              "identical": b_scores[0] == by["c_repeat_of_b_random_0"]["tm_score"]},
        "c_aligned_length_and_rmsd_also_identical": all(
            (by[x]["aligned_length"], by[x]["rmsd"]) == (by[y]["aligned_length"], by[y]["rmsd"])
            for x, y in (("a_reference", "c_repeat_of_a_reference"), ("b_random_0", "c_repeat_of_b_random_0"))),
        "d_timing_all_calls": timing(rows), "d_timing_excluding_call_1": timing(rows[1:]),
    }
    print("\n" + json.dumps(summary, indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gpu_after = gpu_snapshot()
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "genome_len": GENOME_LEN, "model_load_s": model_load_s, "env": env,
                   "gpu_before": gpu_before, "gpu_after": gpu_after, "calls": rows, "summary": summary}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
